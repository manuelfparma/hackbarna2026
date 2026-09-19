# Agent

Movie recommendation assistant built with [LangGraph](https://langchain-ai.github.io/langgraph/), as a **supervisor over a subagent**: `OrquestratorAgent` (the supervisor) embeds `RecommendationAgent`'s compiled graph as a single node. Both pause on `interrupt()` to collect input from the user; the subagent's interrupts propagate up through the supervisor, so the supervisor still mediates every turn without having to know why the subagent is asking.

## Graph: OrquestratorAgent (supervisor)

```
welcome --> conversation --> goodbye --> END
(interrupt)   (subgraph:
             RecommendationAgent)
```

- **welcome** — `interrupt()` asking what the user feels like watching. The one question this layer asks directly; everything after the first message is the subagent's job.
- **conversation** — *is* `RecommendationAgent.graph`, not a function that calls it. The node is the compiled subgraph itself, which is what lets it pause mid-node instead of running to completion before the supervisor sees it.
- **goodbye** — closes the session with a farewell once the subagent reports a `choice`.

## Graph: RecommendationAgent (subagent)

```
classify --> {theme_recommendation | feedback | direct_request
              | social} --> reply --> turn ------------------+
                   ^                                           |
                   +-------------------------------------------+
                              (no pick yet)
```

- **classify** — extracts this turn's entities and a criteria action (`add`, `replace`, `reset`, or `keep`), merges them into `search_criteria`, then resolves the capability. A feedback turn that adds a genre or theme is routed back through retrieval with the full accumulated brief.
- **theme_recommendation** / **feedback** / **direct_request** / **social** — one capability per route (`nodes/`). Generic “recommend a movie” intents share theme search with explicit mood asks. Each produces structured result data; none owns user-facing prose.
- **reply** — the single conversational voice. It receives the raw result, entities, feedback, current shortlist, and the latest six turns of history, then writes a brief `response`. A deterministic fallback keeps the graph usable if this LLM call fails.
- **turn** — `interrupt()` showing the reply and, when `show_options` is true, asking the user to pick a number or say what to change. A valid number ends the graph with `choice` set; anything else feeds back into `classify`.

`movies` and `show_options` are deliberately separate. Small-talk and failed lookup detours retain the selectable shortlist in state without redisplaying an unrelated list or call-to-action.

### Persistent search criteria

`entities` describes only the latest utterance. `search_criteria` is the normalized, session-long search brief. Language cues control how the latest entities change it:

- “also”, “and”, “with” → add
- “instead”, “rather”, or an unqualified new direction → replace
- “start over”, “forget that”, “something completely different” → reset
- greetings, thanks, and factual questions → keep

Explicit genres are canonicalized to catalog labels and applied as a strict AND filter. `Comedy` followed by “also Action” therefore queries movies containing both genres. Free-text moods and themes remain semantic signals; when both are present, theme ranking runs only over movies that satisfy the genre filter. Empty intersections are reported instead of silently dropping a constraint.

## How the interrupt loop works

Each `interrupt(...)` call pauses the graph and hands its argument back to the caller as `event["__interrupt__"][0].value`. The caller shows it to the user, gets an answer, and resumes with `Command(resume=answer)` — execution continues right where it stopped, with `interrupt()` now returning that answer.

This requires a checkpointer (`InMemorySaver`) to save/restore state between pause and resume, and a `thread_id` to identify the conversation.

### Propagation from subagent to supervisor

`RecommendationAgent`'s `turn` node raises its own `interrupt()`. Because `conversation` is the *subgraph itself* as a node, that interrupt is not swallowed — it surfaces as the supervisor's own `event["__interrupt__"]`, and a `Command(resume=...)` sent to the supervisor's graph lands back inside the subagent's `turn` node, exactly where it paused. Two things make this work together:

1. **The node is the compiled subgraph**, not a wrapper function (`orchestrator.py`: `graph.add_node("conversation", self.recommendation_agent.graph)`).
2. **The subagent has no checkpointer of its own** — it's constructed as `RecommendationAgent(llm, checkpointer=None)`, so LangGraph hands it the supervisor's checkpointer. Both then live under the same `thread_id`, which is what lets a resume sent to the supervisor find the paused node inside the subagent. (Standalone, `RecommendationAgent()` defaults to its own `InMemorySaver`, so it still runs on its own — see `nodes/../__main__` and `test_scripts/test_local_llm.py`.)

One consequence worth knowing: **while the subagent is paused, the supervisor's own state doesn't yet contain anything the subagent has written** — a subgraph's writes only land once its node returns, and mid-node it hasn't returned. The interrupt payload (built via `io/turn.py`'s `prompt(text, options)`) is therefore the only channel out during the pause; `OrquestratorAgent.pending_state()` reaches past that when needed (logging, a fallback if the shortlist came back empty) by reading the subagent's in-flight state with `get_state(subgraphs=True)`.

The supervisor's `State` is deliberately a **superset** of the subagent's (`search_criteria`, `result`, `history`, `show_options`, `response`, routing fields, plus its own `farewell`): LangGraph only propagates state keys the parent schema declares, so a narrower parent schema would silently drop everything the subagent produces once it *does* return.

```python
# orchestrator.py
event = self.graph.invoke({}, config)
while "__interrupt__" in event:
    print(event["__interrupt__"][0].value)
    event = self.graph.invoke(Command(resume=input("> ")), config)
```

`OrquestratorAgent.run()` is the terminal-only driving loop. The graph itself doesn't know it's talking to a terminal — swap that loop for the web API (`frontend/api.py`) without touching either graph's nodes.

## Why keep the supervisor, if the subagent owns the interrupts?

Because **the one who pauses is the one who needs the answer** — only `RecommendationAgent` knows when it needs to disambiguate a request or have its shortlist reviewed; that's domain knowledge about recommending movies. If the supervisor formulated those questions itself, it would have to model the subagent's internals, which is the coupling we're trying to avoid. Pushing the interrupt down to where the question originates is what keeps the supervisor thin, not what makes it redundant.

What stays at the supervisor layer, and can't live in the subagent:

- **Session lifecycle** — when the conversation starts and ends, the farewell.
- **I/O modality** — narration vs. rendering (`io/tts.py`, `io/stt.py`, `io/turn.py`'s `prompt()` contract) is a presentation decision, not a recommendation one.
- **Capability composition** — `RecommendationAgent` is the first subagent plugged into `conversation`. The next one (watchlist, "where can I stream this", trailers) becomes another node next to it, rather than being bolted onto the recommender's classifier where it doesn't belong.
- **Cross-cutting policy** — retries, fallbacks when a subagent's shortlist comes back empty, per-session logging — all things that should apply uniformly regardless of which subagent is running.

And it costs nothing to keep both layers: the subagent's interrupts reach the supervisor's `__interrupt__` and resume from the supervisor unchanged, as shown above — the supervisor mediates every turn exactly as before.

## Nodes / subagents

Each intent branch in `RecommendationAgent` is its own class under `nodes/`:

- `nodes/classifier.py` — `Classifier`, routes each message, extracts entities, and selects the criteria merge action.
- `nodes/criteria.py` — pure normalization, merge, summary, and semantic-query helpers for the persisted search brief.
- `nodes/theme_recommender.py` — `ThemeRecommender`, nearest-theme search via pgvector/Supabase, pre-filtered by all accumulated genres. Generic recommendation and unconstrained feedback use this path too. Similarity scores remain internal.
- `nodes/direct_request.py` — `DirectRequestHandler`, chains accumulated catalog filters; multiple genres use PostgreSQL array containment as a strict AND.
- `nodes/feedback.py` / `nodes/social.py` — package context for the reply layer without making their own prose-generation calls.
- `nodes/reply.py` — `ReplyComposer`, the shared conversational voice. Production gives this client temperature `0.4`; routing remains at temperature `0`.

`io/turn.py` holds `prompt(text, options)`, the shape every `interrupt()` payload takes — shared by both graphs so `text` (narrated) and `options` (rendered) stay a stable contract for whatever's driving the conversation (terminal loop, web API, TTS/STT).
