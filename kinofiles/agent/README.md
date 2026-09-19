# Agent

Movie recommendation orchestrator built with [LangGraph](https://langchain-ai.github.io/langgraph/). It drives a small state machine that pauses on `interrupt()` to collect input from the user, then resumes the graph with what they typed.

## Graph

```
user_input --> recommendation --> review --+--> goodbye --> END
                    ^                      |
                    +----------------------+
                 (feedback, loops back)
```

- **user_input** — `interrupt()` asking what the user feels like watching. Plain text for now; will be replaced/extended with other I/O subagents later.
- **recommendation** — calls the `Recommender` subagent (`sugagents/recommender.py`) with the request and the feedback collected so far, gets back a list of movies.
- **review** — `interrupt()` showing the movies. If the user picks a valid number, goes to `goodbye`. Otherwise, whatever they typed is appended to `feedback` and the graph loops back to `recommendation`.
- **goodbye** — closes the session with a farewell message once a movie is picked.

## How the interrupt loop works

Each `interrupt(...)` call pauses the graph and hands its argument back to the caller as `event["__interrupt__"][0].value`. The caller shows it to the user, gets an answer, and resumes the graph with `Command(resume=answer)` — execution continues right where it stopped, with `interrupt()` now returning that answer.

This requires a checkpointer (`InMemorySaver`) so the graph can save/restore its state between the pause and the resume, and a `thread_id` to identify the conversation.

See `OrquestratorAgent.run()` for the loop:

```python
event = self.graph.invoke({}, config)
while "__interrupt__" in event:
    print(event["__interrupt__"][0].value)
    event = self.graph.invoke(Command(resume=input("> ")), config)
```

`run()` is the terminal-only driving loop. The graph itself (`self.graph`) doesn't know it's talking to a terminal — swap that loop for a web/chat frontend without touching the nodes.

## Subagents

`sugagents/recommender.py` holds the `Recommender` class, injected into `OrquestratorAgent` as `self.recommender`. It currently wraps a single LLM call (Nebius, see `agent/llm.py`). When the team's search engine is ready, only its `recommend(request, feedback)` body needs to change — the signature stays the same.
