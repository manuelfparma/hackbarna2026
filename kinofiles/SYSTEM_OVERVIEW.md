# KinoFiles: current capabilities, conversations, and limitations

KinoFiles focuses on one loop: **discover a movie, refine the suggestions, ask
about an option, and choose it**. This guide describes the reduced feature set.
`UPGRADE_PLAN.md` is historical; its superseded tasks are not current capabilities.

## Entry points after the group-mediator integration

- **Default web/orchestrator flow:** asks for 1–4 names, collects each person's
  preferences, merges a group brief, and asks everyone for a numbered vote.
  Ties or no votes lead to refinement. The original group flow has a three-round
  cap and then picks the first available option, or a generic fallback label if
  no options exist. It is a staged voting workflow, not the solo free-form chat.
- **Standalone `RecommendationAgent`:** retains the conversational discovery,
  fact lookup, selection, and deterministic control behavior described below.
  Run it with `uv run python -m agent.recomedation_agent` from `kinofiles/`.
- Both share classification and retrieval helpers. Group field removals use
  classified actions; solo pre-classification reset handlers are not automatically
  used by the group graph. Group classifier failures retry the same stage.

The capability reference and examples below describe the standalone agent and
shared retrieval. Do not assume every solo command works during a group vote.
See `agent/README.md` for the group graph and merge policy.

## 1. Supported capabilities

| Capability | Example | Behavior |
|---|---|---|
| Genre recommendations | “Any comedy?” | Filters catalog records and orders direct matches by rating. |
| Combined/alternative genres | “Action comedy”; “comedy or romance” | Supports AND within genre groups and OR between groups. |
| Actor/director search | “Adam Sandler movies”; “movies by Nolan” | Case-insensitive partial-name matching. |
| Language filtering | “A French-language movie” | Filters stored language metadata. |
| Mood/topic discovery | “Something melancholy about family” | Uses semantic theme retrieval. |
| Plot identification | “An animated movie about a rat in Paris helping a chef” | Searches descriptions and suggests a likely title when confidence is sufficient. |
| Similar-to search | “Something like The Dark Knight” | Finds description neighbors of a resolved seed film. |
| Preference memory | “I love animation” | Records taste for the next search without immediately replacing the shortlist. |
| Seen-film exclusions | “I have seen Ratatouille” | Records seen titles for exclusion from later recommendation searches. |
| Refinements | “Also some action”; “make it darker” | Updates constraints or semantic feedback and searches again. |
| More suggestions | “What else?” | Reuses the search brief while excluding previously shown results. |
| Single-film facts | “What can you tell me about Love and Monsters?” | Looks up the film and answers using its catalog row. |
| Contextual questions | “What is it about?”; “who directed the second one?” | Uses focus/shortlist references, with clarification for ambiguous supported patterns. |
| Streaming lookup/filtering | “Is Ratatouille on Netflix?”; “comedies on Netflix” | Uses the stored **mock** streaming-service array. |
| Open browsing | “Surprise me” | In browse mode with an empty brief, samples eligible top-rated records. |
| Selection | `2`, “the second one”, or an exact displayed title | Picks from the shortlist and ends the conversation. |
| Natural acceptance | “I’ll watch Ratatouille” | Resolves the named choice; asks for clarification if necessary. |
| Exit | “Goodbye” | Ends the conversation without requiring a choice. |
| Search controls | “Start over”; “forget about 2020” | Resets the brief or removes a filter group. |
| Help | “What can you do?” | Returns static help, without querying catalog statistics. |

Most natural-language requests still depend on correct classification and entity
extraction. Deterministic handlers cover common control, reference, and factual
question patterns; they do not replace all language understanding.

## 2. Removed features

The application no longer provides:

- **Session watchlists:** no save/list workflow or watchlist state.
- **Two-film comparisons:** ask about one film at a time instead. Similar-to
  recommendations remain supported and are not a comparison operation.
- **Catalog analytics:** no record counts or inventory of available genres.
  General capability help remains available.
- **Studio filtering:** studio metadata may still exist in imported movie rows
  and single-film facts, but it is not a recommendation constraint.
- **Semantic content exclusions:** requests such as “nothing involving death”
  are not implemented. Explicit genre and actor/director exclusions remain.
- **Audience criteria:** no dedicated viewer/audience field is extracted,
  remembered, or appended to retrieval queries. “With my kids” does not imply
  Family, Animation, excluding Horror, or rating/runtime limits.

Requests classified as unsupported receive a deterministic explanation and leave
the existing brief unchanged. The agent must not claim a removed operation ran.
Detection of arbitrary unsupported paraphrases remains classifier-dependent.

Original user messages are not scrubbed: a viewer mention may still appear in
conversation history or the raw request embedded for semantic search, but it has
no separate criterion or filter. “A family movie” or “animation, no horror” still
supplies explicit genre constraints, not a child-suitability guarantee.

## 3. Filters and result ordering

### Filter semantics

| Constraint | Interpretation |
|---|---|
| Multiple genres | AND by default; all requested genres must be present. |
| Genre OR | “Comedy or romance” accepts either genre. |
| Multiple actors/directors/languages | AND across values; people use partial-name matching. |
| Streaming | Slugs: `netflix`, `hbo`, `prime`, `appletv`. Multiple required services use AND. |
| Exact years | `[2020]` means 2020; multiple exact years are alternatives. |
| Decades | “90s”/“1990s” becomes 1990–1999. |
| Year bounds | “After 2015” means 2016 onward; “before 2020” means through 2019. |
| “Classic” / “old” | Fixed cutoff: 1980 or earlier. |
| “Recent” / “new” / “modern” | Fixed cutoff: 2015 onward, not a rolling window. |
| Runtime | Integer-minute bounds; “under two hours” means at most 119 minutes. |
| Duration labels | Short: under 100; Standard: 100–129; Long: at least 130; Epic: at least 150 minutes. |
| Minimum rating | Catalog 0–5 scale; the classifier is instructed to interpret “highly rated” as at least 4.0. |
| Negative genres | “No horror or war” excludes either genre. |
| Negative people | Supports explicit actor/director exclusions. |

Hard filters are shared across direct, theme, and description retrieval. Once
correctly extracted, constraints are not silently relaxed to manufacture results.
Unsupported negative content labels are not expanded into semantic exclusions.

### Why the same comedies can keep appearing

The direct catalog path effectively runs:

```sql
SELECT name
FROM movies
WHERE genres @> ARRAY['Comedy']
ORDER BY rating DESC
LIMIT 8;
```

It is not random. Repeating the same ordinary direct request can return the same
eight records, because shown-title exclusion is applied to `more` and certain
feedback turns, not every fresh search. Use “What else?” in the same session to
request another batch.

| Retrieval path | Ordering |
|---|---|
| Direct catalog search | Rating descending, up to eight matches. |
| Theme search | Weighted theme overlap, then rating, then title. |
| Plot/similar-to search | Description-embedding similarity. |
| Empty-brief browse | Random sample of up to eight from up to 200 eligible top-rated records. |
| Partial-title resolution fallback | Can choose the highest-rated substring match if exact resolution fails. |

Overall rating is not a measure of how funny a film is. Comedy may be only one
of several genre tags. The import can also contain series/specials; there is no
movie-versus-series filter in this flow.

## 4. Memory and search controls

The latest message’s `entities` and the accumulated `search_criteria` are separate.

- `add` merges list values and updates supplied scalar limits.
- `replace` discards the previous brief in favor of newly extracted criteria.
- `keep` preserves the brief.
- `reset` clears it, optionally followed by a new request.
- `remove` clears a group of related fields.

Taste is session-local search memory, not a trained user profile. For example,
liking an actor can become a hard actor constraint. A preference acknowledgment
is not itself a new recommendation search.

### Deterministic controls

These examples are handled before the classifier:

```text
Start over
Start over. Let's find another movie
Start over. I want a comedy
Never mind
nevermind
No, forget about it
Forget about this, I want to search for other movies
Please reset all filters
Forget about 2020
Forget about the 90s
Remove the date filter
Any year is fine
The year doesn't matter
Forget the runtime
Remove the actor filter
```

A recognized reset/removal can work in an initialized session even during a
classifier outage. A new request appended to it may still need classification.
If that call fails, the explicit state change remains applied and the reply says
that a new search did not run.

### What gets cleared

| State | Full search reset | Remove the date filter |
|---|---|---|
| Search filters | All | Date fields only |
| Actor and genre preferences | Clears | Retains |
| Shortlist and focused movie | Clears | Clears |
| Shown-title tracking | Clears | Clears |
| Cached search route/text | Clears | Clears |
| Feedback and recent history | Clears | Clears |
| Seen-film memory and resolution cache | Retains | Retains |
| Conversation | Remains open | Remains open |

Date removal clears `years`, `time_periods`, `year_min`, and `year_max` together.
Runtime removal clears `duration`, `minute_min`, and `minute_max`. Other supported
removal groups are rating, genres, actors, directors, streaming, languages,
and themes.

These are group removals: “forget about 2020” clears the date group, not just one
value from a multi-year list. A pure control command acknowledges the change;
follow it with a search request or “What else?” to retrieve new options.

A social or factual detour may hide the shortlist without deleting it, so a valid
number can still select a stored option. Reset/removal clears the shortlist, so
old numbers cannot select it. A failed direct search can retain an older hidden
shortlist; use an explicit reset to discard it.

Start a new chat/thread to discard all session memory, including seen films.
Restarting the backend also loses its in-memory sessions.

## 5. Example conversations

These are illustrative flows, not guaranteed verbatim replies, rankings, or
provider availability. Bracketed text describes an expected action or UI result.

### A. Taste, search, refine, choose

```text
User: I love animation films.
Assistant: Got it — I have noted those preferences for your next search.
User: What should I watch?
Assistant: [Shows Animation matches.]
User: Also some comedy.
Assistant: [Shows films with both genres.]
User: What else?
Assistant: [Shows additional eligible titles, or reports no further matches.]
User: the second one
Assistant: Enjoy [the second option]! Goodbye.
```

### B. Identify a film from its plot

```text
User: An animated movie about a rat in Paris helping a chef.
Assistant: Did you mean Ratatouille? That is the closest plot match I found.
User: What is it about?
Assistant: [Answers from the focused film's catalog description.]
User: Ratatouille
Assistant: Enjoy Ratatouille! Goodbye.
```

Plot identification is a suggestion, not certainty. Weak matches prompt another
detail rather than a confident title claim.

### C. Remove the year without losing the rest

```text
User: I'm with my kids. We want a comedy with Adam Sandler from 2020.
Assistant: [May report no matches.]
User: Forget about 2020.
Assistant: I've cleared the date filter and kept your other preferences.
           What would you like to search for?
User: What else?
Assistant: [Searches with Adam Sandler and Comedy, without the year filter.]
```

No audience preference is stored. No Family/Animation/Horror rules are inferred,
and no child-suitability claim is established by the recommendations.

### D. Start fresh and ask a factual question

```text
User: Start over. Let's find another movie.
Assistant: I've cleared your search filters and previous suggestions.
           What would you like to watch?
User: What can you tell me about Love and Monsters?
Assistant: [Looks up that film and answers from its catalog row.]
```

Factual lookup does not constrain the named film by the recommendation brief.
If a title cannot be resolved, a successful description fallback may offer an
explicitly labeled guess; otherwise it reports the failure.

### E. Similar films, filters, and availability

```text
User: Something like The Dark Knight.
Assistant: [Finds description neighbors, excluding the seed.]
User: Start over. Show me comedies or romance films on Netflix,
      under two hours, released after 2015. No horror.
Assistant: [Searches for (Comedy OR Romance), Netflix, minute <= 119,
            year >= 2016, and NOT Horror.]
User: Is Ratatouille on Netflix?
Assistant: [Reports the mock service assignment for that film.]
```

Streaming data is artificial, not a live provider lookup.

### F. Ambiguity and unsupported requests

```text
User: Recommend a thriller.
Assistant: [Shows several options without a single focused title.]
User: What is it about?
Assistant: Which movie do you mean?
User: What is Inception about?
Assistant: [Looks up Inception.]
User: Compare Inception and Interstellar.
Assistant: That operation is not supported. [Offers help with retained features.]
```

An unsupported request does not change the brief or run the removed operation.
For supported movie facts, name one film explicitly when the reference is unclear.

### G. Classification failure

```text
User: [A request requiring classification when the provider call fails.]
Assistant: I could not interpret this request. I have not run a new search.
           Please try again.
User: Start over.
Assistant: I've cleared your search filters and previous suggestions.
           What would you like to watch?
```

Failures are logged separately from genuine social messages. Social, control,
help, unsupported, and error replies do not use an LLM to invent a successful
operation or repeat an old search failure.

## 6. Architecture and retrieval

```text
Reflex UI / group terminal
  -> OrquestratorAgent: welcome -> collect_preferences -> mediate -> group_vote
       group_vote -> goodbye (winner)
       group_vote -> refine -> mediate (retry)

Solo terminal
  -> RecommendationAgent: controls / classify -> capability -> reply -> turn
                                                                  -> select / exit
```

Each entry point owns an in-memory LangGraph checkpointer and independent state.
The group orchestrator calls shared capability helpers directly; it does not
embed the solo graph. Interrupt payloads now include a participant field, and
web responses expose participants/votes and accept a voice selection.

- **Theme retrieval:** embeds the request and semantic context, retrieves up to
  ten theme labels, keeps labels within 0.10 of the best similarity, then ranks
  eligible movies using reciprocal-rank-weighted overlap.
- **Description retrieval:** starts with 40 neighbors and can expand to 1,000
  candidates when filters leave too few. Explicit plot-identification mode uses
  a 0.68 similarity gate and keeps suggestions within 0.10 of the top score.
  The threshold is a heuristic, not a probability of correctness.
- **Similar-to:** uses the last seed/liked entry in the current criteria list,
  embeds its plot with refinements, and excludes the seed itself. This is not a
  learned blend of all liked films.
- **Direct lookup:** case-insensitive exact/substring title lookup and partial
  person-name matching. Explicit years help disambiguate titles; arbitrary typos
  are not guaranteed to resolve.
- **Replies:** successful recommendations and facts can use Nebius. Social,
  preference, help, unsupported, control, error, availability, and guessed-title
  replies use deterministic copy. Generated prose is instructed to stay within
  retrieved data but is not a formal guarantee of factual correctness.

The web interface exposes `POST /api/agent/chat` and `POST /api/voice`. A chat
request with `message: null` starts the welcome flow for its `thread_id`; later
messages resume it. Waiting responses carry reply text, options, and criteria.
The title list is rendered separately from narration.

Voice uses SLNG-backed batch services: Deepgram Nova-3 transcription and Aura-2
synthesis. It is record-and-send, not continuous streaming. TTS call failures can
degrade to text-only replies.

## 7. Limitations and operating boundaries

- The standard builder selects up to **1,000 popular records from 2000 onward**.
  Deployed contents can differ after a separate import. Supported 1990s or
  “classic” filters can legitimately return no matches.
- The import keeps only the first five actor entries per movie. The absence of
  an actor in metadata is not proof they were absent from the film.
- The catalog is a snapshot with potentially incomplete metadata, not a live
  release feed. Records may include series/specials.
- Most interpretation still depends on an LLM. It can misclassify intent or
  extract an unwanted/missing filter. Guards cover specific patterns, primarily
  in English; mixed-language transcription does not guarantee correct parsing.
- Only one primary intent is executed per turn. Arbitrary multi-action requests
  and complex Boolean expressions are not supported; reset-plus-new-request is
  explicitly handled.
- There is no audience, age-rating, or content-advisory filter. There is no
  guarantee of suitability for children or absence of thematic content.
- Similarity and broad theme labels can yield plausible but incorrect matches.
- “More” is exclusion-based retrieval, not a stable cursor. Fresh searches can
  repeat titles, and reset clears shown-title tracking.
- `movies.streaming` is **mock data**, with `netflix`, `hbo`, `prime`, `appletv`.
  Its deterministic backfill assigns 2–3 services for ratings >=3.7, 1–2 for
  ratings >=3.0, and one for lower ratings. There is no region, price, licensing
  date, subscription tier, or actual provider verification.
- State, seen-film memory, and caches are in process memory. They do not survive
  restarts or automatically synchronize across workers. `thread_id` is not an
  authenticated user identity, and the chat endpoint has no application-level
  session access control.
- Selecting a title ends the graph. Playback, trailers, franchise/watch-order
  navigation, awards queries, person biographies, and durable user profiles are
  not implemented. There is no dedicated “unsee” operation.

## 8. Running, verification, and troubleshooting

From `kinofiles/`, start group mediation:

```bash
uv run python -m agent.orchestrator
```

For the standalone single-user chat and its full command loop:

```bash
uv run python -m agent.recomedation_agent
```

Core configuration uses `MISTRAL_API_KEY`, `NEBIUS_API_KEY`, `SUPABASE_URL`, and
`SUPABASE_KEY`. Web voice clients additionally require `SLNG_API_KEY`. See
`.env.template` and `agent/llm.py` for configuration. The embedding model must
remain compatible with stored `mistral-embed` vectors.

The database needs its movie schema, embedding tables/RPCs, and streaming column.
No database columns were removed by the feature purge. Do not run migrations or
backfills against a shared database without approval. Restart a running
CLI/backend after code changes to load the reduced schema and routes.

### Offline checks

```bash
uv run python -m unittest discover tests
uv run python -m unittest tests.test_controls tests.test_simplified_scope
uvx --from ruff==0.12.12 ruff check agent tests --ignore E402
```

Regression coverage includes reset/date removal, Ratatouille-style description
retrieval, facts, preference memory, pagination, and the absence of removed
capabilities. Offline tests are not a natural-language accuracy benchmark.

The separate live test runners and ReDial evaluation script have been removed.
Use the offline regression suite above. For manual interaction with real services,
use the terminal agent command shown earlier; it is not an automated test.

| Symptom | What to check |
|---|---|
| The same comedies recur | Direct requests rank by rating; try “What else?” in the same session. |
| A date persists after reset | Restart the process and inspect `criteria_changed` logs and returned criteria. |
| No matches after removing the date | Inspect other filters and catalog/actor coverage. |
| Classification fails | Inspect `classification_failed` invocation/validation metadata; failures are not social messages. |
| An unsupported operation is requested | Use recommendations or single-film facts; the removed feature is not silently emulated. |
| “It” is ambiguous | Name a film or use a shortlist ordinal. |
| No audio | Check SLNG configuration/availability; synthesis may fall back to text. |
| Memory disappears after restart | Expected with the in-memory checkpointer. |

## 9. Code map

| Area | Implementation |
|---|---|
| Supervisor/lifecycle | `agent/orchestrator.py` |
| State, routing, capabilities | `agent/recomedation_agent.py` |
| Classification and reference handling | `agent/nodes/classifier.py` |
| Deterministic reset/removal | `agent/nodes/criteria_commands.py` |
| Criteria merging/clearing | `agent/nodes/criteria.py` |
| Shared hard filters | `agent/nodes/filters.py` |
| Title resolution and direct retrieval | `agent/nodes/catalog.py`, `agent/nodes/direct_request.py` |
| Theme, browse, similar-to retrieval | `agent/nodes/theme_recommender.py` |
| Shared plot-neighbor search | `agent/nodes/description_search.py` |
| Selection | `agent/nodes/selection.py` |
| Narration and deterministic replies | `agent/nodes/reply.py` |
| Text/voice services | `agent/io/` |
| Web endpoints | `frontend/frontend/api.py`, `frontend/frontend/io_api.py` |
| Data preparation | `data_management/` |
| Offline regression tests | `tests/`, including `test_controls.py`, `test_upgrade.py`, `test_simplified_scope.py` |
