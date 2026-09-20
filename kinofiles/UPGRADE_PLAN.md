# KinoFiles upgrade — historical implementation spec

> Scope update: the subsequent feature purge removed session watchlists,
> two-film comparison, catalog analytics, studio filtering, semantic content
> exclusions, and audience criteria (including the former soft-context field).
> Tasks describing these features below are superseded historical context, not
> instructions to restore them. Static help, positive theme/plot search, and explicit
> genre/people exclusions remain. See `SYSTEM_OVERVIEW.md` for current capabilities.
>
> Test-tooling update: the standalone interactive LLM runner, scripted conversation
> runner, live upgrade assertion runner, and ReDial evaluation script were deleted.
> References to those runners and commands below are historical and obsolete;
> do not execute or recreate them. Current verification uses the offline `tests/` suite.

Self-contained spec for upgrading the query coverage of this repo's
conversational movie recommender. Implement the tasks in order; each lists
the files to touch, the exact contract changes, and how to verify.

## Repo orientation

- Project root: `kinofiles/`. Python ≥3.11, managed with `uv`
  (`pyproject.toml`, `uv.lock`). Run everything from `kinofiles/`.
- Tests are stdlib `unittest` in `tests/` (`test_routing.py`,
  `test_criteria.py`, `test_filter_memory.py`, `test_reply.py`,
  `test_turn.py`, `test_capabilities.py`). They test **pure logic only** —
  routing, criteria merging, fallbacks. Keep them offline: no LLM or
  Supabase calls in unit tests. Run: `uv run python -m unittest discover tests`.
- End-to-end driver: `uv run python test_scripts/test_conversations.py`
  (live Mistral + Nebius + Supabase; needs `.env`).
- `.env` needs `SUPABASE_URL`, `SUPABASE_KEY`, `MISTRAL_API_KEY`,
  `NEBIUS_API_KEY` (see `.env.template`). LLM factories live in
  `agent/llm.py`: Mistral classifies/embeds, Nebius (OpenAI API) writes
  replies. Do not swap the embedding model — stored vectors are in
  `mistral-embed` space.

## Architecture (what exists, and invariants)

`OrquestratorAgent` (`agent/orchestrator.py`) is a thin supervisor:
`welcome` → `conversation` → `goodbye`. `conversation` **is**
`RecommendationAgent.graph` (compiled subgraph as a node, checkpointer
borrowed from the parent — `RecommendationAgent(..., checkpointer=None)`).

Subagent flow (`agent/recomedation_agent.py`):
`classify` → `{theme_recommendation | feedback | direct_request | social}`
→ `reply` → `turn` (interrupt) → back to `classify`. A bare in-range digit
at `turn` sets `choice` → END → `goodbye`.

Invariants — break any of these and the system fails silently:

1. **Parent state is a superset.** LangGraph drops state keys the parent
   doesn't declare. Any new `State` key added to
   `recomedation_agent.State` must also be added to `orchestrator.State`
   and initialized in `welcome()` and in `RecommendationAgent.run()`'s
   initial payload (and `test_scripts/test_conversations.py`'s `INITIAL`).
2. **Capabilities return data, `reply` returns prose.** Every capability
   result follows `{"kind": str, "titles": list[str], "error": str|None,
   ...}`. `movie_fact` results also carry `facts` (full catalog row minus
   `id`). Reply fallbacks branch on `kind`/`facts`/`error` — keep the
   contract.
3. **Never invent titles.** All user-facing titles come from Supabase rows.
   The classifier copies names; it never generates them.
4. **`movies` vs `show_options` stay separate** — a detour hides the list
   without clearing it; a number still selects.
5. **Classifier stays stateless.** All context it needs must be passed
   explicitly (Task 1) — it must not read graph state itself.

Current classifier contract (`agent/nodes/classifier.py`):

- `Intent = Literal["theme_recommendation", "recommendation", "feedback",
  "direct_request", "social"]`
- `MovieRole = Literal["seed", "liked", "seen", "wanted", "asked_about"]`
- `CriteriaAction = Literal["add", "replace", "reset", "keep"]`
- `Entities`: `movies: list[MovieMention{title, year, role}]`, plus lists
  `actors, directors, genres, themes, studios, languages, years,
  time_periods, audience`.
- `Classification{intent, entities, criteria_action}`, plus `as_filter()`
  returning the first populated `FILTER_COLUMNS` value or `("name",
  title)`.
- `Classifier.classify(request)` returns `Classification`; any failure →
  `Classification(intent="social", criteria_action="keep")`.

Criteria (`agent/nodes/criteria.py`): `CRITERIA_FIELDS` mirrors `Entities`
list fields; `CATALOG_ARRAY_FIELDS = (directors, actors, genres, studios,
languages)`; `split_genres` canonicalizes catalog labels and demotes
non-labels to `themes`; `merge_criteria(current, entities, action)`;
`build_theme_query(request, criteria)` composes the embedding query
(seed/liked titles join it; `seen` titles currently don't — Task 4 uses
them).

Retrieval:

- `nodes/theme_recommender.py` — `ThemeRecommender.recommend(request,
  feedback, criteria)`: embeds query → `match_themes` RPC (top 10, keep
  within `SIMILARITY_MARGIN=0.10`, reciprocal-rank weights) → loads
  `movies` (`contains("genres", genres)` when set) → scores → top 5.
  `_similar_to` fires when criteria holds a `seed`/`liked` movie:
  `resolve_movie` → `match_descriptions` RPC (top 25) → drop seed → genre
  post-filter → top 5.
- `nodes/direct_request.py` — `DirectRequestHandler.handle(column, value,
  criteria)`: `genres` → `contains` (strict AND), other array fields →
  `ilike` on generated `*_text` columns (one per value, still AND), `name`
  → `ilike`; `order("rating", desc).limit(5)`. `handle_movie_fact(title,
  year)` → `resolve_movie` → `facts`.
- `nodes/catalog.py` — `MOVIE_COLUMNS` (add `streaming` here in Task 7),
  `clean_title` (strips `"(year)"` suffix and ILIKE wildcards — reuse it
  everywhere user text becomes an ILIKE pattern), `resolve_movie` (exact
  citext → year+substring → best-rated substring).
- `nodes/reply.py` — `ReplyComposer.compose(...)` context dict already
  carries `search_criteria`, `current_shortlist`, `feedback`, `history`.
  Extend `SYSTEM_PROMPT`/`fallback` for new `kind`s.

Routing today is `RecommendationAgent._route_for_criteria(intent,
entities, criteria)` — pure function, unit-tested in `test_routing.py`.
Keep it pure and keep signature-compatible or update tests.

## Task 0 — Eval harness (baseline first)

Create `test_scripts/eval_redial.py`. For each conversation in
`annotations/redial_test_query_labels.json`, drive a fresh
`RecommendationAgent` (unique `thread_id`) feeding only `seeker` turns'
`text_resolved` via the interrupt/resume loop (pattern in
`test_scripts/test_conversations.py`). After each turn, read graph state
(`get_state`) and log: request, produced `intent`, `result.kind`,
`result.error`, `len(movies)`, whether the graph ended.

Label→expected mapping (a turn "succeeds" when the system did the right
*kind* of thing, not exact titles):

| Label | Success means |
|---|---|
| `recommend` | `movies` non-empty (any search ran) |
| `prefer` | criteria grew (entities absorbed) AND no forced error |
| `feedback` + `feedback=accept` | graph ends with a `choice`, or acknowledgment |
| `feedback` (other) | criteria/feedback grew or new non-empty `movies` |
| `ask_item` | `result.kind == "movie_fact"` (or `compare`/`availability`/`description_match` once added) |
| `known_item` | `choice` set, or title found (incl. by description) |
| `more` | new `movies` disjoint from previous |
| `social` | `intent == "social"` |
| `other` | report only, not scored |

Output: per-label success table + total. Support `--intent X --n K` for
iteration. **Run the full baseline before Task 1 and save the table** —
it's the regression metric for everything below. Expect `prefer`,
`more`, `ask_item`-anaphora, `accept`, and described-title lookups to
dominate failures.

## Task 1 — Context-aware classification

Goal: resolve anaphora ("it", "that one", "the second one", corrections
like "I mean X").

- `Classifier.classify(request)` → `classify(request, context: dict)`.
  `context` = `{"shortlist": state["movies"], "recent_history":
  state["history"], "criteria_summary": <compact criteria>}`. Build it in
  `recomedation_agent.classify` — the classifier stays stateless.
- Extend `SYSTEM_PROMPT`: a numbered shortlist and recent turns are
  provided; pronouns and ordinals refer to them; when the user clearly
  references a listed title, emit it in `entities.movies` with the new
  role `referenced`; **never** emit a title absent from both message and
  context.
- `MovieRole` += `"referenced"`. Routing treats `referenced` like
  `asked_about` for `movie_fact` (a pronoun question about a shown item)
  but it must not enter `build_theme_query`'s seed/liked clause.
- Ordinal/title resolution of `referenced` against `state["movies"]`
  happens in `turn`/`accept` (Tasks 3 and 5), not in the classifier.
- Update the fallback path and `test_routing.py`/`test_reply.py` callers
  for the new signature.

Acceptance: "What is it about?" with a shortlist present resolves
`referenced` to the listed title and returns `movie_fact`.

## Task 2 — Described-film lookup ("name that movie")

Motivating failure, reproduced live against the real graph:

```
YOU  > Could you please give me an animation movie of a rat that lives
       in Paris and helps a chef?
     intent=direct_request | route=theme_recommendation
     entities={genres:[Animation], themes:[rat, Paris, chef, helps a chef]}
     titles=[Lilo & Stitch, Tangled, Rio, Monsters, Inc.,
             The Emperor's New Groove]          ← no Ratatouille

YOU  > ¡No, no, no! It's the rat in France and helps a chef to become a
       better cooker. ¿No se me qué?
     intent=direct_request | route=direct_request
     entities={movies:[{title:'the rat in France', role:'asked_about'}]}
     kind=movie_fact | error=I couldn't find 'the rat in France' in the catalog.
```

Root causes:

- **Theme vocabulary is too coarse for plot queries.** `themes_embeddings`
  holds ~109 generic labels ("Kids' animated fun and adventure"). A
  specific plot ("rat + Paris + chef") embeds onto themes shared by dozens
  of animated films; ties break on `rating` and Ratatouille (4.2, #10 of
  113 Animation titles by rating) misses the top 5. The precise tool
  already exists — `match_descriptions` embeds the request against plot
  embeddings — but it is only reachable through `_similar_to` when a seed
  title resolves.
- **The classifier emits descriptions as titles.** "the rat in France" is
  a description, yet it was produced as `movies[0].title` with role
  `asked_about`; `_is_factual_movie_request` then sent it to
  `handle_movie_fact`, `resolve_movie`'s ILIKE missed, and the turn
  dead-ended on the error string.

### 2a — Shared description search

Factor description-embedding search out of `_similar_to` into a helper
(e.g. `nodes/description_search.py` or a `catalog.py` sibling):

```
search_descriptions(supabase, embeddings, query_text, genres=None,
                    exclude_ids=None, match_count=25) -> list[rows]
```

embed `query_text` → `match_descriptions` RPC → drop `exclude_ids` →
genre post-filter (same `id/genres` fetch + casefold intersection
`_similar_to` does today — reuse it, don't copy it) → return rows with
`movie_id`, `name`, `similarity`.

### 2b — Blend description search into `theme_recommendation`

In `recommend()`, when `criteria["themes"]` is non-empty (the request
carried descriptive content — plain genre/person asks don't), also run
`search_descriptions` on `build_theme_query(request, criteria)`:

- If the top similarity ≥ `DESC_MATCH_THRESHOLD` (new constant — calibrate
  against `mistral-embed` cosine scores on a few known plot descriptions;
  start at 0.55 and adjust): return the description-ranked top 5 as
  `{kind: "description_match", titles, themes: <matched themes if
  computed>, genres, error}`.
- Otherwise keep today's theme-scored list.

Exact plot descriptions win over generic-theme voting; mood/vibe asks
stay on the theme path since their description similarity won't cross the
threshold.

### 2c — `movie_fact` resolve fallback

In `DirectRequestHandler.handle_movie_fact`, when `resolve_movie` returns
`None`: run `search_descriptions` on the spoken phrase (needs an
embeddings client — give `DirectRequestHandler` a lazy
`MistralAIEmbeddings` like `ThemeRecommender._connect` builds, or inject
the helper). If top similarity ≥ `DESC_MATCH_THRESHOLD`, return
`{kind: "movie_fact", titles: [nearest names], suggestions: [...],
guessed: True, facts: <nearest row>, error: None}`. Reply narrates it as a
guess ("Did you mean Ratatouille?") — extend `SYSTEM_PROMPT` and
`fallback` for `guessed`/`suggestions`. Below threshold → keep today's
error.

### 2d — Classifier prompt rule

Extend `SYSTEM_PROMPT`: when the user *describes* a film without naming
it ("it's the one where…", "the rat in France that cooks"), the
description goes to `themes` and `movies` stays empty — a noun phrase is
not a title. Keep the existing "actual titles only" rule and add this as
its explicit consequence, with one example.

### Tests

- Unit-test the `guessed`/`suggestions` result contract and the threshold
  gate with a fake `search_descriptions`.
- Add the transcript above as a scripted conversation in
  `test_scripts/test_conversations.py`; expected outcome: turn 1's
  shortlist contains Ratatouille, turn 2 suggests it rather than erroring.
- New label coverage in `eval_redial.py`: described-title `ask_item`/
  `known_item` turns (`title` slot filled but title absent from the text)
  are the regression metric for this task.

## Task 3 — New intents: `prefer`, `accept`, `more`, `exit`

Extend `Intent` and the prompt:

- **`prefer`** — taste statements with no ask ("I loved X", "I love
  Nolan", "I've seen Y", "I'm a horror fan"). `criteria_action=add`. New
  node `preference`: packages `{kind: "preference", absorbed: <what was
  merged>}` for reply — **no search**. Route in `_route_for_criteria`
  before the other branches; `prefer` never reaches retrieval.
- **`accept`** — "I'll take it", "perfect, that one", "I'll watch that".
  New node `accept`: resolve the pick (priority: `referenced`/`wanted`
  entity title → ordinal word → fuzzy `clean_title` match against
  `state["movies"]`). On success `Command(goto=END, update={"choice":
  title})`; on failure `Command(goto="reply")` with a result that asks
  which one.
- **`exit`** — "bye", "I'm done", "that's all". From `classify`:
  `Command(goto=END)` (no capability). `orchestrator.goodbye` must handle
  a missing/`None` `choice` — no-pick farewell variant.
- **`more`** — "what else", "any others". Add `last_route` to `State` (and
  parent `State` — invariant 1); `classify` records each chosen route.
  `more` → `Command(goto=last_route)`; if `last_route` in
  `{"social", "accept", "preference", None}` fall back to
  `theme_recommendation`. Combined with Task 4's exclusions, the same
  criteria produce the *next* five.

`criteria_action` rules: `prefer`→`add`, `accept`/`exit`→`keep`,
`more`→`keep`. `social` keeps its existing `keep` override.

Reply additions: `kind="preference"` acknowledgment line ("Got it — more
Depp in the mix"); `accept` failure prompt. Extend `fallback` for new
kinds.

## Task 4 — Exclusion of seen and shown titles

- New `State` key `shown_movies: list[str]` (append each displayed
  shortlist; parent `State` + initializers per invariant 1).
- `seen`-role movies in `criteria.movies` resolve to names via
  `resolve_movie` (cache per session — don't re-resolve every turn).
- Exclusion set = `shown_movies ∪ resolved(seen titles) ∪ current seed`.
- `theme_recommender.recommend`: fetch ~3× `MOVIE_LIMIT` candidates, drop
  excluded names, then take 5. `_similar_to` and the new
  `search_descriptions` helper (Task 2a): raise `match_count`
  (25→40) and filter before slicing.
- `direct_request.handle`: `.not_.in_("name", excluded)` on the query.

Acceptance: a `more` turn returns zero overlap with the previously shown
list; "I've seen X" removes X from subsequent results.

## Task 5 — Pick by title or ordinal

`turn` currently: `answer.isdigit() and in range` → `choice`. Extend
before the classify fallback:

- Ordinal words (`first`–`fifth`, "number three", "the second one") →
  index. Small static map.
- `clean_title(answer)` substring-match against `state["movies"]`
  (casefold); single unambiguous match → `choice`.
- Ambiguous or no match → `classify` as today.

## Task 6 — Filter layer uses the full schema

All of these are `Entities`/`CRITERIA_FIELDS` additions plus application
in **both** `direct_request.handle` and `theme_recommender`'s movie
query/post-filter. New fields go in `Entities`, `CRITERIA_FIELDS`,
`normalize_criteria`, `merge_criteria` (they merge like the existing
lists).

- **6a Date ranges**: `time_periods`/`years` → `date` filters. Normalize
  in `criteria.py`: `90s`/`1990s` → gte 1990, lte 1999; bare year → eq;
  `classic`/`old` → lte 1980; `new`/`recent`/`modern` → gte 2015 (check
  actual `date` distribution in `data/movies.csv` and document the
  chosen cutoffs). Apply `gte`/`lte`/`eq` in `direct_request`; same
  predicate as a pre-filter next to `contains("genres", ...)` in
  `theme_recommender`.
- **6b Duration**: new entity `duration` ("short", "under two hours",
  "epic"). Inspect `duration_category` labels + `minute` distribution in
  `data/movies.csv` first; map to `duration_category` eq when the label
  matches, else `minute` lte/gte (e.g. `short` → `minute <= 100`).
- **6c Rating floor**: `min_rating` float — only set by explicit quality
  language ("highly rated", "something good"). `gte("rating", floor)` in
  both paths. Not part of the theme query.
- **6d Negation**: `exclude_genres`, `exclude_actors`,
  `exclude_directors`, `exclude_themes`. Canonicalize `exclude_genres`
  through `split_genres` (non-labels demote to `exclude_themes`). SQL:
  `.not_.contains("genres", vals)` and `.not_.ilike(f"{col}_text",
  f"%{clean_title(v)}%")`. `exclude_themes` is semantic: in
  `theme_recommender`, embed the excluded phrases, `match_themes`, drop
  movies whose `themes` intersect the kept matches.
- **6e Disjunction**: `genre_groups: list[list[str]]` — "comedy or
  romance" → `[["Comedy"],["Romance"]]`; "action comedy" →
  `[["Action","Comedy"]]`. In `direct_request`, OR groups via PostgREST
  `or_()` (`or_("genres.cs.{Comedy},genres.cs.{Romance}")`) — verify the
  exact supabase-py call signature before writing it. AND-within-group,
  OR-between-groups. When `genre_groups` is present it replaces (not adds
  to) the `genres` constraint; merge keeps them consistent. Theme path:
  keep existing behavior (genres already prefilter; groups apply the same
  OR as a post-filter on candidates).
- **6f Audience**: keep `audience` semantic, but let the classifier emit
  concrete constraints for clear cases — "for kids" → `genres:
  [Family|Animation]` + `exclude_genres: [Horror]` — documented in the
  prompt with two examples.

Unit-test each in `test_criteria.py`/`test_routing.py` (normalization and
routing are pure); add a `test_filters.py` for criteria→predicate mapping
if the query-building is factored into a helper — factor it if it helps.

## Task 7 — Streaming availability (mock data)

Schema — **already added** to `data_management/create_schema.sql`:

```sql
ALTER TABLE movies ADD COLUMN IF NOT EXISTS streaming TEXT[] DEFAULT '{}';
ALTER TABLE movies ADD COLUMN IF NOT EXISTS streaming_text TEXT
  GENERATED ALWAYS AS (array_to_string_immutable(streaming, ' ')) STORED;
CREATE INDEX IF NOT EXISTS idx_movies_streaming ON movies USING gin (streaming);
```

Backfill — **already written**: `data_management/migrate_streaming.py`
(style of `migrate_posters.py`: env check, paginated `id, rating` fetch,
per-id single-column updates, abort on first schema error). Deterministic
per movie — `random.Random(movie.id)`, count weighted by `rating`
(≥3.7 → 2-3 services, ≥3.0 → 1-2, else 0-1) drawn from `("netflix",
"hbo", "prime", "appletv")`, stored lowercase. Run it once:

```
uv run python data_management/migrate_streaming.py
```

Agent changes (still to implement):

- Add `streaming` to `MOVIE_COLUMNS` in `catalog.py` (so `movie_fact`
  results include it and reply can narrate services).
- New intent `availability` ("where can I watch X", "is it on Netflix").
  Reuse `movie_fact`'s resolve-and-narrate path — the facts row now
  carries `streaming`. Extend the reply prompt with a services line;
  empty array → "I don't have streaming info for that one".
- Filter direction: add `streaming` to `Entities` + `CRITERIA_FIELDS` as
  a service constraint ("a comedy on Netflix"). In
  `direct_request.handle`, treat `streaming` like `genres` —
  `contains("streaming", values)` — extend the `if filter_column ==
  "genres"` branch to `in ("genres", "streaming")`. Do **not** add it to
  `as_filter`'s `FILTER_COLUMNS` preference order ahead of
  person/attribute columns (a named service shouldn't steal the primary
  filter slot from "movies by Nolan"); append it last if included.

## Task 8 — `compare` intent

"which is better, X or Y", "is it like the Godfather" (real `ask_item`
examples in the labels). New intent + `nodes/compare.py`: resolve up to
two `entities.movies` titles via `resolve_movie`; both resolve →
`{kind:"compare", facts:[row1, row2]}`; one resolves → fall back to
`movie_fact` result for it. Reply prompt gets a compare branch:
contrast rating/date/themes of the two rows only. Unit-test the
resolution fallback logic.

## Task 9 — `meta` intent + serendipity + watchlist

- **`meta`**: "what can you do", "which genres do you have", "how many
  horror movies". Node returns `{kind:"meta", answer:...}` — static
  capability blurb, `CATALOG_GENRES` listing, and cheap counts
  (`select("id", count="exact")` or `len` on a filtered query). No prose
  of its own — reply narrates.
- **Serendipity**: in `theme_recommender.recommend`, when criteria is
  empty and the request is generic ("surprise me", "any movie"), return a
  random sample (random offset into top-200 by rating, or `order` by a
  random key if the API supports it) instead of the deterministic
  theme-match top-5.
- **Watchlist**: `watchlist: list[str]` in `State` (+ parent per
  invariant 1). Intent `save` ("keep that one", "add it") resolves like
  `accept` does → append to `watchlist`, acknowledge via reply. "what's
  on my list" → `meta` variant reading `watchlist`. Session-scoped only —
  no table.

## Task 10 — Compound turns (prompt-level only)

Extend `SYSTEM_PROMPT`: when a turn carries multiple acts, pick the
dominant one by this priority — `accept`/`exit` > retrieval intents >
`ask_item`/`availability`/`compare` > `prefer`/`feedback` > `social` —
**but** still extract every entity present, since `prefer`-flavored turns
with an ask ("I loved The Conjuring — got anything like it?") must merge
the taste *and* run the search in one pass (criteria merge already
happens before routing, so this works once extraction is complete).

## Gotchas discovered during gap analysis — respect them

- `merge_criteria` treats `replace` and `reset` identically; `prefer`
  relies on `add` — don't let prefer turns emit `replace`.
- `classify` already forces `action="keep"` for `social` and factual
  requests — extend that same guard to `accept`/`exit`/`more`/`meta`.
- `_is_factual_movie_request` requires *all* movies be `asked_about` with
  no catalog filters — `referenced` titles should satisfy it (Task 1).
- The classifier will emit descriptive noun phrases as `movies[].title`
  ("the rat in France" → `asked_about`) — Task 2d's prompt rule is the
  fix; the Task 2c `movie_fact` fallback is the safety net when it still
  happens.
- `clean_title` must wrap every user string before it becomes an ILIKE
  pattern (it strips `%`/`_`).
- Supabase-py: exclusions are `.not_.in_(...)`/`.not_.contains(...)`/
  `.not_.ilike(...)`; disjunctions are `.or_("postgrest.filter.syntax")`.
  Verify signatures against the installed version before writing.
- Interrupt propagation depends on the subgraph-as-node +
  borrowed-checkpointer setup — do not wrap `conversation` in a function
  or give the subagent its own checkpointer.
- The themes table has ~109 *generic* labels; plot-specific requests will
  always lose to popular films on the rating tiebreak. That's why
  Task 2's `match_descriptions` path matters more than tuning
  `SIMILARITY_MARGIN`.

## Verification

- `uv run python -m unittest discover tests` — all green, plus new tests
  per task.
- `uv run python test_scripts/test_conversations.py` — extend
  `CONVERSATIONS` with: described-film lookup (the Ratatouille
  transcript), prefer-then-search, anaphoric fact question,
  accept-by-name, `more` pagination, exit-without-pick, streaming
  availability, compare.
- `uv run python test_scripts/eval_redial.py` — baseline vs final table
  in the PR/commit summary.

## Out of scope

Franchise/sequel navigation (needs relation data), person bios (catalog
has names only), real streaming-provider data (JustWatch/TMDb — the mock
column's query path is already correct, only the backfill would change),
MPAA content advisories, persistent multi-session watchlists.
