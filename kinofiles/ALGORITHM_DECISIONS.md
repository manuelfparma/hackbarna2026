# Algorithm decisions

Living notes on *why* the theme search and router work this way. Code can change; the constraints below are what the current numbers are solving.

## 1. Embed unique themes, not movies

A movie has many themes (`movies.themes` is a `text[]`). We store **one embedding per unique theme string** in `public.themes_embeddings`, not one vector per film and not one vector per movie–theme row.

**Why**

- Query-time search is “what themes is this sentence close to?”, then “which films carry those themes?”.
- Duplicating the same phrase (e.g. “Epic heroes” on 80 films) would make KNN return the same theme many times instead of N distinct themes.
- The catalogue is small (~109 unique themes in the current movies table). That is cheap to embed with `mistral-embed` (1024 dimensions) and cheap to join in Python.

The pipeline reads `movies.themes`, flattens, dedupes, skips names already in `themes_embeddings`, and upserts. Reruns are safe.

## 2. Retrieval is two hops, not one movie vector

```
user query
  → same embedding model as the table
  → top-N nearest rows in themes_embeddings  (cosine / pgvector <=> )
  → join theme names onto movies.themes
  → rank films
```

**Why not embed each movie** (title + description + themes concatenated): a film is a bag of themes. Embedding the bag blurs a driving movie that also has “superhero” tags into the same region as MCU films. Searching themes first keeps the semantic unit aligned with how people ask (“action and driving”, “melancholy family”).

**Why cosine (`<=>`) and HNSW `vector_cosine_ops`:** that matches how Mistral embeddings are typically compared (direction, not magnitude). Similarity shown to us is `1 - distance` so higher is better.

**Why a SQL function `match_themes`:** the Python/PostgREST table API cannot `ORDER BY embedding <=> $query LIMIT n`. An RPC is the whitelist that actually hits the index. The client only sends a 1024-float list and `match_count`.

Hackathon shortcut: we do **not** use RLS. The pipeline and agent use the service-role key.

## 3. Ranking films is not “count the overlapping themes”

Naive score = `|movie.themes ∩ top_10_themes|`.

That failed in practice. Example: query *“action and car driving movie”*.

- Closest theme: **Adrenaline-fueled action and fast cars** (~0.77).
- Themes 3–10 were generic action / superhero tags (~0.61–0.65).
- MCU titles tagged with five weak action themes **tied or beat** Fast Five / Drive, then sorted alphabetically so Iron Man appeared first.

It looked like the previous turn’s superhero answer had “lagged” one request. The second query *did* run; the ranker just promoted blockbusters that share many loose action labels.

Current ranker (`agent/nodes/theme_recommender.py`):

1. **Keep only themes within `SIMILARITY_MARGIN` (0.10) of the best match.**  
   Mistral scores on this catalogue sit in a **narrow band (~0.60–0.80)**. Absolute thresholds (“similarity > 0.7”) are brittle; a gap from the *best* match drops the unrelated tail. 0.10 still keeps a second related theme (e.g. heists next to fast cars) without keeping “superheroes vs villains”.

2. **Weight remaining themes by reciprocal rank:** `1, 1/2, 1/3, …`  
   Hitting the closest theme outweighs hitting several weaker ones. Broadly tagged blockbusters stop winning on volume.

3. **Tie-break: rating descending, then title.**  
   After filtering, many films share the same one or two themes and would otherwise sort as `2012`, `Atomic Blonde`, `Baby Driver` alphabetically. Rating is already on `movies`.

Constants (`MATCH_COUNT=10`, `MOVIE_LIMIT=5`, `SIMILARITY_MARGIN=0.10`) are knobs, not theory. If the theme catalogue grows or we switch embedding models, re-plot the similarity histogram before changing the margin.

Clients and embeddings objects are **cached on the `ThemeRecommender` instance** so each CLI turn does not rebuild HTTP clients (and so we do not reprint “Connecting to Supabase…” as if a new search backend appeared).

## 4. Router: retrieval vs catalog lookup

`RecommendationAgent` classifies each message, merges its entities into persistent `search_criteria`, runs one capability, passes its structured result through `ReplyComposer`, and pauses in `turn`. A non-numeric answer loops back through classification; a valid shortlist number ends the subgraph. `thread_id` + `InMemorySaver` keep criteria, feedback, movies, and the latest six conversational turns.

| Intent | When | Engine |
|---|---|---|
| `theme_recommendation` | Mood, vibe, topic, thematic content | pgvector + ranker above |
| `recommendation` | Generic “recommend a movie” with no catalog filter | Same theme search; the utterance is the query |
| `feedback` | Reaction to a previous list, no new catalog filter | Theme search again, with prior reactions folded into the query |
| `direct_request` | Catalog filters or facts about a named film | Supabase attribute filters ordered by rating; `asked_about` titles get a `movie_fact` row lookup (§5) |
| `social` | Greetings / off-topic | Shared reply LLM; classifier failures fall back here |

The classifier still distinguishes `recommendation` from `theme_recommendation` (and reply/history keep that label), but both retrieve through `ThemeRecommender`. Catalog attributes (genre, actor, director, studio, language) still take the filter path so an ask for “a comedy” is not an arbitrary KNN neighborhood. **Why not treat “themes of Inception” as theme search:** that is a fact about one title, not a request for similar thematic films.

The classifier uses structured output, so only declared intents can route. After classification we clear checkpointed `result`, `response`, and display flags so a failed capability cannot replay the previous turn.

### Criteria memory and refinement

`entities` is the latest turn; `search_criteria` is the accumulated search brief. The classifier emits a cue-based action:

- `add` for “also”, “and”, “with”
- `replace` for “instead”, “rather”, or an unqualified new direction
- `reset` for explicit start-over language
- `keep` for social and factual detours

Strings deduplicate case-insensitively. Genres are canonicalized to the catalog's exact labels, including aliases such as `sci-fi` → `Science Fiction`.

Explicit genres are hard constraints. Supabase receives all accumulated genres in one array-containment filter, so `Comedy` + `Action` requires both. Themes, moods, audience, and time-period language remain semantic: they form one embedding query, while accumulated genres pre-filter the movies ranked against the matched themes. If the intersection is empty, the result reports that instead of dropping a constraint.

A turn classified as `feedback` but carrying a new genre is rerouted to catalog lookup; a new theme (or a reaction with no new catalog filter) goes through theme search with the accumulated brief and prior reactions.

`OrquestratorAgent` embeds this compiled graph as its `conversation` node. The subagent's interrupt propagates through the parent, so terminal and web clients resume the same paused graph.

## 5. Similar-to and factual lookups ground on the catalog, not the LLM

Both paths resolve a spoken title to a `movies` row through `nodes/catalog.py`'s `resolve_movie`: exact case-insensitive match first, then best-rated substring match, with ReDial-style `"(2001)"` suffixes stripped. A mentioned `year` gets a shot at disambiguating substring hits before rating does.

- **`similar_to`** — a `seed`/`liked` movie in criteria triggers `match_descriptions` on the seed's own description text, with any refinement themes/audience/feedback appended to the embedded query (“like Super Troopers but darker”). Genres post-filter candidates in Python because the RPC signature predates them. If the seed doesn't resolve or has no description, the turn falls back to theme search, where the title text still lands in the query.

- **`movie_fact`** — `asked_about` titles fetch their catalog row and the reply layer narrates it. Facts never write `movies` or `show_options`, so asking “who directed it?” mid-shortlist doesn't clobber the pending pick.

Why description neighbors for similar-to rather than the theme ranker: theme tags only express “shares these labels”, while the description embedding captures plot and tone — closer to what “like Super Troopers” actually means. The LLM `Recommender` node was removed entirely: ungrounded title lists could end the session on a film the catalog doesn't have.

## 6. Conversational replies are separate from capability results

Capability nodes return data, not presentation copy. `ThemeRecommender` keeps cosine scores internal and exposes titles plus readable theme names; direct lookup exposes titles without database IDs. `ReplyComposer` receives that result, entities, feedback, shortlist, and bounded history, then produces at most three short sentences.

This costs one reply-model call per turn, but gives every path one voice and prevents retrieval diagnostics from reaching TTS. Reply generation uses temperature `0.4`; classification remains deterministic. If the reply call fails, intent-specific static copy keeps the interrupt loop usable.

The shortlist and its visibility are separate state. Social or unsuccessful lookup detours preserve an earlier shortlist, but do not redisplay its options or selection prompt. `turn` still accepts a valid number for that preserved shortlist.

## 6. What we explicitly did not do (yet)

- Embed movies or plot text.
- Weighted overlap using raw cosine as the movie score (rank is enough on 109 themes).
- Structured exclusions such as “less MCU” (there is no persisted exclusion field yet).
- Server-side movie ranking in SQL (hackathon: load `movies` in Python).
- Production RLS / connection pooling / HNSW tuning.

When those change, add a short “we tried X because Y, it failed because Z” here rather than only editing code comments.
