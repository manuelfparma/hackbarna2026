# How KinoFiles works

A conversational movie recommender: the user chats (text or voice), the agent
clarifies taste over multiple turns, shows a shortlist, and ends when the user
picks a film. Everything user-facing is grounded in the Supabase catalog —
the LLMs route, merge criteria, and narrate; they never invent titles.

For *why* specific choices were made, see `ALGORITHM_DECISIONS.md`. This
document is the *what*: the moving parts and what happens on each message.

## The big picture

```
Reflex UI (chat + mic)                      Supabase
      |                              ┌──────────────────────────┐
      v                              │ movies (1000 rows):      │
FastAPI /api/agent/chat              │  name citext, rating,    │
      |                              │  date, description,      │
      v                              │  genres/actors/...       │
OrquestratorAgent                    │  + generated *_text cols │
  welcome ──> conversation ──>       │  (pg_trgm indexes)       │
   (int.)     │             goodbye  │                          │
              = subgraph             │ themes_embeddings (109)  │
              RecommendationAgent    │ descriptions_embeddings  │
                                     │  (1000)                  │
                                     └──────────────────────────┘
LLMs: Mistral (classify, embed) · Nebius Qwen (reply)
Voice: SLNG gateway — Deepgram Nova-3 (STT), Aura-2 (TTS)
```

The supervisor owns session lifecycle (welcome, goodbye) and I/O. The actual
conversation is a **subgraph embedded as a node** — its interrupts propagate
up through the supervisor, so the caller just resumes and lands back inside.

## The conversation loop

```
welcome (interrupt: "What do you feel like watching?")
   |
   v
conversation ── RecommendationAgent subgraph ─────────────────────┐
   |                                                              |
   |   classify ──> capability ──> reply ──> turn (interrupt)      |
   |      ^                                        |              |
   |      └──────────── non-numeric answer ────────┘              |
   |                          | numeric pick                     |
   v                          v                                  |
goodbye  <──────────  subgraph ends with `choice`                 |
```

`classify` runs on every message. The capability produces **data**, `reply`
turns it into prose, `turn` pauses the graph and waits for the user.

- A bare number in range (e.g. `2`) ends the subgraph with `choice` set →
  `goodbye` → `END`.
- Anything else becomes the next `request` and loops back to `classify`.

## State carried between turns

| Key | What it holds |
|---|---|
| `request` | The user's latest message |
| `intent` | This turn's classified intent |
| `entities` | Entities from the latest message only |
| `search_criteria` | The accumulated search brief — survives all turns |
| `feedback` | List of past reactions ("too long", "more action") |
| `movies` | Current shortlist (titles the user can pick by number) |
| `result` | Last capability's structured output (titles, themes, facts, error) |
| `history` | Last 6 turns `{user, assistant, intent}` — reply context |
| `show_options` | Whether `turn` displays the shortlist + pick prompt |
| `response` / `choice` | Composed narration / the final pick |

`movies` and `show_options` are deliberately separate: a social or factual
detour hides the list without deleting it, and a number still selects it.

### Criteria memory

`entities` describes this turn; `search_criteria` is the running brief. The
classifier also emits a `criteria_action` from language cues:

| Action | Cues | Effect |
|---|---|---|
| `add` | "also", "and", "with" | Union into existing criteria |
| `replace` | "instead", "rather", new direction | Discard old, keep new |
| `reset` | "start over", "something different" | Clear, keep new |
| `keep` | greetings, factual questions | No change |

Genres are canonicalized to catalog labels (`sci-fi` → `Science Fiction`) and
act as **hard constraints** (array-containment AND). Themes, moods, audience,
and time-periods stay **semantic** — they go into the embedding query.

## Routing

`classify` produces one intent; `_route_for_criteria` may reroute based on
what the merged criteria actually contain:

| Condition | Route |
|---|---|
| intent `social` | `social` |
| `direct_request` + only `asked_about` movies + no filters | `direct_request` → `movie_fact` |
| new `themes` in entities, or theme-ish intent with remembered themes | `theme_recommendation` |
| `feedback` carrying a new catalog filter | `direct_request` |
| `recommendation`/`theme_recommendation` with catalog filters in criteria | `direct_request` |
| `recommendation` with nothing else | `theme_recommendation` |
| otherwise | the classified intent |

## The capabilities

### `theme_recommendation` — mood/vibe search

```
request + remembered themes/audience/periods + seed titles + feedback
  → mistral-embed → match_themes(top 10, cosine)
  → keep themes within 0.10 of the best similarity
  → weight kept themes by reciprocal rank (1, 1/2, 1/3, ...)
  → fetch movies (genre-prefiltered if genres accumulated)
  → score = Σ weights of matched themes; ties → rating, then title
  → top 5
```

### `similar_to` — "a movie like X"

Fires inside `theme_recommendation` whenever criteria holds a `seed` or
`liked` movie:

```
title → resolve_movie() (exact citext → year+substring → best-rated substring)
  → embed "seed description + refinement themes + feedback"
  → match_descriptions(top 25)
  → drop the seed itself → genre post-filter → top 5
```

If the title doesn't resolve or has no description, the turn falls back to
theme search (the title text still lands in the query).

### `direct_request` — catalog filters

```
criteria filters (genres/actors/directors/studios/languages)
  + column/value from the classifier
  → genres: contains() — strict AND on exact labels
  → other arrays: ilike on generated *_text columns — "Nolan" matches
    "Christopher Nolan"; one ilike per value keeps it an AND
  → name (a mentioned title): ilike on citext name — fuzzy title search
  → order by rating desc, limit 5
```

### `movie_fact` — questions about a named film

`asked_about` titles resolve to their catalog row (description, directors,
rating, date, themes...). The reply layer narrates the answer. Never touches
`movies`/`show_options` — asking "who directed it?" mid-shortlist doesn't
clobber the pending pick.

### `feedback`

Reactions accumulate in `feedback` and re-run theme search with the prior
reactions folded into the embedding query. Carrying a *new* genre/theme is
rerouted to retrieval with the full accumulated brief instead.

### `social`

Greetings/small talk → packaged context for the reply layer. Also the
fallback destination when classification itself fails.

### `reply` — the single voice

Every capability result goes through `ReplyComposer` (Nebius, temp 0.4):
request + intent + entities + criteria + result + shortlist + last 6 turns
→ at most three short sentences. Rules: never list titles (they're rendered),
never expose scores/ids/routing, never invent films. If the call fails,
deterministic per-intent fallback copy keeps the loop usable.

### `turn` — the pause

`interrupt({text, options})`: `text` is narrated (TTS), `options` are
rendered only — a numbered list makes terrible speech. With options visible
the text gains "Pick a number (1-N) or tell me what to change."

---

## Worked examples

### 1. Theme search, then a pick

```
user: "something melancholy about family"
classify → intent=theme_recommendation, entities.themes=["melancholy family"],
           action=replace
route    → theme_recommendation
embed    → "something melancholy about family; melancholy family"
match_themes → "Family bonds and grief" (0.78), "Quiet melancholy" (0.74), ...
margin   → keeps the ~2 themes within 0.10 of 0.78
rank     → movies carrying those themes, reciprocal-rank score, rating breaks ties
result   → titles=["Manchester by the Sea", "Aftersun", ...]
reply    → "These sit with family and grief without rushing it."
turn     → options shown; interrupt
user: "2"
turn     → choice="Aftersun" → END → goodbye: "Enjoy Aftersun! Goodbye."
```

### 2. Genre constraints accumulate (AND)

```
user: "I want a comedy"
classify → intent=direct_request, genres=[Comedy], action=add/replace
criteria → {genres: [Comedy]}
handle   → SELECT name FROM movies WHERE genres @> '{Comedy}'
           ORDER BY rating DESC LIMIT 5
turn     → options shown

user: "also some action"
classify → intent=feedback, genres=[Action], action=add
criteria → {genres: [Comedy, Action]}
route    → feedback + new catalog filter → direct_request
handle   → genres @> '{Comedy,Action}'  ← BOTH required
turn     → new options
```

### 3. Similar-to a named film

```
user: "a movie like The Dark Knight"
classify → intent=recommendation, movies=[{title:"The Dark Knight",
           role:"seed"}], action=replace
route    → theme_recommendation (nothing else applies)
recommend() sees a seed → _similar_to:
  resolve  → movies.name ILIKE 'The Dark Knight' → row (id, description)
  embed    → seed description text
  match_descriptions → The Dark Knight Rises, The Batman, Joker, ...
  drop seed → top 5
reply    → "If it's the Gotham-noir vibe you want, these live next door."
```

With a genre: "a comedy like Super Troopers" → same path, but candidates are
post-filtered to `Comedy`.

### 4. Factual question mid-shortlist

```
(shortlist ["Drive", "Heat"] is on screen)
user: "who directed Inception?"
classify → intent=direct_request, movies=[{title:"Inception",
           role:"asked_about"}], action=keep
route    → direct_request → _is_factual_movie_request → handle_movie_fact
resolve  → movies.name ILIKE 'Inception' → row
result   → kind=movie_fact, facts={name, date, directors, rating, ...}
reply    → "Christopher Nolan directed Inception (2010) — it holds a 4.4."
turn     → options hidden; the shortlist is still selectable — "1" picks Drive
```

### 5. Feedback steers the next search

```
(shortlist shown)
user: "these are too mainstream, something darker"
classify → intent=feedback, themes=["darker"], action=add
route    → theme_recommendation
embed    → request + remembered themes + "these are too mainstream,
           something darker" (prior reactions join the query)
rank     → new top 5
reply    → "Less polished, more shadow — try this batch."
```

### 6. Fuzzy person lookup

```
user: "movies by Nolan"
classify → intent=direct_request, directors=["Nolan"]
handle   → directors_text ILIKE '%Nolan%'  (matches "Christopher Nolan")
           ORDER BY rating DESC LIMIT 5
result   → ["The Dark Knight", "Interstellar", "Oppenheimer", ...]
```

### 7. Title search (known item)

```
user: "the movie Heat"
classify → intent=direct_request, movies=[{title:"Heat", role:"wanted"}]
as_filter → ("name", "Heat")   ← mentioned titles win first slot
handle   → name ILIKE '%Heat%' → ["Heat", "The Heat", ...] by rating
```

### 8. Social detour doesn't lose state

```
(shortlist on screen)
user: "thanks!"
classify → intent=social, action=keep → criteria untouched
social   → packages context; reply → "Anytime — still deciding?"
turn     → options hidden, but typing "3" still picks the third film
```

## Known limitations

- **The classifier sees only the current message** — no history or
  shortlist. "Who directed *it*?" or "the second one" without a title
  can't be resolved yet.
- **Selection is digits only** — saying a title or "the first one" goes
  back through classification instead of picking.
- **No exclusions** — `seen`-role movies and already-shown titles aren't
  filtered out; "I've seen that" can't remove an item.
- **No exit intent** — the graph only ends on a pick; "bye" loops socially.
- `years`/`time_periods` are extracted but not yet applied as `date`
  filters; `audience` and duration are semantic-only.
