# Benchmarks

Sends every seeker turn from `annotations/redial_test_query_labels.json` to
both systems and records **how long it took**, **how many tokens it cost**
and — offline, afterwards — **how good the shortlist was**, graded against the
catalog. Every answer is also written to the results JSON next to the human
recommender's reply from the original conversation, to be judged by hand.

Cost and quality are deliberately separate steps. A comparison that reports
only tokens and seconds can only ever favour the cheapest system, and the
scorer must never be able to spend tokens inside a measured turn.

```bash
# from kinofiles/
uv run python -m benchmarks.run                      # every movie-seeking turn (226)
uv run python -m benchmarks.run --limit 20           # first 20
uv run python -m benchmarks.run --intents recommend  # one ReDial intent only
uv run python -m benchmarks.run --systems agent      # one system only
```

Then grade the shortlists against the catalog, and turn the run into a
side-by-side markdown report:

```bash
uv run python -m benchmarks.score                    # newest run
uv run python -m benchmarks.score --intents recommend
```


```bash
uv run python -m benchmarks.report                          # newest run, every query
uv run python -m benchmarks.report --limit 20               # first 20
uv run python -m benchmarks.report --intents recommend      # one intent
uv run python -m benchmarks.report --only-answered          # skip turns nobody answered
uv run python -m benchmarks.report results/xxx.json --out /tmp/report.md
```

## The two systems

| name | what it is |
| --- | --- |
| `agent` | the full LangGraph agent: classify → retrieve (pgvector + catalog filters) → reply |
| `mistral_simple` | one Mistral call, no retrieval — the baseline the agent's extra tokens have to justify |

Each query runs on its own fresh thread: one turn in, one answer out. The
agent's multi-turn behaviour (accumulated criteria, feedback) is out of scope,
which is also why short turns like *"I mean American Pie"* look worse than they
would in a real conversation. `context` in the results carries the two previous
turns so a reader can tell those apart — neither system is sent them.

## What is measured

### Quality (`score.py`, offline)

The catalog is the ground truth, so no annotator is needed:

- **Grounded** — share of recommended titles that exist in `movies`. A title
  that does not resolve cannot be shown, rated or played.
- **Constraints** — where the ReDial annotation records a genre the user
  asked for, it is checked against the `genres` column, never against the
  model's claim about the film. Unresolvable titles fail by definition.
- **Relevance** — cosine between the query embedding and the film's stored
  description embedding, reported penalized (ungrounded titles score 0) and
  grounded-only. The gap is the cost of hallucinating.
- **Agreement with the human** — the ReDial recommender's own answer, as an
  exact hit and as distance in description space. One human's suggestion is
  not a complete relevance judgment, so read it as a relative signal.

Only turns where a system returned a shortlist are scored; `answered turns`
is reported separately so declining to answer neither flatters a system nor
hides.

### Cost

- **Tokens** are provider-reported (`usage_metadata`), never estimated,
  collected by a callback attached to the chat models themselves — so the
  agent's classifier *and* reply calls are both counted, and a node added
  later is counted without touching the harness.
- **Failed LLM calls** are counted too. `ReplyComposer` swallows its
  exceptions and answers with canned copy, so without this a dead provider
  looks like a cheap, fast agent instead of a broken one. A run starts with a
  preflight ping of both providers for the same reason.
- **Embedding calls** are counted separately, by call and by characters:
  `mistral-embed` reports no token usage.
- Retries are included. A flaky provider is a real cost.

Systems are interleaved per query, so a slow afternoon on someone's API lands
on both equally instead of penalising whichever arm ran last.

## Output

`results/<timestamp>.md` — the totals, then one block per query: the user's
turn, what the human recommender answered in the original conversation, and a
table with each system's reply, shortlist, tokens and seconds.

`results/<timestamp>.json` — the same thing as data:

```jsonc
{
  "summary": { "agent": {...}, "mistral_simple": {...} },   // tokens, latency, errors
  "records": [{
    "id": "20001_204171",
    "intent": "recommend",
    "query": "Hi I am looking for a movie like Super Troopers (2001)",
    "context": ["seeker: ...", "recommender: ..."],
    "human_reply": {"text": "You should watch Police Academy (1984)",
                    "titles": ["Police Academy  (1984)"]},
    "agent":          {"reply": "...", "titles": [...], "seconds": 2.09, "total_tokens": 1167, ...},
    "mistral_simple": {"reply": "...", "titles": [...], "seconds": 1.86, "total_tokens": 239,  ...}
  }]
}
```

## Files

| file | role |
| --- | --- |
| `run.py` | CLI: runs both systems over every query, writes the JSON |
| `report.py` | CLI: turns a results JSON into the pair-by-pair markdown report |
| `score.py` | CLI: grades a finished run's shortlists against the catalog |
| `catalog.py` | in-memory catalog snapshot, title resolution, constraint checks |
| `systems.py` | the two arms behind one `answer(query)` interface |
| `dataset.py` | reads the queries out of the ReDial annotations |
| `instrumentation.py` | token / embedding / latency meters |
