# Benchmark

One conversation, two ways of answering it. The conversation is a real
session captured from `agent/orchestrator.py`, in
[`disney_princess_romance.json`](disney_princess_romance.json):

| turn | the user says | what must hold by then |
| --- | --- | --- |
| 1 | *"I want to a disney movie"* | disney |
| 2 | *"princess"* | disney **+** princess |
| 3 | *"with a romantic ending"* | disney **+** princess **+** romantic ending |

Criteria accumulate. A turn that satisfies only its own words has forgotten
the conversation, and that is exactly what this case is built to catch.

```bash
# from kinofiles/
uv run python -m benchmarks.run                    # both arms, then judge
uv run python -m benchmarks.run --systems agent    # one arm only
uv run python -m benchmarks.run --no-judge         # cost and latency only
```

## The two arms

| name | what it is |
| --- | --- |
| `agent` | the real flow: `OrquestratorAgent` driven through its interrupts, classify → merge criteria → retrieve → reply, one round per turn |
| `catalog_prompt` | no agent: one Mistral call per turn with **all 1000 films in the prompt** and the conversation so far |

The baseline is given the whole catalog on purpose. Both arms then pick from
the same films, so what is being compared is not access to the catalog but
the machinery around it — and the price of that machinery shows up where it
belongs, in the tokens.

Each arm carries **its own** previous shortlists into the next turn, not the
other's. Two systems compared across turns have to live with their own
mistakes.

The agent arm is driven exactly as a person drives it from the terminal: the
script answers the name prompt, states the preference, and abstains at each
vote (anything that is not a digit reads as "nobody picked one"), which is
what sends the flow to `refine` and asks what to change. Naming the
participants is recorded apart, as `setup`, because it happens before any
recommendation.

## What is measured

**Cost** — provider-reported tokens (`usage_metadata`, never estimated),
collected by a callback attached to the chat models themselves. The
orchestrator builds its own models, and `Classifier` wraps its one in
`with_structured_output`, so the meter is installed by replacing the factories
in that module's namespace before the graph is built: classifier, reply
composer and anything a node adds later are all counted. Failed calls and
retries count too — a flaky provider is a real cost. Embedding calls are
counted separately, by call, because `mistral-embed` reports no token usage.

**Time** — wall clock per turn, and the total.

**Quality** (`judge.py`) — an LLM judge, because none of this is a column in
`movies`: `studios` is patchy and there is no princess flag. Each shortlist is
graded title by title against the criteria in force at that turn, in JSON,
with a one-line reason so a reader can overrule it. The report gives the count
per criterion and the strict `ALL of them`.

The judge runs **after** both arms have finished, on a model nobody attached a
meter to, so grading can never land in a system's cost.

## Files

| file | role |
| --- | --- |
| `disney_princess_romance.json` | the conversation: question, cumulative criteria, and the captured reference run |
| `run.py` | CLI: replays the conversation on both arms, judges, writes the results JSON |
| `systems.py` | the two arms, and the catalog block the baseline is shown |
| `judge.py` | the LLM judge and its scoring |
| `catalog.py` | reads `movies` once and formats it for the prompt |
| `instrumentation.py` | token / embedding / latency meters |
