"""Score a finished run against the catalog — offline, after the fact.

    uv run python -m benchmarks.score                    # newest run
    uv run python -m benchmarks.score results/xxx.json

Tokens and latency say what an answer costs. They say nothing about whether
it is any good, and a comparison that reports only cost can only ever favour
the cheapest system. This adds the other half, and needs no annotator:

1. **Grounded** — share of recommended titles that exist in `movies`. A title
   we cannot resolve cannot be shown, rated or played; it is a dead end in the
   UI however plausible it sounds.
2. **Constraints** — when the ReDial annotation says the user asked for a
   genre, the check reads the `genres` column, never the model's claim about
   the film. Unresolvable titles fail by definition: unverifiable compliance
   is not compliance.
3. **Relevance** — cosine between the query embedding and the film's stored
   description embedding. Reported penalized (ungrounded titles score 0) and
   grounded-only; the gap between the two is the cost of hallucinating.
4. **Agreement with the human** — the ReDial recommender's own answer, both as
   an exact hit and as distance in description space.

Scored turns are only those where the system returned a shortlist. Turns
nobody answered are counted separately (`answered`) rather than folded in, so
declining to answer neither flatters a system nor is hidden from the reader.

Writes `<run>-scored.json` and prints the table.
"""

import argparse
import json
import re
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
ANNOTATIONS = Path(__file__).resolve().parent.parent / "annotations" / "redial_test_query_labels.json"

ROWS = [
    ("answered turns", "answered_rate", "{:.1%}"),
    ("grounded in catalog", "grounded", "{:.1%}"),
    ("constraints satisfied", "constraints", "{:.1%}"),
    ("relevance (penalized)", "relevance", "{:.3f}"),
    ("relevance (grounded only)", "relevance_grounded", "{:.3f}"),
    ("similarity to human pick", "human_similarity", "{:.3f}"),
    ("exact hit on human pick", "human_hit", "{:.1%}"),
    ("tokens / answered turn", "tokens_per_answer", "{:.0f}"),
]


def latest_results() -> Path:
    runs = sorted(path for path in RESULTS.glob("*.json") if not path.stem.endswith("-scored"))
    if not runs:
        raise SystemExit("no runs in benchmarks/results — run `python -m benchmarks.run` first")
    return runs[-1]


def genre_constraints(catalog) -> dict[str, list[str]]:
    """Genres the annotators recorded, keyed by query id.

    Matched against the catalog's own genre vocabulary rather than the agent's
    alias table: the grader must not share code with a system it grades.
    """
    vocabulary = {genre for row in catalog.rows for genre in (row.get("genres") or [])}
    annotations = json.loads(ANNOTATIONS.read_text())
    constraints = {}
    for conversation in annotations["conversations"]:
        for turn in conversation["turns"]:
            if turn["role"] != "seeker":
                continue
            wanted = sorted(
                {
                    genre
                    for phrase in (turn.get("slots") or {}).get("genres_moods") or []
                    for genre in vocabulary
                    if phrase.strip().casefold() == genre.casefold()
                }
            )
            if wanted:
                constraints[f"{conversation['conversation_id']}_{turn['message_id']}"] = wanted
    return constraints


def mean(values):
    values = [value for value in values if value is not None]
    return statistics.fmean(values) if values else None


def score_answer(catalog, scorer_vector, record, answer, constraints) -> dict:
    from benchmarks.catalog import cosine, satisfies

    titles = answer["titles"]
    rows = [catalog.resolve(title) for title in titles]
    resolved = [row for row in rows if row]
    score = {"n_titles": len(titles), "grounded": len(resolved) / len(titles)}

    if constraints:
        verdicts = [
            bool(row) and all(satisfies(row, {"genres": constraints}).values()) for row in rows
        ]
        score["constraints"] = sum(verdicts) / len(verdicts)

    vectors = catalog.description_vectors([row["id"] for row in resolved])
    similarities = [
        cosine(scorer_vector, vectors.get(row["id"]) or []) if row else 0.0 for row in rows
    ]
    score["relevance"] = statistics.fmean(similarities)
    grounded = [value for value in similarities if value > 0]
    score["relevance_grounded"] = statistics.fmean(grounded) if grounded else None

    human = [
        row
        for row in (catalog.resolve(title) for title in record["human_reply"]["titles"])
        if row
    ]
    if human and resolved:
        human_vectors = catalog.description_vectors([row["id"] for row in human])
        score["human_hit"] = any(row["id"] in {h["id"] for h in human} for row in resolved)
        best = [
            max(
                cosine(vectors.get(row["id"]) or [], human_vectors.get(h["id"]) or [])
                for h in human
            )
            for row in resolved
            if vectors.get(row["id"])
        ]
        score["human_similarity"] = statistics.fmean(best) if best else None
    return score


def summarize(records, system) -> dict:
    scored = [record["scores"][system] for record in records if record["scores"].get(system)]
    answered = len(scored)
    tokens = sum(record[system]["total_tokens"] for record in records)
    return {
        "n_turns": len(records),
        "answered": answered,
        "answered_rate": answered / len(records),
        "grounded": mean(score["grounded"] for score in scored),
        "constraints": mean(score.get("constraints") for score in scored),
        "relevance": mean(score["relevance"] for score in scored),
        "relevance_grounded": mean(score.get("relevance_grounded") for score in scored),
        "human_similarity": mean(score.get("human_similarity") for score in scored),
        "human_hit": mean(
            float(score["human_hit"]) for score in scored if "human_hit" in score
        ),
        "tokens_per_answer": tokens / answered if answered else None,
    }


def table(summary: dict) -> str:
    systems = list(summary)
    width = max(len(label) for label, _, _ in ROWS) + 2
    lines = ["", "  " + "".ljust(width) + "".join(name.rjust(18) for name in systems)]
    for label, key, fmt in ROWS:
        values = [summary[system].get(key) for system in systems]
        if all(value is None for value in values):
            continue
        cells = "".join(
            (fmt.format(value) if value is not None else "–").rjust(18) for value in values
        )
        lines.append("  " + label.ljust(width) + cells)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?")
    parser.add_argument("--intents", default="", help="comma-separated ReDial intents")
    args = parser.parse_args()

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from langchain_mistralai import MistralAIEmbeddings

    from benchmarks.catalog import Catalog
    from data_management.theme_embeddings_pipeline import EMBEDDING_MODEL

    path = Path(args.results) if args.results else latest_results()
    run = json.loads(path.read_text())
    systems = run["systems"]
    records = run["records"]
    intents = set(filter(None, args.intents.split(",")))
    if intents:
        records = [record for record in records if record["intent"] in intents]
    if not records:
        raise SystemExit("no records left after filtering")

    catalog = Catalog()
    constraints = genre_constraints(catalog)
    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    print(f"scoring {len(records)} turns x {len(systems)} systems against {len(catalog)} movies")
    for index, record in enumerate(records, start=1):
        record["scores"] = {}
        if not any(record[system]["titles"] for system in systems):
            continue
        vector = embeddings.embed_query(record["query"])
        for system in systems:
            if record[system]["titles"]:
                record["scores"][system] = score_answer(
                    catalog, vector, record, record[system], constraints.get(record["id"])
                )
        if index % 25 == 0:
            print(f"  {index}/{len(records)}")

    summary = {system: summarize(records, system) for system in systems}
    run["quality"] = summary
    run["records"] = records
    out = path.with_name(f"{path.stem}-scored.json")
    out.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(table(summary))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
