"""Run both systems over the ReDial queries and write the answers to JSON.

    uv run python -m benchmarks.run                  # every seeker turn
    uv run python -m benchmarks.run --limit 20       # first 20
    uv run python -m benchmarks.run --intents recommend,prefer

Measures tokens and latency only. Quality is left to whoever reads
`results/<timestamp>.json`, which holds every answer next to the human
recommender's reply from the original conversation.

Systems are interleaved per query rather than run one after another, so a
slow afternoon on someone's API lands on both equally instead of making
whichever arm ran last look bad.
"""

import argparse
import json
import statistics
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="agent,mistral_simple")
    parser.add_argument("--limit", type=int, default=0, help="first N queries only")
    parser.add_argument(
        "--intents",
        default="",
        help="comma-separated ReDial intents; empty means the usual movie-seeking ones",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.5,
        help="seconds between calls; free API tiers rate-limit hard",
    )
    parser.add_argument("--tag", default="", help="suffix for the output filename")
    return parser.parse_args()


def preflight() -> None:
    """Ping both providers before a run that takes minutes to fail politely.

    The agent catches its own reply-model errors and answers with canned copy,
    so a dead provider produces a plausible-looking run instead of an obvious
    failure. Better to say so up front than to find it in the numbers.
    """
    from agent.llm import build_llm, build_mistral_llm

    for label, factory in (("mistral", build_mistral_llm), ("nebius", build_llm)):
        try:
            factory().invoke("ping")
            print(f"  {label}: ok")
        except Exception as error:
            print(f"  {label}: UNAVAILABLE ({str(error).splitlines()[0][:110]})")
            if label == "nebius":
                print("    -> agent replies fall back to canned copy; shortlists are unaffected")


def summarize(records: list[dict], system: str) -> dict:
    """Totals and percentiles for one system. Failures stay in the numbers."""
    rows = [record[system] for record in records]
    seconds = sorted(row["seconds"] for row in rows)

    def percentile(fraction):
        index = min(len(seconds) - 1, int(round(fraction * (len(seconds) - 1))))
        return round(seconds[index], 3)

    return {
        "n_queries": len(rows),
        "n_errors": sum(1 for row in rows if row["error"]),
        "seconds_mean": round(statistics.fmean(seconds), 3),
        "seconds_p50": percentile(0.50),
        "seconds_p95": percentile(0.95),
        "seconds_total": round(sum(seconds), 1),
        "tokens_mean": round(statistics.fmean(row["total_tokens"] for row in rows), 1),
        "prompt_tokens_mean": round(statistics.fmean(row["prompt_tokens"] for row in rows), 1),
        "completion_tokens_mean": round(
            statistics.fmean(row["completion_tokens"] for row in rows), 1
        ),
        "tokens_total": sum(row["total_tokens"] for row in rows),
        "llm_calls_mean": round(statistics.fmean(row["llm_calls"] for row in rows), 2),
        "llm_errors_total": sum(row["llm_errors"] for row in rows),
        "embedding_calls_mean": round(
            statistics.fmean(row["embedding_calls"] for row in rows), 2
        ),
        "empty_answers": sum(1 for row in rows if not row["titles"]),
    }


def console_table(summary: dict) -> str:
    systems = list(summary)
    rows = [
        ("queries", "{:.0f}", "n_queries"),
        ("errors", "{:.0f}", "n_errors"),
        ("latency mean (s)", "{:.2f}", "seconds_mean"),
        ("latency p50 (s)", "{:.2f}", "seconds_p50"),
        ("latency p95 (s)", "{:.2f}", "seconds_p95"),
        ("tokens / query", "{:.0f}", "tokens_mean"),
        ("  prompt", "{:.0f}", "prompt_tokens_mean"),
        ("  completion", "{:.0f}", "completion_tokens_mean"),
        ("tokens total", "{:.0f}", "tokens_total"),
        ("LLM calls / query", "{:.2f}", "llm_calls_mean"),
        ("failed LLM calls", "{:.0f}", "llm_errors_total"),
        ("embedding calls / query", "{:.2f}", "embedding_calls_mean"),
        ("answers with no titles", "{:.0f}", "empty_answers"),
    ]
    width = max(len(label) for label, _, _ in rows) + 2
    lines = ["", "  " + "".ljust(width) + "".join(name.rjust(18) for name in systems)]
    for label, fmt, key in rows:
        cells = "".join(fmt.format(summary[name][key]).rjust(18) for name in systems)
        lines.append("  " + label.ljust(width) + cells)
    return "\n".join(lines)


def main():
    args = parse_args()

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from benchmarks import dataset
    from benchmarks.systems import Meters, build_systems

    intents = tuple(filter(None, args.intents.split(","))) or dataset.DEFAULT_INTENTS
    queries = dataset.load(intents)
    if args.limit:
        queries = queries[: args.limit]
    if not queries:
        raise SystemExit("no queries selected")

    # Meters are installed before the systems are built: the embedding patch
    # has to be in place before any node constructs its embedder.
    meters = Meters()
    systems = build_systems(args.systems.split(","), meters)

    print("preflight:")
    preflight()
    started = datetime.now(timezone.utc)
    print(f"\n{len(queries)} queries x {len(systems)} systems ({', '.join(s.name for s in systems)})")

    records = []
    for index, query in enumerate(queries, start=1):
        print(f"\n[{index}/{len(queries)}] {query['id']} ({query['intent']}): {query['query'][:70]}")
        record = dict(query)
        for system in systems:
            try:
                answer = system.answer(query["query"])
            except Exception:
                # One system blowing up must not cost us the rest of the run.
                traceback.print_exc()
                from benchmarks.systems import Answer

                answer = Answer(error="exception")
            record[system.name] = answer.as_dict()
            print(
                f"  {system.name:<16} {answer.cost.seconds:5.1f}s "
                f"{answer.cost.usage.total_tokens:>6} tok  {len(answer.titles)} titles"
                + (f"  ERROR: {answer.error}" if answer.error else "")
            )
            if args.sleep:
                time.sleep(args.sleep)
        records.append(record)

    summary = {system.name: summarize(records, system.name) for system in systems}
    run = {
        "started_at": started.isoformat(timespec="seconds"),
        "source": str(dataset.ANNOTATIONS.relative_to(dataset.ANNOTATIONS.parents[2])),
        "intents": list(intents),
        "systems": [system.name for system in systems],
        "n_queries": len(queries),
        "summary": summary,
        "records": records,
    }

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = started.strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    path = RESULTS / f"{stem}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(console_table(summary))
    print(f"\nwrote {path.relative_to(Path.cwd()) if path.is_relative_to(Path.cwd()) else path}")


if __name__ == "__main__":
    main()
