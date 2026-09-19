"""Turn a results JSON into a side-by-side markdown report.

    uv run python -m benchmarks.report                      # newest run
    uv run python -m benchmarks.report results/xxx.json     # a specific one
    uv run python -m benchmarks.report --intents recommend --limit 20

One block per query: what the user asked, what the human recommender answered
in the original conversation, and then each system's reply, shortlist, tokens
and seconds in one table — so a pair can be judged by reading across a row
instead of scrolling between two files.
"""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
MAX_REPLY_CHARS = 400


def latest_results() -> Path:
    runs = sorted(RESULTS.glob("*.json"))
    if not runs:
        raise SystemExit("no runs in benchmarks/results — run `python -m benchmarks.run` first")
    return runs[-1]


def cell(text: str) -> str:
    """Markdown tables are one line per row, so nothing may break the row."""
    clean = " ".join((text or "").split())
    if len(clean) > MAX_REPLY_CHARS:
        clean = clean[: MAX_REPLY_CHARS - 1].rstrip() + "…"
    return clean.replace("|", "\\|") or "–"


def titles_cell(titles: list[str]) -> str:
    if not titles:
        return "_(none)_"
    return "<br>".join(f"{index}. {cell(title)}" for index, title in enumerate(titles, 1))


def summary_table(run: dict) -> list[str]:
    systems = run["systems"]
    rows = [
        ("queries", "n_queries", "{:.0f}"),
        ("errors", "n_errors", "{:.0f}"),
        ("latency mean (s)", "seconds_mean", "{:.2f}"),
        ("latency p50 (s)", "seconds_p50", "{:.2f}"),
        ("latency p95 (s)", "seconds_p95", "{:.2f}"),
        ("tokens / query", "tokens_mean", "{:.0f}"),
        ("&nbsp;&nbsp;prompt", "prompt_tokens_mean", "{:.0f}"),
        ("&nbsp;&nbsp;completion", "completion_tokens_mean", "{:.0f}"),
        ("tokens total", "tokens_total", "{:.0f}"),
        ("LLM calls / query", "llm_calls_mean", "{:.2f}"),
        ("failed LLM calls", "llm_errors_total", "{:.0f}"),
        ("embedding calls / query", "embedding_calls_mean", "{:.2f}"),
        ("answers with no titles", "empty_answers", "{:.0f}"),
    ]
    lines = [
        "| | " + " | ".join(systems) + " |",
        "| --- | " + " | ".join("---" for _ in systems) + " |",
    ]
    for label, key, fmt in rows:
        values = " | ".join(fmt.format(run["summary"][system][key]) for system in systems)
        lines.append(f"| {label} | {values} |")
    return lines


def query_block(record: dict, systems: list[str]) -> list[str]:
    human = record.get("human_reply") or {}
    lines = [
        f"### `{record['id']}` — {record['intent']}",
        "",
        f"> **User:** {cell(record['query'])}",
    ]
    if record.get("context"):
        lines.append(f"> _before this turn:_ {cell(' / '.join(record['context']))}")
    if human.get("text") or human.get("titles"):
        answered = ", ".join(human.get("titles") or []) or "no titles"
        lines.append(f"> **Human recommender:** {cell(human.get('text', ''))} — _{cell(answered)}_")
    lines += [
        "",
        "| | " + " | ".join(systems) + " |",
        "| --- | " + " | ".join("---" for _ in systems) + " |",
        "| reply | " + " | ".join(cell(record[system]["reply"]) for system in systems) + " |",
        "| titles | " + " | ".join(titles_cell(record[system]["titles"]) for system in systems) + " |",
        "| tokens | "
        + " | ".join(
            f"{record[system]['total_tokens']} "
            f"({record[system]['prompt_tokens']}p / {record[system]['completion_tokens']}c)"
            for system in systems
        )
        + " |",
        "| time | " + " | ".join(f"{record[system]['seconds']:.2f}s" for system in systems) + " |",
    ]
    errors = {system: record[system].get("error") for system in systems}
    if any(errors.values()):
        lines.append("| error | " + " | ".join(cell(errors[system] or "") for system in systems) + " |")
    lines.append("")
    return lines


def build(run: dict, records: list[dict]) -> str:
    systems = run["systems"]
    lines = [
        "# Benchmark report",
        "",
        f"- Run: `{run['started_at']}`",
        f"- Source: `{run['source']}`",
        f"- Queries in this report: {len(records)} of {run['n_queries']}",
        f"- Intents: {', '.join(run['intents'])}",
        "",
        "## Totals",
        "",
        *summary_table(run),
        "",
        "## Answers, pair by pair",
        "",
    ]
    for record in records:
        lines += query_block(record, systems)
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="?", help="results JSON (default: the newest)")
    parser.add_argument("--out", default="", help="output path (default: alongside the JSON)")
    parser.add_argument("--limit", type=int, default=0, help="first N queries only")
    parser.add_argument("--intents", default="", help="comma-separated ReDial intents")
    parser.add_argument(
        "--only-answered",
        action="store_true",
        help="skip turns where no system returned a shortlist",
    )
    args = parser.parse_args()

    path = Path(args.results) if args.results else latest_results()
    run = json.loads(path.read_text())
    systems = run["systems"]

    records = run["records"]
    intents = set(filter(None, args.intents.split(",")))
    if intents:
        records = [record for record in records if record["intent"] in intents]
    if args.only_answered:
        records = [
            record for record in records if any(record[system]["titles"] for system in systems)
        ]
    if args.limit:
        records = records[: args.limit]
    if not records:
        raise SystemExit("no records left after filtering")

    out = Path(args.out) if args.out else path.with_suffix(".md")
    out.write_text(build(run, records))
    print(f"wrote {out} — {len(records)} queries, {len(systems)} systems")


if __name__ == "__main__":
    main()
