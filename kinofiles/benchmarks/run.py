"""Run the Disney/princess/romance conversation with and without the agent.

    uv run python -m benchmarks.run                    # both arms, then judge
    uv run python -m benchmarks.run --systems agent    # one arm only
    uv run python -m benchmarks.run --no-judge         # cost and latency only

Measures what the flow costs — tokens and seconds, per turn — and then grades
each shortlist against what the user had asked for by that point: turn 1 must
be Disney, turn 2 Disney and princess, turn 3 Disney, princess and a romantic
ending. Criteria accumulate; that is the whole point of a multi-turn case.

Grading happens after both arms have finished, on a model nobody metered, so
the judge can never spend tokens inside a measured turn.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

CONVERSATION = Path(__file__).resolve().parent / "test_files.json"
RESULTS = Path(__file__).resolve().parent / "results"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", default="agent,catalog_prompt")
    parser.add_argument("--no-judge", action="store_true", help="skip grading")
    parser.add_argument(
        "--llm",
        default="mistral",
        choices=("mistral", "shipped"),
        help="mistral: run the agent on the same model as the baseline (default). "
        "shipped: leave its replies on Nebius, as the app ships.",
    )
    parser.add_argument("--tag", default="", help="suffix for the output filename")
    return parser.parse_args()


def totals(turns: list[dict], setup: dict) -> dict:
    return {
        "seconds": round(sum(t["seconds"] for t in turns) + setup["seconds"], 2),
        "total_tokens": sum(t["total_tokens"] for t in turns) + setup["total_tokens"],
        "prompt_tokens": sum(t["prompt_tokens"] for t in turns) + setup["prompt_tokens"],
        "completion_tokens": sum(t["completion_tokens"] for t in turns)
        + setup["completion_tokens"],
        "llm_calls": sum(t["llm_calls"] for t in turns) + setup["llm_calls"],
        "llm_errors": sum(t["llm_errors"] for t in turns) + setup["llm_errors"],
        "embedding_calls": sum(t["embedding_calls"] for t in turns)
        + setup["embedding_calls"],
    }


def cost_table(run: dict) -> str:
    names = run["systems"]
    rows = [("tokens", "total_tokens"), ("  prompt", "prompt_tokens"),
            ("  completion", "completion_tokens"), ("seconds", "seconds"),
            ("LLM calls", "llm_calls"), ("embedding calls", "embedding_calls")]
    width = 26
    column = max([len(n) for n in names] + [14]) + 2
    lines = ["", "COST", "  " + "".ljust(width) + "".join(n.rjust(column) for n in names)]
    for index, question in enumerate(run["questions"]):
        lines.append("  " + f'turn {index + 1}: "{question}"'[:width].ljust(width))
        for label, key in rows[:4]:
            cells = "".join(
                f"{run['results'][n]['turns'][index][key]:,.0f}".rjust(column)
                if key != "seconds"
                else f"{run['results'][n]['turns'][index][key]:.2f}".rjust(column)
                for n in names
            )
            lines.append("    " + label.ljust(width - 2) + cells)
    lines.append("  " + "TOTAL (incl. setup)".ljust(width))
    for label, key in rows:
        cells = "".join(
            (f"{run['results'][n]['total'][key]:,.0f}" if key != "seconds"
             else f"{run['results'][n]['total'][key]:.2f}").rjust(column)
            for n in names
        )
        lines.append("    " + label.ljust(width - 2) + cells)
    return "\n".join(lines)


def judge_table(run: dict) -> str:
    names = run["systems"]
    column = max([len(n) for n in names] + [14]) + 2
    width = 26
    lines = ["", "JUDGE (titles meeting each criterion, out of 8)"]
    lines.append("  " + "".ljust(width) + "".join(n.rjust(column) for n in names))
    for index, turn in enumerate(run["turns"]):
        lines.append("  " + f"turn {index + 1}: {', '.join(turn['criteria'])}"[:width].ljust(width))
        for name in turn["criteria"] + ["all"]:
            cells = ""
            for system in names:
                score = run["results"][system]["scores"][index]
                cells += f"{score.get(name, 0)}/{score.get('n', 0)}".rjust(column)
            label = "  ALL of them" if name == "all" else f"  {name}"
            lines.append("    " + label.ljust(width - 2) + cells)
    return "\n".join(lines)


def shortlists(run: dict) -> str:
    lines = ["", "SHORTLISTS"]
    for index, turn in enumerate(run["turns"]):
        lines.append(f'\n  turn {index + 1} — "{turn["question"]}"  ({", ".join(turn["criteria"])})')
        for system in run["systems"]:
            result = run["results"][system]
            graded = {row["title"]: row for row in (result["graded"][index] if result.get("graded") else [])}
            titles = []
            for title in result["turns"][index]["titles"]:
                row = graded.get(title)
                mark = "" if not row else (" ✔" if row.get("all") else " ✘")
                titles.append(f"{title}{mark}")
            lines.append(f"    {system:<16} {', '.join(titles) or '(nothing)'}")
    return "\n".join(lines)


def main():
    args = parse_args()

    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")

    from benchmarks.systems import Meters, build_systems

    conversation = json.loads(CONVERSATION.read_text())
    questions = [turn["question"] for turn in conversation]

    # Meters are installed before the systems are built: the embedding patch
    # and the factory patch both have to be in place before any node
    # constructs a model.
    meters = Meters()
    meters.patch_orchestrator(all_mistral=args.llm == "mistral")
    systems = build_systems(args.systems.split(","), meters, questions)

    started = datetime.now(timezone.utc)
    print(f"agent replies on: {'mistral (same model as the baseline)' if args.llm == 'mistral' else 'nebius, as shipped'}")
    results = {}
    for system in systems:
        print(f"\n=== {system.name}")
        turns = system.run()
        for index, turn in enumerate(turns, start=1):
            print(
                f"  turn {index}: {turn.cost.seconds:5.1f}s "
                f"{turn.cost.usage.total_tokens:>6} tok  {len(turn.titles)} titles"
                + (f"  ERROR: {turn.error}" if turn.error else "")
            )
            print(f"           {', '.join(turn.titles) or '(nothing)'}")
        rows = [turn.as_dict() for turn in turns]
        results[system.name] = {
            "turns": rows,
            "setup": system.setup_cost.as_dict(),
            "total": totals(rows, system.setup_cost.as_dict()),
        }

    run = {
        "started_at": started.isoformat(timespec="seconds"),
        "conversation": CONVERSATION.name,
        "questions": questions,
        "turns": [{"question": t["question"], "criteria": t["criteria"]} for t in conversation],
        "systems": [s.name for s in systems],
        "agent_llm": args.llm,
        "results": results,
    }

    if not args.no_judge:
        from benchmarks.judge import Judge, score

        judge = Judge()
        print("\njudging…")
        for name, result in results.items():
            graded, scores = [], []
            for index, turn in enumerate(conversation):
                verdicts = judge.grade(result["turns"][index]["titles"], turn["criteria"])
                graded.append(verdicts)
                scores.append(score(verdicts, turn["criteria"]))
            result["graded"] = graded
            result["scores"] = scores

    print(cost_table(run))
    if not args.no_judge:
        print(judge_table(run))
    print(shortlists(run))

    RESULTS.mkdir(parents=True, exist_ok=True)
    stem = started.strftime("%Y%m%d-%H%M%S") + (f"-{args.tag}" if args.tag else "")
    path = RESULTS / f"{stem}.json"
    path.write_text(json.dumps(run, indent=2, ensure_ascii=False))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
