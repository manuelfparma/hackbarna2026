"""Grades a shortlist against what the user had asked for by that turn.

A judge, not a lookup: "is this a Disney film", "is this a princess film" and
"does it end romantically" are not columns in `movies` — `studios` is patchy
and there is no princess flag — so the question goes to a model that knows
the films. It answers one title at a time, in JSON, with a short reason, so a
reader can overrule it.

Its own tokens are never metered: the judge runs after both arms are done, on
a model nobody attached a meter to, so grading can never land in a system's
cost.
"""

import json
import re

from agent.llm import build_llm

JUDGE_PROMPT = """You are grading a movie recommendation.

The user asked for: {asked}

For each film below, answer each question with true or false, from your own
knowledge of the film. Judge the film itself, not how it was marketed: a
person named as a criterion has to actually be in it (or have directed it),
a genre has to be what the film is rather than a scene it contains, and a
subject has to be what the film is about.

Films:
{films}

Reply with JSON only:
{{"verdicts": [{{"title": "...", {keys}, "why": "<8 words max>"}}]}}"""

# Definitions worth pinning down because a model left to itself is loose
# about them. Anything not listed is asked about by name.
CRITERION_QUESTIONS = {
    "disney": "is it a Walt Disney Pictures / Disney Animation / Pixar production, not merely distributed by Disney?",
    "princess": "is a princess (or a queen-to-be) the central character?",
    "romantic ending": "does it end with a romance resolved happily?",
}


def _question(name: str) -> str:
    """How a criterion is put to the judge."""
    return CRITERION_QUESTIONS.get(name, f'does the film match "{name}"?')


class Judge:
    def __init__(self, llm=None):
        # Deliberately unmetered and separate from both arms' models.
        self._llm = llm or build_llm(temperature=0)

    def grade(self, titles: list[str], criteria: list[str]) -> list[dict]:
        """One verdict dict per title: the criteria as booleans, plus `why`."""
        if not titles:
            return []
        keys = ", ".join(f'"{name}": true|false' for name in criteria)
        asked = "; ".join(f"{name} ({_question(name)})" for name in criteria)
        prompt = JUDGE_PROMPT.format(
            asked=asked,
            films="\n".join(f"- {title}" for title in titles),
            keys=keys,
        )
        try:
            raw = self._llm.invoke(prompt).content
        except Exception as error:
            return [{"title": title, "error": str(error)[:120]} for title in titles]

        verdicts = _parse(raw)
        by_title = {str(v.get("title", "")).casefold(): v for v in verdicts}
        graded = []
        for title in titles:
            verdict = by_title.get(title.casefold(), {})
            graded.append(
                {
                    "title": title,
                    **{name: bool(verdict.get(name)) for name in criteria},
                    "all": all(bool(verdict.get(name)) for name in criteria),
                    "why": str(verdict.get("why", ""))[:80],
                    "judged": bool(verdict),
                }
            )
        return graded


def _parse(raw: str) -> list[dict]:
    blob = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.+?)```", blob, re.S)
    if fenced:
        blob = fenced.group(1).strip()
    brace = re.search(r"\{.*\}", blob, re.S)
    if not brace:
        return []
    try:
        payload = json.loads(brace.group(0))
    except json.JSONDecodeError:
        return []
    return [v for v in payload.get("verdicts", []) if isinstance(v, dict)]


def score(graded: list[dict], criteria: list[str]) -> dict:
    """Per-criterion hit counts for one shortlist, plus the strict all-of."""
    total = len(graded)
    summary = {name: sum(1 for row in graded if row.get(name)) for name in criteria}
    summary["all"] = sum(1 for row in graded if row.get("all"))
    summary["n"] = total
    return summary
