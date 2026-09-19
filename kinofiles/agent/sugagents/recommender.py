"""Recommendation subagent (MOCK): a single LLM call.

Once the team's search engine is ready, only the body of `recommend` needs to
change. The signature stays the same.
"""

import re

PROMPT = """You are a movie recommender.

Request: {request}
Feedback from previous rounds:
{feedback}

Answer with exactly 3 movie titles, one per line, nothing else."""

# Leading bullet or numbering ("- ", "2) "), but not a title like "1917".
BULLET = re.compile(r"^\s*(?:[-*\u2022]|\d+[.)])\s*")


class Recommender:
    def __init__(self, llm):
        self.llm = llm

    def recommend(self, request: str, feedback: list[str]) -> list[str]:
        """Return a list of titles based on the request and the feedback so far."""
        notes = "\n".join(f"- {f}" for f in feedback) or "- none"
        answer = self.llm.invoke(PROMPT.format(request=request, feedback=notes))
        lines = (BULLET.sub("", line).strip() for line in answer.content.splitlines())
        return [line for line in lines if line]
