"""Queries for the benchmark, read straight from the ReDial annotations.

`annotations/redial_test_query_labels.json` is the only source: every seeker
turn is a query, exactly as the user typed it. What the human recommender
answered next comes along for the ride — not scored here, just carried into
the results so quality can be judged by eye afterwards.
"""

import json
import re
from pathlib import Path

ANNOTATIONS = (
    Path(__file__).resolve().parent.parent
    / "annotations"
    / "redial_test_query_labels.json"
)

# Turns where the user is actually after a film. `social` ("hi", "thanks") and
# `other` (corrections, meta) would measure politeness, not recommendation.
DEFAULT_INTENTS = ("recommend", "prefer", "more", "ask_item", "known_item", "feedback")
CONTEXT_TURNS = 2


def _resolved(turn: dict) -> str:
    """Text with @ids swapped for titles — the agent cannot read an id."""
    return (turn.get("text_resolved") or turn.get("text") or "").strip()


def _human_reply(conversation: dict, index: int) -> dict:
    """What the human recommender said next, with the titles they named."""
    text, titles = [], []
    for turn in conversation["turns"][index + 1 :]:
        if turn["role"] == "seeker":
            break
        text.append(_resolved(turn))
        for mention in re.findall(r"@(\d+)", turn["text"]):
            title = conversation["movie_mentions"].get(mention)
            if title:
                titles.append(title.strip())
    return {"text": " ".join(text).strip(), "titles": list(dict.fromkeys(titles))}


def load(intents: tuple[str, ...] = DEFAULT_INTENTS) -> list[dict]:
    annotations = json.loads(ANNOTATIONS.read_text())
    queries = []
    for conversation in annotations["conversations"]:
        turns = conversation["turns"]
        for index, turn in enumerate(turns):
            if turn["role"] != "seeker":
                continue
            if intents and turn.get("intent") not in intents:
                continue
            text = _resolved(turn)
            if not text:
                continue
            queries.append(
                {
                    "id": f"{conversation['conversation_id']}_{turn['message_id']}",
                    "conversation_id": conversation["conversation_id"],
                    "intent": turn.get("intent"),
                    "query": text,
                    # Earlier turns are not sent to either system — this runs
                    # one turn at a time — but a short query like "I mean that
                    # one" is unreadable without them.
                    "context": [
                        f"{earlier['role']}: {_resolved(earlier)}"
                        for earlier in turns[max(0, index - CONTEXT_TURNS) : index]
                    ],
                    "human_reply": _human_reply(conversation, index),
                }
            )
    return queries
