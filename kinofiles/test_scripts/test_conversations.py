"""Drive RecommendationAgent through scripted conversations end-to-end.

Runs the real graph: Mistral classification + embeddings, Nebius replies,
live Supabase. Each conversation feeds a fixed script of user messages and
prints the interrupt payload (narrated text + rendered options) plus the
internal state that produced it, so routing decisions are visible.

Usage (from kinofiles/):  uv run python test_scripts/test_conversations.py
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command

from agent.nodes.criteria import normalize_criteria
from agent.recomedation_agent import RecommendationAgent

INITIAL = {
    "feedback": [],
    "movies": [],
    "search_criteria": normalize_criteria({}),
    "result": {},
    "history": [],
    "show_options": False,
}

CONVERSATIONS = [
    ("C1 simple theme query -> pick", [
        "something melancholy about family",
        "2",
    ]),
    ("C2 genre filter + add another + pick", [
        "I want a comedy",
        "also some action",
        "1",
    ]),
    ("C3 similar-to a named film -> pick", [
        "a movie like The Dark Knight",
        "3",
    ]),
    ("C4 fuzzy person lookup -> pick", [
        "movies by Nolan",
        "2",
    ]),
    ("C5 feedback steers the search", [
        "a heist movie",
        "these are too mainstream, something darker",
        "1",
    ]),
    ("C6 factual question mid-shortlist", [
        "a thriller please",
        "who directed Inception?",
        "1",
    ]),
    ("C7 social detour keeps hidden shortlist", [
        "recommend a horror movie",
        "thanks!",
        "2",
    ]),
    ("C8 genre + theme + fact + pick", [
        "I want a science fiction movie",
        "make it about space exploration",
        "what's Interstellar about?",
        "2",
    ]),
]


def brief_state(agent, config):
    values = agent.graph.get_state(config).values
    criteria = {
        k: v for k, v in normalize_criteria(values.get("search_criteria")).items() if v
    }
    result = values.get("result") or {}
    bits = [
        f"intent={values.get('intent')}",
        f"kind={result.get('kind')}",
    ]
    if criteria:
        bits.append(f"criteria={criteria}")
    if result.get("error"):
        bits.append(f"error={result['error']}")
    return " | ".join(bits)


def run(agent, title, script):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")
    config = {"configurable": {"thread_id": title.split()[0].lower()}}
    event = agent.graph.invoke({**INITIAL, "request": script[0]}, config)
    print(f"YOU > {script[0]}")
    for message in script[1:]:
        if "__interrupt__" not in event:
            break
        pending = event["__interrupt__"][0].value
        print(f"AGENT > {pending['text']}")
        for i, option in enumerate(pending["options"], 1):
            print(f"        {i}. {option}")
        print(f"      [{brief_state(agent, config)}]")
        print(f"YOU > {message}")
        event = agent.graph.invoke(Command(resume=message), config)
    if "__interrupt__" in event:
        pending = event["__interrupt__"][0].value
        print(f"AGENT > {pending['text']}")
        for i, option in enumerate(pending["options"], 1):
            print(f"        {i}. {option}")
        print(f"      [{brief_state(agent, config)}]")
        print("      (conversation still open — script exhausted)")
    else:
        print(f"AGENT > session ended, choice={event.get('choice')!r}")
        print(f"      {event.get('farewell', '')}")


def main():
    agent = RecommendationAgent()
    for title, script in CONVERSATIONS:
        try:
            run(agent, title, script)
        except Exception as e:
            print(f"!! {title} crashed: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
