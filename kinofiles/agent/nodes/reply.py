"""Turn raw capability results into short, conversational narration."""

import json
from typing import Any

MAX_HISTORY_TURNS = 6

SYSTEM_PROMPT = """You are the conversational voice of a movie assistant.
Write a warm, natural reply of at most three short sentences.

Rules:
- The movie choices are rendered separately on screen. Do not repeat them as a list.
- Never mention database IDs, embeddings, similarity scores, ranking, tools, or internal routing.
- Never introduce a movie that is not present in the capability result.
- If the result includes facts about a movie, answer the user's question using only those facts.
- Acknowledge the user's latest request and use recent context only when useful.
- If the result contains an error or no choices, explain that briefly and ask one useful follow-up.
- Do not tell the user how to select a choice; the application adds that instruction.
"""


class ReplyComposer:
    def __init__(self, llm):
        self.llm = llm

    def compose(
        self,
        *,
        request: str,
        intent: str,
        entities: dict[str, Any],
        result: dict[str, Any],
        history: list[dict[str, str]],
        feedback: list[str],
        movies: list[str],
        search_criteria: dict[str, Any] | None = None,
    ) -> str:
        """Generate narration, falling back to deterministic copy on failure."""
        context = {
            "request": request,
            "intent": intent,
            "entities": entities,
            "accumulated_search_criteria": search_criteria or {},
            "capability_result": result,
            "current_shortlist": movies,
            "feedback": feedback,
            "recent_history": history[-MAX_HISTORY_TURNS:],
        }
        try:
            answer = self.llm.invoke(
                f"{SYSTEM_PROMPT}\n\nConversation context:\n"
                f"{json.dumps(context, ensure_ascii=False)}"
            )
            text = answer.content.strip()
            if text:
                return text
        except Exception:
            pass
        return self.fallback(intent, result)

    @staticmethod
    def fallback(intent: str, result: dict[str, Any]) -> str:
        """Return usable copy without another model or external call."""
        error = result.get("error")
        titles = result.get("titles") or []
        if error:
            return f"{error} Could you give me another detail to work with?"
        facts = result.get("facts")
        if facts:
            details = []
            if facts.get("date"):
                details.append(str(facts["date"]))
            directors = facts.get("directors") or []
            if directors:
                details.append("directed by " + ", ".join(directors))
            if facts.get("rating"):
                details.append(f"rated {facts['rating']}")
            suffix = f" ({'; '.join(details)})" if details else ""
            return f"Here's what I have on {facts.get('name', 'that title')}{suffix}."
        if intent == "social":
            return "I'm here to help you find something worth watching. What are you in the mood for?"
        if intent == "feedback":
            return "Got it — I've adjusted the suggestions based on what you said."
        if titles:
            return "I found a few options that fit what you're looking for."
        return "I couldn't find a strong match yet. What mood, genre, or movie should I use as a starting point?"
