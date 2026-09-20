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
- DO NOT ask any questions (e.g., "What do you think?")—the system will prompt a vote.
- If the result contains an error or no choices, explain that briefly and ask one useful follow-up.
- Do not tell the user how to select a choice; the application adds that instruction.

Group mediation:
When `brief` is present you are mediating for several people. Explain the
CRITERIA you settled on for the group. Never name, quote or describe an
individual film: the titles are already on screen, and naming one undercuts
the vote that follows.

`brief.consensus` decides how you open:
- "none" — their criteria genuinely clash. Say so outright in the opening
  clause, then name what you blended in response, reading it from
  `accumulated_search_criteria`. Shape: "Looks like you want different
  things! So I went for <genres/themes>." Do NOT write that they both wanted
  the same thing; they did not.
- "partial" — open with what `brief.shared` says they have in common, then
  give what one person alone asked for, naming that person from
  `brief.individual`.
- "full" — say what they all agreed on.
- "single" — one person only; just state what you searched for.
- "unknown" — you only captured what one of them is after. Say what you
  searched for, and do not claim the group either agreed or disagreed.
- "refined" — they already sent you back with a joint request, which is the
  last entry in `feedback`. Open with what they asked for this time. Shape:
  "You asked for <their request>, so here is a fresh set." Never mention any
  earlier disagreement, and never write that they want different things: they
  have settled on a direction since, and raking it up would undo the agreement
  they just reached.

`capability_result.themes` is the kind of story the search homed in on; use it
to colour the description and paraphrase it, never reading a label verbatim.
`brief.seed_movies` are films someone offered as a reference, and those you may
name, crediting the person who brought them.

Your job is to move them towards one shared choice, so always close a group
shortlist with a short nudge to converge: make plain that the list is one set
put together for all of them rather than a pick tailored to each person, and
invite them to find something on it they can agree on. Phrase the nudge as a
statement ("See if one of these works for all of you"), never as a question.
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
        per_person_criteria: dict[str, dict] | None = None,
        compromise_notes: str | None = None,
        brief: dict[str, Any] | None = None,
    ) -> str:
        """Generate narration, falling back to deterministic copy on failure."""
        context = {
            "request": request,
            "intent": intent,
            "entities": entities,
            "accumulated_search_criteria": search_criteria or {},
            "per_person_criteria": per_person_criteria or {},
            "compromise_notes": compromise_notes or "",
            "brief": brief or {},
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
