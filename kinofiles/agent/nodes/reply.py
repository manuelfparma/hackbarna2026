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
- If the result contains an error or no choices, explain that briefly; the application handles the next prompt.
- Do not tell the user how to select a choice; the application adds that instruction.
- Preference acknowledgments and help replies do not need choices; no choices is not an error there.
- For preference results acknowledge absorbed taste without claiming a new search ran.
- For guessed results ask 'Did you mean TITLE?' rather than asserting an identification.
- Do not claim to save lists, compare films, count catalog records, filter by studio,
  or guarantee the absence of thematic content. These operations are unsupported.
- Genre matches do not establish child suitability; content advisories are unavailable.

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
        if per_person_criteria is not None:
            if result.get("error"):
                return result["error"]
            if not result.get("titles"):
                return "I couldn't find a match for the group's preferences."
            if result.get("guessed"):
                return f"The closest plot match I found is {result['titles'][0]}."
        if (
            result.get("kind")
            in {
                "availability",
                "social",
                "preference",
                "criteria_changed",
                "classification_error",
                "clarification",
                "help",
                "unsupported",
            }
            or result.get("guessed")
            or result.get("error")
            or (not result.get("titles") and not result.get("facts"))
        ):
            return self.fallback(intent, result)
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
        kind = result.get("kind")
        if error:
            change = result.get("criteria_change") or {}
            prefix = (
                "I cleared the previous search filters. "
                if change.get("action") == "reset"
                else ""
            )
            if change.get("action") == "remove":
                prefix = f"I cleared the {', '.join(change.get('fields', []))} filter. "
            if kind in {"classification_error", "clarification"}:
                return prefix + error
            return f"{prefix}{error} Could you give me another detail to work with?"
        if kind == "criteria_changed":
            if result.get("action") == "reset":
                return "I've cleared your search filters and previous suggestions. What would you like to watch?"
            fields = ", ".join(result.get("fields") or [])
            return f"I've cleared the {fields} filter and kept your other preferences. What would you like to search for?"
        facts = result.get("facts")
        if result.get("guessed") and titles:
            return f"Did you mean {titles[0]}? That is the closest plot match I found."
        if kind in {"help", "unsupported"}:
            prefix = (
                "That operation is not supported. " if kind == "unsupported" else ""
            )
            return (
                prefix
                + "I can recommend movies by genre, actor, director, mood or plot, answer movie questions, and refine or reset your search."
            )
        if kind == "preference":
            return "Got it — I have noted those preferences for your next search."
        if kind == "availability":
            services = (facts or {}).get("streaming")
            if services is None:
                return "Streaming information is unavailable. The demo uses mock service assignments."
            names = {
                "netflix": "Netflix",
                "hbo": "HBO",
                "prime": "Prime Video",
                "appletv": "Apple TV+",
            }
            requested = result.get("requested_services") or []
            if requested:
                available = [
                    names.get(service, service)
                    for service in requested
                    if service in services
                ]
                missing = [
                    names.get(service, service)
                    for service in requested
                    if service not in services
                ]
                parts = ([f"listed on {', '.join(available)}"] if available else []) + (
                    [f"not listed on {', '.join(missing)}"] if missing else []
                )
                return f"Mock demo data: {facts['name']} is {'; '.join(parts)}. This is not verified streaming availability."
            return f"Mock demo data: {facts['name']} is listed on {', '.join(names.get(s, s) for s in services) or 'no services'}. This is not verified streaming availability."
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
