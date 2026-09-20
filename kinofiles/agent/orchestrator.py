"""Movie recommendation orchestrator built with LangGraph for group mediation.

Flow:
welcome -> collect_preferences (loop) -> mediate -> group_vote -> {goodbye | refine -> mediate}
"""

import logging
from typing import Literal, TypedDict

from agent.io.turn import prompt
from agent.llm import build_llm, build_mistral_llm
from agent.nodes.classifier import ClassificationError, Classifier
from agent.nodes.criteria import (
    empty_criteria,
    has_catalog_filters,
    merge_criteria,
    merge_group_criteria,
)
from agent.nodes.direct_request import DirectRequestHandler
from agent.nodes.reply import MAX_HISTORY_TURNS, ReplyComposer
from agent.nodes.theme_recommender import ThemeRecommender
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


def _compact(value: dict | None) -> dict:
    """Drop empty fields so log lines only show what the group actually asked for."""
    return {key: item for key, item in (value or {}).items() if item}


class WelcomeExtract(BaseModel):
    """Decide whether the first message names people watching, or is a movie request."""

    kind: Literal["participants", "movie_request"] = Field(
        description=(
            "participants: the user is listing people who will watch together "
            "(friends, family, roommates in the room). "
            "movie_request: the user is asking for films — including naming "
            "actors, directors, titles, genres, or moods. Famous performer "
            "names without social/watching-together context are movie_request, "
            "never participants."
        )
    )
    names: list[str] = Field(
        default_factory=list,
        description=(
            "Watcher names when kind=participants, properly capitalized. "
            "Empty when kind=movie_request. Never put actors or directors here."
        ),
    )


class State(TypedDict):
    # Group
    participants: list[str]
    current_participant: int
    preferences: dict[str, list[str]]
    per_person_criteria: dict[str, dict]

    # Mediation
    group_criteria: dict
    compromise_notes: str
    round: int

    # Recommendation
    movies: list[str]
    result: dict
    show_options: bool
    response: str
    history: list[dict[str, str]]

    # Voting
    votes: dict[str, str]
    choice: str

    # Session
    farewell: str
    group_feedback: list[str]
    inline_feedback: str


class OrquestratorAgent:
    def __init__(self, llm=None, reply_llm=None):
        self.llm = llm or build_llm()
        self.reply_llm = reply_llm or (llm if llm is not None else build_llm(temperature=0.4))
        self.classifier = Classifier(build_mistral_llm())
        self.direct_request_handler = DirectRequestHandler(self.llm)
        self.theme_recommender = ThemeRecommender()
        self.reply_composer = ReplyComposer(self.reply_llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("welcome", self.welcome)
        graph.add_node("collect_preferences", self.collect_preferences)
        graph.add_node("mediate", self.mediate)
        graph.add_node("group_vote", self.group_vote)
        graph.add_node("goodbye", self.goodbye)
        graph.add_node("refine", self.refine)

        graph.set_entry_point("welcome")

        return graph.compile(checkpointer=InMemorySaver())

    def welcome(self, state: State) -> Command:
        msg = (
            "Welcome to KinoFiles! What do you feel like watching? "
            "If you're with friends, tell me their names. "
            "If you want films with a particular actor, say the actor — "
            "not the people in the room."
        )
        if (state.get("result") or {}).get("kind") in {
            "classification_error",
            "clarification",
        }:
            msg = f"{state['response']}\n{msg}"
        extractor = self.reply_llm.with_structured_output(WelcomeExtract)

        while True:
            text = interrupt(prompt(msg))
            if not text.strip():
                return self._welcome_clarification()
            try:
                result = extractor.invoke(
                    "The user just started a movie-night session. Classify "
                    "their first message.\n\n"
                    f"Input: {text!r}\n\n"
                    "kind=participants only when they are naming people who "
                    "will watch with them (e.g. 'I'm here with Alice and Bob', "
                    "'Me, Marta and Joan', first names of friends).\n"
                    "kind=movie_request when they name actors, directors, "
                    "titles, genres, moods, or otherwise ask for films "
                    "(e.g. 'Tom Hanks', 'something with Scarlett Johansson "
                    "and Brad Pitt', 'a comedy'). Well-known film people "
                    "without watching-together context are movie_request.\n"
                    "Return watcher names only for kind=participants."
                )
                kind = result.kind
                names = list(result.names or [])
            except Exception:
                # Do not treat comma-separated words as watchers; the
                # classifier can recover actors, titles, and moods.
                kind = "movie_request"
                names = []

            if kind != "participants" or not names:
                return self._start_movie_request(text)
            if len(names) == 1:
                confirmation = f"Got it, just you ({names[0]}) tonight."
                break
            if len(names) <= 4:
                confirmation = f"I detected {len(names)} people: {', '.join(names)}."
                break
            msg = "That's a lot of people! Could you limit it to 4 names? Let's try again:"

        return Command(
            goto="collect_preferences",
            update={
                "participants": names,
                "current_participant": 0,
                "preferences": {n: [] for n in names},
                "per_person_criteria": {n: empty_criteria() for n in names},
                "history": [],
                "round": 1,
                "response": confirmation,
                "result": {},
            },
        )

    def _start_movie_request(self, text: str) -> Command:
        """Solo session: keep the first message as the search, skip name collection."""
        person = "You"
        try:
            classify_result = self.classifier.classify(text)
        except ClassificationError:
            return self._classification_failure("welcome")
        if classify_result.intent not in {
            "recommendation",
            "theme_recommendation",
            "direct_request",
            "prefer",
            "feedback",
        }:
            return self._welcome_clarification()
        entities = classify_result.entities.model_dump()
        criteria = merge_criteria(
            empty_criteria(),
            entities,
            classify_result.criteria_action,
            classify_result.clear_fields,
        )
        return Command(
            goto="collect_preferences",
            update={
                "participants": [person],
                "current_participant": 1,
                "preferences": {person: [text]},
                "per_person_criteria": {person: criteria},
                "history": [],
                "round": 1,
                "response": "Got it, just you tonight.",
                "result": {},
            },
        )

    @staticmethod
    def _welcome_clarification():
        message = (
            "Please give the names of people watching with you, " "or a movie preference (including actors) to start."
        )
        return Command(
            goto="welcome",
            update={
                "response": message,
                "result": {"kind": "clarification", "titles": [], "error": None},
            },
        )

    @staticmethod
    def _classification_failure(stage):
        message = "I could not interpret that request. Please try again."
        return Command(
            goto=stage,
            update={
                "response": message,
                "result": {
                    "kind": "classification_error",
                    "titles": [],
                    "error": message,
                },
            },
        )

    def collect_preferences(self, state: State) -> Command:
        names = state["participants"]
        idx = state["current_participant"]

        if idx >= len(names):
            return Command(goto="mediate")

        person = names[idx]
        msg = f"{person}, what are you in the mood for?" if person else "What are you in the mood for?"
        if (state.get("result") or {}).get("kind") == "classification_error":
            msg = f"{state['response']}\n{msg}"
        text = interrupt(prompt(msg, participant=person))

        # Classify
        try:
            result = self.classifier.classify(text)
        except ClassificationError:
            return self._classification_failure("collect_preferences")
        entities = result.entities.model_dump()
        action = result.criteria_action

        current_criteria = state["per_person_criteria"][person]
        new_criteria = merge_criteria(current_criteria, entities, action, result.clear_fields)

        prefs = list(state["preferences"][person])
        prefs.append(text)

        return Command(
            goto="collect_preferences",
            update={
                "current_participant": idx + 1,
                "preferences": {**state["preferences"], person: prefs},
                "per_person_criteria": {
                    **state["per_person_criteria"],
                    person: new_criteria,
                },
                "response": "",
                "result": {},
            },
        )

    def mediate(self, state: State) -> Command:
        round_no = state.get("round", 1)
        merged = state.get("group_criteria")
        if not merged:
            merged, notes = merge_group_criteria(state["per_person_criteria"])
            merge_source = "fresh"
        else:
            notes = "Updated search based on group feedback."
            # Rounds 2+ carry the accumulated group criteria; the per-person
            # criteria are never re-consulted after the first merge.
            merge_source = "carried"

        # Flatten feedback from all people
        all_feedback = list(state.get("group_feedback", []))
        for p in state["participants"]:
            all_feedback.extend(state["preferences"][p])

        req = "; ".join(all_feedback) if all_feedback else "recommend something"

        for person in state["participants"]:
            logger.info(
                "mediate | round=%s | input | %s=%s",
                round_no,
                person,
                _compact(state["per_person_criteria"].get(person)),
            )
        logger.info(
            "mediate | round=%s | merge=%s | merged_criteria=%s | notes=%r",
            round_no,
            merge_source,
            _compact(merged),
            notes,
        )
        logger.info("mediate | round=%s | merged_query=%r", round_no, req)

        if has_catalog_filters(merged) and not merged.get("themes"):
            # They want a specific director/actor/etc without mood qualifiers
            route = "direct_request"
            result = self.direct_request_handler.handle(None, None, criteria=merged)
        else:
            # Re-use theme_recommender for semantics (or semantics + strict genre bounds)
            route = "theme_recommender"
            result = self.theme_recommender.recommend(req, feedback=[], criteria=merged)

        movies = result.get("titles", [])
        logger.info(
            "mediate | round=%s | route=%s | kind=%s | titles=%s | error=%s",
            round_no,
            route,
            result.get("kind"),
            movies,
            result.get("error"),
        )

        # Compose response
        response = self.reply_composer.compose(
            request="mediate",
            intent="theme_recommendation",
            entities={},
            result=result,
            history=state.get("history", []),
            feedback=[],
            movies=movies,
            search_criteria=merged,
            per_person_criteria=state["per_person_criteria"],
            compromise_notes=notes,
        )

        history = state.get("history", []) + [
            {
                "user": "Group preferences merged",
                "assistant": response,
                "intent": "mediation",
            }
        ]

        return Command(
            goto="group_vote",
            update={
                "group_criteria": merged,
                "compromise_notes": notes,
                "movies": movies,
                "result": result,
                "show_options": bool(movies),
                "response": response,
                "history": history[-MAX_HISTORY_TURNS:],
                "votes": {},
                "current_participant": 0,
            },
        )

    def group_vote(self, state: State) -> Command:
        movies = state.get("movies", [])
        if not movies:
            return Command(goto="refine")

        names = state["participants"]
        idx = state.get("current_participant", 0)

        if idx >= len(names):
            # Tally votes
            votes = state.get("votes", {})
            tally = {}
            for v in votes.values():
                if v and v != "none":
                    tally[v] = tally.get(v, 0) + 1
            if not tally:
                return Command(goto="refine", update={"response": ""})

            # Check for tie
            max_votes = max(tally.values())
            top_movies = [m for m, v in tally.items() if v == max_votes]
            if len(top_movies) > 1:
                return Command(
                    goto="refine",
                    update={"response": "We have a tie! Let's refine our search."},
                )

            winner = top_movies[0]
            return Command(goto="goodbye", update={"choice": winner})

        person = names[idx]
        msg = f"{person}, which one speaks to you?" if person else "Which one speaks to you?"

        ans = interrupt(prompt(msg, movies, participant=person)).strip()

        vote_val = "none"
        inline_fb = ""
        if ans.isdigit() and 1 <= int(ans) <= len(movies):
            vote_val = movies[int(ans) - 1]
        elif ans:
            # The user typed text instead of a number — that IS their feedback.
            inline_fb = ans

        return Command(
            goto="group_vote",
            update={
                "votes": {**state.get("votes", {}), person: vote_val},
                "current_participant": idx + 1,
                "inline_feedback": inline_fb,
            },
        )

    def refine(self, state: State) -> Command:
        if state.get("round", 1) >= 3:
            # Force the top-voted movie or first from shortlist
            movies = state.get("movies", [])
            fallback = movies[0] if movies else "something fun"
            return Command(goto="goodbye", update={"choice": fallback})

        # If the user already gave text feedback during the vote step, use it
        # directly instead of asking again.
        ans = state.get("inline_feedback", "")
        if not ans:
            msg = "What else are you looking for?"
            movies = state.get("movies", [])
            ans = interrupt(prompt(msg, options=movies))

        group_fb = list(state.get("group_feedback", []))
        group_fb.append(ans)

        try:
            result = self.classifier.classify(ans)
        except ClassificationError:
            return self._classification_failure("refine")
        new_group = merge_criteria(
            state["group_criteria"],
            result.entities.model_dump(),
            result.criteria_action,
            result.clear_fields,
        )
        logger.info(
            "refine | round=%s | feedback=%r | action=%s | new_group_criteria=%s",
            state.get("round", 1),
            ans,
            result.criteria_action,
            _compact(new_group),
        )

        return Command(
            goto="mediate",
            update={
                "group_criteria": new_group,
                "group_feedback": group_fb,
                "round": state.get("round", 1) + 1,
                "inline_feedback": "",
            },
        )

    def goodbye(self, state: State) -> Command:
        return Command(
            goto=END,
            update={"farewell": f"{state['choice']} wins! Enjoy the movie, everyone. 🎬"},
        )

    def run(self, thread_id: str = "1") -> str:
        config = {"configurable": {"thread_id": thread_id}}
        event = self.graph.invoke({}, config)
        while "__interrupt__" in event:
            pending = event["__interrupt__"][0].value
            print(pending["text"])
            for i, option in enumerate(pending["options"], 1):
                print(f"{i}. {option}")
            event = self.graph.invoke(Command(resume=input("> ")), config)

        print(event.get("farewell", "Goodbye!"))
        return event.get("choice", "")


if __name__ == "__main__":
    import logging

    load_dotenv()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    # The Hub relays server-side notices (e.g. the unauthenticated-request
    # nag) through this logger, which also carries its own handler, so the
    # same line lands twice. Silencing the logger suppresses both copies.
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    OrquestratorAgent().run()
