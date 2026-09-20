"""Intent-routing movie assistant built with LangGraph.

Flow: classify (LLM) -> {theme_recommendation | feedback | direct_request |
social} -> reply -> turn (interrupt) -> classify -> ...

The loop closes on `turn`, which is where this agent pauses to ask the user
something. It owns those interrupts on purpose: only this graph knows when it
needs the human — which title of the shortlist, what to change about it. An
orchestrator that asked on its behalf would have to model this graph's
internals, so the pause belongs here.

Two consequences for whoever embeds this as a subgraph:

- While it is paused, the parent sees *nothing* it has written to state. The
  interrupt payload is the only channel, so `turn` puts everything the caller
  needs into it.
- LangGraph drops state keys the parent does not declare, so a parent must
  carry the full reply state (`result`, `history`, `response`, and display
  flags) in its own schema or lose it.
"""

import logging
from typing import TypedDict

from agent.io.turn import prompt
from agent.llm import build_llm, build_mistral_llm
from agent.nodes.classifier import Classifier, ClassificationError, Entities
from agent.nodes.criteria_commands import CriteriaCommand, parse_criteria_command
from agent.nodes.criteria import clear_criteria
from agent.nodes.catalog import clean_title, resolve_movie
from agent.nodes.selection import select_movie, match_title
from agent.nodes.criteria import has_search_terms
from agent.nodes.criteria import (
    has_catalog_filters,
    merge_criteria,
    normalize_criteria,
)
from agent.nodes.direct_request import DirectRequestHandler
from agent.nodes.feedback import FeedbackHandler
from agent.nodes.reply import MAX_HISTORY_TURNS, ReplyComposer
from agent.nodes.social import SocialHandler
from agent.nodes.theme_recommender import ThemeRecommender
from dotenv import load_dotenv
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

# Distinguishes "caller said nothing" from an explicit `checkpointer=None`,
# which is how a parent graph asks to lend this one its own.
_OWN_CHECKPOINTER = object()

logger = logging.getLogger(__name__)


def _compact(value: dict | None) -> dict:
    """Drop empty fields so log lines only show what the turn extracted."""
    return {key: item for key, item in (value or {}).items() if item}


class State(TypedDict):
    request: str
    intent: str
    column: str | None
    value: str | None
    entities: dict
    search_criteria: dict
    feedback: list[str]
    movies: list[str]
    result: dict
    history: list[dict[str, str]]
    show_options: bool
    response: str
    choice: str | None
    shown_movies: list[str]
    seen_movies: list[dict]
    resolved_movies: dict
    last_route: str
    last_search: dict
    search_mode: str
    focused_movie: str | None


def initial_state(request: str = "") -> dict:
    return {
        "request": request,
        "feedback": [],
        "movies": [],
        "search_criteria": normalize_criteria({}),
        "result": {},
        "history": [],
        "show_options": False,
        "response": "",
        "choice": None,
        "shown_movies": [],
        "seen_movies": [],
        "resolved_movies": {},
        "last_route": "",
        "last_search": {},
        "search_mode": "auto",
        "focused_movie": None,
    }


class RecommendationAgent:
    def __init__(
        self,
        llm=None,
        classifier_llm=None,
        reply_llm=None,
        checkpointer=_OWN_CHECKPOINTER,
    ):
        """Pass `checkpointer=None` when embedding this as a subgraph:
        LangGraph then hands it the parent's, which is what lets the parent
        resume the interrupts raised in here. Left alone, it persists on its
        own so it still runs standalone."""
        if checkpointer is _OWN_CHECKPOINTER:
            checkpointer = InMemorySaver()
        self.llm = llm or build_llm()
        self.classifier = Classifier(classifier_llm or build_mistral_llm())
        self.theme_recommender = ThemeRecommender()
        self.feedback_handler = FeedbackHandler()
        self.direct_request_handler = DirectRequestHandler(self.llm)
        self.social_handler = SocialHandler()
        conversational_llm = reply_llm or (
            llm if llm is not None else build_llm(temperature=0.4)
        )
        self.reply_composer = ReplyComposer(conversational_llm)
        self.graph = self._build_graph(checkpointer)

    def _build_graph(self, checkpointer):
        graph = StateGraph(State)
        graph.add_node("classify", self.classify)
        graph.add_node("theme_recommendation", self.theme_recommendation)
        graph.add_node("feedback", self.feedback)
        graph.add_node("direct_request", self.direct_request)
        graph.add_node("social", self.social)
        graph.add_node("reply", self.reply)
        graph.add_node("turn", self.turn)
        for name in ("preference", "availability", "help"):
            graph.add_node(name, getattr(self, name))
            graph.add_edge(name, "reply")
        graph.add_node("accept", self.accept)

        graph.set_entry_point("classify")
        graph.add_edge("theme_recommendation", "reply")
        graph.add_edge("feedback", "reply")
        graph.add_edge("direct_request", "reply")
        graph.add_edge("social", "reply")
        graph.add_edge("reply", "turn")

        return graph.compile(checkpointer=checkpointer)

    @staticmethod
    def _criteria_control_update(state, command):
        criteria = command.apply(state.get("search_criteria"))
        logger.info(
            "criteria_changed | action=%s | fields=%s | criteria=%s",
            command.action,
            command.fields,
            _compact(criteria),
        )
        return {
            "search_criteria": criteria,
            "history": [],
            "feedback": [],
            "movies": [],
            "shown_movies": [],
            "focused_movie": None,
            "last_search": {},
            "last_route": "",
            "column": None,
            "value": None,
            "entities": {},
            "search_mode": "auto",
            "choice": None,
            "show_options": False,
            "response": "",
            "intent": "criteria_update",
            "result": {
                "kind": "criteria_changed",
                "action": command.action,
                "fields": list(command.fields),
                "titles": [],
                "error": None,
            },
        }

    def classify(self, state: State) -> Command:
        """Extract this turn, merge its criteria, and choose a capability."""
        base_update = {}
        control = parse_criteria_command(state["request"])
        if control:
            base_update = self._criteria_control_update(state, control)
            if not control.remainder:
                return Command(goto="reply", update=base_update)
            base_update["request"] = control.remainder
            state = {**state, **base_update}
        context = {
            "shortlist": state.get("movies", []),
            "recent_history": state.get("history", [])[-3:],
            "criteria_summary": _compact(state.get("search_criteria")),
            "focused_movie": state.get("focused_movie"),
            "search_mode": state.get("search_mode", "auto"),
        }
        try:
            result = self.classifier.classify(state["request"], context)
        except ClassificationError:
            return Command(
                goto="reply",
                update={
                    **base_update,
                    "intent": "classification_error",
                    "entities": {},
                    "response": "",
                    "show_options": False,
                    "result": {
                        "kind": "classification_error",
                        "titles": [],
                        "criteria_change": base_update.get("result"),
                        "error": "I could not interpret this request. I have not run a new search. Please try again.",
                    },
                },
            )
        intent = result.intent
        if intent in {"help", "unsupported"}:
            return Command(
                goto="help",
                update={
                    **base_update,
                    "intent": intent,
                    "entities": {},
                    "result": {},
                    "response": "",
                    "show_options": False,
                },
            )
        entities = result.entities.model_dump()
        action = result.criteria_action
        if action in {"reset", "remove"}:
            if action == "remove" and not result.clear_fields:
                return Command(
                    goto="reply",
                    update={
                        **base_update,
                        "show_options": False,
                        "intent": "criteria_update",
                        "result": {
                            "kind": "clarification",
                            "titles": [],
                            "error": "Which filter should I remove?",
                        },
                    },
                )
            model_control = CriteriaCommand(action, tuple(result.clear_fields))
            control_update = self._criteria_control_update(state, model_control)
            base_update.update(control_update)
            state = {**state, **control_update}
            if action == "remove":
                entities = clear_criteria(entities, result.clear_fields)
                result.entities = Entities.model_validate(entities)
            if not has_search_terms(entities):
                return Command(goto="reply", update=base_update)
            action = "add"
            if intent == "social":
                intent = "recommendation"
        if intent in {
            "social",
            "accept",
            "exit",
            "availability",
        } or self._is_factual_movie_request(intent, entities):
            action = "keep"
        elif intent == "prefer":
            action = "add"
        elif intent == "more":
            action = "add" if has_search_terms(entities) else "keep"
        if result.needs_clarification:
            return Command(
                goto="reply",
                update={
                    **base_update,
                    "intent": intent,
                    "entities": entities,
                    "show_options": False,
                    "result": {
                        "kind": "clarification",
                        "titles": [],
                        "error": "Which movie do you mean?",
                    },
                    "response": "",
                },
            )
        criteria = merge_criteria(
            state.get("search_criteria", {}),
            entities,
            action,
        )
        route = self._route_for_criteria(intent, entities, criteria)
        column, value = result.as_filter()
        mode = result.search_mode
        extra = {}
        search_request = state["request"]
        if intent == "exit":
            return Command(
                goto=END,
                update={
                    **base_update,
                    "intent": intent,
                    "choice": None,
                    "show_options": False,
                    "result": {"kind": "exit", "titles": [], "error": None},
                },
            )
        if intent == "more":
            previous = state.get("last_search") or {}
            route = previous.get("route") or self._route_for_criteria(
                "recommendation", {}, criteria
            )
            column, value = previous.get("column"), previous.get("value")
            mode = previous.get(
                "search_mode", "auto" if has_search_terms(criteria) else "browse"
            )
            if action == "add":
                route = self._route_for_criteria("recommendation", entities, criteria)
            search_request = previous.get("request", "Recommend a movie")
        if route in {
            "theme_recommendation",
            "direct_request",
            "feedback",
        } and not self._is_factual_movie_request(intent, entities):
            replay_route = "theme_recommendation" if route == "feedback" else route
            extra.update(
                last_route=replay_route,
                last_search={
                    "route": replay_route,
                    "request": search_request,
                    "column": column,
                    "value": value,
                    "search_mode": mode,
                },
            )
        seen = state.get("seen_movies", []) + [
            movie for movie in entities["movies"] if movie.get("role") == "seen"
        ]
        extra["seen_movies"] = normalize_criteria({"movies": seen})["movies"]
        if action in {"reset", "replace"}:
            extra["feedback"] = []
        elif intent == "feedback" and route != "feedback":
            extra["feedback"] = state.get("feedback", []) + [state["request"]]
        logger.info(
            "request=%r | intent=%s | action=%s | route=%s | entities=%s | criteria=%s",
            state["request"],
            intent,
            action,
            route,
            _compact(entities),
            _compact(criteria),
        )
        return Command(
            goto=route,
            update={
                **base_update,
                **extra,
                "search_mode": mode,
                "intent": intent,
                "column": column,
                "value": value,
                "entities": entities,
                "search_criteria": criteria,
                "result": {},
                "show_options": False,
                "response": "",
            },
        )

    @staticmethod
    def _is_factual_movie_request(intent: str, entities: dict) -> bool:
        if intent != "direct_request":
            return False
        movies = entities.get("movies") or []
        return bool(movies) and all(
            movie.get("role") in {"asked_about", "referenced"} for movie in movies
        )

    @staticmethod
    def _route_for_criteria(intent: str, entities: dict, criteria: dict) -> str:
        """Route refinements by their merged constraints, not only intent."""
        if intent == "prefer":
            return "preference"
        if intent in {"help", "unsupported"}:
            return "help"
        if intent in {
            "social",
            "accept",
            "availability",
            "exit",
            "more",
        }:
            return intent
        if RecommendationAgent._is_factual_movie_request(intent, entities):
            return "direct_request"
        movies = entities.get("movies") or []
        if any(movie.get("role") in {"seed", "liked"} for movie in movies):
            # A seed film always takes the similar_to path, even when the
            # classifier packed catalog filters alongside it — _similar_to
            # post-filters genres on real neighbors.
            return "theme_recommendation"
        if entities.get("themes") or (
            intent in {"theme_recommendation", "recommendation", "feedback"}
            and criteria.get("themes")
        ):
            return "theme_recommendation"
        if intent == "feedback" and has_catalog_filters(entities):
            return "direct_request"
        if intent in {"recommendation", "theme_recommendation"} and has_catalog_filters(
            criteria
        ):
            return "direct_request"
        if intent == "recommendation":
            return "theme_recommendation"
        if (
            intent == "direct_request"
            and not has_catalog_filters(criteria)
            and criteria.get("themes")
        ):
            # The catalog filters didn't survive normalization (e.g. "heist"
            # demoted to a theme) — this is a semantic ask, not a lookup.
            return "theme_recommendation"
        return intent

    def _exclusions(self, state):
        cache = dict(state.get("resolved_movies") or {})
        paginate = state.get("intent") == "more" or (
            state.get("intent") == "feedback"
            and state.get("search_mode") != "description"
        )
        names = list(state.get("shown_movies", [])) if paginate else []
        for mention in state.get("seen_movies", []):
            key = f"{mention['title'].casefold()}:{mention.get('year')}"
            if key not in cache:
                try:
                    row = resolve_movie(
                        self.direct_request_handler.supabase,
                        mention["title"],
                        mention.get("year"),
                    )
                    cache[key] = row["name"] if row else None
                except Exception:
                    names.append(mention["title"])
                    continue
            names.append(cache.get(key) or mention["title"])
        return list(dict.fromkeys(names)), cache

    def help(self, state):
        return {
            "result": {"kind": state["intent"], "titles": [], "error": None},
            "show_options": False,
        }

    def preference(self, state):
        return {
            "result": {
                "kind": "preference",
                "titles": [],
                "absorbed": _compact(state.get("entities")),
                "error": None,
            },
            "show_options": False,
        }

    def _pick(self, state):
        movies = state.get("movies") or []
        choice = select_movie(state["request"], movies)
        mentions = (state.get("entities") or {}).get("movies") or []
        request = " ".join(state["request"].casefold().split())
        explicit = [
            mention
            for mention in mentions
            if " ".join(clean_title(mention["title"]).casefold().split()) in request
        ]
        wanted = [mention for mention in explicit if mention.get("role") == "wanted"]
        mentions = wanted or explicit or mentions
        if choice:
            return choice
        if len(mentions) == 1:
            mention = mentions[0]
            known = movies + (
                [state["focused_movie"]] if state.get("focused_movie") else []
            )
            choice = match_title(mention["title"], known)
            if choice:
                return choice
            if mention.get("role") != "referenced" or mention in explicit:
                try:
                    row = resolve_movie(
                        self.direct_request_handler.supabase,
                        mention["title"],
                        mention.get("year"),
                    )
                    return row["name"] if row else None
                except Exception:
                    return None
        return None

    def accept(self, state):
        choice = self._pick(state)
        if choice:
            return Command(goto=END, update={"choice": choice, "show_options": False})
        return Command(
            goto="reply",
            update={
                "result": {
                    "kind": "accept",
                    "titles": [],
                    "error": "Which movie would you like to watch?",
                },
                "show_options": False,
            },
        )

    def availability(self, state):
        mentions = (state.get("entities") or {}).get("movies") or []
        if not mentions:
            return {
                "show_options": False,
                "result": {
                    "kind": "availability",
                    "titles": [],
                    "error": "Which movie do you want streaming information for?",
                },
            }
        result = self.direct_request_handler.handle_movie_fact(
            mentions[0]["title"], mentions[0].get("year")
        )
        result.update(
            kind="availability",
            mock=True,
            requested_services=state.get("entities", {}).get("streaming", []),
        )
        update = {"result": result, "show_options": False}
        if result.get("titles"):
            update["focused_movie"] = result["titles"][0]
        return update

    def theme_recommendation(self, state: State) -> State:
        """Movie suggestions via nearest theme embeddings."""
        exclude, cache = self._exclusions(state)
        request = (
            state.get("last_search", {}).get("request", state["request"])
            if state.get("intent") == "more"
            else state["request"]
        )
        result = self.theme_recommender.recommend(
            request,
            state.get("feedback", []),
            criteria=state.get("search_criteria", {}),
            exclude_names=exclude,
            search_mode=state.get("search_mode", "auto"),
        )
        movies = result.get("titles", [])
        return {
            "movies": movies,
            "resolved_movies": cache,
            "focused_movie": movies[0]
            if len(movies) == 1 or (movies and result.get("guessed"))
            else None,
            "result": result,
            "show_options": bool(movies),
        }

    def feedback(self, state: State) -> State:
        """Fold feedback into the running list and search again by theme."""
        feedback = state.get("feedback", []) + [state["request"]]
        exclude, cache = self._exclusions(state)
        result = self.theme_recommender.recommend(
            state["request"],
            feedback,
            criteria=state.get("search_criteria", {}),
            exclude_names=exclude,
            search_mode=state.get("search_mode", "auto"),
        )
        packaged = self.feedback_handler.handle(
            state["request"], state.get("movies", [])
        )
        titles = result.get("titles", [])
        packaged["titles"] = titles
        packaged["themes"] = result.get("themes", [])
        packaged["genres"] = result.get("genres", [])
        packaged["error"] = result.get("error")
        if "matches" in result:
            packaged["matches"] = result["matches"]
        return {
            "feedback": feedback,
            "resolved_movies": cache,
            "focused_movie": titles[0] if len(titles) == 1 else None,
            "movies": titles,
            "result": packaged,
            "show_options": bool(titles),
        }

    def direct_request(self, state: State) -> State:
        """Process a direct request using the extracted column and value."""
        entities = state.get("entities", {})
        if self._is_factual_movie_request(state.get("intent", ""), entities):
            # Facts answer a question about a title; they never replace the
            # shortlist, so `movies` and its options stay untouched.
            mention = (entities.get("movies") or [{}])[0]
            result = self.direct_request_handler.handle_movie_fact(
                mention.get("title", ""),
                mention.get("year"),
                query_text=state["request"],
            )
            update = {"result": result, "show_options": False}
            if result.get("titles"):
                update["focused_movie"] = result["titles"][0]
            return update
        exclude, cache = self._exclusions(state)
        result = self.direct_request_handler.handle(
            state.get("column"),
            state.get("value"),
            criteria=state.get("search_criteria", {}),
            exclude_names=exclude if state.get("column") != "name" else [],
        )
        titles = result.get("titles", [])
        update = {
            "result": result,
            "show_options": bool(titles),
            "resolved_movies": cache,
            "focused_movie": titles[0] if len(titles) == 1 else None,
        }
        if titles:
            update["movies"] = titles
        return update

    def social(self, state: State) -> State:
        """Reply to greetings and small talk."""
        return {
            "result": self.social_handler.handle(state["request"]),
            "show_options": False,
        }

    def reply(self, state: State) -> State:
        """Compose the only user-facing prose produced by capability turns."""
        result = state.get("result", {})
        logger.info(
            "result | kind=%s | titles=%s | error=%s",
            result.get("kind"),
            result.get("titles"),
            result.get("error"),
        )
        response = self.reply_composer.compose(
            request=state["request"],
            intent=state["intent"],
            entities=state.get("entities", {}),
            search_criteria=state.get("search_criteria", {}),
            result=state.get("result", {}),
            history=state.get("history", []),
            feedback=state.get("feedback", []),
            movies=state.get("movies", []),
        )
        history = state.get("history", []) + [
            {
                "user": state["request"],
                "assistant": response,
                "intent": state["intent"],
            }
        ]
        if result.get("kind") == "criteria_changed":
            history = state.get("history", [])
        shown = state.get("shown_movies", [])
        if state.get("show_options"):
            shown = list(dict.fromkeys(shown + state.get("movies", [])))
        return {
            "response": response,
            "history": history[-MAX_HISTORY_TURNS:],
            "shown_movies": shown,
        }

    def turn(self, state: State) -> Command:
        """Hand the answer back and wait for the next message.

        A bare number picks from the current shortlist and ends the graph —
        that is the one outcome the parent acts on. Anything else is another
        message, so it goes back through the classifier rather than being
        assumed to be feedback.

        A social or unsuccessful direct-request detour leaves `movies`
        untouched but hides its options. The user may still select it by
        number, while the UI avoids repeating an unrelated shortlist.
        """
        movies = state.get("movies") or []
        text = state.get("response") or ""
        show_options = state.get("show_options", False) and bool(movies)

        if show_options:
            text = (
                f"{text}\n\nPick a number (1-{len(movies)}) or tell me what to change."
            )
        answer = interrupt(
            prompt(
                text,
                movies if show_options else [],
                criteria=state.get("search_criteria", {}),
            )
        ).strip()

        choice = select_movie(answer, movies)
        if choice:
            logger.info("pick | choice=%r", choice)
            return Command(goto=END, update={"choice": choice})
        return Command(goto="classify", update={"request": answer, "response": ""})

    def run(self, request: str, thread_id: str = "1") -> str:
        """Handle a single message and return the assistant's reply.

        One message in, one reply out, even though the graph now pauses
        mid-flight: a fresh thread starts it, a paused one resumes it.
        """
        config = {"configurable": {"thread_id": thread_id}}
        paused = bool(self.graph.get_state(config).next)
        payload = Command(resume=request) if paused else initial_state(request)

        event = self.graph.invoke(payload, config)
        if "__interrupt__" in event:
            return event["__interrupt__"][0].value["text"]
        return f"Enjoy {event['choice']}!" if event.get("choice") else "Goodbye!"


if __name__ == "__main__":
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s | %(message)s"
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpx2").setLevel(logging.WARNING)
    agent = RecommendationAgent()
    while True:
        text = input("> ")
        print(agent.run(text))
