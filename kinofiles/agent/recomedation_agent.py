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

from typing import TypedDict

from agent.io.turn import prompt
from agent.llm import build_llm, build_mistral_llm
from agent.nodes.classifier import Classifier
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
    choice: str


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

        graph.set_entry_point("classify")
        graph.add_edge("theme_recommendation", "reply")
        graph.add_edge("feedback", "reply")
        graph.add_edge("direct_request", "reply")
        graph.add_edge("social", "reply")
        graph.add_edge("reply", "turn")

        return graph.compile(checkpointer=checkpointer)

    def classify(self, state: State) -> Command:
        """Extract this turn, merge its criteria, and choose a capability."""
        result = self.classifier.classify(state["request"])
        intent = result.intent
        entities = result.entities.model_dump()
        action = result.criteria_action
        if intent == "social":
            action = "keep"
        elif self._is_factual_movie_request(intent, entities):
            action = "keep"
        criteria = merge_criteria(
            state.get("search_criteria", {}),
            entities,
            action,
        )
        route = self._route_for_criteria(intent, entities, criteria)
        column, value = result.as_filter()
        return Command(
            goto=route,
            update={
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
            movie.get("role") == "asked_about" for movie in movies
        ) and not has_catalog_filters(entities)

    @staticmethod
    def _route_for_criteria(intent: str, entities: dict, criteria: dict) -> str:
        """Route refinements by their merged constraints, not only intent."""
        if intent == "social":
            return "social"
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

    def theme_recommendation(self, state: State) -> State:
        """Movie suggestions via nearest theme embeddings."""
        result = self.theme_recommender.recommend(
            state["request"],
            state.get("feedback", []),
            criteria=state.get("search_criteria", {}),
        )
        movies = result.get("titles", [])
        return {
            "movies": movies,
            "result": result,
            "show_options": bool(movies),
        }

    def feedback(self, state: State) -> State:
        """Fold feedback into the running list and search again by theme."""
        feedback = state.get("feedback", []) + [state["request"]]
        result = self.theme_recommender.recommend(
            state["request"],
            feedback,
            criteria=state.get("search_criteria", {}),
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
                mention.get("title", ""), mention.get("year")
            )
            return {"result": result, "show_options": False}
        result = self.direct_request_handler.handle(
            state.get("column"),
            state.get("value"),
            criteria=state.get("search_criteria", {}),
        )
        titles = result.get("titles", [])
        update = {"result": result, "show_options": bool(titles)}
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
        return {"response": response, "history": history[-MAX_HISTORY_TURNS:]}

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

        answer = interrupt(prompt(text, movies if show_options else [], criteria=state.get("search_criteria", {}))).strip()

        if movies and answer.isdigit() and 1 <= int(answer) <= len(movies):
            return Command(goto=END, update={"choice": movies[int(answer) - 1]})
        return Command(goto="classify", update={"request": answer, "response": ""})

    def run(self, request: str, thread_id: str = "1") -> str:
        """Handle a single message and return the assistant's reply.

        One message in, one reply out, even though the graph now pauses
        mid-flight: a fresh thread starts it, a paused one resumes it.
        """
        config = {"configurable": {"thread_id": thread_id}}
        paused = bool(self.graph.get_state(config).next)
        payload = (
            Command(resume=request)
            if paused
            else {
                "request": request,
                "feedback": [],
                "movies": [],
                "search_criteria": normalize_criteria({}),
                "result": {},
                "history": [],
                "show_options": False,
            }
        )

        event = self.graph.invoke(payload, config)
        if "__interrupt__" in event:
            return event["__interrupt__"][0].value["text"]
        return f"Enjoy {event['choice']}!"


if __name__ == "__main__":
    load_dotenv()
    agent = RecommendationAgent()
    while True:
        text = input("> ")
        print(agent.run(text))
