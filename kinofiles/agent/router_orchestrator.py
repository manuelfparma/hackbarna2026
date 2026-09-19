"""Intent-routing movie assistant built with LangGraph.

Flow: classify (LLM) -> {theme_recommendation | recommendation | feedback |
direct_request | social} -> turn (interrupt) -> classify -> ...

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
  carry `response`, `intent`, `column` and `value` in its own schema or lose
  them for good.
"""

from typing import TypedDict

from agent.io.turn import prompt
from agent.llm import build_llm, build_mistral_llm
from agent.nodes.classifier import Classifier
from agent.nodes.direct_request import DirectRequestHandler
from agent.nodes.feedback import FeedbackHandler
from agent.nodes.recommender import Recommender
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
    feedback: list[str]
    movies: list[str]
    response: str
    choice: str


class RecommendationAgent:
    def __init__(self, llm=None, classifier_llm=None, checkpointer=_OWN_CHECKPOINTER):
        """Pass `checkpointer=None` when embedding this as a subgraph:
        LangGraph then hands it the parent's, which is what lets the parent
        resume the interrupts raised in here. Left alone, it persists on its
        own so it still runs standalone."""
        if checkpointer is _OWN_CHECKPOINTER:
            checkpointer = InMemorySaver()
        self.llm = llm or build_llm()
        self.classifier = Classifier(classifier_llm or build_mistral_llm())
        self.recommender = Recommender(self.llm)
        self.theme_recommender = ThemeRecommender(self.llm)
        self.feedback_handler = FeedbackHandler(self.llm)
        self.direct_request_handler = DirectRequestHandler(self.llm)
        self.social_handler = SocialHandler(self.llm)
        self.graph = self._build_graph(checkpointer)

    def _build_graph(self, checkpointer):
        graph = StateGraph(State)
        graph.add_node("classify", self.classify)
        graph.add_node("theme_recommendation", self.theme_recommendation)
        graph.add_node("recommendation", self.recommendation)
        graph.add_node("feedback", self.feedback)
        graph.add_node("direct_request", self.direct_request)
        graph.add_node("social", self.social)
        graph.add_node("turn", self.turn)

        graph.set_entry_point("classify")
        graph.add_edge("theme_recommendation", "turn")
        graph.add_edge("recommendation", "turn")
        graph.add_edge("feedback", "turn")
        graph.add_edge("direct_request", "turn")
        graph.add_edge("social", "turn")

        return graph.compile(checkpointer=checkpointer)

    def classify(self, state: State) -> Command:
        """First layer: decide which node should handle this message."""
        result = self.classifier.classify(state["request"])
        intent = result.intent
        column, value = result.as_filter()
        return Command(
            goto=intent,
            update={
                "intent": intent,
                "column": column,
                "value": value,
                "entities": result.entities.model_dump(),
                "response": "",
            },
        )

    def theme_recommendation(self, state: State) -> State:
        """Movie suggestions via nearest theme embeddings."""
        movies, response = self.theme_recommender.recommend(state["request"], state.get("feedback", []))
        return {"movies": movies, "response": response}

    def recommendation(self, state: State) -> State:
        """Produce movie suggestions for the request."""
        movies = self.recommender.recommend(state["request"], state.get("feedback", []))
        # The titles ride in the interrupt's `options`, so the spoken half of
        # the turn only has to introduce them.
        return {"movies": movies, "response": "Here's what I'd suggest:"}

    def feedback(self, state: State) -> State:
        """Fold feedback into the running list and regenerate recommendations."""
        feedback = state.get("feedback", []) + [state["request"]]
        movies = self.recommender.recommend(state["request"], feedback)
        response = self.feedback_handler.handle(state["request"], state.get("movies", []))
        return {"feedback": feedback, "movies": movies, "response": response}

    def direct_request(self, state: State) -> State:
        """Process a direct request using the extracted column and value."""
        response = self.direct_request_handler.handle(state.get("column"), state.get("value"))
        return {"response": response}

    def social(self, state: State) -> State:
        """Reply to greetings and small talk."""
        response = self.social_handler.handle(state["request"])
        return {"response": response}

    def turn(self, state: State) -> Command:
        """Hand the answer back and wait for the next message.

        A bare number picks from the current shortlist and ends the graph —
        that is the one outcome the parent acts on. Anything else is another
        message, so it goes back through the classifier rather than being
        assumed to be feedback.

        `social` and `direct_request` leave `movies` untouched on purpose: a
        detour mid-conversation should not cost the user the shortlist they
        were about to pick from.
        """
        movies = state.get("movies") or []
        text = state.get("response") or ""
        if movies:
            text = f"{text}\n\nPick a number (1-{len(movies)}) or tell me what to change."

        answer = interrupt(prompt(text, movies)).strip()

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
        payload = Command(resume=request) if paused else {"request": request, "feedback": [], "movies": []}

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
