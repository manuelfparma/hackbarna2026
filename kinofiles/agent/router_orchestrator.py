"""Intent-routing movie assistant built with LangGraph.

Flow: classify (LLM) -> {recommendation | feedback | direct_request | social}

Unlike OrquestratorAgent, this agent has no interrupts: each call handles one
message and returns a response right away. A client (e.g. the frontend)
drives the conversation turn by turn, reusing the same thread_id so state
(feedback, last recommendations) persists across calls.
"""

from typing import TypedDict

from agent.sugagents.classifier import Classifier
from agent.sugagents.direct_request import DirectRequestHandler
from agent.sugagents.feedback import FeedbackHandler
from agent.sugagents.recommender import Recommender
from agent.sugagents.social import SocialHandler
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command


class State(TypedDict):
    request: str
    intent: str
    feedback: list[str]
    movies: list[str]
    response: str


class RecommendationAgent:
    def __init__(self, llm=None):
        self.llm = llm or ChatMistralAI(model="ministral-8b-2512", temperature=0)
        self.classifier = Classifier(self.llm)
        self.recommender = Recommender(self.llm)
        self.feedback_handler = FeedbackHandler(self.llm)
        self.direct_request_handler = DirectRequestHandler(self.llm)
        self.social_handler = SocialHandler(self.llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("classify", self.classify)
        graph.add_node("recommendation", self.recommendation)
        graph.add_node("feedback", self.feedback)
        graph.add_node("direct_request", self.direct_request)
        graph.add_node("social", self.social)

        graph.set_entry_point("classify")
        graph.add_edge("recommendation", END)
        graph.add_edge("feedback", END)
        graph.add_edge("direct_request", END)
        graph.add_edge("social", END)

        return graph.compile(checkpointer=InMemorySaver())

    def classify(self, state: State) -> Command:
        """First layer: decide which node should handle this message."""
        intent = self.classifier.classify(state["request"])
        return Command(goto=intent, update={"intent": intent})

    def recommendation(self, state: State) -> State:
        """Produce movie suggestions for the request."""
        movies = self.recommender.recommend(state["request"], state.get("feedback", []))
        return {"movies": movies, "response": "\n".join(movies)}

    def feedback(self, state: State) -> State:
        """Fold feedback into the running list and regenerate recommendations."""
        feedback = state.get("feedback", []) + [state["request"]]
        movies = self.recommender.recommend(state["request"], feedback)
        response = self.feedback_handler.handle(state["request"], state.get("movies", []))
        return {"feedback": feedback, "movies": movies, "response": response}

    def direct_request(self, state: State) -> State:
        """Answer a specific factual question about a movie."""
        response = self.direct_request_handler.handle(state["request"])
        return {"response": response}

    def social(self, state: State) -> State:
        """Reply to greetings and small talk."""
        response = self.social_handler.handle(state["request"])
        return {"response": response}

    def run(self, request: str, thread_id: str = "1") -> str:
        """Handle a single message and return the assistant's response."""
        config = {"configurable": {"thread_id": thread_id}}
        event = self.graph.invoke({"request": request}, config)
        return event["response"]


if __name__ == "__main__":
    load_dotenv()
    agent = RecommendationAgent()
    while True:
        text = input("> ")
        print(agent.run(text))
