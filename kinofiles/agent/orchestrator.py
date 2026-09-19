"""Movie recommendation orchestrator built with LangGraph.

Flow:  input (interrupt) -> recommendation (subagent) -> review (interrupt) -> ...
The input node is plain text for now; later it gets replaced by other I/O
subagents.
"""

import os
from typing import TypedDict
from dotenv import load_dotenv

from langchain_mistralai import ChatMistralAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.sugagents.recommender import Recommender


class State(TypedDict):
    request: str
    feedback: list[str]
    movies: list[str]
    choice: str
    farewell: str


class OrquestratorAgent:
    def __init__(self, llm=None):
        api_key = os.environ.get("MISTRAL_API_KEY")
        endpoint = os.environ.get("MISTRAL_ENDPOINT")
        model = os.environ.get("MISTRAL_MODEL", "ministral-8b-2512")
        
        self.llm = llm or ChatMistralAI(
            model=model,
            temperature=0,
            mistral_api_key=api_key,
            endpoint=endpoint,
        )
        self.recommender = Recommender(self.llm)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("user_input", self.user_input)
        graph.add_node("recommendation", self.recommendation)
        graph.add_node("review", self.review)
        graph.add_node("goodbye", self.goodbye)

        graph.set_entry_point("user_input")
        graph.add_edge("user_input", "recommendation")
        graph.add_edge("recommendation", "review")
        graph.add_edge("goodbye", END)

        return graph.compile(checkpointer=InMemorySaver())

    def user_input(self, state: State) -> State:
        """Ask the user what they feel like watching."""
        text = interrupt("What do you feel like watching?")
        return {"request": text, "feedback": [], "movies": []}

    def recommendation(self, state: State) -> State:
        """Call the recommendation subagent."""
        movies = self.recommender.recommend(state["request"], state["feedback"])
        return {"movies": movies}

    def review(self, state: State) -> Command:
        """Let the user pick a movie, or collect feedback for another round."""
        movies = state["movies"]
        answer = interrupt({
            "movies": movies,
            "question": f"Pick a number (1-{len(movies)}) or tell me what to change.",
        }).strip()

        if answer.isdigit() and 1 <= int(answer) <= len(movies):
            return Command(goto="goodbye", update={"choice": movies[int(answer) - 1]})
        return Command(
            goto="recommendation",
            update={"feedback": state["feedback"] + [answer]},
        )

    def goodbye(self, state: State) -> State:
        """Close the session once the user has picked a movie."""
        return {"farewell": f"Enjoy {state['choice']}! Goodbye."}

    def run(self, thread_id: str = "1") -> str:
        """Drive the graph from the terminal, resuming on every interrupt."""
        config = {"configurable": {"thread_id": thread_id}}
        event = self.graph.invoke({}, config)
        while "__interrupt__" in event:
            print(event["__interrupt__"][0].value)
            event = self.graph.invoke(Command(resume=input("> ")), config)
        print(event["farewell"])
        return event["choice"]


if __name__ == "__main__":
    load_dotenv()
    OrquestratorAgent().run()
