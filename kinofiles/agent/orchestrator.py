"""Movie recommendation orchestrator built with LangGraph.

Flow:  welcome (interrupt) -> conversation (RecommendationAgent subgraph)
       -> goodbye

The conversation node *is* the recommendation subagent, compiled graph and
all. That subagent pauses on its own whenever it needs the user, and those
interrupts surface right here on the parent's `__interrupt__` — resuming from
this graph lands back inside it at the exact node that paused. So handing the
pause down costs the orchestrator nothing: it still mediates every turn.

What stays at this layer is what no single capability can own:

- session lifecycle — when the conversation is over, and the farewell
- I/O modality — speech in and out (`io/tts.py`, `io/stt.py`), and the
  `prompt()` contract splitting what gets narrated from what gets rendered
- capability composition — recommendation is the first subagent; the next
  ones (watchlist, availability, trailers) plug in here rather than being
  bolted into the recommender's classifier

`State` is deliberately a superset of the subagent's: LangGraph only
propagates keys the parent declares, so leaving out `response` or `intent`
would silently discard everything the subagent generated.
"""

from typing import TypedDict
from dotenv import load_dotenv

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.io.turn import prompt
from agent.llm import build_llm
from agent.router_orchestrator import RecommendationAgent


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
    farewell: str


class OrquestratorAgent:
    def __init__(self, llm=None):
        self.llm = llm or build_llm()
        # No checkpointer of its own: it borrows this graph's, so a resume
        # sent here reaches the interrupt waiting inside it.
        self.recommendation_agent = RecommendationAgent(self.llm, checkpointer=None)
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(State)
        graph.add_node("welcome", self.welcome)
        graph.add_node("conversation", self.recommendation_agent.graph)
        graph.add_node("goodbye", self.goodbye)

        graph.set_entry_point("welcome")
        graph.add_edge("welcome", "conversation")
        graph.add_edge("conversation", "goodbye")
        graph.add_edge("goodbye", END)

        return graph.compile(checkpointer=InMemorySaver())

    def welcome(self, state: State) -> State:
        """Open the session and collect the first message.

        The only question this layer asks: starting a session is its job,
        while every follow-up is the subagent's, because only the subagent
        knows what it still needs to know.
        """
        text = interrupt(prompt("What do you feel like watching?"))
        return {"request": text, "feedback": [], "movies": [], "response": ""}

    def goodbye(self, state: State) -> State:
        """Close the session once the subagent reports a pick."""
        return {"farewell": f"Enjoy {state['choice']}! Goodbye."}

    def run(self, thread_id: str = "1") -> str:
        """Drive the graph from the terminal, resuming on every interrupt."""
        config = {"configurable": {"thread_id": thread_id}}
        event = self.graph.invoke({}, config)
        while "__interrupt__" in event:
            pending = event["__interrupt__"][0].value
            print(pending["text"])
            for i, option in enumerate(pending["options"], 1):
                print(f"{i}. {option}")
            event = self.graph.invoke(Command(resume=input("> ")), config)
        print(event["farewell"])
        return event["choice"]

    def pending_state(self, thread_id: str = "1") -> dict:
        """Everything the subagent holds while it sits paused.

        A paused subgraph publishes nothing to the parent — its writes only
        land once its node returns — so the interrupt payload is normally the
        whole story. This reaches past that when the orchestrator needs the
        subagent's working state to apply policy (logging, a fallback when
        the shortlist came back empty) without the subagent having to
        anticipate it. Returns `{}` when nothing is paused.
        """
        config = {"configurable": {"thread_id": thread_id}}
        for task in self.graph.get_state(config, subgraphs=True).tasks:
            if task.state:
                return dict(task.state.values)
        return {}


if __name__ == "__main__":
    load_dotenv()
    OrquestratorAgent().run()
