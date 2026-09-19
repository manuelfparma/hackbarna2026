"""Movie recommendation orchestrator built with LangGraph for group mediation.

Flow:
welcome -> collect_preferences (loop) -> mediate -> group_vote -> {goodbye | refine -> mediate}
"""

import re
from typing import TypedDict
from dotenv import load_dotenv

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from agent.io.turn import prompt
from agent.llm import build_llm, build_mistral_llm
from agent.nodes.classifier import Classifier
from agent.nodes.criteria import has_catalog_filters, merge_criteria, merge_group_criteria, empty_criteria
from agent.nodes.direct_request import DirectRequestHandler
from agent.nodes.theme_recommender import ThemeRecommender
from agent.nodes.reply import ReplyComposer, MAX_HISTORY_TURNS

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

class OrquestratorAgent:
    def __init__(self, llm=None, reply_llm=None):
        self.llm = llm or build_llm()
        self.reply_llm = reply_llm or (
            llm if llm is not None else build_llm(temperature=0.4)
        )
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
        msg = "Welcome to KinoFiles! Who's picking tonight?\nTell me everyone's names (1–4 people)."
        while True:
            text = interrupt(prompt(msg))
            raw_names = re.split(r',|\band\b', text)
            names = [n.strip() for n in raw_names if n.strip()]
            
            if 1 <= len(names) <= 4:
                break
            msg = "Please give between 1 and 4 names, separated by commas."
            
        return Command(
            goto="collect_preferences",
            update={
                "participants": names,
                "current_participant": 0,
                "preferences": {n: [] for n in names},
                "per_person_criteria": {n: empty_criteria() for n in names},
                "history": [],
                "round": 1,
            }
        )

    def collect_preferences(self, state: State) -> Command:
        names = state["participants"]
        idx = state["current_participant"]
        
        if idx >= len(names):
            return Command(goto="mediate")
            
        person = names[idx]
        msg = f"{person}, what are you in the mood for?"
        text = interrupt(prompt(msg, participant=person))
        
        # Classify
        result = self.classifier.classify(text)
        entities = result.entities.model_dump()
        action = result.criteria_action
        
        current_criteria = state["per_person_criteria"][person]
        new_criteria = merge_criteria(current_criteria, entities, action)
        
        prefs = list(state["preferences"][person])
        prefs.append(text)
        
        return Command(
            goto="collect_preferences",
            update={
                "current_participant": idx + 1,
                "preferences": {**state["preferences"], person: prefs},
                "per_person_criteria": {**state["per_person_criteria"], person: new_criteria}
            }
        )

    def mediate(self, state: State) -> Command:
        merged = state.get("group_criteria")
        if not merged:
            merged, notes = merge_group_criteria(state["per_person_criteria"])
        else:
            notes = "Updated search based on group feedback."
        
        # Flatten feedback from all people
        all_feedback = list(state.get("group_feedback", []))
        for p in state["participants"]:
            all_feedback.extend(state["preferences"][p])
            
        req = "; ".join(all_feedback) if all_feedback else "recommend something"
        
        if has_catalog_filters(merged) and not merged.get("themes"):
            # They want a specific director/actor/etc without mood qualifiers
            result = self.direct_request_handler.handle(None, None, criteria=merged)
        else:
            # Re-use theme_recommender for semantics (or semantics + strict genre bounds)
            result = self.theme_recommender.recommend(req, feedback=[], criteria=merged)
            
        movies = result.get("titles", [])
        
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
            compromise_notes=notes
        )
        
        history = state.get("history", []) + [
            {"user": "Group preferences merged", "assistant": response, "intent": "mediation"}
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
                "current_participant": 0
            }
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
                return Command(goto="refine", update={"response": "Nobody liked those options."})
            
            # Check for tie
            max_votes = max(tally.values())
            top_movies = [m for m, v in tally.items() if v == max_votes]
            if len(top_movies) > 1:
                return Command(goto="refine", update={"response": "We have a tie! Let's refine our search."})
            
            winner = top_movies[0]
            return Command(goto="goodbye", update={"choice": winner})
            
        person = names[idx]
        text = state.get("response", "") if idx == 0 else ""
        msg = f"{text}\n\n{person}, which one speaks to you? (pick 1-{len(movies)} or say 'none')".strip()
        
        ans = interrupt(prompt(msg, movies, participant=person)).strip()
        
        vote_val = "none"
        if ans.isdigit() and 1 <= int(ans) <= len(movies):
            vote_val = movies[int(ans) - 1]
            
        return Command(
            goto="group_vote",
            update={
                "votes": {**state.get("votes", {}), person: vote_val},
                "current_participant": idx + 1
            }
        )

    def refine(self, state: State) -> Command:
        if state.get("round", 1) >= 3:
            # Force the top-voted movie or first from shortlist
            movies = state.get("movies", [])
            fallback = movies[0] if movies else "something fun"
            return Command(goto="goodbye", update={"choice": fallback})
            
        text = state.get("response", "Let's try again.")
        msg = f"{text}\nWhat should we change? (e.g. 'less sci-fi, more comedy')"
        ans = interrupt(prompt(msg))
        
        group_fb = list(state.get("group_feedback", []))
        group_fb.append(ans)
        
        result = self.classifier.classify(ans)
        new_group = merge_criteria(state["group_criteria"], result.entities.model_dump(), result.criteria_action)
        
        return Command(
            goto="mediate",
            update={
                "group_criteria": new_group,
                "group_feedback": group_fb,
                "round": state.get("round", 1) + 1
            }
        )

    def goodbye(self, state: State) -> Command:
        return Command(
            goto=END,
            update={"farewell": f"{state['choice']} wins! Enjoy the movie, everyone. 🎬"}
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
    load_dotenv()
    OrquestratorAgent().run()
