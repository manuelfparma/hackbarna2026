import json
import unittest
from unittest.mock import Mock

from langgraph.types import Command

from agent.nodes.classifier import Classification
from agent.nodes.criteria import normalize_criteria
from agent.nodes.reply import ReplyComposer
from agent.recomedation_agent import initial_state
from tests.test_upgrade import make_agent as make_base_agent, LLM


def make_agent(*results):
    agent = make_base_agent(*results)
    agent.theme_recommender = Mock()
    agent.theme_recommender.recommend.return_value = {
        "kind": "theme_recommendation",
        "titles": ["Fallback movie"],
        "error": None,
    }
    return agent


class ControlTests(unittest.TestCase):
    def dirty_state(self, request):
        return {
            **initial_state(request),
            "search_criteria": normalize_criteria(
                {
                    "genres": ["Comedy"],
                    "years": [2020],
                    "time_periods": ["90s"],
                    "actors": ["Adam Sandler"],
                }
            ),
            "movies": ["Old suggestion"],
            "focused_movie": "Old suggestion",
            "last_search": {
                "route": "theme_recommendation",
                "request": "1990s comedy",
                "search_mode": "theme",
            },
            "last_route": "theme_recommendation",
            "feedback": ["Only 2020"],
            "history": [
                {
                    "user": "Only 2020",
                    "assistant": "No 2020 matches",
                    "intent": "direct_request",
                }
            ],
            "seen_movies": [{"title": "Heat", "role": "seen"}],
        }

    def test_full_reset_commands_work_even_when_classifier_is_unavailable(self):
        for request in [
            "Start over. Let's find another movie",
            "nevermind",
            "Never mind",
            "No, forget about it",
            "Forget about this, I want to search for other movies",
            "Please reset all filters",
        ]:
            with self.subTest(request=request):
                agent = make_agent()
                agent.classifier.llm.invoke = Mock(side_effect=RuntimeError("offline"))
                config = {"configurable": {"thread_id": request}}
                event = agent.graph.invoke(self.dirty_state(request), config)
                state = agent.graph.get_state(config).values
                self.assertEqual(state["search_criteria"], normalize_criteria({}))
                self.assertEqual(state["movies"], [])
                self.assertEqual(state["last_search"], {})
                self.assertEqual(state["feedback"], [])
                self.assertEqual(state["history"], [])
                self.assertIsNone(state["focused_movie"])
                self.assertNotIn("watchlist", state)
                self.assertEqual(len(state["seen_movies"]), 1)
                self.assertEqual(event["__interrupt__"][0].value["options"], [])
                self.assertIn("cleared", state["response"].lower())
                agent.classifier.llm.invoke.assert_not_called()

    def test_date_removal_clears_all_representations_but_keeps_other_preferences(self):
        for request in [
            "Forget about 2020",
            "Forget about the 90s",
            "Remove the date filter",
            "Any year is fine",
            "The year doesn't matter",
        ]:
            with self.subTest(request=request):
                agent = make_agent()
                config = {"configurable": {"thread_id": request}}
                agent.graph.invoke(self.dirty_state(request), config)
                state = agent.graph.get_state(config).values
                c = state["search_criteria"]
                self.assertEqual(c["years"], [])
                self.assertEqual(c["time_periods"], [])
                self.assertIsNone(c["year_min"])
                self.assertIsNone(c["year_max"])
                self.assertEqual(c["actors"], ["Adam Sandler"])
                self.assertEqual(c["genres"], ["Comedy"])
                self.assertNotIn("audience", c)
                self.assertEqual(state["last_search"], {})
                self.assertEqual(state["history"], [])
                self.assertIn("date", state["response"].lower())

    def test_reset_and_new_search_classifies_only_new_request_against_clean_context(
        self,
    ):
        agent = make_agent(
            Classification(
                intent="direct_request",
                entities={"genres": ["Comedy"]},
                criteria_action="add",
            )
        )
        config = {"configurable": {"thread_id": "compound"}}
        agent.graph.invoke(self.dirty_state("Start over. I want a comedy"), config)
        criteria = agent.theme_recommender.recommend.call_args.kwargs["criteria"]
        self.assertEqual(criteria["genres"], ["Comedy"])
        self.assertEqual(criteria["years"], [])
        self.assertEqual(criteria["actors"], [])
        prompt = json.loads(agent.classifier.llm.prompts[0][-1].content)
        self.assertEqual(prompt["request"], "I want a comedy")
        self.assertEqual(prompt["context"]["criteria_summary"], {})
        self.assertEqual(prompt["context"]["recent_history"], [])

    def test_more_after_removing_date_does_not_replay_old_query(self):
        agent = make_agent(Classification(intent="more", criteria_action="keep"))
        agent.direct_request_handler.handle = Mock(
            return_value={
                "kind": "direct_request",
                "titles": ["Fresh film"],
                "error": None,
            }
        )
        config = {"configurable": {"thread_id": "remove-more"}}
        state = self.dirty_state("Forget about 2020")
        state["seen_movies"] = []
        agent.graph.invoke(state, config)
        agent.graph.invoke(Command(resume="What else?"), config)
        agent.direct_request_handler.handle.assert_called_once()
        criteria = agent.direct_request_handler.handle.call_args.kwargs["criteria"]
        self.assertEqual(criteria["years"], [])
        self.assertIsNone(criteria["year_min"])
        self.assertEqual(criteria["actors"], ["Adam Sandler"])

    def test_social_intent_does_not_override_explicit_reset_action(self):
        agent = make_agent(Classification(intent="social", criteria_action="reset"))
        config = {"configurable": {"thread_id": "model-reset"}}
        agent.graph.invoke(self.dirty_state("Wipe the slate clean"), config)
        self.assertEqual(
            agent.graph.get_state(config).values["search_criteria"],
            normalize_criteria({}),
        )

    def test_classifier_failures_are_logged_and_not_social(self):
        for response in [RuntimeError("secret-provider-value"), {"unexpected": True}]:
            with self.subTest(response=type(response).__name__):
                agent = make_agent()
                agent.classifier.llm.invoke = (
                    Mock(side_effect=response)
                    if isinstance(response, Exception)
                    else Mock(return_value=response)
                )
                config = {"configurable": {"thread_id": "failure"}}
                before = self.dirty_state("An unusual request")
                with self.assertLogs(
                    "agent.nodes.classifier", level="ERROR"
                ) as captured:
                    agent.graph.invoke(before, config)
                state = agent.graph.get_state(config).values
                self.assertEqual(state["result"]["kind"], "classification_error")
                self.assertEqual(state["search_criteria"], before["search_criteria"])
                self.assertNotIn(
                    "secret-provider-value", str(captured.output) + state["response"]
                )
                self.assertIn("not run", state["response"].lower())

    def test_named_factual_question_does_not_depend_on_llm_routing(self):
        agent = make_agent()
        agent.classifier.llm.invoke = Mock(side_effect=RuntimeError("offline"))
        agent.direct_request_handler.handle_movie_fact = Mock(
            return_value={
                "kind": "movie_fact",
                "titles": ["Love and Monsters"],
                "facts": {"name": "Love and Monsters", "description": "Catalog plot"},
                "error": None,
            }
        )
        config = {"configurable": {"thread_id": "named-fact"}}
        agent.graph.invoke(
            self.dirty_state("What can you tell me about Love and Monsters?"), config
        )
        state = agent.graph.get_state(config).values
        self.assertEqual(state["result"]["kind"], "movie_fact")
        self.assertEqual(
            agent.direct_request_handler.handle_movie_fact.call_args.args[0],
            "Love and Monsters",
        )
        agent.classifier.llm.invoke.assert_not_called()

    def test_social_reply_cannot_invent_facts_or_repeat_stale_search_failure(self):
        llm = LLM()
        llm.invoke = Mock(
            return_value=Mock(
                content="Love and Monsters is a 2020 comedy. No matches found."
            )
        )
        reply = ReplyComposer(llm).compose(
            request="thanks",
            intent="social",
            entities={},
            result={"kind": "social", "titles": [], "error": None},
            history=[
                {"user": "2020", "assistant": "No matches", "intent": "direct_request"}
            ],
            feedback=[],
            movies=[],
            search_criteria={"years": [2020]},
        )
        self.assertNotIn("2020", reply)
        self.assertNotIn("Love and Monsters", reply)
        self.assertNotIn("No matches", reply)
        llm.invoke.assert_not_called()

    def test_model_removal_is_not_overridden_by_social_and_does_not_readd_year(self):
        agent = make_agent(
            Classification(
                intent="social",
                criteria_action="remove",
                clear_fields=["date"],
                entities={"years": [2020]},
            )
        )
        config = {"configurable": {"thread_id": "model-remove"}}
        agent.graph.invoke(self.dirty_state("2020 is no longer a requirement"), config)
        state = agent.graph.get_state(config).values
        self.assertEqual(state["search_criteria"]["years"], [])
        self.assertIsNone(state["search_criteria"]["year_min"])
        self.assertEqual(state["search_criteria"]["actors"], ["Adam Sandler"])

    def test_compound_reset_is_applied_even_if_new_query_cannot_be_classified(self):
        agent = make_agent()
        agent.classifier.llm.invoke = Mock(side_effect=RuntimeError("offline"))
        config = {"configurable": {"thread_id": "reset-error"}}
        with self.assertLogs("agent.nodes.classifier", level="ERROR"):
            agent.graph.invoke(self.dirty_state("Start over. I want a comedy"), config)
        state = agent.graph.get_state(config).values
        self.assertEqual(state["search_criteria"], normalize_criteria({}))
        self.assertEqual(state["result"]["kind"], "classification_error")
        self.assertIn("cleared", state["response"])
        self.assertIn("not run", state["response"])

    def test_duration_removal_does_not_rebuild_bounds_from_words(self):
        agent = make_agent()
        state = self.dirty_state("Forget the runtime")
        state["search_criteria"] = normalize_criteria(
            {
                **state["search_criteria"],
                "duration": ["under two hours"],
                "minute_min": 80,
            }
        )
        config = {"configurable": {"thread_id": "duration-remove"}}
        agent.graph.invoke(state, config)
        criteria = agent.graph.get_state(config).values["search_criteria"]
        self.assertEqual(criteria["duration"], [])
        self.assertIsNone(criteria["minute_min"])
        self.assertIsNone(criteria["minute_max"])
        self.assertEqual(criteria["years"], [2020])

    def test_transcript_reset_prevents_date_leaking_into_sandler_search(self):
        agent = make_agent(
            Classification(
                intent="direct_request",
                criteria_action="add",
                entities={
                    "actors": ["Adam Sandler"],
                    "genres": ["Comedy"],
                },
            )
        )
        agent.direct_request_handler.handle = Mock(
            return_value={
                "kind": "direct_request",
                "titles": ["A new movie"],
                "error": None,
            }
        )
        agent.direct_request_handler.handle_movie_fact = Mock(
            return_value={
                "kind": "movie_fact",
                "titles": ["Love and Monsters"],
                "facts": {"name": "Love and Monsters"},
                "error": None,
            }
        )
        config = {"configurable": {"thread_id": "transcript"}}
        agent.graph.invoke(
            self.dirty_state("What can you tell me about Love and Monsters?"), config
        )
        agent.graph.invoke(
            Command(resume="Forget about this, I want to search for other movies"),
            config,
        )
        agent.graph.invoke(
            Command(
                resume="I'm with my kids, we wanted to watch a comedy where Adam Sandler appears"
            ),
            config,
        )
        criteria = agent.direct_request_handler.handle.call_args.kwargs["criteria"]
        self.assertEqual(criteria["years"], [])
        self.assertEqual(criteria["time_periods"], [])
        self.assertEqual(criteria["actors"], ["Adam Sandler"])
        self.assertEqual(criteria["genres"], ["Comedy"])
        agent.graph.invoke(
            Command(resume="Start over. Let's find another movie"), config
        )
        self.assertEqual(
            agent.graph.get_state(config).values["search_criteria"],
            normalize_criteria({}),
        )

    def test_genuine_retrieval_failure_cannot_be_rewritten_as_success(self):
        llm = LLM()
        llm.invoke = Mock(return_value=Mock(content="I found many movies."))
        response = ReplyComposer(llm).compose(
            request="search",
            intent="direct_request",
            entities={},
            result={
                "kind": "direct_request",
                "titles": [],
                "error": "The catalog is unavailable.",
            },
            history=[],
            feedback=[],
            movies=["Old movie"],
        )
        self.assertIn("unavailable", response)
        self.assertNotIn("found many", response)
        llm.invoke.assert_not_called()

    def test_negated_reset_and_titles_are_not_control_commands(self):
        for request in [
            "Don't forget about 2020",
            "Don't start over",
            "Tell me about Forgetting Sarah Marshall",
        ]:
            with self.subTest(request=request):
                agent = make_agent(
                    Classification(intent="social", criteria_action="keep")
                )
                agent.direct_request_handler.handle_movie_fact = Mock(
                    return_value={
                        "kind": "movie_fact",
                        "titles": [],
                        "error": "Not found",
                    }
                )
                config = {"configurable": {"thread_id": request}}
                state = self.dirty_state(request)
                agent.graph.invoke(state, config)
                self.assertEqual(
                    agent.graph.get_state(config).values["search_criteria"],
                    state["search_criteria"],
                )


if __name__ == "__main__":
    unittest.main()
