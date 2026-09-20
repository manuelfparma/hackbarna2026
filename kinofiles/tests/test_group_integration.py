import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from langgraph.types import Command

from agent.nodes.classifier import Classification
from agent.nodes.criteria import merge_group_criteria, normalize_criteria
from agent.orchestrator import OrquestratorAgent
from tests.test_upgrade import LLM


class GroupReplyLLM(LLM):
    def with_structured_output(self, schema):
        return Mock(invoke=Mock(return_value=SimpleNamespace(names=["Alice", "Bob"])))


def make_group(*classifications):
    reply = GroupReplyLLM()
    with (
        patch(
            "agent.orchestrator.build_mistral_llm", return_value=LLM(*classifications)
        ),
        patch("agent.nodes.direct_request.load_dotenv"),
        patch.dict("os.environ", {}, clear=True),
    ):
        agent = OrquestratorAgent(llm=reply, reply_llm=reply)
    result = {
        "kind": "direct_request",
        "titles": ["First movie", "Second movie"],
        "descriptions": {"First movie": "First plot", "Second movie": "Second plot"},
        "error": None,
    }
    agent.direct_request_handler.handle = Mock(return_value=result)
    agent.theme_recommender.recommend = Mock(return_value=result)
    return agent


class GroupIntegrationTests(unittest.TestCase):
    def test_group_flow_collects_independent_preferences_and_votes(self):
        agent = make_group(
            Classification(
                intent="direct_request",
                entities={"genres": ["Comedy"], "minute_max": 90},
            ),
            Classification(
                intent="direct_request",
                entities={"genres": ["Comedy", "Action"], "minute_max": 120},
            ),
        )
        config = {"configurable": {"thread_id": "group-vote"}}
        welcome = agent.graph.invoke({}, config)
        self.assertIn("names", welcome["__interrupt__"][0].value["text"])
        event = agent.graph.invoke(Command(resume="Alice and Bob"), config)
        self.assertEqual(event["__interrupt__"][0].value["participant"], "Alice")
        event = agent.graph.invoke(Command(resume="Comedy under 91 minutes"), config)
        self.assertEqual(event["__interrupt__"][0].value["participant"], "Bob")
        event = agent.graph.invoke(
            Command(resume="Action comedy under 121 minutes"), config
        )
        self.assertEqual(event["__interrupt__"][0].value["participant"], "Alice")
        self.assertEqual(
            event["__interrupt__"][0].value["options"], ["First movie", "Second movie"]
        )
        criteria = agent.direct_request_handler.handle.call_args.kwargs["criteria"]
        self.assertEqual(criteria["genres"], ["Comedy"])
        self.assertEqual(criteria["minute_max"], 90)
        context = json.loads(
            agent.reply_llm.prompts[-1].split("Conversation context:\n")[1]
        )
        self.assertEqual(set(context["per_person_criteria"]), {"Alice", "Bob"})
        self.assertTrue(context["compromise_notes"])
        event = agent.graph.invoke(Command(resume="1"), config)
        self.assertEqual(event["__interrupt__"][0].value["participant"], "Bob")
        event = agent.graph.invoke(Command(resume="1"), config)
        self.assertNotIn("__interrupt__", event)
        self.assertEqual(event["choice"], "First movie")
        self.assertEqual(event["votes"], {"Alice": "First movie", "Bob": "First movie"})
        self.assertIn("wins", event["farewell"])

    def test_group_merge_keeps_upstream_genre_policy_and_new_scalar_bounds(self):
        criteria, notes = merge_group_criteria(
            {
                "Alice": {
                    "genres": ["Comedy"],
                    "year_min": 2010,
                    "minute_max": 90,
                    "min_rating": 3.5,
                },
                "Bob": {
                    "genres": ["Action"],
                    "year_min": 2015,
                    "minute_max": 120,
                    "min_rating": 4,
                },
            }
        )
        self.assertEqual(set(criteria["genres"]), {"Comedy", "Action"})
        self.assertEqual(criteria["year_min"], 2015)
        self.assertEqual(criteria["minute_max"], 90)
        self.assertEqual(criteria["min_rating"], 4)
        self.assertIn("Combined different genre preferences", notes)

    def test_refinement_passes_clear_fields_to_shared_merger(self):
        agent = make_group(
            Classification(
                intent="direct_request", criteria_action="remove", clear_fields=["date"]
            )
        )
        state = {
            "round": 1,
            "group_criteria": normalize_criteria(
                {"years": [2020], "genres": ["Comedy"]}
            ),
            "group_feedback": [],
        }
        with patch("agent.orchestrator.interrupt", return_value="Forget about 2020"):
            command = agent.refine(state)
        self.assertEqual(command.update["group_criteria"]["years"], [])
        self.assertEqual(command.update["group_criteria"]["genres"], ["Comedy"])
        self.assertEqual(command.goto, "mediate")

    def test_classifier_failure_retries_current_group_stage_without_losing_preferences(
        self,
    ):
        for stage in ("collect_preferences", "refine"):
            with self.subTest(stage=stage):
                agent = make_group()
                agent.classifier.llm.invoke = Mock(side_effect=RuntimeError("offline"))
                state = {
                    "participants": ["Alice", "Bob"],
                    "current_participant": 0,
                    "per_person_criteria": {
                        "Alice": normalize_criteria({"genres": ["Comedy"]})
                    },
                    "preferences": {"Alice": []},
                    "group_criteria": normalize_criteria({"genres": ["Comedy"]}),
                    "round": 1,
                }
                with (
                    patch(
                        "agent.orchestrator.interrupt", return_value="something funny"
                    ),
                    self.assertLogs("agent.nodes.classifier", level="ERROR"),
                ):
                    command = getattr(agent, stage)(state)
                self.assertEqual(command.goto, stage)
                self.assertNotIn("current_participant", command.update)
                self.assertNotIn("per_person_criteria", command.update)
                self.assertNotIn("group_criteria", command.update)
                self.assertNotIn("round", command.update)
                self.assertIn("try again", command.update["response"].lower())

    def test_group_fallback_does_not_append_a_second_question(self):
        agent = make_group()
        result = {
            "kind": "theme_recommendation",
            "titles": [],
            "error": "No catalog matches.",
        }
        response = agent.reply_composer.compose(
            request="mediate",
            intent="theme_recommendation",
            entities={},
            result=result,
            history=[],
            feedback=[],
            movies=[],
            per_person_criteria={"Alice": {}},
        )
        self.assertEqual(response, "No catalog matches.")
        self.assertEqual(agent.reply_llm.prompts, [])

    def test_vote_tie_uses_colleague_refinement_flow(self):
        agent = make_group()
        command = agent.group_vote(
            {
                "participants": ["Alice", "Bob"],
                "current_participant": 2,
                "movies": ["First movie", "Second movie"],
                "votes": {"Alice": "First movie", "Bob": "Second movie"},
            }
        )
        self.assertEqual(command.goto, "refine")
        self.assertIn("tie", command.update["response"].lower())


if __name__ == "__main__":
    unittest.main()
