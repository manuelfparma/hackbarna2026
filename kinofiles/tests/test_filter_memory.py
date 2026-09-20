import unittest

from langgraph.types import Command

from agent.nodes.classifier import Classification
from agent.recomedation_agent import RecommendationAgent


class Response:
    def __init__(self, content):
        self.content = content


class FakeLLM:
    def __init__(self, content):
        self.content = content

    def invoke(self, _prompt):
        return Response(self.content)


class SequencedClassifierLLM:
    def __init__(self):
        self.results = [
            Classification(
                intent="direct_request",
                entities={"genres": ["Comedy"]},
                criteria_action="replace",
            ),
            Classification(
                intent="feedback",
                entities={"genres": ["Action"]},
                criteria_action="add",
            ),
        ]

    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return self.results.pop(0)


class Result:
    def __init__(self, data):
        self.data = data


class RecordingQuery:
    def __init__(self, data, calls):
        self.data = data
        self.calls = calls

    def select(self, *_args):
        return self

    def contains(self, column, values):
        self.calls.append((column, values))
        return self

    def neq(self, *_args):
        return self

    def order(self, *_args, **_kwargs):
        return self

    def limit(self, *_args):
        return self

    def execute(self):
        return Result(self.data)


class RecordingSupabase:
    MOVIES = [
        {
            "name": "The Nice Guys",
            "rating": 4.0,
            "themes": ["Funny"],
            "description": "",
        },
        {"name": "Hot Fuzz", "rating": 3.9, "themes": ["Funny"], "description": ""},
    ]

    def __init__(self):
        self.calls = []

    def table(self, _name):
        return RecordingQuery(self.MOVIES, self.calls)

    def rpc(self, _name, _params):
        return RecordingQuery([{"theme": "Funny", "similarity": 0.8}], self.calls)


class FakeEmbeddings:
    def embed_query(self, _text):
        return [0.0] * 1024


class FilterMemoryGraphTests(unittest.TestCase):
    def test_comedy_then_also_action_uses_both_genres(self):
        agent = RecommendationAgent(
            llm=FakeLLM("unused"),
            classifier_llm=SequencedClassifierLLM(),
            reply_llm=FakeLLM("These match the filters you've given me."),
        )
        supabase = RecordingSupabase()
        agent.direct_request_handler.supabase = supabase
        # A bare genre is ranked semantically, so the same catalog filter now
        # arrives through the theme recommender.
        agent.theme_recommender._supabase = supabase
        agent.theme_recommender._embeddings = FakeEmbeddings()
        config = {"configurable": {"thread_id": "criteria-two-turn"}}

        first = agent.graph.invoke(
            {
                "request": "I want a comedy movie",
                "feedback": [],
                "movies": [],
                "search_criteria": {},
                "result": {},
                "history": [],
                "show_options": False,
            },
            config,
        )
        self.assertIn("__interrupt__", first)

        second = agent.graph.invoke(
            Command(resume="Also, it should have some action in it"),
            config,
        )

        self.assertIn("__interrupt__", second)
        self.assertEqual(
            supabase.calls,
            [
                ("genres", ["Comedy"]),
                ("genres", ["Comedy", "Action"]),
            ],
        )
        state = agent.graph.get_state(config).values
        self.assertEqual(state["search_criteria"]["genres"], ["Comedy", "Action"])
        self.assertEqual(
            second["__interrupt__"][0].value["options"],
            ["The Nice Guys", "Hot Fuzz"],
        )

    def test_generic_recommendation_routes_to_theme_search(self):
        self.assertEqual(
            RecommendationAgent._route_for_criteria("recommendation", {}, {}),
            "theme_recommendation",
        )

    def test_feedback_without_new_filters_stays_on_feedback(self):
        self.assertEqual(
            RecommendationAgent._route_for_criteria("feedback", {}, {}),
            "feedback",
        )


if __name__ == "__main__":
    unittest.main()
