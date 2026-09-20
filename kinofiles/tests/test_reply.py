import unittest

from agent.nodes.classifier import Classification
from agent.nodes.reply import MAX_HISTORY_TURNS, ReplyComposer
from agent.recomedation_agent import RecommendationAgent


class Response:
    def __init__(self, content: str):
        self.content = content


class FakeLLM:
    def __init__(self, content: str = ""):
        self.content = content
        self.prompts: list[str] = []

    def invoke(self, prompt: str):
        self.prompts.append(prompt)
        return Response(self.content)


class FakeClassifierLLM:
    def with_structured_output(self, _schema):
        return self

    def invoke(self, _messages):
        return Classification(intent="theme_recommendation")


class ReplyComposerTests(unittest.TestCase):
    def test_composer_passes_structured_context_without_debug_scores(self):
        llm = FakeLLM("These lean into the tense, rain-soaked mood you asked for.")
        composer = ReplyComposer(llm)

        reply = composer.compose(
            request="Something moody",
            intent="theme_recommendation",
            entities={"themes": ["moody"]},
            result={
                "kind": "theme_recommendation",
                "titles": ["Drive"],
                "themes": ["Neon-lit isolation"],
                "error": None,
            },
            history=[],
            feedback=[],
            movies=["Drive"],
        )

        self.assertEqual(
            reply, "These lean into the tense, rain-soaked mood you asked for."
        )
        context = llm.prompts[0].split("Conversation context:\n", 1)[1]
        self.assertIn("Neon-lit isolation", context)
        self.assertNotIn("similarity", context)
        self.assertNotIn("0.7", context)

    def test_fallback_renders_movie_facts(self):
        reply = ReplyComposer.fallback(
            "direct_request",
            {
                "facts": {
                    "name": "Heat",
                    "date": 1995,
                    "directors": ["Michael Mann"],
                    "rating": 8.3,
                }
            },
        )

        self.assertIn("Heat", reply)
        self.assertIn("1995", reply)
        self.assertIn("Michael Mann", reply)

    def test_reply_node_caps_history(self):
        agent = RecommendationAgent.__new__(RecommendationAgent)
        agent.reply_composer = ReplyComposer(FakeLLM("A concise reply."))
        old_history = [
            {"user": str(index), "assistant": "old", "intent": "social"}
            for index in range(MAX_HISTORY_TURNS)
        ]

        update = agent.reply(
            {
                "request": "new",
                "intent": "social",
                "entities": {},
                "result": {"kind": "social", "titles": [], "error": None},
                "history": old_history,
                "feedback": [],
                "movies": [],
            }
        )

        self.assertEqual(len(update["history"]), MAX_HISTORY_TURNS)
        self.assertEqual(update["history"][-1]["user"], "new")
        self.assertEqual(update["history"][0]["user"], "1")

    def test_compiled_graph_routes_capability_through_reply(self):
        class FakeThemeRecommender:
            def recommend(self, request, feedback=None, criteria=None, **kwargs):
                return {
                    "kind": "theme_recommendation",
                    "titles": ["Drive", "Heat", "Ronin"],
                    "themes": ["Adrenaline-fueled action and fast cars"],
                    "error": None,
                }

        agent = RecommendationAgent(
            llm=FakeLLM("unused"),
            classifier_llm=FakeClassifierLLM(),
            reply_llm=FakeLLM("I picked a few propulsive crime movies."),
        )
        agent.theme_recommender = FakeThemeRecommender()

        event = agent.graph.invoke(
            {
                "request": "Recommend something",
                "feedback": [],
                "movies": [],
                "result": {},
                "history": [],
                "show_options": False,
            },
            {"configurable": {"thread_id": "reply-test"}},
        )

        pending = event["__interrupt__"][0].value
        self.assertEqual(pending["options"], ["Drive", "Heat", "Ronin"])
        self.assertIn("I picked a few", pending["text"])
        self.assertIn("Pick a number (1-3)", pending["text"])


if __name__ == "__main__":
    unittest.main()
