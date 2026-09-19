import unittest
from unittest.mock import patch

from agent.io.turn import prompt
from agent.nodes.social import SocialHandler
from agent.recomedation_agent import RecommendationAgent


class TurnTests(unittest.TestCase):
    def setUp(self):
        self.agent = RecommendationAgent.__new__(RecommendationAgent)
        self.agent.social_handler = SocialHandler()

    def test_prompt_keeps_titles_out_of_narration(self):
        payload = prompt("I found a few options.", ["Drive", "Heat"])

        self.assertEqual(payload["text"], "I found a few options.")
        self.assertEqual(payload["options"], ["Drive", "Heat"])
        self.assertNotIn("Drive", payload["text"])

    def test_visible_shortlist_has_deterministic_selection_instruction(self):
        captured = {}

        def answer(payload):
            captured.update(payload)
            return "2"

        with patch("agent.recomedation_agent.interrupt", side_effect=answer):
            command = self.agent.turn(
                {
                    "response": "These fit your request.",
                    "movies": ["Drive", "Heat"],
                    "show_options": True,
                }
            )

        self.assertEqual(captured["options"], ["Drive", "Heat"])
        self.assertIn("Pick a number (1-2)", captured["text"])
        self.assertEqual(command.update["choice"], "Heat")

    def test_social_detour_hides_but_preserves_selectable_shortlist(self):
        social_update = self.agent.social(
            {"request": "thanks", "movies": ["Drive", "Heat"]}
        )
        self.assertNotIn("movies", social_update)
        self.assertFalse(social_update["show_options"])

        captured = {}

        def answer(payload):
            captured.update(payload)
            return "1"

        with patch("agent.recomedation_agent.interrupt", side_effect=answer):
            command = self.agent.turn(
                {
                    "response": "You're welcome!",
                    "movies": ["Drive", "Heat"],
                    "show_options": False,
                }
            )

        self.assertEqual(captured["options"], [])
        self.assertNotIn("Pick a number", captured["text"])
        self.assertEqual(command.update["choice"], "Drive")


if __name__ == "__main__":
    unittest.main()
