import unittest

from agent.nodes.classifier import Classification
from agent.nodes.criteria import normalize_criteria
from agent.recomedation_agent import RecommendationAgent

route = RecommendationAgent._route_for_criteria


class RouteTests(unittest.TestCase):
    def test_seed_movie_takes_similar_to_even_with_filters(self):
        entities = {
            "movies": [{"title": "The Dark Knight", "role": "seed"}],
            "genres": ["Action", "Thriller"],
        }
        criteria = normalize_criteria(entities)

        self.assertEqual(
            route("recommendation", entities, criteria), "theme_recommendation"
        )

    def test_direct_request_without_surviving_filters_goes_semantic(self):
        entities = {"genres": ["Heist"]}
        criteria = normalize_criteria(entities)  # "Heist" demotes to themes

        self.assertEqual(
            route("direct_request", entities, criteria), "theme_recommendation"
        )

    def test_direct_request_with_real_filters_stays(self):
        entities = {"directors": ["Nolan"]}
        criteria = normalize_criteria(entities)

        self.assertEqual(route("direct_request", entities, criteria), "direct_request")

    def test_factual_request_keeps_fact_path(self):
        entities = {"movies": [{"title": "Inception", "role": "asked_about"}]}
        criteria = normalize_criteria(entities)

        self.assertEqual(route("direct_request", entities, criteria), "direct_request")

    def test_wanted_title_does_not_trigger_similar_to(self):
        entities = {"movies": [{"title": "Heat", "role": "wanted"}]}

        self.assertEqual(
            route("direct_request", entities, normalize_criteria(entities)),
            "direct_request",
        )


class AsFilterTests(unittest.TestCase):
    def test_attributes_beat_title_when_both_present(self):
        result = Classification(
            intent="direct_request",
            entities={
                "movies": [{"title": "Nolan", "role": "asked_about"}],
                "directors": ["Nolan"],
            },
        )

        self.assertEqual(result.as_filter(), ("directors", "Nolan"))

    def test_title_is_fallback_when_nothing_else(self):
        result = Classification(
            intent="direct_request",
            entities={"movies": [{"title": "Heat", "role": "wanted"}]},
        )

        self.assertEqual(result.as_filter(), ("name", "Heat"))


if __name__ == "__main__":
    unittest.main()
