import unittest
from typing import get_args
from unittest.mock import Mock, patch

from agent.nodes.theme_recommender import ThemeRecommender

from agent.nodes.classifier import Classification, Classifier, Entities, Intent
from agent.nodes.criteria import (
    normalize_criteria,
    has_catalog_filters,
    merge_criteria,
    build_theme_query,
    ClearField,
    CLEAR_GROUPS,
)
from agent.nodes.criteria_commands import parse_criteria_command
from agent.nodes.filters import apply_filters, matches_filters
from agent.nodes.reply import ReplyComposer
from agent.recomedation_agent import State, initial_state
from tests.test_upgrade import LLM, make_agent


class SimplifiedScopeTests(unittest.TestCase):
    def test_audience_is_absent_from_schema_memory_query_and_clear_commands(self):
        self.assertNotIn("audience", Entities.model_fields)
        self.assertNotIn("audience", get_args(ClearField))
        self.assertNotIn("audience", CLEAR_GROUPS)
        legacy = {"audience": ["kids"], "genres": ["Comedy"]}
        self.assertNotIn("audience", Entities.model_validate(legacy).model_dump())
        self.assertNotIn("audience", normalize_criteria(legacy))
        for action in ("keep", "add", "replace", "reset"):
            self.assertNotIn("audience", merge_criteria(legacy, legacy, action))
        self.assertNotIn("kids", build_theme_query("something funny", legacy))
        self.assertIsNone(parse_criteria_command("Remove the audience filter"))

    def test_similar_search_does_not_append_legacy_audience(self):
        recommender = ThemeRecommender()
        recommender._supabase, recommender._embeddings = Mock(), Mock()
        with (
            patch(
                "agent.nodes.theme_recommender.resolve_movie",
                return_value={
                    "id": 1,
                    "name": "Seed",
                    "description": "A story about friendship",
                },
            ),
            patch(
                "agent.nodes.theme_recommender.search_descriptions",
                return_value=[{"movie_id": 2, "name": "Another film"}],
            ) as search,
        ):
            result = recommender.recommend(
                "like Seed",
                criteria={
                    "movies": [{"title": "Seed", "role": "seed"}],
                    "audience": ["kids"],
                },
            )
        self.assertEqual(result["titles"], ["Another film"])
        self.assertNotIn("kids", search.call_args.args[2])
        self.assertNotIn("audience", search.call_args.kwargs["criteria"])

    def test_removed_capabilities_have_no_schema_state_or_graph_nodes(self):
        self.assertTrue({"save", "compare", "meta"}.isdisjoint(get_args(Intent)))
        self.assertNotIn("meta_action", Classification.model_fields)
        self.assertNotIn("studios", Entities.model_fields)
        self.assertNotIn("exclude_themes", Entities.model_fields)
        self.assertNotIn("watchlist", State.__annotations__)
        self.assertNotIn("watchlist", initial_state())
        agent = make_agent()
        for node in ("save", "compare", "meta"):
            self.assertNotIn(node, agent.graph.nodes)
            self.assertFalse(hasattr(agent, node))

    def test_old_filter_fields_are_ignored_by_both_query_paths(self):
        legacy = {
            "studios": ["Pixar"],
            "exclude_themes": ["death"],
            "excluded_theme_labels": ["Death"],
            "genres": ["Comedy"],
        }
        criteria = normalize_criteria(legacy)
        self.assertNotIn("studios", criteria)
        self.assertNotIn("exclude_themes", criteria)
        self.assertNotIn("excluded_theme_labels", criteria)
        self.assertFalse(
            has_catalog_filters({"studios": ["Pixar"], "exclude_themes": ["death"]})
        )
        query = Mock()
        query.contains.return_value = query
        apply_filters(query, legacy)
        query.contains.assert_called_once_with("genres", ["Comedy"])
        query.ilike.assert_not_called()
        query.not_.contains.assert_not_called()
        self.assertTrue(
            matches_filters({"genres": ["Comedy"], "themes": ["Death"]}, legacy)
        )

    def test_negative_genres_remain_supported_without_semantic_expansion(self):
        criteria = normalize_criteria({"exclude_genres": ["Horror", "gore"]})
        self.assertEqual(criteria["exclude_genres"], ["Horror"])
        self.assertNotIn("exclude_themes", criteria)
        self.assertFalse(matches_filters({"genres": ["Horror"]}, criteria))
        self.assertTrue(matches_filters({"genres": ["Comedy"]}, criteria))

    def test_explicit_audience_does_not_infer_hard_constraints(self):
        llm = LLM(
            Classification(
                intent="direct_request",
                entities={
                    "audience": ["kids"],
                    "actors": ["Adam Sandler"],
                    "genres": ["Comedy", "Family"],
                    "genre_groups": [["Family"], ["Animation"]],
                    "exclude_genres": ["Horror"],
                },
            )
        )
        result = Classifier(llm).classify(
            "I'm with my kids, we want a comedy with Adam Sandler"
        )
        self.assertNotIn("audience", result.entities.model_dump())
        self.assertEqual(result.entities.genres, ["Comedy"])
        self.assertEqual(result.entities.genre_groups, [])
        self.assertEqual(result.entities.exclude_genres, [])

    def test_audience_inference_cannot_reenter_through_themes_or_missing_audience_slot(
        self,
    ):
        for audience in ([], ["family"]):
            with self.subTest(audience=audience):
                llm = LLM(
                    Classification(
                        intent="recommendation",
                        entities={
                            "audience": audience,
                            "genres": ["Family", "Animation"],
                            "themes": ["animation films", "family movies"],
                            "exclude_genres": ["Horror"],
                        },
                    )
                )
                result = Classifier(llm).classify("Something to watch with my family")
                criteria = normalize_criteria(result.entities.model_dump())
                self.assertEqual(criteria["genres"], [])
                self.assertEqual(criteria["exclude_genres"], [])

    def test_semantic_exclusion_is_not_demoted_into_a_different_search(self):
        llm = LLM(
            Classification(
                intent="direct_request",
                entities={"genres": ["Comedy"], "exclude_genres": ["gore"]},
            )
        )
        result = Classifier(llm).classify("A comedy without gore")
        self.assertEqual(result.intent, "unsupported")
        self.assertEqual(result.criteria_action, "keep")

    def test_explicit_family_animation_and_no_horror_are_not_removed(self):
        llm = LLM(
            Classification(
                intent="direct_request",
                entities={
                    "audience": ["kids"],
                    "genre_groups": [["Family"], ["Animation"]],
                    "exclude_genres": ["Horror"],
                },
            )
        )
        result = Classifier(llm).classify(
            "A family movie or animation for my kids, no horror"
        )
        self.assertEqual(result.entities.genre_groups, [["Family"], ["Animation"]])
        self.assertEqual(result.entities.exclude_genres, ["Horror"])

    def test_unsupported_request_keeps_brief_and_does_not_access_catalog(self):
        agent = make_agent(
            Classification(
                intent="unsupported",
                criteria_action="reset",
                entities={"genres": ["Drama"]},
            )
        )
        agent.direct_request_handler = Mock()
        agent.theme_recommender = Mock()
        config = {"configurable": {"thread_id": "unsupported"}}
        state = {
            **initial_state("Save this to my watchlist"),
            "search_criteria": normalize_criteria({"genres": ["Comedy"]}),
        }
        event = agent.graph.invoke(state, config)
        after = agent.graph.get_state(config).values
        self.assertEqual(after["search_criteria"], state["search_criteria"])
        self.assertEqual(after["result"]["kind"], "unsupported")
        self.assertEqual(event["__interrupt__"][0].value["options"], [])
        self.assertIn("not supported", after["response"])
        agent.direct_request_handler.handle.assert_not_called()
        agent.theme_recommender.recommend.assert_not_called()

    def test_help_is_static_and_only_advertises_retained_features(self):
        agent = make_agent(Classification(intent="help"))
        agent.direct_request_handler = Mock()
        config = {"configurable": {"thread_id": "help"}}
        agent.graph.invoke(initial_state("What can you do?"), config)
        response = agent.graph.get_state(config).values["response"].lower()
        for removed in ("watchlist", "compare", "studio", "count"):
            self.assertNotIn(removed, response)
        self.assertIn("recommend", response)
        self.assertIn("plot", response)
        agent.direct_request_handler.supabase.table.assert_not_called()

    def test_unsupported_reply_cannot_pretend_operation_succeeded(self):
        llm = Mock()
        response = ReplyComposer(llm).compose(
            request="Compare these two",
            intent="unsupported",
            entities={},
            result={"kind": "unsupported", "titles": [], "error": None},
            history=[],
            feedback=[],
            movies=[],
        )
        self.assertIn("not supported", response)
        llm.invoke.assert_not_called()


if __name__ == "__main__":
    unittest.main()
