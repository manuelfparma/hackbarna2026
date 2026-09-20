import json
import unittest
from unittest.mock import Mock, patch
from types import SimpleNamespace

from langgraph.types import Command
from agent.nodes.classifier import (
    Classification,
    Classifier,
    spoken_numbers_to_digits,
)
from agent.nodes.criteria import normalize_criteria, merge_criteria, has_catalog_filters
from agent.nodes.filters import apply_filters, matches_filters
from agent.nodes.description_search import search_descriptions
from agent.nodes.direct_request import DirectRequestHandler
from agent.nodes.theme_recommender import ThemeRecommender
from agent.nodes.reply import ReplyComposer
from agent.nodes.selection import select_movie
from agent.recomedation_agent import RecommendationAgent, initial_state


class LLM:
    def __init__(self, *results):
        self.results = list(results)
        self.prompts = []

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return (
            self.results.pop(0) if self.results else SimpleNamespace(content="Noted.")
        )


def make_agent(*results):
    with (
        patch("agent.nodes.direct_request.load_dotenv"),
        patch.dict("os.environ", {}, clear=True),
    ):
        return RecommendationAgent(
            llm=LLM(), classifier_llm=LLM(*results), reply_llm=LLM()
        )


class CriteriaUpgradeTests(unittest.TestCase):
    def test_correction_does_not_invert_previous_plot_constraints(self):
        llm = LLM(
            Classification(
                intent="theme_recommendation",
                search_mode="description",
                criteria_action="replace",
                entities={
                    "exclude_genres": ["Animation"],
                    "themes": ["rat in France helping a chef"],
                    "movies": [{"title": "Ratatouille", "role": "referenced"}],
                },
            )
        )
        result = Classifier(llm).classify(
            "¡No, no, no! It's the rat in France and helps a chef.",
            {"search_mode": "description"},
        )
        self.assertEqual(result.criteria_action, "add")
        self.assertEqual(result.entities.exclude_genres, [])
        self.assertEqual(result.entities.movies, [])

    def test_explicit_negative_constraints_survive_validation(self):
        llm = LLM(
            Classification(
                intent="direct_request", entities={"exclude_genres": ["Horror", "War"]}
            )
        )
        result = Classifier(llm).classify("No horror or war movies")
        self.assertEqual(result.entities.exclude_genres, ["Horror", "War"])

    def test_reference_uses_focus_without_model_guessing(self):
        llm = LLM()
        result = Classifier(llm).classify(
            "What is it about?",
            {"shortlist": ["Heat", "Drive"], "focused_movie": "Heat"},
        )
        self.assertEqual(result.entities.movies[0].title, "Heat")
        self.assertEqual(result.intent, "direct_request")
        self.assertEqual(llm.prompts, [])
        result = Classifier(llm).classify(
            "What is it about?", {"shortlist": ["Heat", "Drive"]}
        )
        self.assertTrue(result.needs_clarification)

    def test_animation_as_theme_is_recovered_without_turning_family_into_genre(self):
        c = normalize_criteria({"themes": ["animation films", "family", "grief"]})
        self.assertEqual(c["genres"], ["Animation"])
        self.assertEqual(c["themes"], ["family", "grief"])

    def test_year_bounds_do_not_become_invented_equality_filters(self):
        llm = LLM(
            Classification(
                intent="direct_request", entities={"years": [2016], "year_min": 2015}
            )
        )
        result = Classifier(llm).classify("Comedy movies released after 2015")
        self.assertEqual(result.entities.year_min, 2016)
        self.assertEqual(result.entities.years, [])
        llm = LLM(Classification(intent="direct_request", entities={"years": [2015]}))
        self.assertEqual(
            Classifier(llm).classify("A movie from 2015").entities.years, [2015]
        )

    def test_dictated_year_and_bare_runtime_become_year_and_lower_bound(self):
        llm = LLM(
            Classification(
                intent="direct_request",
                entities={"actors": ["Josh Brolin"], "minute_max": 180},
            )
        )
        result = Classifier(llm).classify(
            "I would like a movie that is from the twenty fifteen. "
            "It's three hours of duration and stars Josh Brolin"
        )
        self.assertEqual(result.entities.years, [2015])
        self.assertEqual(result.entities.minute_min, 180)
        self.assertIsNone(result.entities.minute_max)
        self.assertEqual(result.entities.actors, ["Josh Brolin"])

    def test_spoken_year_forms_convert_without_swallowing_counts(self):
        self.assertEqual(
            spoken_numbers_to_digits("a movie from nineteen ninety nine"),
            "a movie from 1999",
        )
        self.assertEqual(
            spoken_numbers_to_digits("something from two thousand ten"),
            "something from 2010",
        )
        self.assertEqual(
            spoken_numbers_to_digits("twenty twenty four comedies"),
            "2024 comedies",
        )
        self.assertEqual(
            spoken_numbers_to_digits("ninety minutes of comedy"),
            "90 minutes of comedy",
        )
        self.assertEqual(
            spoken_numbers_to_digits("I'm here with twenty five friends"),
            "I'm here with twenty five friends",
        )

    def test_explicit_runtime_ceilings_are_left_alone(self):
        for request, expected in (
            ("A comedy under two hours", ("minute_max", 119)),
            ("A comedy of at most 100 minutes", ("minute_max", 100)),
        ):
            with self.subTest(request=request):
                field, value = expected
                llm = LLM(
                    Classification(
                        intent="direct_request", entities={field: value}
                    )
                )
                entities = Classifier(llm).classify(request).entities
                self.assertEqual(getattr(entities, field), value)
                self.assertIsNone(entities.minute_min)

    def test_bare_duration_text_is_a_minimum(self):
        self.assertEqual(
            normalize_criteria({"duration": ["3 hours"]})["minute_min"], 180
        )
        self.assertIsNone(normalize_criteria({"duration": ["3 hours"]})["minute_max"])
        self.assertEqual(
            normalize_criteria({"duration": ["90 minutes long"]})["minute_min"], 90
        )

    def test_not_seen_does_not_create_seen_exclusion(self):
        llm = LLM(
            Classification(
                intent="prefer",
                entities={"movies": [{"title": "Heat", "role": "seen"}]},
            )
        )
        self.assertEqual(
            Classifier(llm).classify("I have not seen Heat").entities.movies, []
        )

    def test_word_number_runtime_and_explicit_or(self):
        self.assertEqual(
            normalize_criteria({"duration": ["under two hours"]})["minute_max"], 119
        )
        llm = LLM(
            Classification(
                intent="direct_request", entities={"genres": ["Comedy", "Romance"]}
            )
        )
        result = Classifier(llm).classify("Comedies or romance films")
        self.assertEqual(result.entities.genre_groups, [["Comedy"], ["Romance"]])
        self.assertEqual(result.entities.genres, [])
        llm = LLM(
            Classification(
                intent="direct_request", entities={"exclude_genres": ["Horror", "War"]}
            )
        )
        result = Classifier(llm).classify("No horror or war")
        self.assertEqual(result.entities.genre_groups, [])

    def test_duration_labels_match_dataset_categories(self):
        self.assertEqual(normalize_criteria({"duration": ["Short"]})["minute_max"], 99)
        standard = normalize_criteria({"duration": ["Standard"]})
        self.assertEqual((standard["minute_min"], standard["minute_max"]), (100, 129))
        self.assertEqual(normalize_criteria({"duration": ["Long"]})["minute_min"], 130)

    def test_scalar_filters_and_aliases(self):
        c = normalize_criteria(
            {
                "time_periods": ["90s"],
                "duration": ["under 2 hours"],
                "min_rating": 4,
                "streaming": ["Apple TV+", "Amazon Prime"],
                "exclude_genres": ["horror", "gore"],
            }
        )
        self.assertEqual(
            (c["year_min"], c["year_max"], c["minute_max"]), (1990, 1999, 119)
        )
        self.assertEqual(c["streaming"], ["appletv", "prime"])
        self.assertEqual(c["exclude_genres"], ["Horror"])
        self.assertNotIn("exclude_themes", c)
        self.assertTrue(has_catalog_filters({"min_rating": 4}))

    def test_or_then_and_distributes_without_weakening(self):
        c = merge_criteria(
            {"genre_groups": [["Comedy"], ["Romance"]]},
            {"genres": ["Action"], "minute_max": 90},
            "add",
        )
        self.assertEqual(
            c["genre_groups"], [["Comedy", "Action"], ["Romance", "Action"]]
        )
        self.assertEqual(c["genres"], [])
        self.assertEqual(c["minute_max"], 90)
        c = merge_criteria(c, {"minute_max": 120}, "add")
        self.assertEqual(c["minute_max"], 120)
        self.assertIsNone(
            merge_criteria(c, {"genres": ["Drama"]}, "replace")["minute_max"]
        )

    def test_seen_and_liked_roles_both_survive(self):
        c = merge_criteria(
            {"movies": [{"title": "Heat", "role": "liked"}]},
            {"movies": [{"title": "Heat", "role": "seen"}]},
            "add",
        )
        self.assertEqual(len(c["movies"]), 2)

    def test_predicates_reject_any_excluded_genre(self):
        self.assertFalse(
            matches_filters(
                {"genres": ["Horror"]}, {"exclude_genres": ["Horror", "War"]}
            )
        )
        self.assertTrue(
            matches_filters(
                {
                    "genres": ["Comedy"],
                    "minute": 89,
                    "date": 1995,
                    "rating": 4.2,
                    "streaming": ["netflix"],
                },
                {
                    "genre_groups": [["Comedy"], ["Romance"]],
                    "time_periods": ["90s"],
                    "minute_max": 90,
                    "min_rating": 4,
                    "streaming": ["netflix"],
                },
            )
        )
        self.assertFalse(matches_filters({"minute": None}, {"minute_max": 90}))

    def test_sql_negation_is_per_genre_and_supports_scalar_only(self):
        query = Mock()
        query.not_.contains.return_value = query
        query.gte.return_value = query
        query.lte.return_value = query
        apply_filters(
            query,
            {"exclude_genres": ["Horror", "War"], "year_min": 1990, "minute_max": 90},
        )
        self.assertEqual(query.not_.contains.call_count, 2)
        query.gte.assert_called_once_with("date", 1990)
        query.lte.assert_called_once_with("minute", 90)


class ConversationUpgradeTests(unittest.TestCase):
    def test_context_and_preference_do_not_search(self):
        a = make_agent(
            Classification(
                intent="prefer",
                entities={"genres": ["Animation"]},
                criteria_action="replace",
            )
        )
        a.theme_recommender = Mock()
        config = {"configurable": {"thread_id": "prefer"}}
        a.graph.invoke(initial_state("I love animation"), config)
        state = a.graph.get_state(config).values
        self.assertEqual(state["search_criteria"]["genres"], ["Animation"])
        self.assertEqual(state["result"]["kind"], "preference")
        a.theme_recommender.recommend.assert_not_called()
        prompt = a.classifier.llm.prompts[0][-1].content
        self.assertIn("shortlist", json.loads(prompt)["context"])

    def test_preference_is_consumed_by_next_search(self):
        a = make_agent(
            Classification(intent="prefer", entities={"themes": ["found family"]}),
            Classification(intent="recommendation", criteria_action="keep"),
        )
        a.theme_recommender = Mock()
        a.theme_recommender.recommend.return_value = {
            "kind": "theme_recommendation",
            "titles": ["A"],
            "error": None,
        }
        config = {"configurable": {"thread_id": "taste-search"}}
        a.graph.invoke(initial_state("I love found family stories"), config)
        a.graph.invoke(Command(resume="What should I watch?"), config)
        self.assertEqual(
            a.theme_recommender.recommend.call_args.kwargs["criteria"]["themes"],
            ["found family"],
        )

    def test_more_keeps_search_after_social_detour(self):
        a = make_agent(
            Classification(
                intent="theme_recommendation", entities={"themes": ["heist"]}
            ),
            Classification(intent="social"),
            Classification(intent="more"),
        )
        a.theme_recommender = Mock()
        a.theme_recommender.recommend.side_effect = [
            {"kind": "theme_recommendation", "titles": ["A", "B"], "error": None},
            {"kind": "theme_recommendation", "titles": ["C"], "error": None},
        ]
        config = {"configurable": {"thread_id": "more"}}
        a.graph.invoke(initial_state("heist films"), config)
        a.graph.invoke(Command(resume="thanks"), config)
        a.graph.invoke(Command(resume="what else"), config)
        call = a.theme_recommender.recommend.call_args
        self.assertEqual(call.args[0], "heist films")
        self.assertEqual(call.kwargs["exclude_names"], ["A", "B"])
        self.assertEqual(a.graph.get_state(config).values["movies"], ["C"])

    def test_selection_does_not_eat_questions_or_negations(self):
        for answer in [
            "What is Heat about?",
            "not Heat",
            "Is the second one scary?",
            "I have seen Heat",
        ]:
            self.assertIsNone(select_movie(answer, ["Drive", "Heat"]))
        self.assertEqual(select_movie("the second one", ["Drive", "Heat"]), "Heat")
        self.assertEqual(select_movie("Heat", ["Drive", "Heat"]), "Heat")
        self.assertIsNone(select_movie("9", ["Drive", "Heat"]))

    def test_standalone_exit_without_choice(self):
        a = make_agent(Classification(intent="exit"))
        self.assertEqual(a.run("bye", thread_id="bye"), "Goodbye!")
        snapshot = a.graph.get_state({"configurable": {"thread_id": "bye"}})
        self.assertFalse(snapshot.next)
        self.assertIsNone(snapshot.values["choice"])

    def test_accept_by_name_and_ambiguous_accept(self):
        a = make_agent()
        state = {
            **initial_state(),
            "request": "I'll take Heat",
            "movies": ["Heat", "Drive"],
            "entities": {"movies": [{"title": "Heat", "role": "wanted"}]},
        }
        self.assertEqual(a.accept(state).update["choice"], "Heat")
        state["entities"] = {}
        self.assertEqual(a.accept(state).goto, "reply")

    def test_explicit_named_pick_resolves_even_if_role_is_referenced(self):
        a = make_agent()
        state = {
            **initial_state("I'll watch Ratatouille"),
            "entities": {
                "movies": [
                    {"title": "Ratatouille", "role": "referenced"},
                    {"title": "Heat", "role": "liked"},
                ]
            },
        }
        with patch(
            "agent.recomedation_agent.resolve_movie",
            return_value={"name": "Ratatouille"},
        ):
            self.assertEqual(a.accept(state).update["choice"], "Ratatouille")

    def test_anaphoric_fact_retains_shortlist(self):
        a = make_agent(
            Classification(
                intent="direct_request",
                entities={"movies": [{"title": "Heat", "role": "referenced"}]},
            )
        )
        a.direct_request_handler.handle_movie_fact = Mock(
            return_value={
                "kind": "movie_fact",
                "titles": ["Heat"],
                "facts": {"name": "Heat"},
                "error": None,
            }
        )
        c = {"configurable": {"thread_id": "fact"}}
        a.graph.invoke(
            {
                **initial_state("What is it about?"),
                "movies": ["Drive", "Heat"],
                "focused_movie": "Heat",
            },
            c,
        )
        s = a.graph.get_state(c).values
        self.assertEqual(s["movies"], ["Drive", "Heat"])
        self.assertFalse(s["show_options"])
        self.assertEqual(s["focused_movie"], "Heat")

    def test_ambiguous_reference_clarifies(self):
        a = make_agent(
            Classification(intent="direct_request", needs_clarification=True)
        )
        a.direct_request_handler = Mock()
        c = {"configurable": {"thread_id": "ambiguous"}}
        a.graph.invoke(
            {**initial_state("what is it about?"), "movies": ["Drive", "Heat"]}, c
        )
        self.assertEqual(a.graph.get_state(c).values["result"]["kind"], "clarification")
        a.direct_request_handler.handle_movie_fact.assert_not_called()

    def test_seen_resolution_cached(self):
        a = make_agent()
        state = {**initial_state(), "seen_movies": [{"title": "Heat", "role": "seen"}]}
        with patch(
            "agent.recomedation_agent.resolve_movie", return_value={"name": "Heat"}
        ) as resolver:
            names, cache = a._exclusions(state)
            a._exclusions({**state, "resolved_movies": cache})
            resolver.assert_called_once()
        self.assertEqual(names, ["Heat"])


class RetrievalUpgradeTests(unittest.TestCase):
    def test_similar_search_preserves_seed_year(self):
        r = ThemeRecommender()
        r._supabase, r._embeddings = Mock(), Mock()
        with (
            patch(
                "agent.nodes.theme_recommender.resolve_movie",
                return_value={"id": 1, "name": "It", "description": "A clown"},
            ) as resolver,
            patch("agent.nodes.theme_recommender.search_descriptions", return_value=[]),
        ):
            r.recommend(
                "like It from 2017",
                criteria={"movies": [{"title": "It", "year": 2017, "role": "seed"}]},
            )
        resolver.assert_called_once_with(r._supabase, "It", 2017)

    def test_description_search_enforces_all_filters(self):
        db, embeddings = Mock(), Mock()
        db.rpc.return_value.execute.return_value.data = [
            {"movie_id": 1, "name": "Ratatouille", "similarity": 0.9},
            {"movie_id": 2, "name": "Wrong", "similarity": 0.8},
        ]
        db.table.return_value.select.return_value.in_.return_value.execute.return_value.data = [
            {"id": 1, "genres": ["Animation"], "streaming": ["netflix"], "minute": 111},
            {"id": 2, "genres": ["Animation"], "streaming": ["prime"], "minute": 120},
        ]
        rows = search_descriptions(
            db,
            embeddings,
            "rat in Paris",
            criteria={
                "genres": ["Animation"],
                "streaming": ["netflix"],
                "minute_max": 115,
            },
        )
        self.assertEqual([row["name"] for row in rows], ["Ratatouille"])

    def test_plot_search_and_confidence_gate(self):
        r = ThemeRecommender()
        r._supabase, r._embeddings = Mock(), Mock()
        for score, expected in [(0.9, ["Ratatouille"]), (0.3, [])]:
            with patch(
                "agent.nodes.theme_recommender.search_descriptions",
                return_value=[{"name": "Ratatouille", "similarity": score}],
            ):
                result = r.recommend(
                    "rat in Paris helps a chef",
                    criteria={"genres": ["Animation"], "themes": ["rat in Paris"]},
                    search_mode="description",
                )
            self.assertEqual(result["titles"], expected)

    def test_fact_failure_suggests_but_does_not_claim_identity(self):
        handler = DirectRequestHandler.__new__(DirectRequestHandler)
        handler.supabase, handler._embeddings = Mock(), Mock()
        handler.supabase.table.return_value.select.return_value.eq.return_value.execute.return_value.data = [
            {"id": 1, "name": "Ratatouille"}
        ]
        with (
            patch("agent.nodes.direct_request.resolve_movie", return_value=None),
            patch(
                "agent.nodes.direct_request.search_descriptions",
                return_value=[
                    {"movie_id": 1, "name": "Ratatouille", "similarity": 0.9}
                ],
            ),
        ):
            result = handler.handle_movie_fact(
                "rat in France", query_text="rat in France helps a chef"
            )
        self.assertTrue(result["guessed"])
        self.assertNotIn("id", result["facts"])
        self.assertIn(
            "Did you mean Ratatouille", ReplyComposer.fallback("direct_request", result)
        )

    def test_streaming_mock_disclosure_and_requested_service(self):
        reply = ReplyComposer.fallback(
            "availability",
            {
                "kind": "availability",
                "titles": ["Heat"],
                "facts": {"name": "Heat", "streaming": ["prime"]},
                "requested_services": ["netflix"],
            },
        )
        self.assertIn("not listed on Netflix", reply)
        self.assertIn("Mock", reply)

    def test_browse_samples_only_eligible_rows(self):
        r = ThemeRecommender()
        r._supabase, r._embeddings = Mock(), Mock()
        r._supabase.table.return_value.select.return_value.order.return_value.limit.return_value.execute.return_value.data = [
            {"name": str(i)} for i in range(20)
        ]
        result = r.recommend("surprise me", search_mode="browse")
        self.assertEqual(len(set(result["titles"])), 8)
        self.assertEqual(result["kind"], "browse")
        r._embeddings.embed_query.assert_not_called()


if __name__ == "__main__":
    unittest.main()
