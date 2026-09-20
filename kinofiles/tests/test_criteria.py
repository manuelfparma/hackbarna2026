import unittest

from agent.nodes.criteria import (
    build_theme_query,
    merge_criteria,
    normalize_criteria,
    normalize_genres,
    split_genres,
)


class CriteriaTests(unittest.TestCase):
    def test_add_keeps_prior_genre_and_deduplicates_case_insensitively(self):
        current = {"genres": ["Comedy"]}
        turn = {"genres": ["comedy", "Action"]}

        merged = merge_criteria(current, turn, "add")

        self.assertEqual(merged["genres"], ["Comedy", "Action"])

    def test_replace_discards_prior_direction(self):
        merged = merge_criteria(
            {"genres": ["Comedy"], "themes": ["found family"]},
            {"genres": ["Thriller"]},
            "replace",
        )

        self.assertEqual(merged["genres"], ["Thriller"])
        self.assertEqual(merged["themes"], [])

    def test_reset_clears_old_criteria_and_keeps_new_turn(self):
        merged = merge_criteria(
            {"genres": ["Comedy"]},
            {"themes": ["something completely different"]},
            "reset",
        )

        self.assertEqual(merged["genres"], [])
        self.assertEqual(merged["themes"], ["something completely different"])

    def test_keep_ignores_social_entities(self):
        merged = merge_criteria(
            {"genres": ["Comedy"]},
            {"themes": ["thanks"]},
            "keep",
        )

        self.assertEqual(merged["genres"], ["Comedy"])
        self.assertEqual(merged["themes"], [])

    def test_genre_aliases_use_catalog_labels(self):
        self.assertEqual(
            normalize_genres(["sci-fi", "action movies", "COMEDY"]),
            ["Science Fiction", "Action", "Comedy"],
        )

    def test_non_catalog_genres_demote_to_themes(self):
        canonical, demoted = split_genres(["Heist", "comedy", "space"])

        self.assertEqual(canonical, ["Comedy"])
        self.assertEqual(demoted, ["Heist", "space"])

    def test_normalize_criteria_moves_bogus_genres_into_themes(self):
        normalized = normalize_criteria({"genres": ["Heist", "Crime"]})

        self.assertEqual(normalized["genres"], ["Crime"])
        self.assertEqual(normalized["themes"], ["Heist"])

    def test_theme_query_combines_latest_request_with_remembered_moods(self):
        query = build_theme_query(
            "also make it exciting",
            {
                "genres": ["Comedy"],
                "themes": ["found family"],
                "audience": ["friends"],
            },
        )

        self.assertIn("also make it exciting", query)
        self.assertIn("found family", query)
        self.assertNotIn("friends", query)
        self.assertNotIn("Comedy", query)


if __name__ == "__main__":
    unittest.main()
