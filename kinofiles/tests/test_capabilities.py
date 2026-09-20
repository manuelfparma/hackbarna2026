import unittest

from agent.nodes.direct_request import DirectRequestHandler
from agent.nodes.theme_recommender import ThemeRecommender


class Result:
    def __init__(self, data):
        self.data = data


class Query:
    def __init__(self, data, contains_calls=None, order_calls=None):
        self.data = data
        self.contains_calls = contains_calls if contains_calls is not None else []
        self.order_calls = order_calls if order_calls is not None else []

    def select(self, *_args):
        return self

    def contains(self, column, values):
        self.contains_calls.append((column, values))
        return self

    def ilike(self, *_args):
        return self

    def in_(self, *_args):
        return self

    def eq(self, *_args):
        return self

    def order(self, *args, **kwargs):
        self.order_calls.append((args, kwargs))
        return self

    def limit(self, *_args):
        return self

    def execute(self):
        return Result(self.data)


class FakeSupabase:
    def __init__(self, themes, movies):
        self.themes = themes
        self.movies = movies
        self.contains_calls = []
        self.order_calls = []

    def rpc(self, *_args, **_kwargs):
        return Query(self.themes)

    def table(self, *_args):
        return Query(self.movies, self.contains_calls, self.order_calls)


class FakeEmbeddings:
    def __init__(self):
        self.queries = []

    def embed_query(self, request):
        self.queries.append(request)
        return [0.1, 0.2]


class CapabilityTests(unittest.TestCase):
    def test_theme_result_contains_evidence_but_not_scores(self):
        recommender = ThemeRecommender()
        recommender._supabase = FakeSupabase(
            themes=[
                {"theme": "Fast cars", "similarity": 0.8},
                {"theme": "Heists", "similarity": 0.75},
            ],
            movies=[
                {
                    "name": "Drive",
                    "rating": 8.0,
                    "themes": ["Fast cars", "Heists"],
                    "description": "A getaway driver faces a dangerous job.",
                }
            ],
        )
        recommender._embeddings = FakeEmbeddings()

        result = recommender.recommend("cars and crime")

        self.assertEqual(result["titles"], ["Drive"])
        self.assertEqual(
            result["descriptions"], {"Drive": "A getaway driver faces a dangerous job."}
        )
        self.assertEqual(result["themes"], ["Fast cars", "Heists"])
        self.assertNotIn("similarity", result)
        self.assertNotIn("scores", result)

    def test_direct_request_returns_titles_without_catalog_ids(self):
        handler = DirectRequestHandler.__new__(DirectRequestHandler)
        handler.supabase = FakeSupabase(
            [],
            [{"id": 42, "name": "Arrival", "description": "Visitors arrive on Earth."}],
        )

        result = handler.handle("directors", "Denis Villeneuve")

        self.assertEqual(result["titles"], ["Arrival"])
        self.assertEqual(
            result["descriptions"], {"Arrival": "Visitors arrive on Earth."}
        )
        self.assertNotIn("id", result)
        self.assertNotIn("42", str(result))

    def test_direct_request_requires_all_accumulated_genres(self):
        handler = DirectRequestHandler.__new__(DirectRequestHandler)
        supabase = FakeSupabase([], [{"name": "The Nice Guys"}])
        handler.supabase = supabase

        result = handler.handle(
            "genres",
            "Action",
            criteria={"genres": ["comedy", "Action"]},
        )

        self.assertEqual(result["titles"], ["The Nice Guys"])
        self.assertEqual(
            supabase.contains_calls,
            [("genres", ["Comedy", "Action"])],
        )

    def test_theme_search_applies_genres_and_composes_remembered_themes(self):
        recommender = ThemeRecommender()
        supabase = FakeSupabase(
            themes=[{"theme": "Found family", "similarity": 0.8}],
            movies=[
                {
                    "name": "Guardians of the Galaxy",
                    "rating": 8.0,
                    "themes": ["Found family"],
                }
            ],
        )
        embeddings = FakeEmbeddings()
        recommender._supabase = supabase
        recommender._embeddings = embeddings

        result = recommender.recommend(
            "also make it funny",
            criteria={"genres": ["Action", "Comedy"], "themes": ["found family"]},
        )

        self.assertEqual(result["titles"], ["Guardians of the Galaxy"])
        self.assertEqual(
            supabase.contains_calls,
            [("genres", ["Action", "Comedy"])],
        )
        self.assertIn("found family", embeddings.queries[0])

    def test_theme_search_includes_prior_feedback_in_the_query(self):
        recommender = ThemeRecommender()
        embeddings = FakeEmbeddings()
        recommender._supabase = FakeSupabase(
            themes=[{"theme": "Slow-burn crime", "similarity": 0.8}],
            movies=[
                {
                    "name": "Heat",
                    "rating": 8.0,
                    "themes": ["Slow-burn crime"],
                }
            ],
        )
        recommender._embeddings = embeddings

        result = recommender.recommend(
            "too loud",
            feedback=["too much CGI", "too loud"],
            criteria={"themes": ["heist"]},
        )

        self.assertEqual(result["titles"], ["Heat"])
        self.assertIn("too loud", embeddings.queries[0])
        self.assertIn("too much CGI", embeddings.queries[0])
        self.assertEqual(embeddings.queries[0].count("too loud"), 1)

    def test_direct_request_orders_by_rating(self):
        handler = DirectRequestHandler.__new__(DirectRequestHandler)
        supabase = FakeSupabase([], [{"name": "Heat"}])
        handler.supabase = supabase

        handler.handle("genres", "Crime")

        self.assertEqual(supabase.order_calls, [(("rating",), {"desc": True})])

    def test_movie_fact_returns_row_without_id(self):
        handler = DirectRequestHandler.__new__(DirectRequestHandler)
        handler.supabase = FakeSupabase(
            [],
            [
                {
                    "id": 42,
                    "name": "Heat",
                    "date": 1995,
                    "directors": ["Michael Mann"],
                    "rating": 8.3,
                    "description": "Cop and robber.",
                }
            ],
        )

        result = handler.handle_movie_fact("heat")

        self.assertEqual(result["titles"], ["Heat"])
        self.assertEqual(result["facts"]["directors"], ["Michael Mann"])
        self.assertNotIn("id", result["facts"])
        self.assertNotIn("42", str(result))

    def test_movie_fact_unknown_title_reports_error(self):
        handler = DirectRequestHandler.__new__(DirectRequestHandler)
        handler.supabase = FakeSupabase([], [])

        result = handler.handle_movie_fact("Nonexistent Film")

        self.assertEqual(result["titles"], [])
        self.assertIn("Nonexistent Film", result["error"])


class DescQuery(Query):
    """Query whose execute() consumes a queue of result sets."""

    def __init__(self, queue):
        super().__init__(None)
        self.queue = queue

    def execute(self):
        return Result(self.queue.pop(0) if self.queue else [])


class DescFakeSupabase:
    """Serves queued table reads and a fixed match_descriptions response."""

    def __init__(self, table_results, desc_matches):
        self.table_results = table_results
        self.desc_matches = desc_matches
        self.rpc_calls = []

    def rpc(self, fn, _params):
        self.rpc_calls.append(fn)
        return Query(self.desc_matches)

    def table(self, *_args):
        return DescQuery(self.table_results)


class SimilarToTests(unittest.TestCase):
    SEED = {
        "id": 1,
        "name": "Super Troopers",
        "description": "Prank-loving highway cops.",
        "rating": 7.0,
        "genres": ["Comedy"],
        "themes": [],
    }

    def _recommender(self, supabase):
        recommender = ThemeRecommender()
        recommender._supabase = supabase
        recommender._embeddings = FakeEmbeddings()
        return recommender

    def test_seed_movie_searches_description_neighbors(self):
        supabase = DescFakeSupabase(
            table_results=[[self.SEED]],
            desc_matches=[
                {"movie_id": 1, "name": "Super Troopers", "similarity": 1.0},
                {"movie_id": 2, "name": "Beerfest", "similarity": 0.9},
                {"movie_id": 3, "name": "Hot Fuzz", "similarity": 0.8},
            ],
        )

        result = self._recommender(supabase).recommend(
            "a movie like Super Troopers",
            criteria={"movies": [{"title": "Super Troopers", "role": "seed"}]},
        )

        self.assertEqual(result["kind"], "similar_to")
        self.assertEqual(result["titles"], ["Beerfest", "Hot Fuzz"])
        self.assertEqual(result["seed"], "Super Troopers")
        self.assertEqual(supabase.rpc_calls, ["match_descriptions"])

    def test_similar_to_applies_genre_filter(self):
        supabase = DescFakeSupabase(
            table_results=[
                [self.SEED],
                [
                    {"id": 2, "genres": ["Comedy"]},
                    {"id": 3, "genres": ["Horror"]},
                ],
            ],
            desc_matches=[
                {"movie_id": 2, "name": "Beerfest", "similarity": 0.9},
                {"movie_id": 3, "name": "Hereditary", "similarity": 0.8},
            ],
        )

        result = self._recommender(supabase).recommend(
            "a comedy like Super Troopers",
            criteria={
                "movies": [{"title": "Super Troopers", "role": "seed"}],
                "genres": ["Comedy"],
            },
        )

        self.assertEqual(result["titles"], ["Beerfest"])

    def test_unresolved_seed_falls_back_to_theme_search(self):
        supabase = DescFakeSupabase(table_results=[[]], desc_matches=[])
        themes = [{"theme": "Cops", "similarity": 0.8}]
        movies = [{"name": "Hot Fuzz", "rating": 8.0, "themes": ["Cops"]}]

        def dispatch(fn, params):
            if fn == "match_themes":
                return Query(themes)
            return Query([])

        calls = {"seed_lookup": False}

        def table(_name):
            if not calls["seed_lookup"]:
                calls["seed_lookup"] = True
                return DescQuery([[]])
            return Query(movies)

        supabase.rpc = dispatch
        supabase.table = table

        result = self._recommender(supabase).recommend(
            "something like an unknown title",
            criteria={"movies": [{"title": "Unknown Title", "role": "seed"}]},
        )

        self.assertEqual(result["kind"], "theme_recommendation")
        self.assertEqual(result["titles"], ["Hot Fuzz"])


if __name__ == "__main__":
    unittest.main()
