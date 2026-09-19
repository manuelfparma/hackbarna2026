"""Theme-based recommendation: nearest themes via pgvector, then movies."""

from data_management.theme_embeddings_pipeline import EMBEDDING_MODEL, _client, _load_env
from langchain_mistralai import MistralAIEmbeddings

from agent.nodes.catalog import resolve_movie
from agent.nodes.criteria import build_theme_query, normalize_criteria

MATCH_COUNT = 10
MOVIE_LIMIT = 5
SIMILARITY_MARGIN = 0.10
# Similar-to candidates are post-filtered by genre, so ask for extra rows
# to have something left after the filter bites.
SIMILAR_MATCH_COUNT = 25


class ThemeRecommender:
    def __init__(self):
        self._supabase = None
        self._embeddings = None

    def _connect(self):
        if self._supabase is None:
            _load_env()
            self._supabase = _client()
            self._embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
        return self._supabase

    def recommend(
        self,
        request: str,
        feedback: list[str] | None = None,
        criteria: dict | None = None,
    ) -> dict:
        """Return ranked titles and human-readable evidence, not presentation copy."""
        normalized_criteria = normalize_criteria(criteria)
        genres = normalized_criteria["genres"]
        supabase = self._connect()
        if supabase is None:
            return {
                "kind": "theme_recommendation",
                "titles": [],
                "themes": [],
                "error": "I couldn't connect to the movie catalog.",
            }

        seeds = [
            movie["title"]
            for movie in normalized_criteria["movies"]
            if movie.get("role") in {"seed", "liked"} and movie.get("title")
        ]
        if seeds:
            similar = self._similar_to(
                supabase, seeds[0], request, feedback, normalized_criteria, genres
            )
            if similar is not None:
                return similar
            # Seed didn't resolve; fall through so its title still reaches
            # the theme query below.

        query_text = build_theme_query(request, normalized_criteria)
        notes = [item for item in (feedback or []) if item and item != request]
        if notes:
            query_text = f"{query_text}; {'; '.join(notes)}"
        vector = self._embeddings.embed_query(query_text)
        try:
            matches = (
                supabase.rpc(
                    "match_themes",
                    {"query_embedding": vector, "match_count": MATCH_COUNT},
                )
                .execute()
                .data
                or []
            )
        except Exception:
            return {
                "kind": "theme_recommendation",
                "titles": [],
                "themes": [],
                "error": "I couldn't search by theme right now.",
            }

        if not matches:
            return {
                "kind": "theme_recommendation",
                "titles": [],
                "themes": [],
                "error": "I couldn't find a matching theme.",
            }

        cutoff = matches[0]["similarity"] - SIMILARITY_MARGIN
        kept = [row for row in matches if row["similarity"] >= cutoff]

        # Reciprocal rank: matching the closest theme outweighs matching a
        # handful of weaker ones, otherwise broad blockbusters always win.
        weights = {row["theme"]: 1 / rank for rank, row in enumerate(kept, start=1)}

        movie_query = supabase.table("movies").select("name, rating, themes, description")
        if genres:
            movie_query = movie_query.contains("genres", genres)
        try:
            movies = movie_query.execute().data or []
        except Exception:
            return {
                "kind": "theme_recommendation",
                "titles": [],
                "themes": [row["theme"] for row in kept],
                "genres": genres,
                "error": "I couldn't load the movie catalog right now.",
            }
        if not movies:
            label = " + ".join(genres) if genres else "those constraints"
            return {
                "kind": "theme_recommendation",
                "titles": [],
                "themes": [row["theme"] for row in kept],
                "genres": genres,
                "error": f"I couldn't find {label} movies matching those themes.",
            }
        scored: list[tuple[float, float, str, list[str], str]] = []
        for movie in movies:
            hit = sorted(set(movie.get("themes") or []).intersection(weights))
            if hit:
                score = sum(weights[theme] for theme in hit)
                scored.append((score, movie.get("rating") or 0.0, movie["name"], hit, movie.get("description", "")))
        # Ties on theme overlap are common, so the better-rated film wins.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        top = scored[:MOVIE_LIMIT]
        titles = [name for _, _, name, _, _ in top]
        if not titles:
            label = " + ".join(genres) if genres else "those constraints"
            return {
                "kind": "theme_recommendation",
                "titles": [],
                "themes": [row["theme"] for row in kept],
                "genres": genres,
                "error": f"I couldn't find {label} movies matching those themes.",
            }
        return {
            "kind": "theme_recommendation",
            "titles": titles,
            "themes": [row["theme"] for row in kept],
            "genres": genres,
            "matches": {name: hit for _, _, name, hit, _ in top},
            "descriptions": {name: desc for _, _, name, _, desc in top},
            "error": None,
        }

    def _similar_to(
        self,
        supabase,
        seed_title: str,
        request: str,
        feedback: list[str] | None,
        criteria: dict,
        genres: list[str],
    ) -> dict | None:
        """Nearest description-embedding neighbors of a resolved seed film.

        Returns None when the seed can't be resolved or searched, so the
        caller can fall back to theme search with the title still in the
        query. The query vector embeds the seed's own description plus any
        refinement language ("...but darker") the turn carried.
        """
        try:
            seed = resolve_movie(supabase, seed_title)
        except Exception:
            return None
        if not seed or not (seed.get("description") or "").strip():
            return None

        parts = [seed["description"].strip()]
        parts.extend(criteria["themes"])
        parts.extend(criteria["audience"])
        parts.extend(criteria["time_periods"])
        parts.extend(item for item in (feedback or []) if item and item != request)
        query_text = "; ".join(dict.fromkeys(part for part in parts if part))

        vector = self._embeddings.embed_query(query_text)
        try:
            matches = (
                supabase.rpc(
                    "match_descriptions",
                    {
                        "query_embedding": vector,
                        "match_count": SIMILAR_MATCH_COUNT,
                    },
                )
                .execute()
                .data
                or []
            )
        except Exception:
            return None

        candidates = [row for row in matches if row["movie_id"] != seed["id"]]
        if genres and candidates:
            ids = [row["movie_id"] for row in candidates]
            try:
                rows = (
                    supabase.table("movies")
                    .select("id, genres")
                    .in_("id", ids)
                    .execute()
                    .data
                    or []
                )
            except Exception:
                return None
            wanted = {genre.casefold() for genre in genres}
            allowed = {
                row["id"]
                for row in rows
                if wanted <= {g.casefold() for g in (row.get("genres") or [])}
            }
            candidates = [row for row in candidates if row["movie_id"] in allowed]

        top = candidates[:MOVIE_LIMIT]
        titles = [row["name"] for row in top]
        
        descriptions = {}
        if top:
            ids_to_fetch = [row["movie_id"] for row in top]
            try:
                rows = (
                    supabase.table("movies")
                    .select("name, description")
                    .in_("id", ids_to_fetch)
                    .execute()
                    .data
                    or []
                )
                descriptions = {r["name"]: r.get("description", "") for r in rows}
            except Exception:
                pass
                
        if not titles:
            label = " + ".join(genres) if genres else "those constraints"
            return {
                "kind": "similar_to",
                "titles": [],
                "seed": seed["name"],
                "genres": genres,
                "error": f"I couldn't find {label} movies close to {seed['name']}.",
            }
        return {
            "kind": "similar_to",
            "titles": titles,
            "seed": seed["name"],
            "genres": genres,
            "descriptions": descriptions,
            "error": None,
        }
