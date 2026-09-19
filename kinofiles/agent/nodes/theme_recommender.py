"""Theme-based recommendation: nearest themes via pgvector, then movies."""

from data_management.embeddings_pipeline import EMBEDDING_MODEL, _client, _load_env
from langchain_mistralai import MistralAIEmbeddings

MATCH_COUNT = 10
MOVIE_LIMIT = 5
SIMILARITY_MARGIN = 0.10


class ThemeRecommender:
    def __init__(self, llm=None):
        self.llm = llm
        self._supabase = None
        self._embeddings = None

    def _connect(self):
        if self._supabase is None:
            _load_env()
            self._supabase = _client()
            self._embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
        return self._supabase

    def recommend(self, request: str, feedback: list[str] | None = None) -> tuple[list[str], str]:
        """Return (movie titles, user-facing reply) from theme similarity search."""
        del feedback
        supabase = self._connect()
        if supabase is None:
            return [], "Could not connect to the movie database."

        vector = self._embeddings.embed_query(request)
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
        except Exception as e:
            return [], f"Theme search failed: {e}"

        if not matches:
            return [], "I could not find matching themes for that request."

        cutoff = matches[0]["similarity"] - SIMILARITY_MARGIN
        kept = [row for row in matches if row["similarity"] >= cutoff]

        # Reciprocal rank: matching the closest theme outweighs matching a
        # handful of weaker ones, otherwise broad blockbusters always win.
        weights = {row["theme"]: 1 / rank for rank, row in enumerate(kept, start=1)}

        movies = supabase.table("movies").select("name, rating, themes").execute().data or []
        scored: list[tuple[float, float, str, list[str]]] = []
        for movie in movies:
            hit = sorted(set(movie.get("themes") or []).intersection(weights))
            if hit:
                score = sum(weights[theme] for theme in hit)
                scored.append((score, movie.get("rating") or 0.0, movie["name"], hit))
        # Ties on theme overlap are common, so the better-rated film wins.
        scored.sort(key=lambda item: (-item[0], -item[1], item[2]))
        top = scored[:MOVIE_LIMIT]
        titles = [name for _, _, name, _ in top]

        lines = ["Closest themes:"]
        lines += [f"- {row['theme']} ({row['similarity']:.3f})" for row in kept]
        lines.append("")
        lines.append("Movies with the most matching themes:")
        for score, rating, name, hit in top:
            lines.append(f"- {name} [{rating:.1f}] ({score:.2f}: {', '.join(hit)})")
        return titles, "\n".join(lines)
