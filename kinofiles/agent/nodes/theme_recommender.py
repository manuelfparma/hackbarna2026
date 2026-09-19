"""Theme-based recommendation: nearest themes via pgvector, then movies."""

from data_management.embeddings_pipeline import EMBEDDING_MODEL, _client, _load_env
from langchain_mistralai import MistralAIEmbeddings

MATCH_COUNT = 10
MOVIE_LIMIT = 5


class ThemeRecommender:
    def __init__(self, llm=None):
        self.llm = llm

    def recommend(self, request: str, feedback: list[str] | None = None) -> tuple[list[str], str]:
        """Return (movie titles, user-facing reply) from theme similarity search."""
        del feedback
        _load_env()
        supabase = _client()
        if supabase is None:
            return [], "Could not connect to the movie database."

        vector = MistralAIEmbeddings(model=EMBEDDING_MODEL).embed_query(request)
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

        theme_names = [row["theme"] for row in matches]
        movies = supabase.table("movies").select("name, themes").execute().data or []
        scored: list[tuple[int, str, list[str]]] = []
        for movie in movies:
            hit = set(movie.get("themes") or []).intersection(theme_names)
            if hit:
                scored.append((len(hit), movie["name"], sorted(hit)))
        scored.sort(key=lambda item: (-item[0], item[1]))
        top = scored[:MOVIE_LIMIT]
        titles = [name for _, name, _ in top]

        lines = ["Closest themes:"]
        for row in matches:
            lines.append(f"- {row['theme']}")
        lines.append("")
        lines.append("Movies with the most matching themes:")
        for count, name, hit in top:
            lines.append(f"- {name} ({count}: {', '.join(hit)})")
        return titles, "\n".join(lines)
