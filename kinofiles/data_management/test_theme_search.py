"""Nearest-theme search against public.themes_embeddings (HNSW cosine).

Usage (from kinofiles):

  uv run python data_management/test_theme_search.py "slow melancholy family drama"
"""

import argparse

from embeddings_pipeline import EMBEDDING_MODEL, _client, _load_env
from langchain_mistralai import MistralAIEmbeddings

MATCH_COUNT = 10
MOVIE_LIMIT = 5


def main() -> None:
    parser = argparse.ArgumentParser(description="Nearest-theme search against public.themes_embeddings (HNSW cosine).")
    parser.add_argument("query", nargs="+", help="Search query text")
    args = parser.parse_args()
    query = " ".join(args.query).strip()
    _load_env()

    supabase = _client()
    if supabase is None:
        return

    print(f"Query: {query}\n")
    vector = MistralAIEmbeddings(model=EMBEDDING_MODEL).embed_query(query)

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
        print("RPC match_themes failed. Create the SQL function in the file docstring.")
        print(e)
        return

    if not matches:
        print("No matches.")
        return

    print(f"Top {len(matches)} themes:")
    for i, row in enumerate(matches, start=1):
        print(f"  {i:2}. {row['similarity']:.3f}  {row['theme']}")

    theme_names = [row["theme"] for row in matches]
    movies = supabase.table("movies").select("name, themes").execute().data or []
    scored = []
    for movie in movies:
        movie_themes = set(movie.get("themes") or [])
        hit = movie_themes.intersection(theme_names)
        if hit:
            scored.append((len(hit), movie["name"], sorted(hit)))
    scored.sort(key=lambda item: (-item[0], item[1]))

    print(f"\nMovies with the most matching themes (top {MOVIE_LIMIT}):")
    for count, name, hit in scored[:MOVIE_LIMIT]:
        print(f"  {count}  {name}")
        print(f"      {', '.join(hit)}")


if __name__ == "__main__":
    main()
