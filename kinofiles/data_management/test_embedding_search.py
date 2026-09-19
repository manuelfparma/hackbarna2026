"""Nearest-neighbor search against theme and description embeddings (HNSW cosine).

Usage (from kinofiles):

  uv run python data_management/test_embedding_search.py "slow melancholy family drama"
  uv run python data_management/test_embedding_search.py --kind theme "fast cars"
  uv run python data_management/test_embedding_search.py --kind description "heist in paris"
"""

import argparse

from langchain_mistralai import MistralAIEmbeddings
from supabase import Client
from theme_embeddings_pipeline import EMBEDDING_MODEL, _client, _load_env

MATCH_COUNT = 10
MOVIE_LIMIT = 5
SNIPPET_LEN = 160


def _rpc(supabase: Client, fn: str, vector: list[float]) -> list[dict] | None:
    try:
        return (
            supabase.rpc(
                fn,
                {"query_embedding": vector, "match_count": MATCH_COUNT},
            )
            .execute()
            .data
            or []
        )
    except Exception as e:
        print(f"RPC {fn} failed. Run the SQL in create_schema.sql.")
        print(e)
        return None


def _print_theme_matches(supabase: Client, matches: list[dict]) -> None:
    print(f"Top {len(matches)} themes:")
    for i, row in enumerate(matches, start=1):
        print(f"  {i:2}. {row['similarity']:.3f}  {row['theme']}")

    theme_names = [row["theme"] for row in matches]
    movies = supabase.table("movies").select("name, themes").execute().data or []
    scored = []
    for movie in movies:
        hit = set(movie.get("themes") or []).intersection(theme_names)
        if hit:
            scored.append((len(hit), movie["name"], sorted(hit)))
    scored.sort(key=lambda item: (-item[0], item[1]))

    print(f"\nMovies with the most matching themes (top {MOVIE_LIMIT}):")
    for count, name, hit in scored[:MOVIE_LIMIT]:
        print(f"  {count}  {name}")
        print(f"      {', '.join(hit)}")


def _print_description_matches(matches: list[dict]) -> None:
    print(f"Top {len(matches)} movies:")
    for i, row in enumerate(matches, start=1):
        snippet = (row.get("description") or "").replace("\n", " ").strip()
        if len(snippet) > SNIPPET_LEN:
            snippet = snippet[: SNIPPET_LEN - 3] + "..."
        print(f"  {i:2}. {row['similarity']:.3f}  {row['name']} (id={row['movie_id']})")
        print(f"      {snippet}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Nearest-neighbor search against theme and/or description embeddings."
    )
    parser.add_argument("query", nargs="+", help="Search query text")
    parser.add_argument(
        "--kind",
        choices=("both", "theme", "description"),
        default="both",
        help="Which embedding index to search (default: both)",
    )
    args = parser.parse_args()
    query = " ".join(args.query).strip()
    _load_env()

    supabase = _client()
    if supabase is None:
        return

    print(f"Query: {query}\n")
    vector = MistralAIEmbeddings(model=EMBEDDING_MODEL).embed_query(query)

    run_theme = args.kind in ("both", "theme")
    run_description = args.kind in ("both", "description")

    if run_theme:
        if args.kind == "both":
            print("=== Themes ===")
        matches = _rpc(supabase, "match_themes", vector)
        if matches is None:
            return
        if not matches:
            print("No theme matches.")
        else:
            _print_theme_matches(supabase, matches)

    if run_description:
        if args.kind == "both":
            print("\n=== Descriptions ===")
        matches = _rpc(supabase, "match_descriptions", vector)
        if matches is None:
            return
        if not matches:
            print("No description matches.")
        else:
            _print_description_matches(matches)


if __name__ == "__main__":
    main()
