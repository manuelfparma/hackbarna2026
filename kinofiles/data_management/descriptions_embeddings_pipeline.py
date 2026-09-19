import os

from theme_embeddings_pipeline import (
    EMBED_BATCH_SIZE,
    EMBEDDING_MODEL,
    UPSERT_BATCH_SIZE,
    _client,
    _load_env,
)
from langchain_mistralai import MistralAIEmbeddings
from supabase import Client

PAGE_SIZE = 1000


def _fetch_all(supabase: Client, table: str, columns: str) -> list[dict]:
    rows: list[dict] = []
    start = 0
    while True:
        end = start + PAGE_SIZE - 1
        chunk = (
            supabase.table(table)
            .select(columns)
            .range(start, end)
            .execute()
            .data
            or []
        )
        rows.extend(chunk)
        if len(chunk) < PAGE_SIZE:
            break
        start += PAGE_SIZE
    return rows


def embed_descriptions(
    movies_table: str = "movies",
    embeddings_table: str = "descriptions_embeddings",
) -> None:
    _load_env()

    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY is missing.")
        print("Please add it to your .env file.")
        return

    supabase = _client()
    if supabase is None:
        return

    print(f"Reading descriptions from '{movies_table}'...")
    movies = _fetch_all(supabase, movies_table, "id, description")
    with_text = [
        row
        for row in movies
        if row.get("id") is not None and (row.get("description") or "").strip()
    ]
    print(f"Found {len(with_text)} movies with a description ({len(movies)} total).")

    print(f"Checking existing rows in '{embeddings_table}'...")
    already_stored = {
        row["movie_id"]
        for row in _fetch_all(supabase, embeddings_table, "movie_id")
        if row.get("movie_id") is not None
    }
    pending = [row for row in with_text if row["id"] not in already_stored]
    print(f"{len(already_stored)} already embedded, {len(pending)} to embed.")

    if not pending:
        print("Nothing to embed.")
        return

    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    for i in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[i : i + EMBED_BATCH_SIZE]
        texts = [row["description"] for row in batch]
        print(
            f"Embedding descriptions {i + 1} to {i + len(batch)} of {len(pending)}..."
        )
        try:
            vectors = embeddings.embed_documents(texts)
        except Exception as e:
            print(f"An error occurred while calling Mistral: {e}")
            return

        records = [
            {
                "movie_id": row["id"],
                "description": row["description"],
                "embedding": vector,
            }
            for row, vector in zip(batch, vectors)
        ]
        for j in range(0, len(records), UPSERT_BATCH_SIZE):
            chunk = records[j : j + UPSERT_BATCH_SIZE]
            try:
                supabase.table(embeddings_table).upsert(
                    chunk, on_conflict="movie_id"
                ).execute()
            except Exception as e:
                print(f"An error occurred during upload: {e}")
                return

    print("Upload completed successfully!")


if __name__ == "__main__":
    embed_descriptions()
