import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_mistralai import MistralAIEmbeddings
from supabase import Client, create_client

logger = logging.getLogger(__name__)

EMBED_BATCH_SIZE = 50
UPSERT_BATCH_SIZE = 100
EMBEDDING_MODEL = "mistral-embed"


def _load_env():
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _client() -> Client | None:
    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")
    if not url or not key or key == "your-api-key-here":
        print("Error: SUPABASE_URL or SUPABASE_KEY is missing or invalid.")
        print("Please update your .env file with the correct credentials.")
        return None
    logger.debug("Connecting to Supabase at %s", url)
    return create_client(url, key)


def _unique_themes_from_movies(supabase: Client, movies_table: str) -> list[str]:
    rows = supabase.table(movies_table).select("themes").execute().data or []
    unique: set[str] = set()
    for row in rows:
        for theme in row.get("themes") or []:
            if theme:
                unique.add(theme)
    return sorted(unique)


def embed_themes(
    movies_table: str = "movies",
    embeddings_table: str = "themes_embeddings",
) -> None:
    _load_env()

    if not os.environ.get("MISTRAL_API_KEY"):
        print("Error: MISTRAL_API_KEY is missing.")
        print("Please add it to your .env file.")
        return

    supabase = _client()
    if supabase is None:
        return

    print(f"Reading unique themes from '{movies_table}.themes'...")
    unique_themes = _unique_themes_from_movies(supabase, movies_table)
    print(f"Found {len(unique_themes)} unique themes.")

    print(f"Checking existing rows in '{embeddings_table}'...")
    already_stored = {
        row["theme"]
        for row in (supabase.table(embeddings_table).select("theme").execute().data or [])
        if row.get("theme")
    }
    pending = [theme for theme in unique_themes if theme not in already_stored]
    print(f"{len(already_stored)} already embedded, {len(pending)} to embed.")

    if not pending:
        print("Nothing to embed.")
        return

    embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)

    for i in range(0, len(pending), EMBED_BATCH_SIZE):
        batch = pending[i : i + EMBED_BATCH_SIZE]
        print(f"Embedding themes {i + 1} to {i + len(batch)} of {len(pending)}...")
        try:
            vectors = embeddings.embed_documents(batch)
        except Exception as e:
            print(f"An error occurred while calling Mistral: {e}")
            return

        records = [{"theme": theme, "embedding": vector} for theme, vector in zip(batch, vectors)]
        for j in range(0, len(records), UPSERT_BATCH_SIZE):
            chunk = records[j : j + UPSERT_BATCH_SIZE]
            try:
                supabase.table(embeddings_table).upsert(chunk, on_conflict="theme").execute()
            except Exception as e:
                print(f"An error occurred during upload: {e}")
                return

    print("Upload completed successfully!")


if __name__ == "__main__":
    embed_themes()
