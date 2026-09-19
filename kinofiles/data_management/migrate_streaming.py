import os
import random

from dotenv import load_dotenv
from supabase import Client, create_client

SERVICES = ("netflix", "hbo", "prime", "appletv")
PAGE_SIZE = 1000


def assign_streaming(movie_id: int, rating: float | None) -> list[str]:
    """Deterministic mock services, weighted by rating.

    Seeded on the movie id, so re-runs produce identical data. Top-rated
    films land on 2-3 services, tail films on 0-1.
    """
    rng = random.Random(movie_id)
    r = rating or 0.0
    if r >= 3.7:
        count = rng.choice((2, 3))
    elif r >= 3.0:
        count = rng.choice((1, 2))
    else:
        count = 1
    return list(rng.sample(SERVICES, count))


def migrate_streaming_db(table_name="movies"):
    load_dotenv()

    url = os.environ.get("SUPABASE_URL")
    key = os.environ.get("SUPABASE_KEY")

    if not url or not key or key == "your-api-key-here":
        print("Error: SUPABASE_URL or SUPABASE_KEY is missing or invalid.")
        print("Please update your .env file with the correct credentials.")
        return

    print(f"Connecting to Supabase at {url}...")
    supabase: Client = create_client(url, key)

    print("\n" + "=" * 60)
    print("IMPORTANT: Ensure you have added the 'streaming' column!")
    print("If you haven't, run this in the Supabase SQL Editor:")
    print("ALTER TABLE movies ADD COLUMN IF NOT EXISTS streaming TEXT[] DEFAULT '{}';")
    print("=" * 60 + "\n")

    print("Fetching movie ids and ratings...")
    rows = []
    offset = 0
    while True:
        page = (
            supabase.table(table_name).select("id, rating").range(offset, offset + PAGE_SIZE - 1).execute().data or []
        )
        rows.extend(page)
        if len(page) < PAGE_SIZE:
            break
        offset += PAGE_SIZE

    print(f"Found {len(rows)} movies. Assigning mock streaming services...")
    print("Performing explicit single-column updates to guarantee no other columns are affected.")

    success_count = 0
    error_count = 0

    for row in rows:
        streaming = assign_streaming(row["id"], row.get("rating"))
        try:
            supabase.table(table_name).update({"streaming": streaming}).eq("id", row["id"]).execute()
            success_count += 1
            if success_count % 100 == 0:
                print(f"Updated {success_count}/{len(rows)} movies...")
        except Exception as e:
            print(f"Error updating movie ID {row['id']}: {e}")
            error_count += 1
            if error_count == 1:
                print("Hint: Did you forget to add the 'streaming' column in Supabase?")
                break

    print(f"\nMigration completed! Successfully updated {success_count} rows. Errors: {error_count}")


if __name__ == "__main__":
    migrate_streaming_db()
