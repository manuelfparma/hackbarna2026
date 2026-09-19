"""Poster art for a movie title, looked up in Supabase."""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables (from kinofiles root)
load_dotenv(os.path.join(os.path.dirname(__file__), "../../.env"))
url = os.environ.get("SUPABASE_URL", "https://wbocilfhiuydyjytsukj.supabase.co")
key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ.get("SUPABASE_KEY", "")

supabase = None
if url and key:
    supabase = create_client(url, key)

_cache: dict[str, str] = {}


def poster_urls(titles: list[str]) -> dict[str, str]:
    """Map each title to a poster URL, "" when Supabase has none."""
    wanted = [title for title in titles if title not in _cache]

    if wanted and supabase:
        try:
            # Query Supabase for the posters of the wanted titles
            response = supabase.table("movies").select("name, poster").in_("name", wanted).execute()
            if response.data:
                for row in response.data:
                    _cache[row["name"]] = row.get("poster", "")
        except Exception as e:
            print(f"Error fetching posters from Supabase: {e}")

    # Set default empty string for anything still missing (not in DB)
    for title in titles:
        _cache.setdefault(title, "")
        
    return {title: _cache[title] for title in titles}
