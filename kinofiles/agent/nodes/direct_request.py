"""Direct request subagent: queries Supabase.

Searches for movies by a specific attribute (e.g. director, genre)
using the entity extracted by the Classifier.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

from agent.nodes.catalog import clean_title, resolve_movie
from agent.nodes.criteria import CATALOG_ARRAY_FIELDS, normalize_criteria


class DirectRequestHandler:
    def __init__(self, llm=None):
        self.llm = llm  # Kept for signature compatibility if needed
        load_dotenv()
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_KEY")
        if url and key:
            self.supabase: Client = create_client(url, key)
        else:
            self.supabase = None

    def handle(
        self,
        column: str | None,
        value: str | None,
        criteria: dict | None = None,
    ) -> dict:
        """Return catalog matches as data for the reply layer."""
        normalized = normalize_criteria(criteria)
        filters = {
            field: normalized[field]
            for field in CATALOG_ARRAY_FIELDS
            if normalized[field]
        }
        title = None
        if column == "name" and value:
            title = clean_title(value)
        elif column and value and column in CATALOG_ARRAY_FIELDS and column not in filters:
            filters[column] = [value]
        result = {
            "kind": "direct_request",
            "titles": [],
            "filters": {**filters, **({"name": [value]} if title else {})},
            "error": None,
        }
        if not filters and not title:
            result["error"] = (
                "I couldn't identify the actor, director, genre, studio, language, or title."
            )
            return result

        if not self.supabase:
            result["error"] = "The movie catalog isn't available right now."
            return result

        try:
            query = self.supabase.table("movies").select("name")
            if title:
                # name is citext + pg_trgm: fuzzy title search.
                query = query.ilike("name", f"%{title}%")
            for filter_column, values in filters.items():
                if filter_column == "genres":
                    # Canonical labels: strict array containment keeps the
                    # multi-genre AND intact.
                    query = query.contains("genres", values)
                else:
                    # Free-form names use the generated *_text columns so
                    # partial mentions still match ("Nolan" -> "Christopher
                    # Nolan"); one ilike per value keeps it an AND.
                    for item in values:
                        query = query.ilike(
                            f"{filter_column}_text", f"%{clean_title(item)}%"
                        )
            res = query.order("rating", desc=True).limit(5).execute()
            movies = res.data

            if not movies:
                labels = ", ".join(
                    str(item) for values in result["filters"].values() for item in values
                )
                result["error"] = f"I couldn't find movies matching {labels}."
                return result

            result["titles"] = [movie["name"] for movie in movies]
            return result
        except Exception:
            result["error"] = "I couldn't search the movie catalog right now."
            return result

    def handle_movie_fact(self, title: str, year: int | None = None) -> dict:
        """Fetch a named film's catalog row for the reply layer to narrate."""
        result = {
            "kind": "movie_fact",
            "titles": [],
            "facts": None,
            "error": None,
        }
        if not self.supabase:
            result["error"] = "The movie catalog isn't available right now."
            return result
        if not (title or "").strip():
            result["error"] = "I couldn't tell which movie you meant."
            return result
        try:
            movie = resolve_movie(self.supabase, title, year)
        except Exception:
            result["error"] = "I couldn't search the movie catalog right now."
            return result
        if not movie:
            result["error"] = f"I couldn't find '{title}' in the catalog."
            return result
        # Same contract as `handle`: titles only, no database ids upstream.
        movie.pop("id", None)
        result["titles"] = [movie["name"]]
        result["facts"] = movie
        return result
