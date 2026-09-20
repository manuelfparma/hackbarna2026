"""Direct request subagent: queries Supabase.

Searches for movies by a specific attribute (e.g. director, genre)
using the entity extracted by the Classifier.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

from agent.nodes.catalog import clean_title, resolve_movie
from agent.nodes.criteria import (
    CATALOG_ARRAY_FIELDS,
    normalize_criteria,
    has_catalog_filters,
)
from agent.nodes.description_search import (
    DESC_MATCH_THRESHOLD,
    search_descriptions,
)
from agent.nodes.filters import apply_filters
from langchain_mistralai import MistralAIEmbeddings
from data_management.theme_embeddings_pipeline import EMBEDDING_MODEL


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
        exclude_names: list[str] | None = None,
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
        elif (
            column
            and value
            and column in CATALOG_ARRAY_FIELDS
            and column not in filters
        ):
            # Normalize the lone value too — a non-catalog "genre" demotes
            # instead of becoming a containment filter that matches nothing.
            extra = normalize_criteria({column: [value]})
            if extra[column]:
                filters[column] = extra[column]
        normalized.update(filters)
        result = {
            "kind": "direct_request",
            "titles": [],
            "filters": {**filters, **({"name": [value]} if title else {})},
            "error": None,
        }
        if not has_catalog_filters(normalized) and not title:
            result["error"] = "I couldn't identify a supported movie filter or title."
            return result

        if not self.supabase:
            result["error"] = "The movie catalog isn't available right now."
            return result

        try:
            query = self.supabase.table("movies").select("name, description")
            if title:
                # name is citext + pg_trgm: fuzzy title search.
                query = query.ilike("name", f"%{title}%")
            # Canonical labels: strict array containment keeps the
            # multi-genre AND intact.
            # Free-form names use the generated *_text columns so
            # partial mentions still match ("Nolan" -> "Christopher
            # Nolan"); one ilike per value keeps it an AND.
            query = apply_filters(query, normalized, exclude_names)
            res = query.order("rating", desc=True).limit(8).execute()
            movies = res.data

            if not movies:
                labels = ", ".join(
                    str(item)
                    for values in result["filters"].values()
                    for item in values
                )
                result["error"] = (
                    f"I couldn't find movies matching {labels or 'all those constraints'}."
                )
                return result

            result["titles"] = [movie["name"] for movie in movies]
            result["descriptions"] = {
                movie["name"]: movie.get("description", "") for movie in movies
            }
            return result
        except Exception:
            result["error"] = "I couldn't search the movie catalog right now."
            return result

    def embeddings(self):
        if not getattr(self, "_embeddings", None):
            self._embeddings = MistralAIEmbeddings(model=EMBEDDING_MODEL)
        return self._embeddings

    def handle_movie_fact(
        self, title: str, year: int | None = None, query_text: str | None = None
    ) -> dict:
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
        if not movie and query_text:
            try:
                matches = search_descriptions(
                    self.supabase, self.embeddings(), query_text
                )
                if matches and matches[0].get("similarity", 0) >= DESC_MATCH_THRESHOLD:
                    rows = (
                        self.supabase.table("movies")
                        .select("*")
                        .eq("id", matches[0]["movie_id"])
                        .execute()
                        .data
                        or []
                    )
                    movie = rows[0] if rows else None
                    result.update(
                        guessed=True, suggestions=[row["name"] for row in matches]
                    )
            except Exception:
                pass
        if not movie:
            result["error"] = f"I couldn't find '{title}' in the catalog."
            return result
        movie = dict(movie)
        # Same contract as `handle`: titles only, no database ids upstream.
        movie.pop("id", None)
        result["titles"] = [movie["name"]]
        result["facts"] = movie
        return result
