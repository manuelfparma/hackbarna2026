"""Shared catalog lookups: resolve a spoken title to a `movies` row."""

import re

MOVIE_COLUMNS = (
    "id, name, date, tagline, description, minute, rating,"
    " genres, themes, studios, languages, actors, directors, poster, streaming"
)

# Resolved ReDial mentions arrive as "Super Troopers (2001)"; the catalog
# stores bare names, so the parenthetical year has to come off before match.
_YEAR_SUFFIX = re.compile(r"\s*\(\d{4}\)\s*$")
_LIKE_WILDCARDS = re.compile(r"[%_]")


def clean_title(title: str) -> str:
    """Strip a trailing '(year)' and ILIKE wildcards from a spoken title."""
    return _LIKE_WILDCARDS.sub(" ", _YEAR_SUFFIX.sub("", title or "")).strip()


def resolve_movie(supabase, title: str, year: int | None = None) -> dict | None:
    """Best catalog row for a mentioned title, or None when nothing matches.

    Exact case-insensitive match wins; otherwise the best-rated row whose
    name contains the title. `year` (when the user said one) gets a shot at
    disambiguating substring matches before popularity does.
    """
    clean = clean_title(title)
    if not clean:
        return None

    suffix = re.search(r"\((\d{4})\)\s*$", title)
    year = year or (int(suffix[1]) if suffix else None)
    query = supabase.table("movies").select(MOVIE_COLUMNS).ilike("name", clean)
    if year is not None:
        query = query.eq("date", year)
    rows = query.limit(1).execute().data or []
    if rows:
        return rows[0]

    if year:
        rows = (
            supabase.table("movies")
            .select(MOVIE_COLUMNS)
            .ilike("name", f"%{clean}%")
            .eq("date", year)
            .limit(1)
            .execute()
            .data
            or []
        )
        return rows[0] if rows else None

    rows = (
        supabase.table("movies")
        .select(MOVIE_COLUMNS)
        .ilike("name", f"%{clean}%")
        .order("rating", desc=True)
        .limit(1)
        .execute()
        .data
        or []
    )
    return rows[0] if rows else None
