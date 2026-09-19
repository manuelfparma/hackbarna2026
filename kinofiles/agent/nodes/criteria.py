"""Persistent, normalized search criteria for multi-turn recommendations."""

from copy import deepcopy
from typing import Literal

CriteriaAction = Literal["add", "replace", "reset", "keep"]

CRITERIA_FIELDS = (
    "movies",
    "actors",
    "directors",
    "genres",
    "themes",
    "studios",
    "languages",
    "years",
    "time_periods",
    "audience",
)

CATALOG_ARRAY_FIELDS = (
    "directors",
    "actors",
    "genres",
    "studios",
    "languages",
)

CATALOG_GENRES = (
    "Action",
    "Adventure",
    "Animation",
    "Comedy",
    "Crime",
    "Drama",
    "Family",
    "Fantasy",
    "History",
    "Horror",
    "Music",
    "Mystery",
    "Romance",
    "Science Fiction",
    "Thriller",
    "TV Movie",
    "War",
    "Western",
)

_GENRE_ALIASES = {
    **{genre.casefold(): genre for genre in CATALOG_GENRES},
    "sci-fi": "Science Fiction",
    "sci fi": "Science Fiction",
    "science-fiction": "Science Fiction",
    "rom-com": "Romance",
    "romcom": "Romance",
    "action movie": "Action",
    "action movies": "Action",
    "comedy movie": "Comedy",
    "comedy movies": "Comedy",
}


def empty_criteria() -> dict:
    """Return a new empty criteria object matching the classifier entities."""
    return {field: [] for field in CRITERIA_FIELDS}


def split_genres(genres: list[str]) -> tuple[list[str], list[str]]:
    """Split genre words into (canonical catalog labels, everything else).

    Anything that doesn't map to a catalog label is not a genre the filter
    layer can use — "heist" as a genres containment filter matches nothing
    and errors the lookup. Those words carry real meaning though, so callers
    demote them to themes where they become a semantic query.
    """
    canonical, demoted = [], []
    seen = set()
    for genre in genres:
        clean = genre.strip()
        label = _GENRE_ALIASES.get(clean.casefold())
        if label:
            if label.casefold() not in seen:
                seen.add(label.casefold())
                canonical.append(label)
        elif clean:
            demoted.append(clean)
    return canonical, demoted


def normalize_genres(genres: list[str]) -> list[str]:
    """Map common genre spellings to the catalog's exact labels."""
    return split_genres(genres)[0]


def _dedupe(values: list) -> list:
    result = []
    seen = set()
    for value in values:
        if isinstance(value, dict):
            key = str(value.get("title", value)).casefold()
        elif isinstance(value, str):
            key = value.strip().casefold()
            value = value.strip()
        else:
            key = value
        if key not in seen:
            seen.add(key)
            result.append(value)
    return result


def normalize_criteria(criteria: dict | None) -> dict:
    """Fill missing fields and normalize values copied from structured output.

    Genre words that aren't catalog labels demote to `themes`, so a bogus
    "genre" (heist, space) becomes a semantic signal instead of an empty
    containment filter.
    """
    normalized = empty_criteria()
    for field in CRITERIA_FIELDS:
        normalized[field] = _dedupe(deepcopy((criteria or {}).get(field) or []))
    canonical, demoted = split_genres(normalized["genres"])
    normalized["genres"] = canonical
    normalized["themes"] = _dedupe(normalized["themes"] + demoted)
    return normalized


def merge_criteria(
    current: dict | None,
    turn_entities: dict | None,
    action: CriteriaAction,
) -> dict:
    """Apply the classifier's cue-based operation to persisted criteria."""
    existing = normalize_criteria(current)
    incoming = normalize_criteria(turn_entities)
    if action == "keep":
        return existing
    if action in {"replace", "reset"}:
        return incoming

    merged = empty_criteria()
    for field in CRITERIA_FIELDS:
        merged[field] = _dedupe(existing[field] + incoming[field])
    merged["genres"] = normalize_genres(merged["genres"])
    return merged


def has_search_terms(criteria: dict | None) -> bool:
    normalized = normalize_criteria(criteria)
    return any(normalized[field] for field in CRITERIA_FIELDS)


def has_catalog_filters(criteria: dict | None) -> bool:
    normalized = normalize_criteria(criteria)
    return any(normalized[field] for field in CATALOG_ARRAY_FIELDS)


def build_theme_query(request: str, criteria: dict | None) -> str:
    """Compose one semantic query while leaving genres as strict filters."""
    normalized = normalize_criteria(criteria)
    parts = [request.strip()]
    parts.extend(normalized["themes"])
    parts.extend(normalized["audience"])
    parts.extend(normalized["time_periods"])
    parts.extend(
        movie["title"]
        for movie in normalized["movies"]
        if movie.get("role") in {"seed", "liked"}
    )
    return "; ".join(_dedupe([part for part in parts if part]))
