"""Persistent, normalized search criteria for multi-turn recommendations."""

from copy import deepcopy
import re
from typing import Literal

CriteriaAction = Literal["add", "replace", "reset", "keep", "remove"]
ClearField = Literal[
    "date",
    "duration",
    "rating",
    "genres",
    "actors",
    "directors",
    "streaming",
    "languages",
    "themes",
]
CLEAR_GROUPS = {
    "date": ("years", "time_periods", "year_min", "year_max"),
    "duration": ("duration", "minute_min", "minute_max"),
    "rating": ("min_rating",),
    "genres": ("genres", "genre_groups", "exclude_genres"),
    "actors": ("actors", "exclude_actors"),
    "directors": ("directors", "exclude_directors"),
    "streaming": ("streaming",),
    "languages": ("languages",),
    "themes": ("themes",),
}

CRITERIA_FIELDS = (
    "movies",
    "actors",
    "directors",
    "genres",
    "themes",
    "languages",
    "years",
    "time_periods",
    "streaming",
    "genre_groups",
    "duration",
    "exclude_genres",
    "exclude_actors",
    "exclude_directors",
)

SCALAR_FIELDS = ("year_min", "year_max", "minute_min", "minute_max", "min_rating")
SERVICE_ALIASES = {
    "netflix": "netflix",
    "hbo": "hbo",
    "max": "hbo",
    "hbo max": "hbo",
    "prime": "prime",
    "amazon prime": "prime",
    "prime video": "prime",
    "appletv": "appletv",
    "apple tv": "appletv",
    "apple tv+": "appletv",
}

CATALOG_ARRAY_FIELDS = (
    "directors",
    "actors",
    "genres",
    "languages",
    "streaming",
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
    return {
        **{field: [] for field in CRITERIA_FIELDS},
        **{field: None for field in SCALAR_FIELDS},
    }


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
            key = (
                str(value.get("title", value)).casefold(),
                value.get("year"),
                value.get("role"),
            )
        elif isinstance(value, list):
            key = tuple(str(item).casefold() for item in value)
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
    remaining = []
    for theme in normalized["themes"]:
        label = re.sub(
            r"\s+(?:movies?|films?)$", "", theme.strip(), flags=re.IGNORECASE
        )
        genres = normalize_genres([label])
        if genres and (label.casefold() == "animation" or label != theme.strip()):
            normalized["genres"] = _dedupe(normalized["genres"] + genres)
        else:
            remaining.append(theme)
    normalized["themes"] = remaining
    normalized["exclude_genres"] = normalize_genres(normalized["exclude_genres"])
    normalized["genre_groups"] = _dedupe(
        [
            normalize_genres(group)
            for group in normalized["genre_groups"]
            if normalize_genres(group)
        ]
    )
    if len(normalized["genre_groups"]) == 1:
        normalized["genres"] = normalized["genre_groups"][0]
        normalized["genre_groups"] = []
    elif normalized["genre_groups"]:
        normalized["genres"] = []
    normalized["streaming"] = _dedupe(
        [
            SERVICE_ALIASES.get(v.casefold(), v.casefold())
            for v in normalized["streaming"]
        ]
    )
    for field in SCALAR_FIELDS:
        normalized[field] = (criteria or {}).get(field)
    for period in normalized["time_periods"]:
        match = re.fullmatch(r"(\d{2}|\d{4})['’]?s", period.casefold().strip())
        low = high = None
        if match:
            low = int(match[1])
            low = low + (2000 if low < 30 else 1900) if low < 100 else low
            high = low + 9
        elif period.casefold() in {"old", "classic"}:
            high = 1980
        elif period.casefold() in {"new", "recent", "modern"}:
            low = 2015
        if low is not None:
            normalized["year_min"] = max(normalized["year_min"] or low, low)
        if high is not None:
            normalized["year_max"] = min(normalized["year_max"] or high, high)
    for duration in normalized["duration"]:
        text = duration.casefold().strip()
        for word, number in {
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "ninety": "90",
        }.items():
            text = re.sub(r"\b" + word + r"\b", number, text)
        if text == "short" and normalized["minute_max"] is None:
            normalized["minute_max"] = 99
        elif text == "standard":
            normalized["minute_min"] = (
                normalized["minute_min"]
                if normalized["minute_min"] is not None
                else 100
            )
            normalized["minute_max"] = (
                normalized["minute_max"]
                if normalized["minute_max"] is not None
                else 129
            )
        elif text in {"long", "epic"} and normalized["minute_min"] is None:
            normalized["minute_min"] = 150 if text == "epic" else 130
        match = re.fullmatch(
            r"(under|over|at most|at least) (\d+(?:\.\d+)?) (minutes?|hours?)", text
        )
        if match:
            minutes = float(match[2]) * (60 if match[3].startswith("hour") else 1)
            field = "minute_max" if match[1] in {"under", "at most"} else "minute_min"
            normalized[field] = int(minutes) + (
                -1 if match[1] == "under" else 1 if match[1] == "over" else 0
            )
    return normalized


def clear_criteria(criteria: dict | None, fields: list[str] | tuple[str, ...]) -> dict:
    cleared = normalize_criteria(criteria)
    defaults = empty_criteria()
    for group in fields:
        for field in CLEAR_GROUPS[group]:
            cleared[field] = deepcopy(defaults[field])
    return normalize_criteria(cleared)


def merge_criteria(
    current: dict | None,
    turn_entities: dict | None,
    action: CriteriaAction,
    clear_fields: list[str] | tuple[str, ...] = (),
) -> dict:
    """Apply the classifier's cue-based operation to persisted criteria."""
    if action == "remove":
        return merge_criteria(
            clear_criteria(current, clear_fields),
            clear_criteria(turn_entities, clear_fields),
            "add",
        )
    existing = normalize_criteria(current)
    incoming = normalize_criteria(turn_entities)
    if action == "keep":
        return existing
    if action in {"replace", "reset"}:
        return incoming

    merged = empty_criteria()
    for field in CRITERIA_FIELDS:
        merged[field] = _dedupe(existing[field] + incoming[field])
    for field in SCALAR_FIELDS:
        merged[field] = (
            incoming[field] if incoming[field] is not None else existing[field]
        )
    old_groups = existing["genre_groups"] or (
        [existing["genres"]] if existing["genres"] else []
    )
    new_groups = incoming["genre_groups"] or (
        [incoming["genres"]] if incoming["genres"] else []
    )
    if existing["genre_groups"] or incoming["genre_groups"]:
        merged["genre_groups"] = (
            [_dedupe(a + b) for a in old_groups for b in new_groups]
            if old_groups and new_groups
            else old_groups or new_groups
        )
        merged["genres"] = []
    merged["genres"] = normalize_genres(merged["genres"])
    return normalize_criteria(merged)


def has_search_terms(criteria: dict | None) -> bool:
    normalized = normalize_criteria(criteria)
    return any(normalized[field] for field in CRITERIA_FIELDS) or any(
        normalized[field] is not None for field in SCALAR_FIELDS
    )


def has_catalog_filters(criteria: dict | None) -> bool:
    normalized = normalize_criteria(criteria)
    return any(
        normalized[field]
        for field in (
            *CATALOG_ARRAY_FIELDS,
            "genre_groups",
            "years",
            "exclude_genres",
            "exclude_actors",
            "exclude_directors",
        )
    ) or any(normalized[field] is not None for field in SCALAR_FIELDS)


def build_theme_query(request: str, criteria: dict | None) -> str:
    """Compose one semantic query while leaving genres as strict filters."""
    normalized = normalize_criteria(criteria)
    parts = [request.strip()]
    parts.extend(normalized["themes"])
    parts.extend(normalized["time_periods"])
    parts.extend(
        movie["title"]
        for movie in normalized["movies"]
        if movie.get("role") in {"seed", "liked"}
    )
    return "; ".join(_dedupe([part for part in parts if part]))


def merge_group_criteria(per_person: dict[str, dict]) -> tuple[dict, str]:
    """Merge N people's normalized criteria into one brief.

    Returns (merged_criteria, compromise_notes).

    Strategy:
    - Genres: intersection if non-empty, else union (relaxed match)
    - Themes: union (cast a wide net)
    - Directors/actors and others: union (any match counts)
    - Movies (seeds): all kept
    """
    if not per_person:
        return empty_criteria(), "No preferences provided."

    normalized_per_person = {
        name: normalize_criteria(crit) for name, crit in per_person.items()
    }

    merged = empty_criteria()

    all_genres_sets = []
    for name, crit in normalized_per_person.items():
        if crit.get("genres"):
            all_genres_sets.append(set(crit["genres"]))

    if all_genres_sets:
        intersect = set.intersection(*all_genres_sets)
        if intersect:
            merged["genres"] = list(intersect)
            genre_note = f"Found common genres: {', '.join(intersect)}."
        else:
            union = set.union(*all_genres_sets)
            merged["genres"] = list(union)
            genre_note = f"Combined different genre preferences: {', '.join(union)}."
    else:
        genre_note = "No specific genre constraints."

    for field in CRITERIA_FIELDS:
        if field == "genres":
            continue
        all_vals = []
        for crit in normalized_per_person.values():
            all_vals.extend(crit.get(field) or [])
        merged[field] = _dedupe(all_vals)

    for field in SCALAR_FIELDS:
        values = [
            crit[field]
            for crit in normalized_per_person.values()
            if crit[field] is not None
        ]
        if values:
            merged[field] = min(values) if field.endswith("_max") else max(values)

    notes = f"Merged preferences for {len(per_person)} people. {genre_note}"
    return merged, notes
