"""Persistent, normalized search criteria for multi-turn recommendations."""

import logging
from copy import deepcopy
from typing import Literal

logger = logging.getLogger(__name__)

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


def promote_genre_themes(criteria: dict | None) -> dict:
    """Lift theme words that are really catalog genres into `genres`.

    A bare answer ("action") gets its genre read as a mood, which would keep
    it out of the genre merge that decides what the group has in common.
    `normalize_criteria` only ever demotes in the other direction, so the lift
    has to happen here.
    """
    normalized = normalize_criteria(criteria)
    promoted, remaining = split_genres(normalized["themes"])
    if promoted:
        normalized["genres"] = normalize_genres(normalized["genres"] + promoted)
        normalized["themes"] = remaining
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
    """Compose one semantic query while leaving genres as strict filters.

    The request arrives pre-joined (the mediator concatenates every
    participant's raw utterance with "; "), while the criteria below hold the
    classifier's normalized reading of those same utterances. Splitting the
    request back apart is what lets the dedupe at the bottom collapse the two
    copies — against one joined blob it can only compare the whole string.
    """
    normalized = normalize_criteria(criteria)
    parts = [segment.strip() for segment in request.split(";")]
    parts.extend(normalized["themes"])
    parts.extend(normalized["audience"])
    parts.extend(normalized["time_periods"])
    parts.extend(
        movie["title"]
        for movie in normalized["movies"]
        if movie.get("role") in {"seed", "liked"}
    )
    return "; ".join(_dedupe([part for part in parts if part]))


# Everything a person can state a preference about, except seed movies, which
# carry a role and are summarised separately.
BRIEF_FIELDS = tuple(field for field in CRITERIA_FIELDS if field != "movies")


def group_brief(per_person: dict[str, dict], refined: bool = False) -> dict:
    """Recover the shape of the group's agreement behind the merged query.

    Mediation flattens everyone into one search, which leaves nothing able to
    say *why* that search is what it is. Counting how many people asked for
    each value splits the brief into what the group converged on and what a
    single person brought, which is the explanation worth speaking aloud.
    """
    normalized = {
        person: normalize_criteria(criteria)
        for person, criteria in (per_person or {}).items()
    }

    # (field, casefolded value) -> everyone who asked for it.
    askers: dict[tuple[str, str], list[str]] = {}
    labels: dict[tuple[str, str], str] = {}
    for person, criteria in normalized.items():
        for field in BRIEF_FIELDS:
            for value in criteria[field]:
                if not isinstance(value, str) or not value.strip():
                    continue
                key = (field, value.strip().casefold())
                if person not in askers.setdefault(key, []):
                    askers[key].append(person)
                labels.setdefault(key, value.strip())

    shared, individual = [], {}
    for key, people in askers.items():
        entry = {"field": key[0], "value": labels[key]}
        if len(people) > 1:
            shared.append({**entry, "wanted_by": people})
        else:
            individual.setdefault(people[0], []).append(entry)

    seeds = [
        {"title": movie["title"], "from": person, "role": movie.get("role")}
        for person, criteria in normalized.items()
        for movie in criteria["movies"]
        if isinstance(movie, dict) and movie.get("title")
    ]

    # Named rather than left for the reply model to infer: whether the group
    # actually agreed is the one thing it must not get wrong, and "no overlap
    # at all" is the case worth saying out loud instead of papering over.
    contributors = {person for people in askers.values() for person in people}
    if refined:
        # They came back with a joint request, so whatever split they started
        # from has been settled between them; leading with it again would be
        # stale and would undo the agreement they just reached. Clearing the
        # split matters more than the rule in the prompt: the reply model
        # follows the data it is handed, and while the round-1 divergence is
        # still in the payload it keeps writing the divergence sentence.
        consensus = "refined"
        shared, individual = [], {}
    elif len(normalized) < 2:
        consensus = "single"
    elif len(contributors) < 2:
        # Only one person's wishes were captured. That is missing input, not a
        # clash, and announcing divergence here would be a plain falsehood.
        consensus = "unknown"
    elif not shared:
        consensus = "none"
    elif not individual:
        consensus = "full"
    else:
        consensus = "partial"

    return {
        "shared": shared,
        "individual": individual,
        "seed_movies": seeds,
        "consensus": consensus,
    }


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
        
    normalized_per_person = {name: normalize_criteria(crit) for name, crit in per_person.items()}

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
            logger.info(
                "merge_group | genres=intersection | sets=%s | merged=%s",
                [sorted(genre_set) for genre_set in all_genres_sets],
                sorted(intersect),
            )
        else:
            union = set.union(*all_genres_sets)
            merged["genres"] = list(union)
            genre_note = f"Combined different genre preferences: {', '.join(union)}."
            # The catalog applies genres with array containment (AND), so a
            # union of disjoint genres narrows the search instead of relaxing it.
            logger.warning(
                "merge_group | genres=union(no overlap) | sets=%s | merged=%s "
                "| downstream filter is AND, expect few or zero matches",
                [sorted(genre_set) for genre_set in all_genres_sets],
                sorted(union),
            )
    else:
        genre_note = "No specific genre constraints."
        logger.info("merge_group | genres=none | no genre filter applied")

    for field in CRITERIA_FIELDS:
        if field == "genres":
            continue
        all_vals = []
        for crit in normalized_per_person.values():
            all_vals.extend(crit.get(field) or [])
        merged[field] = _dedupe(all_vals)

    notes = f"Merged preferences for {len(per_person)} people. {genre_note}"
    logger.info(
        "merge_group | people=%s | merged=%s",
        list(per_person),
        {field: value for field, value in merged.items() if value},
    )
    return merged, notes

