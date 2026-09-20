import json

from agent.nodes.catalog import clean_title
from agent.nodes.criteria import CATALOG_ARRAY_FIELDS, normalize_criteria


def apply_filters(query, criteria, exclude_names=None):
    c = normalize_criteria(criteria)
    for field in CATALOG_ARRAY_FIELDS:
        values = c[field]
        if not values:
            continue
        if field in {"genres", "streaming"}:
            query = query.contains(field, values)
        else:
            for value in values:
                query = query.ilike(f"{field}_text", f"%{clean_title(value)}%")
    if c["genre_groups"]:
        groups = [
            ",".join(json.dumps(value) for value in group)
            for group in c["genre_groups"]
        ]
        query = query.or_(",".join(f"genres.cs.{{{group}}}" for group in groups))
    if c["years"]:
        query = query.in_("date", c["years"])
    for key, column, operator in [
        ("year_min", "date", "gte"),
        ("year_max", "date", "lte"),
        ("minute_min", "minute", "gte"),
        ("minute_max", "minute", "lte"),
        ("min_rating", "rating", "gte"),
    ]:
        if c[key] is not None:
            query = getattr(query, operator)(column, c[key])
    for field in ("genres", "actors", "directors"):
        for value in c[f"exclude_{field}"]:
            if field == "genres":
                query = query.not_.contains(field, [value])
            else:
                query = query.not_.ilike(f"{field}_text", f"%{clean_title(value)}%")
    for name in exclude_names or []:
        query = query.neq("name", name)
    return query


def matches_filters(movie, criteria):
    c = normalize_criteria(criteria)
    for field in CATALOG_ARRAY_FIELDS:
        actual = [v.casefold() for v in movie.get(field) or []]
        if field in {"genres", "streaming"}:
            if not {v.casefold() for v in c[field]} <= set(actual):
                return False
        elif any(not any(v.casefold() in item for item in actual) for v in c[field]):
            return False
    genres = set(v.casefold() for v in movie.get("genres") or [])
    if c["genre_groups"] and not any(
        {v.casefold() for v in group} <= genres for group in c["genre_groups"]
    ):
        return False
    for field in ("genres", "actors", "directors"):
        actual = [v.casefold() for v in movie.get(field) or []]
        if any(
            any(
                v.casefold() == item if field == "genres" else v.casefold() in item
                for item in actual
            )
            for v in c[f"exclude_{field}"]
        ):
            return False
    if c["years"] and movie.get("date") not in c["years"]:
        return False
    for field, column, lower in [
        ("year_min", "date", True),
        ("year_max", "date", False),
        ("minute_min", "minute", True),
        ("minute_max", "minute", False),
        ("min_rating", "rating", True),
    ]:
        limit = c[field]
        value = movie.get(column)
        if limit is not None and (
            value is None or (value < limit if lower else value > limit)
        ):
            return False
    return True
