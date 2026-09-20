import logging

from agent.nodes.criteria import has_catalog_filters, normalize_criteria
from agent.nodes.filters import matches_filters

logger = logging.getLogger(__name__)

DESC_MATCH_THRESHOLD = 0.68


def search_descriptions(
    supabase,
    embeddings,
    query_text,
    genres=None,
    exclude_ids=None,
    match_count=40,
    criteria=None,
    exclude_names=None,
    limit=8,
):
    c = dict(criteria or {})
    if genres and not c.get("genre_groups"):
        c["genres"] = genres
    vector = embeddings.embed_query(query_text)
    excluded_ids = set(exclude_ids or [])
    excluded_names = {name.casefold() for name in exclude_names or []}
    count = match_count
    while True:
        matches = (
            supabase.rpc(
                "match_descriptions", {"query_embedding": vector, "match_count": count}
            )
            .execute()
            .data
            or []
        )
        candidates = [
            row
            for row in matches
            if "movie_id" in row
            and row["movie_id"] not in excluded_ids
            and row["name"].casefold() not in excluded_names
        ]
        if candidates and has_catalog_filters(c):
            rows = (
                supabase.table("movies")
                .select("*")
                .in_("id", [row["movie_id"] for row in candidates])
                .execute()
                .data
                or []
            )
            allowed = {row["id"] for row in rows if matches_filters(row, c)}
            candidates = [row for row in candidates if row["movie_id"] in allowed]
        logger.info(
            "description_search | criteria=%s | excluded_ids=%s | excluded_names=%s | kept=%s/%s candidates",
            {key: value for key, value in normalize_criteria(c).items() if value},
            len(excluded_ids),
            len(excluded_names),
            len(candidates),
            len(matches),
        )
        candidates.sort(key=lambda row: -row.get("similarity", 0))
        if len(candidates) >= limit or len(matches) < count or count >= 1000:
            return candidates[:limit]
        count = min(count * 4, 1000)
