"""The catalog, read once: the pool both arms choose from.

The agent reaches into `movies` through pgvector and column filters; the
baseline gets the same rows pasted into its prompt. So this module does one
thing — pull every row and format it — and nothing else. Scoring lives in
`judge.py`, which never touches Supabase.
"""

MOVIE_COLUMNS = "name, date, genres"
# Enough to tell a Disney animation from a Ghibli one at a glance, short
# enough that 1k lines stay a prompt rather than a novel.
GENRES_PER_LINE = 3


def load_movies() -> list[dict]:
    """Every row of `movies`, paged, as plain dicts."""
    from data_management.theme_embeddings_pipeline import _client, _load_env

    _load_env()
    client = _client()
    if client is None:
        raise RuntimeError("Supabase credentials missing — check kinofiles/.env")

    rows, page, size = [], 0, 1000
    while True:
        batch = (
            client.table("movies")
            .select(MOVIE_COLUMNS)
            .range(page * size, page * size + size - 1)
            .execute()
            .data
            or []
        )
        rows.extend(batch)
        if len(batch) < size:
            break
        page += 1
    return rows


def catalog_lines(rows: list[dict]) -> str:
    """The catalog as the baseline sees it: one film per line.

    Sorted by title, not by rating: a popularity ordering would hand the
    baseline a ranking it did not earn, and where a film sits in a 1k-line
    prompt is not nothing.
    """
    lines = []
    for row in sorted(
        rows, key=lambda row: ((row.get("name") or "").casefold(), row.get("date") or 0)
    ):
        year = f" ({row['date']})" if row.get("date") else ""
        genres = ", ".join((row.get("genres") or [])[:GENRES_PER_LINE])
        lines.append(f"{row['name']}{year}" + (f" | {genres}" if genres else ""))
    return "\n".join(lines)
