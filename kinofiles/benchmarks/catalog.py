"""Read-only catalog snapshot used to score answers after a run.

Scoring has to be cheap and identical for every system, so the whole `movies`
table (1k rows) is pulled once and matched in memory. Description embeddings
are the exception: they are fetched lazily for the films a run actually
mentions, because 1k vectors of 1024 floats is a lot of JSON to move for
nothing.

Nothing here writes to Supabase, and none of it runs during a benchmark — it
grades the JSON afterwards, so the scorer's own API calls can never land in a
system's cost numbers.
"""

import json
import math
import re
import unicodedata

MOVIE_COLUMNS = "id, name, date, rating, genres, themes, directors, actors, languages"
_YEAR_SUFFIX = re.compile(r"\s*\(\d{4}\)\s*$")
_ARTICLE = re.compile(r"^(the|a|an)\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")
# Shorter than this, containment stops meaning anything: "X" is a substring of
# "The Matrix", and "Up" of half the catalog. Both sides have to clear it.
_MIN_FUZZY_LEN = 5
# Catalogs and the films people name disagree about release years by about
# this much (festival vs. wide release), and no more.
_YEAR_TOLERANCE = 1


def normalize_title(title: str) -> str:
    """Fold a spoken or generated title to its comparable core."""
    text = _YEAR_SUFFIX.sub("", title or "").strip().lower()
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    text = _ARTICLE.sub("", text)
    return _NON_ALNUM.sub(" ", text).strip()


def title_year(title: str) -> int | None:
    match = re.search(r"\((\d{4})\)\s*$", title or "")
    return int(match.group(1)) if match else None


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class Catalog:
    """In-memory view of `movies`, plus lazy description vectors."""

    def __init__(self, client=None):
        if client is None:
            from data_management.theme_embeddings_pipeline import _client, _load_env

            _load_env()
            client = _client()
            if client is None:
                raise RuntimeError("Supabase credentials missing — check kinofiles/.env")
        self._client = client
        self._rows: list[dict] = []
        self._by_title: dict[str, list[dict]] = {}
        self._vectors: dict[int, list[float]] = {}
        self._load()

    def _load(self) -> None:
        page, size = 0, 1000
        while True:
            rows = (
                self._client.table("movies")
                .select(MOVIE_COLUMNS)
                .range(page * size, page * size + size - 1)
                .execute()
                .data
                or []
            )
            self._rows.extend(rows)
            if len(rows) < size:
                break
            page += 1
        for row in self._rows:
            # Remakes share a normalized title, so every row is kept: a stated
            # year has to be able to pick between them.
            self._by_title.setdefault(normalize_title(row["name"]), []).append(row)

    def __len__(self) -> int:
        return len(self._rows)

    @property
    def rows(self) -> list[dict]:
        return self._rows

    def resolve(self, title: str) -> dict | None:
        """Catalog row for a title as some system wrote it, or None.

        Forgiving about spelling — a baseline answering "Avengers: Infinity
        War" should not be marked ungrounded over punctuation — and strict
        about identity. A stated year that matches nothing means the film
        named is not the one we stock: "The Lion King (1994)" is not the 2019
        remake, and answering with it is exactly the failure this metric
        exists to catch.
        """
        key = normalize_title(title)
        if not key:
            return None
        year = title_year(title)

        candidates = list(self._by_title.get(key, []))
        if not candidates and len(key) >= _MIN_FUZZY_LEN:
            candidates = [
                row
                for other_key, rows in self._by_title.items()
                if len(other_key) >= _MIN_FUZZY_LEN
                and (key in other_key or other_key in key)
                for row in rows
            ]
        if not candidates:
            return None

        if year:
            dated = [
                row
                for row in candidates
                if row.get("date") and abs(row["date"] - year) <= _YEAR_TOLERANCE
            ]
            if dated:
                return self._best(dated)
            if any(row.get("date") for row in candidates):
                return None
        return self._best(candidates)

    @staticmethod
    def _best(rows: list[dict]) -> dict:
        """Ties go to the better-rated film — what a bare title usually means."""
        return max(rows, key=lambda row: row.get("rating") or 0.0)

    def description_vectors(self, movie_ids: list[int]) -> dict[int, list[float]]:
        """Stored description embeddings, fetched once per movie and cached."""
        missing = [mid for mid in dict.fromkeys(movie_ids) if mid not in self._vectors]
        for start in range(0, len(missing), 100):
            batch = missing[start : start + 100]
            rows = (
                self._client.table("descriptions_embeddings")
                .select("movie_id, embedding")
                .in_("movie_id", batch)
                .execute()
                .data
                or []
            )
            for row in rows:
                vector = row["embedding"]
                # pgvector comes back as a JSON string over PostgREST.
                if isinstance(vector, str):
                    vector = json.loads(vector)
                self._vectors[row["movie_id"]] = vector
            for mid in batch:
                self._vectors.setdefault(mid, [])
        return {mid: self._vectors.get(mid) or [] for mid in movie_ids}


def satisfies(row: dict, constraints: dict) -> dict[str, bool]:
    """Per-constraint verdicts for one catalog row.

    Every check reads the catalog, never the model's prose: that is the whole
    point — a system claiming "this is a horror film" is graded against the
    `genres` column, not against its own confidence.
    """
    verdicts: dict[str, bool] = {}
    for field in ("genres", "directors", "actors", "languages"):
        wanted = constraints.get(field) or []
        if not wanted:
            continue
        have = {str(value).casefold() for value in (row.get(field) or [])}
        verdicts[field] = all(
            any(want.casefold() in value or value in want.casefold() for value in have)
            for want in wanted
        )
    return verdicts
