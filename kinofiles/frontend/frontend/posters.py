"""Poster art for a movie title, looked up in `assets/posters.csv`.

The agent only ever hands the frontend a title, but the CSV's Letterboxd
URLs carry the film's slug in the filename
(`.../277064-barbie-0-230-0-345-crop.jpg`), so slugifying the title the same
way Letterboxd does is enough to find its poster.

The file holds ~940k rows, so it is never held in memory: a lookup streams it
once (~1s) resolving every pending title in the same pass, and the answers are
cached. Misses are cached too, so a title nobody has art for costs one scan.
"""

import re
import unicodedata
from pathlib import Path

CSV = Path(__file__).resolve().parent.parent / "assets" / "posters.csv"

# Every poster URL ends with the same crop spec; what precedes it is
# "<film id>-<slug>".
SUFFIX = "-0-230-0-345-crop.jpg"
YEAR = re.compile(r"-(?:19|20)\d{2}$")
TRAILING_YEAR = re.compile(r"\s*\((?:19|20)\d{2}\)\s*$")

_cache: dict[str, str] = {}


def _slugs(title: str) -> list[str]:
    """The slugs a title could be filed under, best guess first.

    This catalogue spells an apostrophe as its own hyphen ("schindler-s-list"),
    but that is not universal, so the elided form is tried as well.
    """
    title = TRAILING_YEAR.sub("", title).strip()
    plain = unicodedata.normalize("NFKD", title.replace("’", "'"))
    plain = plain.encode("ascii", "ignore").decode().lower()
    kept = re.sub(r"[^a-z0-9]+", "-", plain).strip("-")
    dropped = re.sub(r"[^a-z0-9]+", "-", plain.replace("'", "")).strip("-")
    return [slug for slug in dict.fromkeys([kept, dropped]) if slug]


def _slug_of(link: str) -> str:
    """The film slug inside a poster URL, or "" for the unnamed upload variants."""
    if "/film-poster/" not in link:
        return ""
    name = link.rsplit("/", 1)[-1].split("?", 1)[0]
    if not name.endswith(SUFFIX):
        return ""
    core = name[: -len(SUFFIX)]
    film_id, _, slug = core.partition("-")
    return slug if film_id.isdigit() else ""


def poster_urls(titles: list[str]) -> dict[str, str]:
    """Map each title to a poster URL, "" when the catalogue has none."""
    wanted: dict[str, str] = {}
    for title in titles:
        if title not in _cache:
            for slug in _slugs(title):
                wanted.setdefault(slug, title)

    if wanted and CSV.exists():
        pending = {title for title in wanted.values()}
        # A slug carries a disambiguating year only when several films share a
        # name ("dune-2021"), so an exact hit always beats a year-stripped one.
        loose: dict[str, str] = {}
        with CSV.open() as rows:
            next(rows, None)
            for row in rows:
                link = row[row.find(",") + 1 :].strip()
                slug = _slug_of(link)
                if not slug:
                    continue
                title = wanted.get(slug)
                if title is not None:
                    _cache.setdefault(title, link)
                    pending.discard(title)
                    if not pending:
                        break
                elif (base := YEAR.sub("", slug)) in wanted:
                    loose.setdefault(base, link)
        for slug, link in loose.items():
            _cache.setdefault(wanted[slug], link)

    for title in titles:
        _cache.setdefault(title, "")
    return {title: _cache[title] for title in titles}
