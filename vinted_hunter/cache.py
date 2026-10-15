"""SQLite cache for extractions and model specs, plus saved hunt files.

Item pages are not cached: availability (sold, reserved) has to come from a live fetch, so a
cached page could show a listing as available after it sold.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import Extraction, HuntResult, IdentifyDraft, ListingSummary, ModelSpecs

log = logging.getLogger(__name__)

SEARCH_CACHE_TTL_S = 30 * 60  # identical searches within 30 minutes reuse the stored page
NONEXACT_SPECS_TTL_S = 72 * 3600  # non-exact web specs and "no specs found" answers expire after 72 hours

_SCHEMA = """
CREATE TABLE IF NOT EXISTS extractions (
    item_id TEXT NOT NULL,
    model TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL,
    PRIMARY KEY (item_id, model, content_hash)
);
CREATE TABLE IF NOT EXISTS identify_cache (
    query_key TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS search_pages (
    cache_key TEXT PRIMARY KEY,
    fetched_at REAL NOT NULL,
    payload TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS model_specs (
    model_key TEXT PRIMARY KEY,
    created_at REAL NOT NULL,
    payload TEXT NOT NULL
);
"""


class HuntCache:
    """Small sqlite-backed cache. Each call opens its own connection, so it is safe to share."""

    def __init__(
        self,
        path: Path,
        *,
        clock: Callable[[], float] = time.time,
        specs_ttl_s: float = NONEXACT_SPECS_TTL_S,
    ) -> None:
        self.path = path
        self._clock = clock
        self._specs_ttl_s = specs_ttl_s
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """One connection per operation. It is always closed; the transaction commits on success
        and rolls back on error."""
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def get_extraction(self, item_id: str, model: str, content_hash: str) -> Extraction | None:
        with self._db() as conn:
            row = conn.execute(
                "SELECT payload FROM extractions WHERE item_id = ? AND model = ? AND content_hash = ?",
                (item_id, model, content_hash),
            ).fetchone()
        return _load(Extraction, row["payload"]) if row else None

    def put_extraction(self, extraction: Extraction, model: str, content_hash: str) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO extractions (item_id, model, content_hash, created_at, payload) "
                "VALUES (?, ?, ?, ?, ?)",
                (extraction.item_id, model, content_hash, self._clock(), extraction.model_dump_json()),
            )

    def get_search_page(self, cache_key: str, *, max_age_s: float = SEARCH_CACHE_TTL_S) -> list[ListingSummary] | None:
        """Listings from a search page fetched within max_age_s, or None."""
        with self._db() as conn:
            row = conn.execute(
                "SELECT fetched_at, payload FROM search_pages WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        if row is None or self._clock() - row["fetched_at"] > max_age_s:
            return None
        try:
            return [ListingSummary.model_validate(item) for item in json.loads(row["payload"])]
        except (ValueError, ValidationError):
            log.warning("ignoring stale search cache entry")
            return None

    def put_search_page(self, cache_key: str, listings: list[ListingSummary]) -> None:
        payload = json.dumps([listing.model_dump(mode="json") for listing in listings], ensure_ascii=False)
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO search_pages (cache_key, fetched_at, payload) VALUES (?, ?, ?)",
                (cache_key, self._clock(), payload),
            )

    def get_identify(self, query_key: str) -> IdentifyDraft | None:
        with self._db() as conn:
            row = conn.execute("SELECT payload FROM identify_cache WHERE query_key = ?", (query_key,)).fetchone()
        return _load(IdentifyDraft, row["payload"]) if row else None

    def put_identify(self, query_key: str, draft: IdentifyDraft) -> None:
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO identify_cache (query_key, created_at, payload) VALUES (?, ?, ?)",
                (query_key, self._clock(), draft.model_dump_json()),
            )

    def get_specs(self, cache_key: str) -> ModelSpecs | None:
        """Specs stored under cache_key (the model key plus the spec-key set, see pipeline.specs_cache_key).

        Exact-variant web specs with values never expire. Every other entry (non-exact variant, or no specs
        found) is a miss once it is older than the TTL. A payload written before the generic spec format has
        no `specs` map. It is a miss, not an answer, so it is looked up again.
        """
        with self._db() as conn:
            row = conn.execute(
                "SELECT created_at, payload FROM model_specs WHERE model_key = ?", (cache_key,)
            ).fetchone()
        if row is None:
            return None
        try:
            if "specs" not in json.loads(row["payload"]):
                return None
        except (TypeError, ValueError):
            return None
        specs = _load(ModelSpecs, row["payload"])
        if specs is None:
            return None
        if _specs_expire(specs) and self._clock() - row["created_at"] > self._specs_ttl_s:
            return None
        return specs

    def put_specs(self, cache_key: str, specs: ModelSpecs) -> None:
        # The column keeps its name for compatibility with existing cache files; it holds the cache key.
        with self._db() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO model_specs (model_key, created_at, payload) VALUES (?, ?, ?)",
                (cache_key, self._clock(), specs.model_dump_json()),
            )


def _specs_expire(specs: ModelSpecs) -> bool:
    """Only exact-variant web specs that hold values keep for ever. A non-exact answer, a model-knowledge answer,
    and a web answer with no values ("no specs found") expire after the TTL."""
    return not (specs.lookup_mode == "web" and specs.variant_match == "exact" and bool(specs.specs))


def _load(model_cls: Any, payload: str) -> Any:
    try:
        return model_cls.model_validate_json(payload)
    except ValidationError:
        log.warning("ignoring stale cache entry for %s", model_cls.__name__)
        return None


# --- Saved hunts -------------------------------------------------------------


def hunts_dir(data_dir: Path) -> Path:
    return data_dir / "hunts"


def save_hunt(data_dir: Path, hunt_id: str, result: HuntResult) -> Path:
    directory = hunts_dir(data_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{hunt_id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(result.model_dump_json(indent=2), encoding="utf-8")
    tmp.replace(path)
    return path


def load_hunt(data_dir: Path, hunt_id: str) -> HuntResult | None:
    path = hunts_dir(data_dir) / f"{hunt_id}.json"
    if not path.is_file():
        return None
    try:
        return HuntResult.model_validate_json(path.read_text(encoding="utf-8"))
    except (OSError, ValidationError):
        log.warning("could not read saved hunt %s", hunt_id)
        return None


def _summary_of(path: Path) -> dict[str, Any] | None:
    """Summary of one saved hunt, or None if the file is malformed."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return None
        ranked = data.get("ranked") or []
        cost = data.get("cost") or {}
        top_title = None
        if ranked:
            top_title = ranked[0]["listing"]["title"]
        return {
            "hunt_id": path.stem,  # the file name is the id the API serves the hunt under
            "request": str(data.get("request", "")),
            "budget_eur": data.get("budget_eur"),
            "created_at": str(data.get("created_at", "")),
            "ranked_count": len(ranked),
            "top_title": top_title,
            "total_usd": cost.get("total_usd") if isinstance(cost, dict) else None,
        }
    except (OSError, ValueError, KeyError, TypeError, AttributeError, IndexError):
        log.warning("skipping malformed saved hunt %s", path.name)
        return None


def list_hunts(data_dir: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Summaries of saved hunts, newest first. Malformed files are skipped."""
    directory = hunts_dir(data_dir)
    if not directory.is_dir():
        return []
    summaries = [s for s in (_summary_of(p) for p in directory.glob("*.json")) if s is not None]
    summaries.sort(key=lambda s: s["created_at"], reverse=True)
    return summaries[:limit]
