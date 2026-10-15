import json
import sqlite3
import time

import pytest
from helpers import make_detail, make_summary

from vinted_hunter.cache import HuntCache, list_hunts, load_hunt, save_hunt
from vinted_hunter.models import (
    CostReport,
    Extraction,
    HuntResult,
    ModelSpecs,
    RankedEntry,
    RankedListing,
    SearchPlan,
)


class Clock:
    def __init__(self, now=1_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


def make_cache(tmp_path, clock=None):
    return HuntCache(tmp_path / "cache.sqlite3", clock=clock or Clock())


def test_extraction_is_keyed_by_model_and_content_hash(tmp_path):
    cache = make_cache(tmp_path)
    extraction = Extraction(item_id="7", brand="Acme", model="X1", model_confidence=0.8)
    cache.put_extraction(extraction, "claude-opus-5-5", "hash1")

    assert cache.get_extraction("7", "claude-opus-5-5", "hash1") == extraction
    assert cache.get_extraction("7", "claude-opus-5-5", "hash2") is None
    assert cache.get_extraction("7", "claude-haiku-5-5", "hash1") is None


def test_specs_round_trip_without_expiry(tmp_path):
    clock = Clock()
    cache = make_cache(tmp_path, clock)
    specs = ModelSpecs(
        model_key="acme x1", brand="Acme", model="X1", specs={"resolution": "1920x1080"},
        sources=["https://e.x"], variant_match="exact",
    )
    cache.put_specs("acme x1|h", specs)
    clock.now += 365 * 24 * 3600
    assert cache.get_specs("acme x1|h") == specs
    assert cache.get_specs("acme x1|other") is None


def test_stale_payload_is_ignored(tmp_path):
    cache = make_cache(tmp_path)
    with cache._db() as conn:
        conn.execute(
            "INSERT INTO model_specs (model_key, created_at, payload) VALUES (?, ?, ?)",
            ("bad", 0, '{"specs": "very bright"}'),
        )
    assert cache.get_specs("bad") is None


def _result(hunt_id: str, title: str, created_at: str) -> HuntResult:
    listing = make_detail(make_summary("1", title, 20.0))
    return HuntResult(
        hunt_id=hunt_id,
        request="reflex camera",
        budget_eur=50,
        created_at=created_at,
        plan=SearchPlan(product_summary="p", queries=["q"], budget_eur=50),
        ranked=[
            RankedEntry(
                rank=1,
                listing=listing,
                ranking=RankedListing(
                    item_id="1",
                    score=80,
                    verdict="top_pick",
                    meets_hard_requirements=True,
                    headline="h",
                    value_rationale="r",
                ),
            )
        ],
        cost=CostReport(total_usd=0.5),
    )


def test_saved_hunts_round_trip_and_list_newest_first(tmp_path):
    hid_old = "a" * 32
    hid_new = "b" * 32
    save_hunt(tmp_path, hid_old, _result(hid_old, "Old", "2026-01-01T10:00:00+00:00"))
    save_hunt(tmp_path, hid_new, _result(hid_new, "New", "2026-02-01T10:00:00+00:00"))

    loaded = load_hunt(tmp_path, hid_new)
    assert loaded is not None and loaded.ranked[0].listing.title == "New"
    assert load_hunt(tmp_path, "c" * 32) is None

    summaries = list_hunts(tmp_path)
    assert [s["hunt_id"] for s in summaries] == [hid_new, hid_old]
    assert summaries[0]["top_title"] == "New"
    assert summaries[0]["total_usd"] == 0.5


def test_list_hunts_skips_malformed_files(tmp_path):
    good = "d" * 32
    save_hunt(tmp_path, good, _result(good, "Good", "2026-03-01T10:00:00+00:00"))
    directory = tmp_path / "hunts"
    (directory / ("e" * 32 + ".json")).write_text("{not json", encoding="utf-8")
    (directory / ("f" * 32 + ".json")).write_text("[1, 2, 3]", encoding="utf-8")
    (directory / ("0" * 32 + ".json")).write_text('{"ranked": "oops", "created_at": "2026-01-01"}', encoding="utf-8")
    (directory / ("1" * 32 + ".json")).write_text('{"ranked": [{"listing": 5}]}', encoding="utf-8")

    summaries = list_hunts(tmp_path)
    assert [s["hunt_id"] for s in summaries] == [good]


def test_list_hunts_without_directory_is_empty(tmp_path):
    assert list_hunts(tmp_path) == []


def test_list_hunts_uses_the_file_name_as_the_hunt_id(tmp_path):
    name = "9" * 32
    save_hunt(tmp_path, name, _result("0" * 32, "Renamed inside", "2026-04-01T10:00:00+00:00"))
    summaries = list_hunts(tmp_path)
    assert [s["hunt_id"] for s in summaries] == [name]


def test_every_cache_operation_closes_its_connection(tmp_path, monkeypatch):
    import sqlite3

    opened = []
    real_connect = sqlite3.connect

    def tracking_connect(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        opened.append(conn)
        return conn

    monkeypatch.setattr("vinted_hunter.cache.sqlite3.connect", tracking_connect)
    cache = make_cache(tmp_path)
    cache.put_extraction(Extraction(item_id="1", brand="A", model="B", model_confidence=0.5), "m", "h")
    cache.get_extraction("1", "m", "h")
    cache.put_specs("a b|h", ModelSpecs(model_key="a b", brand="A", model="B"))
    cache.get_specs("a b|h")
    assert opened, "expected connections to be opened"
    for conn in opened:
        with pytest.raises(sqlite3.ProgrammingError):
            conn.execute("select 1")  # raises only once the connection is closed


def test_cache_write_rolls_back_on_error(tmp_path):
    cache = make_cache(tmp_path)
    with pytest.raises(RuntimeError):
        with cache._db() as conn:
            conn.execute("INSERT INTO model_specs (model_key, created_at, payload) VALUES ('x', 0, '{}')")
            raise RuntimeError("boom")
    assert cache.get_specs("x") is None


def test_search_page_cache_round_trip_and_thirty_minute_expiry(tmp_path):
    clock = Clock()
    cache = make_cache(tmp_path, clock)
    listings = [make_summary("1", "Appareil photo", 30.0)]
    cache.put_search_page("k", listings)
    assert cache.get_search_page("k") == listings
    clock.now += 29 * 60
    assert cache.get_search_page("k") == listings
    clock.now += 2 * 60
    assert cache.get_search_page("k") is None
    assert cache.get_search_page("other") is None


def test_model_specs_saved_before_the_generic_spec_format_are_a_miss(tmp_path):
    cache = HuntCache(tmp_path / "cache.sqlite3")
    legacy = {"brand": "Acme", "model": "X1", "native_resolution": "1920x1080", "lookup_mode": "web"}
    conn = sqlite3.connect(cache.path)
    try:
        conn.execute(
            "INSERT INTO model_specs (model_key, created_at, payload) VALUES (?, ?, ?)",
            ("acme x1|h", 0.0, json.dumps({**legacy, "model_key": "acme x1"})),
        )
        conn.commit()
    finally:
        conn.close()
    assert cache.get_specs("acme x1|h") is None  # looked up again, never read as empty specs


def test_model_specs_without_a_variant_match_load_as_unknown(tmp_path):
    cache = HuntCache(tmp_path / "cache.sqlite3")
    payload = {"brand": "Acme", "model": "X1", "specs": {"resolution": "1280x720"}, "lookup_mode": "web"}
    conn = sqlite3.connect(cache.path)
    try:
        conn.execute(
            "INSERT INTO model_specs (model_key, created_at, payload) VALUES (?, ?, ?)",
            ("acme x1|h", time.time(), json.dumps({**payload, "model_key": "acme x1"})),
        )
        conn.commit()
    finally:
        conn.close()
    specs = cache.get_specs("acme x1|h")
    assert specs is not None
    assert specs.variant_match == "unknown"
    assert specs.specs == {"resolution": "1280x720"}


def test_exact_web_specs_with_values_never_expire(tmp_path):
    clock = Clock()
    cache = make_cache(tmp_path, clock)
    specs = ModelSpecs(model_key="k", specs={"ram": "8 GB"}, variant_match="exact")
    cache.put_specs("k|a", specs)
    clock.now += 10 * 365 * 24 * 3600
    assert cache.get_specs("k|a") == specs


def test_non_exact_specs_are_a_hit_within_the_ttl_and_a_miss_after_it(tmp_path):
    clock = Clock()
    cache = HuntCache(tmp_path / "cache.sqlite3", clock=clock, specs_ttl_s=72 * 3600)
    specs = ModelSpecs(model_key="k", specs={"ram": "8 GB"}, variant_match="base_model")
    cache.put_specs("k|a", specs)
    clock.now += 71 * 3600
    assert cache.get_specs("k|a") == specs
    clock.now += 2 * 3600
    assert cache.get_specs("k|a") is None


def test_unknown_variant_specs_expire_like_base_model_specs(tmp_path):
    clock = Clock()
    cache = HuntCache(tmp_path / "cache.sqlite3", clock=clock, specs_ttl_s=3600)
    cache.put_specs("k|a", ModelSpecs(model_key="k", specs={"ram": "8 GB"}, variant_match="unknown"))
    clock.now += 3601
    assert cache.get_specs("k|a") is None


def test_no_specs_found_expires_after_the_ttl_even_for_an_exact_variant(tmp_path):
    clock = Clock()
    cache = HuntCache(tmp_path / "cache.sqlite3", clock=clock, specs_ttl_s=72 * 3600)
    cache.put_specs("k|a", ModelSpecs(model_key="k", specs={}, variant_match="exact"))
    clock.now += 71 * 3600
    assert cache.get_specs("k|a") is not None
    clock.now += 2 * 3600
    assert cache.get_specs("k|a") is None


def test_model_knowledge_specs_expire_even_when_they_hold_values(tmp_path):
    clock = Clock()
    cache = HuntCache(tmp_path / "cache.sqlite3", clock=clock, specs_ttl_s=3600)
    specs = ModelSpecs(
        model_key="k", specs={"ram": "8 GB"}, variant_match="exact", lookup_mode="model_knowledge"
    )
    cache.put_specs("k|a", specs)
    clock.now += 3601
    assert cache.get_specs("k|a") is None


def test_the_default_specs_ttl_is_seventy_two_hours(tmp_path):
    from vinted_hunter.cache import NONEXACT_SPECS_TTL_S

    assert NONEXACT_SPECS_TTL_S == 72 * 3600
