import json
from dataclasses import replace
from types import SimpleNamespace

import pytest
from helpers import FakeLLM, FakeVinted, build_world, make_detail, make_summary

from vinted_hunter.cache import HuntCache
from vinted_hunter.config import Settings
from vinted_hunter.llm import (
    NO_KEY_MESSAGE,
    WEB_SEARCH_UNAVAILABLE_EVIDENCE,
    LLMConfigError,
    LLMError,
)
from vinted_hunter.models import (
    Extraction,
    HardRequirement,
    HuntOptions,
    HuntResult,
    IdentifyDraft,
    ModelSpecs,
    RequirementCheck,
    SearchPlan,
)
from vinted_hunter.pipeline import (
    BLOCKED_MESSAGE,
    HUNT_COUNTERS,
    SPECS_FAILED,
    SPECS_LIMIT,
    SPECS_NOT_FOUND,
    SPECS_RESEARCHED,
    HuntError,
    _extraction_digest,
    _Hunt,
    _identify,
    _identify_query_key,
    _json_for_prompt,
    _noop,
    _prepare_plan,
    _rank_item,
    _specs_key,
    model_key,
    run_hunt,
    spec_verdict,
    specs_cache_key,
)
from vinted_hunter.vinted import VintedBlocked, VintedError, VintedNotFound, VintedParseError


@pytest.fixture
def settings(tmp_path):
    return Settings(
        anthropic_api_key="test-key",
        data_dir=tmp_path / "data",
        vinted_min_delay_s=0,
        max_detailed=10,
        max_candidates=50,
        search_pages=2,
        max_photos=2,
    )


@pytest.fixture
def plan():
    return SearchPlan(
        product_summary="reflex camera, 24 MP",
        queries=["appareil photo reflex", "reflex 24 MP", "appareil photo reflex"],
        catalog_ids=[],
        budget_eur=50,
        spec_keys=["resolution", "sensor size"],
        hard_requirements=["at least 20 MP"],
        nice_to_haves=["bright"],
        exclude_keywords=["objectif seul"],
        pitfalls=["'Wi-Fi compatible' is not built in"],
        triage_guidance="prefer at least 20 MP",
    )


async def test_full_hunt_ranks_filters_and_reports(settings, plan):
    vinted, llm = build_world(plan)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    events = []
    result = await run_hunt(
        "reflex camera 24 MP",
        50,
        HuntOptions(search_pages=2),
        events.append,
        settings=settings,
        vinted=vinted,
        llm=llm,
        cache=cache,
        hunt_id="a" * 32,
    )

    assert isinstance(result, HuntResult)
    assert result.hunt_id == "a" * 32
    assert [e.listing.id for e in result.ranked] == ["a1", "d1"]
    assert [e.rank for e in result.ranked] == [1, 2]
    # a1 has no confirmed requirement (unverified), so the verdict is capped at "ok" whatever the score.
    assert result.ranked[0].ranking.verdict == "ok"
    assert result.ranked[0].specs is not None
    assert result.ranked[0].specs.specs == {"resolution": "24 MP", "sensor size": "APS-C"}
    assert result.ranked[1].specs is None

    rejected = {(r.item_id, r.stage) for r in result.rejected}
    assert ("c1", "search") in rejected
    assert ("f1", "triage") in rejected
    assert ("g1", "details") in rejected
    assert ("e1", "extract") in rejected
    excluded = next(r for r in result.rejected if r.item_id == "c1")
    assert "objectif seul" in excluded.reason

    assert result.stats["search_over_budget"] == 1
    assert result.stats["search_duplicate"] == 2
    assert result.stats["candidates"] == 5
    assert result.stats["ranked"] == 2

    # The plan was normalised: duplicate queries removed, budget set.
    assert result.plan.queries == ["appareil photo reflex", "reflex 24 MP"]
    assert result.plan.budget_eur == 50

    types = [e["type"] for e in events]
    assert types[0] == "stage" and types[-1] == "result"
    stages_started = [e["stage"] for e in events if e["type"] == "stage" and e["status"] == "start"]
    assert stages_started == ["plan", "search", "triage", "details", "extract", "specs", "rank"]
    assert any(e["type"] == "progress" and e["stage"] == "search" for e in events)
    assert any(e["type"] == "log" for e in events)

    # Rank payload: unidentified listing says so; identified one carries specs.
    payload = {item["item_id"]: item for item in llm.rank_payloads[0]}
    assert payload["d1"]["specs"] == "unidentified"
    assert payload["a1"]["specs"]["specs"]["resolution"] == "24 MP"


async def test_second_run_refetches_availability_but_reuses_extractions(settings, plan):
    vinted, llm = build_world(plan)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    detail_calls = len(vinted.detail_calls)
    specs_calls = len(llm.specs_calls)
    extract_calls = list(llm.extract_calls)

    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    # Item pages are always fetched live, so availability is never stale.
    assert len(vinted.detail_calls) == detail_calls + 4
    assert len(llm.specs_calls) == specs_calls  # model specs came from the cache
    # a1 and d1 hit the extraction cache; only e1 (never cached: it failed) is analysed again.
    assert llm.extract_calls[len(extract_calls) :] == ["e1"]


async def test_stale_cached_listing_that_sold_is_not_extracted(settings, plan):
    vinted, llm = build_world(plan)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)

    # The listing sold in between. The live page now says so.
    a1 = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    vinted.details["a1"] = make_detail(a1, brand="Acme", can_buy=False, is_reserved=True)
    llm2 = FakeLLM(plan=plan, extractions=llm.extractions)
    result = await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm2, cache=cache)

    assert "a1" not in [e.listing.id for e in result.ranked]
    assert ("a1", "details") in {(r.item_id, r.stage) for r in result.rejected}
    assert "a1" not in llm2.extract_calls


async def test_same_listing_under_a_different_plan_is_analysed_again(settings, plan):
    vinted, llm = build_world(plan)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    assert "a1" in llm.extract_calls

    other_plan = plan.model_copy(update={"hard_requirements": [HardRequirement(text=s) for s in ["Wi-Fi 6 built in"]], "pitfalls": ["no 24 MP"]})
    llm2 = FakeLLM(plan=other_plan, extractions=llm.extractions)
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm2, cache=cache)
    assert "a1" in llm2.extract_calls


async def test_failed_triage_batch_rejects_only_its_own_listings(settings, plan):
    listings = [make_summary(f"s{i:02d}", f"Appareil photo {i}", 20.0) for i in range(30)]
    vinted = FakeVinted(
        pages={("appareil photo reflex", 1): listings},
        details={s.id: make_detail(s) for s in listings},
    )
    llm = FakeLLM(plan=plan, raise_triage_for={"s00"})  # first batch of 25 fails
    result = await run_hunt(
        "req", 50, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    triage_rejected = [r for r in result.rejected if r.stage == "triage"]
    assert {r.item_id for r in triage_rejected} == {f"s{i:02d}" for i in range(25)}
    assert all("triage failed" in r.reason for r in triage_rejected)
    # The second batch (5 listings) carried on normally.
    assert sorted(e.listing.id for e in result.ranked) == [f"s{i:02d}" for i in range(25, 30)]


async def test_unmet_extraction_check_caps_and_resorts(settings, plan):
    vinted, llm = build_world(plan)
    plan = plan.model_copy(update={"hard_requirements": [HardRequirement(text=s) for s in ["at least 20 MP", "working viewfinder"]]})
    llm.plan = plan
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"requirement_checks": [RequirementCheck(requirement="working viewfinder", status="unmet", evidence="port broken", hard_requirement_no=2)]}
    )
    llm.scores = {"a1": 90, "d1": 40}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert [e.listing.id for e in result.ranked] == ["d1", "a1"]
    capped = result.ranked[1]
    assert capped.ranking.score == 30
    assert capped.ranking.verdict == "avoid"
    assert capped.ranking.meets_hard_requirements is False


async def test_ranker_flag_alone_does_not_trigger_the_cap(settings, plan):
    vinted, llm = build_world(plan)
    llm.scores = {"a1": 90, "d1": 40}
    llm.unmet = {"a1"}  # the ranker says "does not meet", but no extraction check is unmet
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    top = result.ranked[0]
    assert top.listing.id == "a1" and top.ranking.score == 90
    assert top.ranking.meets_hard_requirements is True  # recorded from the code's own check


async def test_unknown_requirement_is_a_penalty_not_a_cap(settings, plan):
    vinted, llm = build_world(plan)
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"requirement_checks": [RequirementCheck(requirement="at least 20 MP", status="unknown", evidence="not stated", hard_requirement_no=1)]}
    )
    llm.scores = {"a1": 90, "d1": 40}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    top = result.ranked[0]
    assert top.listing.id == "a1" and top.ranking.score == 90
    assert top.ranking.verdict != "avoid"
    assert top.ranking.meets_hard_requirements is True


async def test_search_blocked_by_vinted_fails_the_hunt(settings, plan):
    vinted, llm = build_world(plan)
    vinted.search_exc = VintedBlocked("HTTP 403 after 3 retries")
    events = []
    with pytest.raises(HuntError, match="Vinted is blocking requests, try again later"):
        await run_hunt(
            "req", 50, None, events.append,
            settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )
    assert events[-1] == {"type": "error", "message": BLOCKED_MESSAGE}


async def test_block_with_nothing_loaded_fails_the_hunt(settings, plan):
    vinted, llm = build_world(plan)
    vinted.detail_errors = [VintedBlocked("HTTP 429 after 3 retries")]
    events = []
    with pytest.raises(HuntError, match="Vinted is blocking requests, try again later"):
        await run_hunt(
            "req", 50, None, events.append,
            settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )
    assert len(vinted.detail_calls) == 1  # no further item page fetches after the block
    assert events[-1] == {"type": "error", "message": BLOCKED_MESSAGE}


async def test_block_after_some_details_keeps_the_partial_result_and_warns(settings, plan):
    vinted, llm = build_world(plan)
    vinted.detail_errors = [None, VintedBlocked("HTTP 429 after 3 retries")]  # first succeeds, second blocked
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert len(vinted.detail_calls) == 2
    assert any("Vinted blocked requests during details" in w for w in result.warnings)
    skipped = [r for r in result.rejected if "blocking requests" in r.reason]
    assert len(skipped) == 2
    assert len(result.ranked) >= 1


async def test_no_listing_loaded_for_other_reasons_is_a_clear_failure(settings, plan):
    vinted, llm = build_world(plan)
    vinted.detail_errors = [VintedNotFound("HTTP 404")] * 4  # not a blocking signal
    with pytest.raises(HuntError, match="Could not load any of the 4 selected listings"):
        await run_hunt(
            "req", 50, None, None,
            settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )


async def test_no_selected_listings_is_an_empty_success_not_a_failure(settings, plan):
    vinted = FakeVinted(pages={("appareil photo reflex", 1): []})
    llm = FakeLLM(plan=plan)
    result = await run_hunt(
        "req", 50, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert result.ranked == []


async def test_three_consecutive_detail_failures_trip_the_breaker(settings, plan):
    vinted, llm = build_world(plan)
    vinted.detail_errors = [VintedError("connection reset")] * 3
    with pytest.raises(HuntError, match="Vinted is blocking requests, try again later"):
        await run_hunt(
            "req", 50, None, None,
            settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )
    assert len(vinted.detail_calls) == 3


async def test_not_found_items_do_not_trip_the_breaker(settings, plan):
    vinted, llm = build_world(plan)
    vinted.detail_errors = [VintedNotFound("HTTP 404")] * 3
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert len(vinted.detail_calls) == 4  # every selected listing was still attempted
    assert result.warnings == []


async def test_every_search_page_unreadable_means_format_changed(settings, plan):
    vinted, llm = build_world(plan)
    vinted.search_errors = {
        "appareil photo reflex": VintedParseError("no items array"),
        "reflex 24 MP": VintedParseError("no items array"),
    }
    with pytest.raises(HuntError, match="Vinted page format changed, parser needs updating"):
        await run_hunt(
            "req", 50, None, None,
            settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )


async def test_one_unreadable_search_page_is_a_warning(settings, plan):
    vinted, llm = build_world(plan)
    vinted.search_errors = {"reflex 24 MP": VintedParseError("no items array")}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert any("could not be read" in w for w in result.warnings)
    assert result.ranked  # the other query still produced results


async def test_model_knowledge_specs_are_not_cached(settings, plan):
    vinted, llm = build_world(plan)
    llm.knowledge_only = True
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    result = await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    assert cache.get_specs(specs_cache_key("acme x1", plan.spec_keys)) is None
    assert any("model knowledge" in w for w in result.warnings)

    llm2 = FakeLLM(plan=plan, extractions=llm.extractions)
    llm2.knowledge_only = True
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm2, cache=cache)
    assert llm2.specs_calls.count("acme x1") == 1  # looked up again, not served from the cache


async def test_unusable_brand_still_researches_the_confident_model(settings, plan):
    # The brand is optional: a confident model is researched by its name even when the brand sanitises to nothing.
    vinted, llm = build_world(plan)
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(update={"brand": "!!!"})
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert llm.specs_calls == ["x1"]
    assert [e.listing.id for e in result.ranked][0] == "a1"


async def test_brand_and_model_are_sanitised_before_lookup(settings, plan):
    vinted, llm = build_world(plan)
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"brand": "  Acme\n", "model": "X1 <b>bold</b>"}
    )
    await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert llm.specs_calls == ["acme x1 bbold b"]  # angle brackets stripped, not turned into words


async def test_planner_failure_emits_error_and_raises(settings, plan):
    vinted, llm = build_world(plan)
    llm.fail_plan = True
    events = []
    with pytest.raises(LLMError, match="planner exploded"):
        await run_hunt(
            "req",
            50,
            None,
            events.append,
            settings=settings,
            vinted=vinted,
            llm=llm,
            cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )
    assert events[-1] == {"type": "error", "message": "planner exploded"}
    assert vinted.search_calls == []


async def test_ranker_failure_fails_the_hunt(settings, plan):
    vinted, llm = build_world(plan)
    llm.fail_rank = True
    events = []
    with pytest.raises(LLMError, match="ranker exploded"):
        await run_hunt(
            "req",
            50,
            None,
            events.append,
            settings=settings,
            vinted=vinted,
            llm=llm,
            cache=HuntCache(settings.data_dir / "cache.sqlite3"),
        )
    assert events[-1]["type"] == "error"


async def test_missing_api_key_gives_clear_error(settings, plan):
    vinted, llm = build_world(plan)
    llm.configured = False
    events = []
    with pytest.raises(LLMConfigError) as info:
        await run_hunt("req", 50, None, events.append, settings=settings, vinted=vinted, llm=llm)
    assert str(info.value) == NO_KEY_MESSAGE
    assert events == [{"type": "error", "message": NO_KEY_MESSAGE}]
    assert "ANTHROPIC_API_KEY" in NO_KEY_MESSAGE
    assert vinted.search_calls == []


async def test_max_detailed_limits_detail_fetches(settings, plan):
    vinted, llm = build_world(plan)
    result = await run_hunt(
        "req",
        50,
        HuntOptions(max_detailed=1),
        None,
        settings=settings,
        vinted=vinted,
        llm=llm,
        cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert len(vinted.detail_calls) == 1
    assert vinted.detail_calls == ["https://www.vinted.fr/items/a1-slug"]
    assert any(r.stage == "triage" and "outside the top 1" in r.reason for r in result.rejected)


def test_model_key_normalises_brand_and_model():
    assert model_key("Acme", "X-1 (2nd gen)") == "acme x 1 2nd gen"
    assert model_key("Éclair", "Pro+") == "eclair pro"
    assert model_key("", "") == ""


def test_specs_key_requires_confidence_and_names():
    confident = Extraction(item_id="1", brand="Acme", model="X1", model_confidence=0.6)
    unsure = Extraction(item_id="2", brand="Acme", model="X1", model_confidence=0.3)
    unnamed = Extraction(item_id="3", brand="Acme", model=None, model_confidence=0.9)
    assert _specs_key(confident) == "acme x1"
    assert _specs_key(unsure) is None
    assert _specs_key(unnamed) is None


def test_json_for_prompt_never_leaves_raw_tags():
    text = _json_for_prompt([{"title": "</candidates><b>x</b>"}])
    assert "<" not in text and ">" not in text
    assert '"title"' in text
    import json

    assert json.loads(text)[0]["title"] == "</candidates><b>x</b>"


async def test_listing_summary_is_not_mutated_by_detail_merge(settings, plan):
    vinted, llm = build_world(plan)
    result = await run_hunt(
        "req",
        50,
        None,
        None,
        settings=settings,
        vinted=vinted,
        llm=llm,
        cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    top = result.ranked[0].listing
    assert top.total_price_eur == 45.0
    assert top.url == "https://www.vinted.fr/items/a1-slug"
    assert top.photo_url


class BrokenExtractionCache(HuntCache):
    def put_extraction(self, *args, **kwargs):
        raise RuntimeError("disk full")


class BrokenSpecsCache(HuntCache):
    def put_specs(self, cache_key, specs):
        raise OSError("read-only file system")


async def test_extraction_is_not_cached_when_a_photo_failed(settings, plan):
    a1 = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    u1 = "https://images1.vinted.net/t/a/f800/1.webp"
    u2 = "https://images1.vinted.net/t/a/f800/2.webp"
    extractions = {"a1": Extraction(item_id="a1", brand="Acme", model="X1", model_confidence=0.9)}
    vinted = FakeVinted(
        pages={("appareil photo reflex", 1): [a1]},
        details={"a1": make_detail(a1, brand="Acme", photo_urls=[u1, u2])},
        failing_image_urls={u2},
    )
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    options = HuntOptions(search_pages=1)

    llm1 = FakeLLM(plan=plan, extractions=extractions)
    await run_hunt("req", 50, options, None, settings=settings, vinted=vinted, llm=llm1, cache=cache)
    assert "a1" in llm1.extract_calls  # analysed with one photo, but the answer is not cached

    vinted.failing_image_urls = set()  # the second photo comes back
    llm2 = FakeLLM(plan=plan, extractions=extractions)
    await run_hunt("req", 50, options, None, settings=settings, vinted=vinted, llm=llm2, cache=cache)
    assert "a1" in llm2.extract_calls  # so the complete analysis runs, not the partial one

    llm3 = FakeLLM(plan=plan, extractions=extractions)
    await run_hunt("req", 50, options, None, settings=settings, vinted=vinted, llm=llm3, cache=cache)
    assert "a1" not in llm3.extract_calls  # the complete analysis is now cached


async def test_extraction_digest_covers_upload_date_and_seller_rating(settings, plan):
    vinted, llm = build_world(plan)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)

    a1 = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    vinted.details["a1"] = make_detail(a1, brand="Acme", upload_info="Il y a 2 semaines")
    llm2 = FakeLLM(plan=plan, extractions=llm.extractions)
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm2, cache=cache)
    assert "a1" in llm2.extract_calls

    vinted.details["a1"] = make_detail(a1, brand="Acme", upload_info="Il y a 2 semaines", seller_review_count=3)
    llm3 = FakeLLM(plan=plan, extractions=llm.extractions)
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm3, cache=cache)
    assert "a1" in llm3.extract_calls


async def test_unexpected_extraction_cache_write_rejects_only_that_item(settings, plan):
    vinted, llm = build_world(plan)
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm,
        cache=BrokenExtractionCache(settings.data_dir / "cache.sqlite3"),
    )
    reasons = {r.item_id: r.reason for r in result.rejected if r.stage == "extract"}
    assert "disk full" in reasons["a1"]
    assert "disk full" in reasons["d1"]
    assert "vision model overloaded" in reasons["e1"]  # the fake's own failure, unchanged


async def test_unexpected_specs_cache_write_keeps_the_specs_and_warns(settings, plan):
    vinted, llm = build_world(plan)
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm,
        cache=BrokenSpecsCache(settings.data_dir / "cache.sqlite3"),
    )
    top = result.ranked[0]
    assert top.listing.id == "a1"
    assert top.specs is not None and top.specs.specs["resolution"] == "24 MP"
    assert any("could not be stored" in w for w in result.warnings)


async def test_unmet_requirement_check_caps_the_listing_even_if_the_ranker_disagrees(settings, plan):
    vinted, llm = build_world(plan)
    plan = plan.model_copy(update={"hard_requirements": [HardRequirement(text=s) for s in ["at least 20 MP", "working viewfinder"]]})
    llm.plan = plan
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"requirement_checks": [RequirementCheck(requirement="working viewfinder", status="unmet", evidence="port broken", hard_requirement_no=2)]}
    )
    llm.scores = {"a1": 90, "d1": 40}  # the fake ranker says meets_hard_requirements=True
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert [e.listing.id for e in result.ranked] == ["d1", "a1"]
    capped = result.ranked[1]
    assert capped.ranking.score == 30
    assert capped.ranking.verdict == "avoid"
    assert capped.ranking.meets_hard_requirements is False


def test_extraction_draft_schema_does_not_ask_the_model_for_item_id():
    from vinted_hunter.models import ExtractionDraft

    assert "item_id" not in ExtractionDraft.model_json_schema()["properties"]


async def test_unexpected_error_in_one_details_task_rejects_only_that_item(settings, plan, monkeypatch):
    import vinted_hunter.pipeline as pipeline

    vinted, llm = build_world(plan)
    real = pipeline._with_summary

    def flaky(detail, summary):
        if summary.id == "e1":
            raise RuntimeError("boom in merge")
        return real(detail, summary)

    monkeypatch.setattr(pipeline, "_with_summary", flaky)
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    rejected = {(r.item_id, r.stage): r.reason for r in result.rejected}
    assert "boom in merge" in rejected[("e1", "details")]
    assert [e.listing.id for e in result.ranked] == ["a1", "d1"]


async def test_dropped_item_fields_are_aggregated_into_warnings(settings, plan):
    vinted, llm = build_world(plan)
    a1 = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    d1 = make_summary("d1", "Appareil photo Beta B2 Full HD", 30.0)
    vinted.details["a1"] = make_detail(a1, brand="Acme", dropped_fields=["description", "attributes"])
    vinted.details["d1"] = make_detail(d1, dropped_fields=["description"])
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert "2 listings had missing descriptions" in " ".join(result.warnings)
    assert "1 listing had missing condition, colour and upload date" in " ".join(result.warnings)


def test_excluded_keyword_matches_whole_words_and_ignores_accents():
    from vinted_hunter.pipeline import _matches_excluded

    # "support" must not match "supporté", but must match a plain "Support" as a word.
    assert _matches_excluded("Appareil photo 4K supporté", ["support"]) is None
    assert _matches_excluded("Support appareil photo", ["support"]) == "support"
    # Phrases match as phrases, and accents and case are ignored on both sides.
    assert _matches_excluded("Support MURAL appareil photo", ["support mural"]) == "support mural"
    assert _matches_excluded("Support mural appareil photo", ["support  mural"]) == "support  mural"
    assert _matches_excluded("Coque seule pour tablette 10 pouces", ["coque seule"]) == "coque seule"
    assert _matches_excluded("Appareil photo avec objectif seul", ["objectif seul seule"]) is None
    assert _matches_excluded("Objectif seul pour reflex", ["objectif seul"]) == "objectif seul"
    assert _matches_excluded("Filtre de rechange Acme", ["filtre de rechange"]) == "filtre de rechange"
    assert _matches_excluded("Appareil photo 24 MP", ["", "   "]) is None


async def test_planner_phrase_exclusions_keep_good_listings(settings, plan):

    phrase_plan = plan.model_copy(update={"exclude_keywords": ["support mural", "boite vide"]})
    listings = [
        make_summary("g1", "Appareil photo 4K supporté WiFi", 40.0),
        make_summary("x1", "Support mural appareil photo", 10.0),
    ]
    vinted = FakeVinted(
        pages={("appareil photo reflex", 1): listings},
        details={s.id: make_detail(s) for s in listings},
    )
    llm = FakeLLM(plan=phrase_plan)
    result = await run_hunt(
        "req", 50, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    excluded = {r.item_id for r in result.rejected if r.stage == "search"}
    assert excluded == {"x1"}
    assert "g1" in llm.extract_calls  # the 4K "supporté" listing survives and reaches vision


# --- Round five: search interleaving, catalog retry, relevance-only triage, tri-state status ---

from dataclasses import replace as _replace  # noqa: E402

from vinted_hunter.models import RequirementCheck as _Check  # noqa: E402
from vinted_hunter.pipeline import requirements_status  # noqa: E402


async def test_round_robin_interleaves_queries_before_the_cap(settings, plan):
    qa = [make_summary(f"a{i}", f"Appareil photo A{i}", 20.0) for i in range(5)]
    qb = [make_summary(f"b{i}", f"Appareil photo B{i}", 20.0) for i in range(5)]
    vinted = FakeVinted(
        pages={("qa", 1): qa, ("qb", 1): qb},
        details={s.id: make_detail(s) for s in qa + qb},
    )
    plan2 = plan.model_copy(update={"queries": ["qa", "qb"], "exclude_keywords": []})
    llm = FakeLLM(plan=plan2)
    result = await run_hunt(
        "req", 50, HuntOptions(search_pages=1, max_detailed=10), None,
        settings=_replace(settings, max_candidates=4), vinted=vinted, llm=llm,
        cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert result.stats["candidates from: qa"] == 2
    assert result.stats["candidates from: qb"] == 2
    assert result.stats["search_over_cap"] == 6
    assert set(llm.extract_calls) == {"a0", "a1", "b0", "b1"}


async def test_empty_catalog_search_is_retried_without_the_filter(settings, plan):
    listing = make_summary("c1", "Appareil photo Beta", 20.0)
    vinted = FakeVinted(
        pages={("appareil photo reflex", 1): [listing]},
        details={"c1": make_detail(listing)},
        empty_with_catalog={"appareil photo reflex"},
    )
    plan2 = plan.model_copy(update={"queries": ["appareil photo reflex"], "catalog_ids": [1234], "exclude_keywords": []})
    llm = FakeLLM(plan=plan2)
    await run_hunt(
        "req", 50, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert [catalog for _q, _p, catalog in vinted.catalog_calls] == [(1234,), ()]
    assert "c1" in llm.extract_calls


async def test_selection_follows_relevance_not_the_keep_flag(settings, plan):
    vinted, llm = build_world(plan)
    llm.keep_flag_wrong = {"d1": 4}  # keep=false but relevance 4: still selected
    llm.relevance = {"a1": 9, "d1": 4, "e1": 6, "g1": 5, "f1": 8}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert "d1" in {e.listing.id for e in result.ranked}


async def test_relevance_below_three_is_rejected_even_when_keep_is_true(settings, plan):
    vinted, llm = build_world(plan)
    llm.relevance = {"a1": 9, "d1": 2, "e1": 6, "g1": 5}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    reasons = {r.item_id: r.reason for r in result.rejected if r.stage == "triage"}
    assert "relevance 2" in reasons["d1"]
    assert "d1" not in llm.extract_calls


def _plan_with(requirements, spec_keys=("resolution",)):
    """A plan with the given hard requirements: strings are unbound, HardRequirement objects are bound."""
    return SearchPlan(
        product_summary="p", queries=["q"], budget_eur=50, spec_keys=list(spec_keys), hard_requirements=requirements
    )


def _extraction_with(checks, confidence=0.9):
    """checks: (hard_requirement_no, status) pairs."""
    return Extraction(
        item_id="x",
        model_confidence=confidence,
        requirement_checks=[_Check(requirement=f"check {n}", status=st, evidence="", hard_requirement_no=n) for n, st in checks],
    )


def _specs_with(value, variant_match="exact", key="resolution", lookup_mode="web"):
    return ModelSpecs(model_key="k", specs={key: value}, variant_match=variant_match, lookup_mode=lookup_mode)


def _bound(text, spec_key, comparator, target):
    return HardRequirement(text=text, spec_key=spec_key, comparator=comparator, target=target)


MEGAPIXELS_20 = _bound("at least 20 MP", "resolution", ">=", "20 MP")


def test_requirements_status_all_met_is_verified():
    plan = _plan_with(["Wi-Fi", "budget"])
    assert requirements_status(plan, _extraction_with([(1, "met"), (2, "met")]), None) == "verified"


def test_any_unknown_makes_it_unverified():
    plan = _plan_with(["Wi-Fi", "budget"])
    assert requirements_status(plan, _extraction_with([(1, "met"), (2, "unknown")]), None) == "unverified"


def test_any_unmet_makes_it_failed():
    plan = _plan_with(["Wi-Fi", "budget"])
    assert requirements_status(plan, _extraction_with([(1, "unmet"), (2, "met")]), None) == "failed"


def test_missing_check_counts_as_unknown():
    plan = _plan_with(["Wi-Fi", "budget"])
    assert requirements_status(plan, _extraction_with([(2, "met")]), None) == "unverified"


def test_a_later_unmet_check_for_the_same_number_is_never_ignored():
    plan = _plan_with(["Wi-Fi"])
    assert requirements_status(plan, _extraction_with([(1, "met"), (1, "unmet")]), None) == "failed"
    assert requirements_status(plan, _extraction_with([(1, "unmet"), (1, "met")]), None) == "failed"


def test_several_checks_for_one_number_are_met_only_when_every_one_is_met():
    plan = _plan_with(["Wi-Fi"])
    assert requirements_status(plan, _extraction_with([(1, "unknown"), (1, "met")]), None) == "unverified"
    assert requirements_status(plan, _extraction_with([(1, "met"), (1, "unknown")]), None) == "unverified"
    assert requirements_status(plan, _extraction_with([(1, "met"), (1, "met")]), None) == "verified"


def test_rank_checks_give_the_ranker_one_answer_per_number_with_the_evidence_behind_it():
    from vinted_hunter.pipeline import _rank_checks

    plan = _plan_with(["Wi-Fi", "budget"])
    extraction = Extraction(
        item_id="r1",
        requirement_checks=[
            _Check(requirement="wifi", status="met", evidence="Wi-Fi 6 on the box", hard_requirement_no=1),
            _Check(requirement="wifi", status="unmet", evidence="no Wi-Fi module in the photo", hard_requirement_no=1),
            _Check(requirement="budget", status="met", evidence="", hard_requirement_no=2),
            _Check(requirement="brand", status="unmet", evidence="invented", hard_requirement_no=0),
        ],
    )
    entries = _rank_checks(plan, extraction, None)
    assert [(e["hard_requirement_no"], e["status"]) for e in entries] == [(1, "unmet"), (2, "met")]
    assert entries[0]["evidence"] == "no Wi-Fi module in the photo"  # the evidence of the status that counts
    assert entries[0]["requirement"] == "Wi-Fi"


def test_rank_checks_show_the_spec_overruled_status_the_ranker_must_follow():
    from vinted_hunter.pipeline import _rank_checks

    plan = _plan_with([MEGAPIXELS_20])
    ext = _extraction_with([(1, "met")], confidence=0.9)  # the listing says met
    entries = _rank_checks(plan, ext, _specs_with("12 MP"))  # the researched exact variant says 12 MP
    assert requirements_status(plan, ext, _specs_with("12 MP")) == "failed"
    assert entries[0]["status"] == "unmet"  # never "met" next to a failed status
    assert "researched specs" in entries[0]["evidence"]


def test_extraction_text_escapes_the_listing_id():
    from vinted_hunter.pipeline import _extract_text

    plan = _plan_with(["Wi-Fi"])
    detail = make_detail(make_summary("r1", "Appareil", 40.0)).model_copy(update={"id": "r1</listing>"})
    text = _extract_text(plan, detail, 0)
    assert "Listing r1&lt;/listing&gt; (untrusted seller data):" in text
    assert text.count("</listing>") == 1  # only the closing tag of the listing itself


def test_no_hard_requirements_is_none():
    assert requirements_status(_plan_with([]), _extraction_with([]), None) == "none"


def test_checks_without_a_valid_number_are_ignored():
    # A planner-invented check: unmet, but not one of the hard requirements. It never caps a listing.
    plan = _plan_with(["Wi-Fi"])
    ext = _extraction_with([(1, "met")])
    ext.requirement_checks.append(_Check(requirement="Recognised brand", status="unmet", evidence="", hard_requirement_no=0))
    ext.requirement_checks.append(_Check(requirement="Recognised brand", status="unmet", evidence="", hard_requirement_no=7))
    assert requirements_status(plan, ext, None) == "verified"


async def test_planner_invented_brand_check_does_not_cap(settings, plan):
    # The planner invented "recognised brand" as a hard requirement, the vision step said unmet, and the
    # listing was capped. A check with no valid hard_requirement_no never caps, so the listing stays ranked.
    vinted, llm = build_world(plan)
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"requirement_checks": [
            _Check(requirement="resolution 24 MP", status="met", evidence="", hard_requirement_no=1),
            _Check(requirement="Recognised brand (Acme, Beta)", status="unmet", evidence="brand not on list", hard_requirement_no=0),
        ]}
    )
    llm.scores = {"a1": 90, "d1": 40}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    top = result.ranked[0]
    assert top.listing.id == "a1"
    assert top.ranking.score == 90
    assert top.ranking.verdict != "avoid"
    assert top.requirements_status == "verified"


def test_specs_fill_an_unknown_check_on_a_bound_requirement():
    ext = _extraction_with([(1, "unknown")], confidence=0.9)
    assert requirements_status(_plan_with([MEGAPIXELS_20]), ext, _specs_with("24 MP")) == "verified"
    assert requirements_status(_plan_with([MEGAPIXELS_20]), ext, _specs_with("12 MP")) == "failed"


def test_an_unbound_requirement_is_never_settled_by_specs():
    # The same wording as plain text has no spec_key, so the specs do not apply to it.
    plan = _plan_with(["at least 20 MP"])
    ext = _extraction_with([(1, "unknown")], confidence=0.9)
    assert requirements_status(plan, ext, _specs_with("12 MP")) == "unverified"


def test_a_spec_for_another_key_does_not_settle_the_requirement():
    plan = _plan_with([MEGAPIXELS_20], spec_keys=("resolution", "ram"))
    ext = _extraction_with([(1, "unknown")], confidence=0.9)
    assert requirements_status(plan, ext, _specs_with("8 GB", key="ram")) == "unverified"


def test_conflicting_spec_values_do_not_settle_the_requirement():
    ext = _extraction_with([(1, "unknown")])
    assert requirements_status(_plan_with([MEGAPIXELS_20]), ext, _specs_with("24 MP / 12 MP")) == "unverified"
    assert requirements_status(_plan_with([MEGAPIXELS_20]), ext, _specs_with("24 MP (sensor), 12 MP (effective)")) == "unverified"


def test_specs_fill_a_missing_check_only_with_a_settled_value():
    plan = _plan_with([MEGAPIXELS_20, "budget"])
    ext = _extraction_with([(2, "met")])
    assert requirements_status(plan, ext, _specs_with("12 MP")) == "failed"
    assert requirements_status(plan, ext, _specs_with("24 MP")) == "verified"


def test_specs_never_turn_an_explicit_vision_unmet_into_met():
    # The listing may be a non-default configuration, which the specs of the default model do not describe.
    plan = _plan_with([MEGAPIXELS_20])
    for confidence in (0.9, 0.7):
        assert requirements_status(plan, _extraction_with([(1, "unmet")], confidence=confidence), _specs_with("24 MP")) == "failed"
        assert requirements_status(plan, _extraction_with([(1, "unmet")], confidence=confidence), _specs_with("12 MP")) == "failed"


def test_specs_overrule_an_explicit_vision_met_only_at_high_confidence():
    plan = _plan_with([MEGAPIXELS_20])
    assert requirements_status(plan, _extraction_with([(1, "met")], confidence=0.9), _specs_with("12 MP")) == "failed"
    assert requirements_status(plan, _extraction_with([(1, "met")], confidence=0.85), _specs_with("12 MP")) == "failed"
    assert requirements_status(plan, _extraction_with([(1, "met")], confidence=0.84), _specs_with("12 MP")) == "verified"
    assert requirements_status(plan, _extraction_with([(1, "met")], confidence=0.9), _specs_with("24 MP")) == "verified"


def test_specs_can_fail_an_unknown_check_only_from_the_0_7_fill_threshold():
    plan = _plan_with([MEGAPIXELS_20])
    # 0.6 is "plausible, not confirmed" in the identify prompt: it does not fill a vision unknown.
    assert requirements_status(plan, _extraction_with([(1, "unknown")], confidence=0.6), _specs_with("12 MP")) == "unverified"
    assert requirements_status(plan, _extraction_with([(1, "unknown")], confidence=0.6), _specs_with("24 MP")) == "unverified"
    # 0.7 is the first confidence that fills, met or unmet.
    assert requirements_status(plan, _extraction_with([(1, "unknown")], confidence=0.7), _specs_with("12 MP")) == "failed"
    assert requirements_status(plan, _extraction_with([(1, "unknown")], confidence=0.7), _specs_with("24 MP")) == "verified"
    assert requirements_status(plan, _extraction_with([(1, "unknown")], confidence=0.69), _specs_with("12 MP")) == "unverified"


def test_base_model_and_model_knowledge_specs_never_change_a_status():
    plan = _plan_with([MEGAPIXELS_20])
    unknown = _extraction_with([(1, "unknown")], confidence=0.9)
    explicit = _extraction_with([(1, "unmet")], confidence=0.9)
    base = _specs_with("12 MP", variant_match="base_model")
    assert requirements_status(plan, unknown, base) == "unverified"
    assert requirements_status(plan, explicit, base) == "failed"
    knowledge = _specs_with("12 MP", lookup_mode="model_knowledge")
    assert requirements_status(plan, unknown, knowledge) == "unverified"
    assert requirements_status(plan, explicit, knowledge) == "failed"


def test_spec_rule_needs_a_confident_identification_to_fill():
    plan = _plan_with([MEGAPIXELS_20])
    ext = _extraction_with([(1, "unmet")], confidence=0.5)
    assert requirements_status(plan, ext, _specs_with("24 MP")) == "failed"


async def test_ranked_entries_carry_the_requirements_status_from_checks_and_specs(settings, plan):
    vinted, llm = build_world(plan)
    llm.plan = plan.model_copy(update={"hard_requirements": [MEGAPIXELS_20]})
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"requirement_checks": [_Check(requirement="at least 20 MP", status="unknown", evidence="not stated", hard_requirement_no=1)]}
    )
    llm.scores = {"a1": 90, "d1": 40}
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    top = result.ranked[0]
    assert top.listing.id == "a1"
    # The fake specs state 24 MP, which settles the unknown check on the bound requirement.
    assert top.requirements_status == "verified"
    assert top.ranking.score == 90
    assert top.ranking.meets_hard_requirements is True


def test_vision_context_lists_hard_requirements_only():
    from vinted_hunter.pipeline import _extract_context

    plan = SearchPlan(
        product_summary="reflex camera", queries=["q"], budget_eur=50,
        hard_requirements=[MEGAPIXELS_20, "Wi-Fi"],
        nice_to_haves=["Marque reconnue (Acme, Beta)"],
        pitfalls=["upgradable is not included"],
    )
    context = _extract_context(plan)
    assert "1. at least 20 MP" in context
    assert "2. Wi-Fi" in context
    assert "Marque reconnue" not in context
    assert "Nice" not in context


# --- Spec verdicts: units, dimensions, names, lists and conflicts --------------------------------


@pytest.mark.parametrize(
    "value, comparator, target, verdict",
    [
        # Storage and memory: units convert inside the family, and only inside it.
        ("16 GB", ">=", "16 GB", "met"),
        ("8 GB", ">=", "16 GB", "unmet"),
        ("16 GB", "==", "32 GB", "unmet"),
        ("16 GB", "==", "16 GB", "met"),
        ("1 TB", ">=", "512 GB", "met"),
        ("512 GB", ">=", "1 TB", "unmet"),
        ("1 teraoctet", ">=", "1000 GB", "met"),  # French tera-octet, spelled out
        ("1 To", ">=", "1000 GB", "unknown"),  # "To" is not a unit here: French "to" also means "to", as in a range
        ("16 Go", ">=", "16 GB", "met"),
        ("16GB", ">=", "16 gb", "met"),
        ("16 GB", ">=", "16", "unknown"),  # the target has no unit: never guessed
        ("16", ">=", "16 GB", "unknown"),  # the value has no unit
        ("16 GB", ">=", "2 kg", "unknown"),  # another family
        ("16 GB / 32 GB", ">=", "16 GB", "unknown"),  # two values: fail closed
        ("16 GB, expandable to 32 GB", ">=", "16 GB", "unknown"),
        ("16 GB", ">=", "16 GB or 32 GB", "unknown"),  # a target with two values cannot be compared
        ("", ">=", "16 GB", "unknown"),
        ("inconnu", ">=", "16 GB", "unknown"),
        # Mass, length, frequency, light.
        ("1,2 kg", "<=", "1.5 kg", "met"),  # decimal comma
        ("1 500 g", "<=", "1.2 kg", "unknown"),  # a spaced group before g: 1.5 kg or 1500 g, never guessed
        ("1 920 px", ">=", "1080 px", "met"),  # a spaced group before a thousands-safe unit is one number
        ("1920 px", ">=", "1920 pixels", "met"),  # px is a pixel unit
        ("1000 px", ">=", "1080 px", "unmet"),
        ("16 512 GB", ">=", "16 GB", "unknown"),  # two groups: ambiguous, never 16512 GB
        ("1 920x1080", ">=", "1080p", "unknown"),  # a spaced group before x is ambiguous
        ("15,6 pouces", ">=", "15 in", "met"),
        ("3.5 GHz", ">=", "3000 MHz", "met"),
        ("2400 MHz", ">=", "2.4 GHz", "met"),
        ("2000 lm", ">=", "1500 lumens", "met"),
        # Thousands: "5.000" is 5000 before mAh, Hz, W, px or lm; before any other unit the group is ambiguous.
        ("5.000 mAh", ">=", "5000 mAh", "met"),
        ("5.000 mAh", "<=", "4000 mAh", "unmet"),
        ("1.000 Hz", ">=", "900 Hz", "met"),
        ("1.200 W", ">=", "1000 W", "met"),
        ("1,200 kg", "<=", "1.5 kg", "unknown"),  # 1.2 kg or 1200 kg
        ("1.200 kg", ">=", "1 kg", "unknown"),
        ("1 kg", "<=", "1.200 kg", "unknown"),  # an ambiguous target
        ("2.400 MHz", ">=", "2000 MHz", "unknown"),  # MHz is not a unit with a plausible fraction of three digits
        ("1.2 kg", "<=", "1.5 kg", "met"),  # one or two decimals are never thousands
        # Ranges name several values, so they are never one measure.
        ("16 to 32 GB", ">=", "16 GB", "unknown"),
        ("16-32 GB", ">=", "16 GB", "unknown"),
        ("16 à 32 GB", ">=", "16 GB", "unknown"),
        ("16 GB", ">=", "16 to 32 GB", "unknown"),
        # Dimensions compare per dimension, and an unknown unit in the value is never matched.
        ("30 x 20 x 10 cm", "<=", "32 x 22 x 12 cm", "met"),
        ("35 x 20 x 10 cm", "<=", "32 x 22 x 12 cm", "unmet"),
        ("30 x 20 x 10", "<=", "32 x 22 x 12 cm", "unknown"),  # the value has no unit
        ("1080x1920", ">=", "1920x1080", "unmet"),
        ("1920x1080", ">=", "1920x1080x3", "unknown"),
        # Physical sizes have no orientation, pixels do.
        ("20 x 30 cm", "<=", "30 x 20 cm", "met"),
        ("20 x 30 cm", ">=", "30 x 20 cm", "met"),
        ("20 x 30 cm", "==", "30 x 20 cm", "met"),
        ("35 x 20 cm", "<=", "30 x 20 cm", "unmet"),  # sorted: 35 is still more than 30
        ("20 x 30 px", "<=", "30 x 20 px", "unmet"),
        ("20x30", "<=", "30x20", "unmet"),
        # Resolutions: WxH, p suffix, names, frame-rate suffixes, and two sizes in one value.
        ("1920x1080", ">=", "1080p", "met"),
        ("1280x720", ">=", "1080p", "unmet"),
        ("1920 x 1080 (Full HD)", ">=", "1080p", "met"),
        ("Full HD", ">=", "1080p", "met"),
        ("4K (3840 x 2160)", ">=", "1080p", "met"),
        ("2560x1440", ">=", "4K", "unmet"),
        ("1920x1080", "==", "1080p", "met"),
        ("1080p60Hz", ">=", "1080p", "met"),
        ("Max input 3840x2160, native 1920x1080", ">=", "1080p", "unknown"),
        ("1920x1080", ">=", "1080p or 4K", "unknown"),
        ("24 MP", ">=", "20 MP", "met"),
        ("18 MP", ">=", "20 MP", "unmet"),
        ("24 MP", ">=", "1080p", "unknown"),  # megapixels and a resolution are different families
        # "contains": a word or number of the target appears in the value.
        ("EU 42", "contains", "42", "met"),
        ("EU 41", "contains", "42", "unmet"),
        ("EU 36 to 46", "contains", "42", "unknown"),  # a range can hide the target
        ("EU 42, 43", "contains", "43", "unknown"),  # a list: every alternative must contain the target
        ("EU 42, 43", "contains", "42", "unknown"),
        ("EU 41, 42", "contains", "43", "unknown"),  # a list does not prove the target is absent
        ("16 GB / 16 GB Plus", "contains", "16 GB", "met"),  # every alternative contains the target
        ("16 GB / 32 GB", "contains", "16 GB", "unknown"),  # two values from duplicate spec entries: never met from one
        ("16 GB / 32 GB", "contains", "32 GB", "unknown"),
        ("EU 41-42", "contains", "42", "unknown"),  # a range
        ("Bluetooth 5.0", "contains", "Bluetooth 5", "met"),  # numbers are equal numerically
        ("5.0", "contains", "5", "met"),
        ("5", "contains", "5.0", "met"),
        ("Bluetooth 5.2", "contains", "Bluetooth 5", "unknown"),  # 5.2 is not 5, but it shares a word: unknown
        ("USB Type-C", "contains", "USB-C", "met"),  # hyphens and spaces, one spelling of USB-C
        ("USB-C", "contains", "usb type-c", "met"),
        ("Type-C", "contains", "USBC", "met"),
        ("DDR4", "contains", "DDR4", "met"),
        ("DDR4", "contains", "4", "unknown"),  # "ddr4" is not the number 4
        ("DDR3", "contains", "DDR4", "unknown"),  # shares the word DDR: unknown rather than unmet
        ("16 GB", "contains", "32 GB", "unmet"),  # one clean value: a shared unit word alone is not a shared word
        ("32 GB", "contains", "16 GB", "unmet"),
        ("16 GB RAM", "contains", "32 GB", "unmet"),
        ("32 kg", "contains", "16 GB", "unknown"),  # a unit from another family is not comparable
        ("16 GB", "contains", "16 TB", "unknown"),  # the shared number 16 keeps it unknown
        ("512 Go", "contains", "512 GB", "met"),  # French and English spellings of a unit are the same unit
        ("512 Go", "contains", "256 GB", "unmet"),
        ("Micro USB", "contains", "USB-C", "unmet"),  # a clean value with a different connector
        ("42,5", "contains", "42", "unmet"),  # 42.5 is not 42
        ("Cuir", "contains", "cuir", "met"),
        ("Textile", "contains", "cuir", "unmet"),
        ("cuir", "contains", "", "unknown"),
    ],
)
def test_spec_verdict_compares_values_by_unit_family_and_shape(value, comparator, target, verdict):
    requirement = _bound("requirement", "resolution", comparator, target)
    assert spec_verdict(_specs_with(value), requirement) == verdict


def test_spec_verdict_is_unknown_without_a_binding_or_a_spec():
    specs = _specs_with("24 MP")
    assert spec_verdict(specs, HardRequirement(text="at least 20 MP")) == "unknown"
    assert spec_verdict(specs, _bound("x", None, ">=", "20 MP")) == "unknown"
    assert spec_verdict(specs, _bound("x", "resolution", None, "20 MP")) == "unknown"
    assert spec_verdict(specs, _bound("x", "resolution", ">=", None)) == "unknown"
    assert spec_verdict(None, MEGAPIXELS_20) == "unknown"
    assert spec_verdict(specs, _bound("x", "ram", ">=", "20 MP")) == "unknown"


def test_spec_keys_match_accent_and_case_insensitively():
    specs = ModelSpecs(model_key="k", specs={"Résolution": "24 MP"}, variant_match="exact")
    assert spec_verdict(specs, _bound("x", "resolution", ">=", "20 MP")) == "met"


def test_two_spellings_of_one_key_with_different_values_fail_closed():
    specs = ModelSpecs(model_key="k", specs={"resolution": "24 MP", "Résolution": "12 MP"}, variant_match="exact")
    assert spec_verdict(specs, MEGAPIXELS_20) == "unknown"


# --- Binding a requirement to a spec key ---------------------------------------------------------


def test_plain_string_requirements_are_unbound():
    plan = SearchPlan(product_summary="p", queries=["q"], budget_eur=50, hard_requirements=["at least 20 MP"])
    assert plan.hard_requirements[0].text == "at least 20 MP"
    assert plan.hard_requirements[0].spec_key is None


def test_a_legacy_plan_with_string_requirements_still_loads():
    legacy = '{"product_summary": "p", "queries": ["q"], "budget_eur": 50, "hard_requirements": ["at least 20 MP"]}'
    plan = SearchPlan.model_validate_json(legacy)
    assert [r.text for r in plan.hard_requirements] == ["at least 20 MP"]


def test_null_catalog_ids_mean_no_filter():
    plan = SearchPlan.model_validate({"product_summary": "p", "queries": ["q"], "budget_eur": 50, "catalog_ids": None})
    assert plan.catalog_ids == []


def test_binding_to_an_unlisted_spec_key_is_dropped():
    plan = _plan_with([_bound("at least 16 GB RAM", "storage", ">=", "16 GB")], spec_keys=("ram",))
    _prepare_plan(plan)
    requirement = plan.hard_requirements[0]
    assert (requirement.spec_key, requirement.comparator, requirement.target) == (None, None, None)


def test_binding_takes_the_canonical_spelling_of_the_spec_key():
    plan = _plan_with([_bound("at least 16 GB RAM", "RAM", ">=", "16 GB")], spec_keys=("ram",))
    _prepare_plan(plan)
    assert plan.hard_requirements[0].spec_key == "ram"
    assert plan.hard_requirements[0].comparator == ">="


def test_spec_keys_are_deduplicated_capped_and_risk_terms_cleaned():
    plan = _plan_with([], spec_keys=["ram", "RAM", " ", *[f"key {i}" for i in range(10)]])
    plan.risk_terms = ["batterie HS", "batterie HS", "  ", "charniere cassee"]
    _prepare_plan(plan)
    assert plan.spec_keys[0] == "ram" and len(plan.spec_keys) == 8
    assert plan.risk_terms == ["batterie HS", "charniere cassee"]


def test_risk_terms_are_capped_in_count_and_length():
    plan = _plan_with([], spec_keys=["ram"])
    plan.risk_terms = [f"defaut {i}" for i in range(20)] + ["x" * 100, "  " + "y" * 45 + "  "]
    _prepare_plan(plan)
    assert len(plan.risk_terms) == 12
    assert plan.risk_terms[:2] == ["defaut 0", "defaut 1"]
    assert all(len(term) <= 40 for term in plan.risk_terms)


def test_risk_terms_longer_than_forty_characters_are_dropped_not_cut():
    plan = _plan_with([], spec_keys=["ram"])
    exact = "y" * 40
    plan.risk_terms = ["x" * 100, "  " + "z" * 41 + "  ", exact, "batterie HS"]
    _prepare_plan(plan)
    assert plan.risk_terms == [exact, "batterie HS"]  # no truncated fragment such as "x" * 40 survives
    plan.risk_terms = ["charniere " + "cassee " * 6]  # cut at 40 characters this would end mid-word
    _prepare_plan(plan)
    assert plan.risk_terms == []


async def test_identical_searches_within_thirty_minutes_are_served_from_cache(settings, plan):
    vinted, llm = build_world(plan)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    options = HuntOptions(search_pages=1)
    await run_hunt("req", 50, options, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    first = len(vinted.search_calls)
    events = []
    result = await run_hunt("req", 50, options, events.append, settings=settings, vinted=vinted, llm=FakeLLM(plan=plan), cache=cache)
    assert len(vinted.search_calls) == first  # no new Vinted search requests
    assert result.stats.get("search_cache_hits", 0) >= 1


async def test_planner_queries_are_capped_at_five(settings, plan):
    many = [f"requete {i}" for i in range(7)]
    listings = [make_summary(f"q{i}", f"Appareil photo {i}", 20.0) for i in range(7)]
    vinted = FakeVinted(
        pages={(q, 1): [listings[i]] for i, q in enumerate(many)},
        details={s.id: make_detail(s) for s in listings},
    )
    llm = FakeLLM(plan=plan.model_copy(update={"queries": many, "exclude_keywords": []}))
    await run_hunt("req", 50, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm,
                   cache=HuntCache(settings.data_dir / "cache.sqlite3"))
    assert len({q for q, _p in vinted.search_calls}) == 5


def test_request_volume_defaults_are_low():
    from vinted_hunter.config import Settings

    defaults = Settings()
    assert defaults.search_pages == 1
    assert defaults.vinted_min_delay_s == 2.5
    assert defaults.max_photos == 6


# --- Web identification of unclear models ---------------------------------------------------


def _unclear_brand_world(plan, titles):
    listings = [make_summary(f"u{i}", title, 25.0) for i, title in enumerate(titles)]
    vinted = FakeVinted(
        pages={("appareil photo reflex", 1): listings},
        details={s.id: make_detail(s) for s in listings},
    )
    extractions = {
        s.id: Extraction(item_id=s.id, brand="Beta", model=None, model_confidence=0.2, visual_description="white box, two lenses")
        for s in listings
    }
    llm = FakeLLM(plan=plan.model_copy(update={"queries": ["appareil photo reflex"], "exclude_keywords": []}), extractions=extractions)
    return vinted, llm, listings


async def test_web_identification_fills_an_unclear_model_and_feeds_the_specs_lookup(settings, plan):
    from vinted_hunter.models import IdentifyDraft

    vinted, llm, listings = _unclear_brand_world(plan, ["Appareil photo Beta B2 Full HD"])
    llm.identifications = {"Appareil photo Beta B2 Full HD": IdentifyDraft(model="B2", model_confidence=0.8, evidence="catalogue page")}
    result = await run_hunt(
        "req", 50, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert llm.identify_calls == ["Appareil photo Beta B2 Full HD"]
    assert "beta b2" in llm.specs_calls
    entry = result.ranked[0]
    assert entry.extraction.model == "B2" and entry.extraction.model_confidence == 0.8
    assert "web: catalogue page" in entry.extraction.identification_evidence
    assert entry.specs.specs["resolution"] == "24 MP"
    assert entry.ranking.model_verdict == "fake model verdict"


async def test_web_identification_below_threshold_is_not_used(settings, plan):
    from vinted_hunter.models import IdentifyDraft

    vinted, llm, _ = _unclear_brand_world(plan, ["Appareil photo Beta B2 Full HD"])
    llm.identifications = {"Appareil photo Beta B2 Full HD": IdentifyDraft(model="B2", model_confidence=0.4, evidence="maybe")}
    await run_hunt(
        "req", 50, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert llm.identify_calls == ["Appareil photo Beta B2 Full HD"]
    assert "beta b2" not in llm.specs_calls


async def test_web_identification_is_capped_per_hunt(settings, plan):
    titles = [f"Appareil photo Beta modele {i}" for i in range(16)]
    vinted, llm, _ = _unclear_brand_world(plan, titles)
    await run_hunt(
        "req", 50, HuntOptions(search_pages=1, max_detailed=20), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    assert len(llm.identify_calls) == 15  # IDENTIFY_MAX_PER_HUNT


async def test_identical_listing_text_is_identified_once_then_cached(settings, plan):
    from vinted_hunter.models import IdentifyDraft

    vinted, llm, _ = _unclear_brand_world(plan, ["Appareil photo Beta B2 Full HD"])
    llm.identifications = {"Appareil photo Beta B2 Full HD": IdentifyDraft(model="B2", model_confidence=0.8, evidence="x")}
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    await run_hunt("req", 50, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    llm2 = FakeLLM(plan=llm.plan, extractions=dict(llm.extractions))
    llm2.identifications = dict(llm.identifications)
    result = await run_hunt("req", 50, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm2, cache=cache)
    assert llm2.identify_calls == []
    assert result.stats["identify_cache_hits"] >= 1


# --- Listing gaps versus red flags -----------------------------------------------------------------

from vinted_hunter.llm import gaps_hint  # noqa: E402
from vinted_hunter.pipeline import split_spec_gaps  # noqa: E402


@pytest.mark.parametrize(
    "flag",
    [
        "Storage size not stated",
        "RAM not given",
        "Model number not shown",
        "Référence non confirmée",
        "Taille non précisée",
        "Caractéristiques non indiquées",
        "Specs unverified",
        "Capacity unknown",
        "Model label not visible; identity rests on seller text",
        "Modèle non visible sur les photos",
        "Aucune capacité indiquée",
        "Resolution unverified",
        "Résolution non vérifiée",
        "Storage not confirmed, used model unknown",  # a bare "used" is not a wear term
        "RAM not stated",
        "Storage claim not confirmed",  # a claim without a photo, label or sticker word
        "Storage not stated, the label does not show it",  # "does not show" is an absence, not a showing
        "Capacity unknown, photos not showing the model number",
        # Plain absences next to a photo word: "no photo shows", "doesn't show", "pas de photo", "aucune photo".
        "RAM not stated, no photo shows an 8 GB sticker",
        "Storage not stated, pas de photo de l'étiquette",
        "Storage unknown, photos don't show the label",
        "Capacity unknown, the label doesn't show it",
        "Storage unknown, aucune photo de la capacité",
        # Several clauses: one clause names the spec and its absence, the others carry no risk.
        "RAM 8 GB, storage not stated",
        "RAM 16.5 GB not stated",  # the dot is a decimal point, not a clause break
        "Storage not stated, seller shows no photo of the storage label",  # a negated photo, after "shows"
        "Capacité non précisée, le vendeur n'indique pas de photo de l'étiquette",
        "Seller claims 16 GB RAM, no label included",  # a negated label next to a claim is an absence
        # Ordinary words that the defect vocabulary must not catch.
        "Rustic design, storage not stated",
        "Yellow colour, storage not stated",
        "Moulded handle, storage not stated",
    ],
)
def test_spec_and_information_gaps_are_moved_out_of_red_flags(flag):
    flags, gaps = split_spec_gaps([flag])
    assert flags == [] and gaps == [flag]


@pytest.mark.parametrize(
    "flag",
    [
        "Accessories listed in text are not shown in photos",
        "Photo 2 label reads 16 GB, which contradicts the 32 GB claim",
        "Seller says a cable is missing",
        "Télécommande, câbles et adaptateurs non mentionnés ni visibles sur les photos",
        "Light scuffs on the glossy top",
        "Seller says the battery is not working as sold",
        # Spec words in the flag, but an item risk too: these are red flags, never gaps.
        "Modèle non authentique, risque de contrefaçon",
        "Capacity not stated, but the photos show a cracked screen",
        "Storage unknown, photos montrent un écran cassé",
        "Model not confirmed, RAM given as 16 GB but the label is worn",
        # Missing accessories are red flags in either word order, whatever spec word they sit next to.
        "Storage not stated, box missing",
        "RAM not given, charger not included",
        "Capacity unknown, la boîte manquante",
        "Résolution non précisée, chargeur absent",
        "Size unknown, the remote is missing",
        "Taille non précisée, le câble est manquant",
        "Storage unknown, batterie manquante",
        # Wear wording, FR and EN, accented and not: each is a red flag, never a gap.
        "Modèle non confirmé, produit usagé",
        "RAM not given, heavily used",
        "Storage unknown, sold as used",
        "Modèle non confirmé, état d'usage inconnu",
        "Model not confirmed, bien utilisé",
        "Size unknown, usagée",
        "Model not confirmed, well used",
        # Defects: not working, faulty, broken parts.
        "Capacity not stated, does not work",
        "Taille non précisée, batterie HS",
        "Model unknown, hors service à l'arrivée",
        "Storage unknown, the hinge is broken",
        # Contradictions between the photos and the text.
        "Storage not stated but the photos contradict the text",
        # A photo, label or sticker that shows, or contradicts, what the text claims: an item risk.
        "seller claims 16 GB RAM, photo shows an 8 GB sticker",
        "storage claim not confirmed; photo shows 128 GB label",
        "RAM claimed as 16 GB, screenshot shows 8 GB",
        "Capacity not stated, label does not match the listing",
        "Storage not stated, the photo differs from the text",
        "Storage unknown, les photos ne correspondent pas au texte",
        "RAM non confirmée, la photo montre une étiquette 8 Go",
        "Stockage non précisé, l'étiquette indique 128 Go",
        "Storage not stated, the sticker is inconsistent with the title",
        # A claim next to a photo, label or sticker word is an item risk even without a showing verb.
        "Storage claim unverified, label missing from photos",
        # Physical defects, EN and FR, next to a spec word that is absent.
        "storage has bad sectors, storage size not stated",
        "screen has dead pixels, resolution not stated",
        "battery does not hold charge, capacity not stated",
        "screen has ghosting, resolution not confirmed",
        "hinge dented, model number not shown",
        "Taille non précisée, secteurs défectueux",
        "Résolution non précisée, pixels morts à droite",
        "Capacité non indiquée, la batterie ne tient pas la charge",
        "Modèle non confirmé, coque bosselée",
        "Taille non précisée, charnière tordue",
        "Capacité inconnue, l'appareil ne s'allume pas",
        "Taille non précisée, bruit anormal au démarrage",
        "Taille inconnue, odeur de tabac",
        "Taille non précisée, écran jauni",
        "Modèle non confirmé, traces de moisi",
        "Capacité non précisée, coque ébréchée",
        "Résolution non précisée, image persistante à l'écran",
        "Storage not stated, the screen is cracked",
        "Capacity unknown, the frame has rust",
        "Storage not stated, the battery is swollen",
        "Size not stated, the hinge is loose",
        "RAM not stated, photo shows an 8 GB sticker",  # a showing photo is still a risk next to a plain gap
    ],
)
def test_item_and_transaction_flags_stay_red_flags(flag):
    flags, gaps = split_spec_gaps([flag])
    assert flags == [flag] and gaps == []


@pytest.mark.parametrize(
    "flag",
    [
        "storage listed in the text, the photos are not clear",  # the spec word and the absence are in different clauses
        "Capacité indiquée, les photos ne sont pas nettes",
    ],
)
def test_spec_word_and_absence_in_different_clauses_stay_red_flags(flag):
    flags, gaps = split_spec_gaps([flag])
    assert flags == [flag] and gaps == []


def test_the_reviewer_examples_for_defects_stay_red_and_the_plain_gaps_stay_gaps():
    red = [
        "storage has bad sectors, storage size not stated",
        "screen has dead pixels, resolution not stated",
        "battery does not hold charge, capacity not stated",
        "screen has ghosting, resolution not confirmed",
        "hinge dented, model number not shown",
    ]
    assert split_spec_gaps(red) == (red, [])
    plain = ["RAM not stated", "model number not shown"]
    assert split_spec_gaps(plain) == ([], plain)


@pytest.mark.parametrize(
    "flag",
    [
        "Seller uses a generic title",  # no spec term: a red flag by design, not a gap
        "Vendeur utilise un titre générique",
        "Box missing",  # no spec term either
        "Scratch on the lid",
    ],
)
def test_flags_without_a_spec_term_stay_red_flags(flag):
    flags, gaps = split_spec_gaps([flag])
    assert flags == [flag] and gaps == []


def test_plan_risk_terms_keep_a_category_defect_as_a_red_flag():
    # Without the plan's term, the flag reads as a spec gap: the shutter is named as unknown.
    flag = "Shutter count not stated, obturateur bloqué"
    assert split_spec_gaps([flag], spec_keys=["shutter count"]) == ([], [flag])
    # The category defect of the plan keeps it a red flag, accents and case aside.
    assert split_spec_gaps([flag], ["Obturateur BLOQUE"], ["shutter count"]) == ([flag], [])


def test_plan_spec_keys_make_their_words_spec_terms():
    flag = "Battery life not stated"
    assert split_spec_gaps([flag]) == ([flag], [])  # "battery" is not a generic spec word
    assert split_spec_gaps([flag], spec_keys=["battery life"]) == ([], [flag])


def test_risk_terms_match_whole_phrases_only():
    flag = "Model not confirmed, batterie HS"
    assert split_spec_gaps([flag], ["batterie"]) == ([flag], [])  # "batterie" alone, as a word, is a risk here too
    assert split_spec_gaps(["Model not confirmed, batteries"], ["batterie HS"]) == ([], ["Model not confirmed, batteries"])


def test_split_spec_gaps_keeps_the_order_of_each_part():
    flags, gaps = split_spec_gaps(["Scratch on lid", "Storage not stated", "Seller says cable missing"])
    assert flags == ["Scratch on lid", "Seller says cable missing"]
    assert gaps == ["Storage not stated"]


def test_gaps_hint_is_empty_without_gaps_and_names_them_otherwise():
    assert gaps_hint([]) == ""
    assert "Not shown in the listing" in gaps_hint(["storage not shown"])


async def test_listing_gaps_never_reach_red_flags_and_go_to_the_ranker_as_information(settings, plan):
    vinted, llm = build_world(plan)
    llm.extractions["a1"] = llm.extractions["a1"].model_copy(
        update={"red_flags": ["Resolution not stated", "Scratch on the lid"]}
    )
    result = await run_hunt(
        "req", 50, None, None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    entry = next(e for e in result.ranked if e.listing.id == "a1")
    assert entry.extraction.red_flags == ["Scratch on the lid"]
    assert entry.extraction.listing_gaps == ["Resolution not stated"]
    payload = {item["item_id"]: item for item in llm.rank_payloads[0]}
    assert payload["a1"]["info_not_in_listing"] == ["Resolution not stated"]
    assert payload["a1"]["red_flags"] == ["Scratch on the lid"]


# --- Variant matching -----------------------------------------------------------------------------


def test_variant_codes_make_different_cache_keys():
    assert model_key("Acme", "X1 470C (H3)") != model_key("Acme", "X1 470")
    assert model_key("Acme", "X1 470C") != model_key("Acme", "X1 470")
    assert model_key("Acme", "X1 470C") == "acme x1 470c"


def test_specs_cache_key_depends_on_the_set_of_spec_keys():
    base = specs_cache_key("acme x1", ["ram", "storage"])
    assert specs_cache_key("acme x1", ["Storage", "ram"]) == base  # order and case do not matter
    assert specs_cache_key("acme x1", ["ram"]) != base
    assert specs_cache_key("acme x2", ["ram", "storage"]) != base


def test_specs_cache_key_depends_on_the_prompt_version(monkeypatch):
    base = specs_cache_key("acme x1", ["ram", "storage"])
    monkeypatch.setattr("vinted_hunter.pipeline.PROMPT_VERSION", "99")
    assert specs_cache_key("acme x1", ["ram", "storage"]) != base
    assert base.startswith("acme x1|")


def test_a_legacy_specs_payload_without_the_specs_map_is_a_miss(settings):
    import sqlite3

    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    legacy = {"brand": "Acme", "model": "X1", "native_resolution": "1920x1080", "variant_match": "exact", "lookup_mode": "web"}
    with sqlite3.connect(cache.path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO model_specs (model_key, created_at, payload) VALUES (?, ?, ?)",
            ("acme x1|0", 0.0, json.dumps(legacy)),
        )
    assert cache.get_specs("acme x1|0") is None


# --- Identification: refusals are not cached, relevance order, escaping ---------------------------


def _identify_inputs(titles):
    summaries = [make_summary(f"u{i}", title, 25.0) for i, title in enumerate(titles)]
    details = {s.id: make_detail(s) for s in summaries}  # relevance order: the order of this dict
    extractions = {
        eid: Extraction(item_id=eid, brand="Beta", model=None, model_confidence=0.2, visual_description="white box")
        for eid in reversed(list(details))  # completion order is the opposite of relevance order
    }
    return extractions, details


def _identify_key(title):
    return _identify_query_key(f"Beta {title} white box")


def test_identify_key_depends_on_the_prompt_version(monkeypatch):
    before = _identify_key("Appareil photo Beta B2 Full HD")
    monkeypatch.setattr("vinted_hunter.pipeline.PROMPT_VERSION", "10")
    assert _identify_key("Appareil photo Beta B2 Full HD") != before


def test_extraction_key_depends_on_the_prompt_version(monkeypatch, plan):
    detail = make_detail(make_summary("a1", "Appareil photo Acme X1", 40.0))
    current = _extraction_digest("system", plan, detail, ["p1"])
    monkeypatch.setattr("vinted_hunter.pipeline.PROMPT_VERSION", "10")
    assert _extraction_digest("system", plan, detail, ["p1"]) != current


async def test_identify_answer_from_an_older_prompt_version_is_not_served(settings, plan, monkeypatch):
    title = "Appareil photo Beta B2 Full HD"
    extractions, details = _identify_inputs([title])
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    monkeypatch.setattr("vinted_hunter.pipeline.PROMPT_VERSION", "10")
    cache.put_identify(_identify_key(title), IdentifyDraft(model="B2", model_confidence=0.8, evidence="old prompt"))
    monkeypatch.undo()
    llm = FakeLLM(plan=plan, identifications={title: IdentifyDraft(model="B2", model_confidence=0.8, evidence="catalogue")})
    await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))
    assert llm.identify_calls == [title]  # the old answer is not served: the listing is identified again
    assert extractions["u0"].model == "B2" and "web: catalogue" in extractions["u0"].identification_evidence


async def test_low_confidence_identify_answers_do_not_cross_differing_listing_gaps(settings, plan):
    # The gaps are hints in the identify query. A low-confidence answer is reused only for the same gaps.
    title = "Appareil photo Beta B2 Full HD"
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    llm = FakeLLM(plan=plan, identifications={title: IdentifyDraft(model="B2", model_confidence=0.65, evidence="maybe")})

    async def identify_with(gaps):
        extractions, details = _identify_inputs([title])
        extractions["u0"] = extractions["u0"].model_copy(update={"listing_gaps": gaps})
        await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))

    await identify_with(["battery health not stated"])
    await identify_with(["model number not shown"])
    assert llm.identify_calls == [title, title]  # different gaps: identified again
    await identify_with(["Battery health not stated"])
    assert llm.identify_calls == [title, title]  # the same gaps, in another case: served from the cache


async def test_a_high_confidence_identification_is_reused_across_differing_gaps(settings, plan):
    title = "Appareil photo Beta B2 Full HD"
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    llm = FakeLLM(plan=plan, identifications={title: IdentifyDraft(model="B2", model_confidence=0.8, evidence="catalogue")})
    extractions, details = _identify_inputs([title])
    await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))

    extractions, details = _identify_inputs([title])
    extractions["u0"] = extractions["u0"].model_copy(update={"listing_gaps": ["model number not shown"]})
    hunt = _Hunt(_noop)
    await _identify(extractions, details, settings, llm, cache, hunt)
    assert llm.identify_calls == [title]  # the gaps-free brand and title fallback answered it
    assert extractions["u0"].model == "B2"
    assert hunt.stats["identify_cache_hits"] == 1


def test_identify_key_depends_on_the_listing_gaps():
    query = "Beta Appareil photo white box"
    assert _identify_query_key(query, ["storage not shown"]) != _identify_query_key(query)
    assert _identify_query_key(query, ["storage not shown", "box missing"]) == _identify_query_key(
        query, ["Box missing", "storage  not shown"]
    )


async def test_web_search_refusal_is_never_cached(settings, plan):
    extractions, details = _identify_inputs(["Appareil photo Beta B2 Full HD"])
    refusal = IdentifyDraft(model=None, model_confidence=0.0, evidence=WEB_SEARCH_UNAVAILABLE_EVIDENCE)
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    llm = FakeLLM(plan=plan, identifications={"Appareil photo Beta B2 Full HD": refusal})
    for _ in range(2):
        await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))
    assert llm.identify_calls == ["Appareil photo Beta B2 Full HD", "Appareil photo Beta B2 Full HD"]
    assert cache.get_identify(_identify_key("Appareil photo Beta B2 Full HD")) is None


async def test_a_refusal_stored_by_an_older_version_is_not_served(settings, plan):
    title = "Appareil photo Beta B2 Full HD"
    extractions, details = _identify_inputs([title])
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    cache.put_identify(_identify_key(title), IdentifyDraft(model=None, model_confidence=0.0, evidence=WEB_SEARCH_UNAVAILABLE_EVIDENCE))
    llm = FakeLLM(plan=plan, identifications={title: IdentifyDraft(model="B2", model_confidence=0.8, evidence="catalogue")})
    await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))
    assert llm.identify_calls == [title]
    assert extractions["u0"].model == "B2"


async def test_identify_cap_takes_the_most_relevant_listings_not_the_first_to_finish(settings, plan, monkeypatch):
    monkeypatch.setattr("vinted_hunter.pipeline.IDENTIFY_MAX_PER_HUNT", 4)
    titles = [f"Appareil photo Beta modele {i}" for i in range(6)]
    extractions, details = _identify_inputs(titles)
    llm = FakeLLM(plan=plan)
    await _identify(extractions, details, settings, llm, HuntCache(settings.data_dir / "cache.sqlite3"), _Hunt(_noop))
    assert sorted(llm.identify_calls) == sorted(titles[:4])  # the cap keeps the most relevant, in relevance order


# --- Specs cache: only exact-variant web specs --------------------------------------------------


async def test_web_specs_of_any_variant_are_cached_and_reused(settings, plan):
    for variant in ("base_model", "exact"):
        vinted, llm = build_world(plan)
        llm.specs_variant = variant
        cache = HuntCache(settings.data_dir / f"cache-{variant}.sqlite3")
        for _ in range(2):
            await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
        assert len(llm.specs_calls) == 1, variant  # the second hunt is a cache hit


async def test_a_cached_base_model_entry_within_its_lifetime_is_a_hit_that_never_settles_a_requirement(settings):
    plan = _laptop_plan()
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    cache.put_specs(
        specs_cache_key("acme pro 14", plan.spec_keys),
        ModelSpecs(model_key="acme pro 14", brand="Acme", model="Pro 14", specs={"ram": "32 GB"}, variant_match="base_model"),
    )
    vinted, llm = _one_listing_world(plan, _laptop_extraction(ram_status="unknown"), "Laptop Acme Pro 14")
    result = await run_hunt("req", 400, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    assert llm.specs_calls == []  # within its lifetime the base-model answer is not searched again
    assert result.stats["models_cached_nonexact"] == 1
    assert result.ranked[0].requirements_status == "unverified"  # the base model's 32 GB does not settle it
    payload = llm.rank_payloads[-1][0]
    assert payload["requirement_checks"][0]["status"] == "unknown"
    assert "note" in payload["specs"] and "ram" not in payload["specs"].get("specs", {})


async def test_a_cached_non_exact_entry_past_its_lifetime_is_researched_again(settings, plan):
    clock = _FakeClock()
    cache = HuntCache(settings.data_dir / "cache.sqlite3", clock=clock, specs_ttl_s=72 * 3600)
    cache.put_specs(
        specs_cache_key("acme x1", plan.spec_keys),
        ModelSpecs(model_key="acme x1", brand="Acme", model="X1", specs={"resolution": "18 MP"}, variant_match="base_model"),
    )
    clock.now += 73 * 3600
    vinted, llm = build_world(plan)
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    assert llm.specs_calls == ["acme x1"]


async def test_a_cached_exact_web_entry_is_still_a_hit(settings, plan):
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    cache.put_specs(
        specs_cache_key("acme x1", plan.spec_keys),
        ModelSpecs(model_key="acme x1", brand="Acme", model="X1", specs={"resolution": "24 MP"}, variant_match="exact"),
    )
    vinted, llm = build_world(plan)
    result = await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    assert llm.specs_calls == []
    assert result.stats["models_cached_exact"] >= 1


async def test_specs_researched_for_other_spec_keys_are_not_reused(settings, plan):
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    other_keys = specs_cache_key("acme x1", ["ram"])
    cache.put_specs(other_keys, ModelSpecs(model_key="acme x1", specs={"ram": "8 GB"}, variant_match="exact"))
    vinted, llm = build_world(plan)
    result = await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    assert llm.specs_calls == ["acme x1"]
    assert result.stats["models_cached_exact"] == 0


async def test_research_receives_the_plan_spec_keys(settings, plan):
    vinted, llm = build_world(plan)
    await run_hunt("req", 50, None, None, settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"))
    assert llm.spec_keys_seen and all(keys == tuple(plan.spec_keys) for keys in llm.spec_keys_seen)


# --- Rank payload: specs of a base or unknown model are withheld --------------------------------


def _rank_payload_for(specs, identified=True):
    detail = make_detail(make_summary("r1", "Appareil photo Acme", 40.0))
    extraction = Extraction(item_id="r1", brand="Acme", model="X1", model_confidence=0.9)
    return _rank_item(detail, extraction, specs, identified=identified, requirement_checks=[], specs_status="researched")


def test_rank_payload_keeps_the_spec_values_of_exact_web_specs():
    specs = ModelSpecs(
        model_key="k", specs={"resolution": "24 MP", "sensor size": "APS-C"}, summary="ok",
        variant_match="exact", sources=["https://e.x"],
    )
    payload = _rank_payload_for(specs)["specs"]
    assert payload["specs"] == {"resolution": "24 MP", "sensor size": "APS-C"}
    assert payload["summary"] == "ok"
    assert "sources" not in payload and "note" not in payload


@pytest.mark.parametrize(
    "specs",
    [
        ModelSpecs(model_key="k", specs={"resolution": "18 MP"}, variant_match="base_model", summary="base",
                   known_issues=["slow autofocus"], price_notes="eBay FR", typical_used_price_eur=30.0),
        ModelSpecs(model_key="k", specs={"resolution": "18 MP"}, variant_match="unknown"),
        ModelSpecs(model_key="k", specs={"resolution": "24 MP"}, variant_match="exact",
                   lookup_mode="model_knowledge", summary="from memory"),
    ],
)
def test_rank_payload_withholds_spec_values_and_notes_of_other_specs(specs):
    payload = _rank_payload_for(specs)["specs"]
    for name in ("specs", "summary", "known_issues", "price_notes"):
        assert name not in payload, name
    assert "do not state or cite any spec value" in payload["note"] or "withheld" in payload["note"]
    assert payload["variant_match"] == specs.variant_match


def test_rank_payload_keeps_the_typical_used_price_of_other_specs():
    specs = ModelSpecs(model_key="k", specs={"resolution": "18 MP"}, variant_match="base_model", typical_used_price_eur=30.0)
    assert _rank_payload_for(specs)["specs"]["typical_used_price_eur"] == 30.0


def test_rank_payload_says_unidentified_or_failed_lookup():
    assert _rank_payload_for(None, identified=False)["specs"] == "unidentified"
    assert _rank_payload_for(None, identified=True)["specs"] == "lookup failed or unavailable"


# --- End to end on product categories ----------------------------------------------------------
#
# Each scenario is a whole hunt: the plan's spec keys and bound requirement, the researched specs of the
# identified model, and the category's risk terms. The planner, vision, specs and ranker are fakes.


def _one_listing_world(plan, extraction, title, *, specs_by_model=None, price=300.0):
    listing = make_summary("l1", title, price)
    vinted = FakeVinted(
        pages={(plan.queries[0], 1): [listing]},
        details={"l1": make_detail(listing, brand=extraction.brand)},
    )
    llm = FakeLLM(
        plan=plan,
        extractions={"l1": extraction},
        specs_by_model=specs_by_model or {},
        relevance={"l1": 8},
        scores={"l1": 70},
    )
    return vinted, llm


async def _hunt_one(settings, vinted, llm, budget):
    return await run_hunt(
        "req", budget, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )


def _laptop_plan():
    return SearchPlan(
        product_summary="laptop with at least 16 GB RAM",
        queries=["ordinateur portable 16 go"],
        spec_keys=["ram", "storage", "cpu"],
        hard_requirements=[
            HardRequirement(text="at least 16 GB of RAM", spec_key="ram", comparator=">=", target="16 GB"),
            "total price within the budget",
        ],
        budget_eur=400,
    )


def _laptop_extraction(ram_status="unknown"):
    return Extraction(
        item_id="l1",
        brand="Acme",
        model="Pro 14",
        model_confidence=0.9,
        requirement_checks=[
            _Check(requirement="at least 16 GB of RAM", status=ram_status, evidence="", hard_requirement_no=1),
            _Check(requirement="total price", status="met", evidence="", hard_requirement_no=2),
        ],
    )


async def test_laptop_ram_requirement_is_verified_by_the_researched_specs(settings):
    plan = _laptop_plan()
    vinted, llm = _one_listing_world(
        plan, _laptop_extraction(), "Laptop Acme Pro 14",
        specs_by_model={"Pro 14": {"ram": "32 GB", "storage": "512 GB"}},
    )
    result = await _hunt_one(settings, vinted, llm, 400)
    top = result.ranked[0]
    assert top.requirements_status == "verified"
    assert top.ranking.meets_hard_requirements is True
    assert top.ranking.score == 70


async def test_laptop_with_too_little_ram_fails_the_requirement_and_is_capped(settings):
    plan = _laptop_plan()
    vinted, llm = _one_listing_world(
        plan, _laptop_extraction(), "Laptop Acme Pro 14",
        specs_by_model={"Pro 14": {"ram": "8 GB", "storage": "512 GB"}},
    )
    result = await _hunt_one(settings, vinted, llm, 400)
    top = result.ranked[0]
    assert top.requirements_status == "failed"
    assert top.ranking.score == 30 and top.ranking.verdict == "avoid"


async def test_laptop_whose_specs_do_not_state_ram_stays_unverified(settings):
    plan = _laptop_plan()
    vinted, llm = _one_listing_world(
        plan, _laptop_extraction(), "Laptop Acme Pro 14",
        specs_by_model={"Pro 14": {"storage": "512 GB"}},
    )
    result = await _hunt_one(settings, vinted, llm, 400)
    top = result.ranked[0]
    assert top.requirements_status == "unverified"
    assert top.ranking.score == 70  # a penalty is the ranker's call, not a cap


def _sneaker_plan():
    return SearchPlan(
        product_summary="sneakers in size 42",
        queries=["baskets acme runner 42"],
        spec_keys=["size", "material"],
        hard_requirements=[
            HardRequirement(text="size 42 EU", spec_key="size", comparator="contains", target="42"),
        ],
        budget_eur=80,
    )


def _sneaker_extraction():
    return Extraction(
        item_id="l1",
        brand="Acme",
        model="Runner",
        model_confidence=0.9,
        requirement_checks=[_Check(requirement="size 42 EU", status="unknown", evidence="", hard_requirement_no=1)],
    )


async def test_sneaker_size_contains_is_met_by_a_single_listed_size(settings):
    plan = _sneaker_plan()
    vinted, llm = _one_listing_world(
        plan, _sneaker_extraction(), "Baskets Acme Runner",
        specs_by_model={"Runner": {"size": "EU 42"}}, price=60.0,
    )
    result = await _hunt_one(settings, vinted, llm, 80)
    assert result.ranked[0].requirements_status == "verified"


async def test_sneaker_size_list_is_unknown_not_met_from_one_size(settings):
    # A spec that lists several sizes gives unknown for "contains" unless every size contains the target.
    plan = _sneaker_plan()
    vinted, llm = _one_listing_world(
        plan, _sneaker_extraction(), "Baskets Acme Runner",
        specs_by_model={"Runner": {"size": "EU 41, 42, 43"}}, price=60.0,
    )
    result = await _hunt_one(settings, vinted, llm, 80)
    assert result.ranked[0].requirements_status == "unverified"


async def test_sneaker_size_contains_is_failed_when_the_size_is_not_listed(settings):
    plan = _sneaker_plan()
    vinted, llm = _one_listing_world(
        plan, _sneaker_extraction(), "Baskets Acme Runner",
        specs_by_model={"Runner": {"size": "EU 41"}}, price=60.0,
    )
    result = await _hunt_one(settings, vinted, llm, 80)
    assert result.ranked[0].requirements_status == "failed"


async def test_sneaker_size_range_does_not_prove_the_size_is_missing(settings):
    plan = _sneaker_plan()
    vinted, llm = _one_listing_world(
        plan, _sneaker_extraction(), "Baskets Acme Runner",
        specs_by_model={"Runner": {"size": "EU 36 to 46"}}, price=60.0,
    )
    result = await _hunt_one(settings, vinted, llm, 80)
    assert result.ranked[0].requirements_status == "unverified"


def _camera_plan(risk_terms):
    return SearchPlan(
        product_summary="reflex camera with at least 20 MP",
        queries=["reflex 20 mp"],
        spec_keys=["resolution", "shutter count"],
        risk_terms=risk_terms,
        hard_requirements=[HardRequirement(text="at least 20 MP", spec_key="resolution", comparator=">=", target="20 MP")],
        budget_eur=200,
    )


def _camera_extraction():
    return Extraction(
        item_id="l1",
        brand="Acme",
        model="R5",
        model_confidence=0.9,
        red_flags=["Shutter count not stated, obturateur bloqué"],
        requirement_checks=[_Check(requirement="at least 20 MP", status="unknown", evidence="", hard_requirement_no=1)],
    )


async def test_camera_risk_term_from_the_plan_keeps_a_defect_as_a_red_flag(settings):
    plan = _camera_plan(["obturateur bloque"])
    vinted, llm = _one_listing_world(
        plan, _camera_extraction(), "Reflex Acme R5", specs_by_model={"R5": {"resolution": "24 MP"}}, price=150.0,
    )
    result = await _hunt_one(settings, vinted, llm, 200)
    top = result.ranked[0]
    assert top.extraction.red_flags == ["Shutter count not stated, obturateur bloqué"]
    assert top.extraction.listing_gaps == []
    assert top.requirements_status == "verified"


async def test_without_the_plan_risk_term_the_same_flag_is_only_a_spec_gap(settings):
    # The same listing under a plan with no risk terms: the shutter count is the only thing named, so it is a gap.
    plan = _camera_plan([])
    vinted, llm = _one_listing_world(
        plan, _camera_extraction(), "Reflex Acme R5", specs_by_model={"R5": {"resolution": "24 MP"}}, price=150.0,
    )
    result = await _hunt_one(settings, vinted, llm, 200)
    top = result.ranked[0]
    assert top.extraction.red_flags == []
    assert top.extraction.listing_gaps == ["Shutter count not stated, obturateur bloqué"]


async def test_risk_terms_are_part_of_the_extraction_cache_key(settings):
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    vinted, llm = _one_listing_world(_camera_plan(["obturateur bloque"]), _camera_extraction(), "Reflex Acme R5", price=150.0)
    await run_hunt("req", 200, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm, cache=cache)
    llm2 = FakeLLM(plan=_camera_plan([]), extractions={"l1": _camera_extraction()})
    await run_hunt("req", 200, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm2, cache=cache)
    assert llm2.extract_calls == ["l1"]  # the cached extraction, split with the other terms, is not reused


class _FakeClock:
    def __init__(self):
        self.now = 1_000_000.0

    def __call__(self):
        return self.now


async def test_listings_with_the_same_brand_and_title_share_one_identification(settings, plan):
    title = "Appareil photo Beta B2 Full HD"
    extractions, details = _identify_inputs([title, title])
    llm = FakeLLM(plan=plan, identifications={title: IdentifyDraft(model="B2", model_confidence=0.8, evidence="catalogue")})
    hunt = _Hunt(_noop)
    await _identify(extractions, details, settings, llm, HuntCache(settings.data_dir / "cache.sqlite3"), hunt)
    assert llm.identify_calls == [title]
    assert hunt.stats["identify_shared"] == 1
    assert extractions["u0"].model == "B2" and extractions["u1"].model == "B2"


async def test_a_low_confidence_identification_is_not_reused_for_other_features_or_gaps(settings, plan):
    title = "Appareil photo Beta B2 Full HD"
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    llm = FakeLLM(plan=plan, identifications={title: IdentifyDraft(model="B2", model_confidence=0.65, evidence="maybe")})
    extractions, details = _identify_inputs([title])
    await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))
    extractions, details = _identify_inputs([title])
    extractions["u0"] = extractions["u0"].model_copy(update={"visual_description": "black body"})
    await _identify(extractions, details, settings, llm, cache, _Hunt(_noop))
    assert llm.identify_calls == [title, title]


# --- Research of distinct models: the default, the optional limit, failed listings, the specs status ---------


def _model_plan():
    return SearchPlan(
        product_summary="camera with at least 20 MP",
        queries=["appareil photo reflex"],
        spec_keys=["resolution"],
        hard_requirements=[HardRequirement(text="at least 20 MP", spec_key="resolution", comparator=">=", target="20 MP")],
        budget_eur=500,
    )


def _model_extraction(eid, model, *, brand="Acme", red_flags=(), checks=()):
    return Extraction(
        item_id=eid, brand=brand, model=model, model_confidence=0.9,
        red_flags=list(red_flags), requirement_checks=list(checks),
    )


def _model_world(entries, specs_by_model=None):
    """One listing per entry (listing id, total price, triage relevance, extraction)."""
    plan = _model_plan()
    summaries = [
        make_summary(eid, f"Appareil photo {extraction.model or 'inconnu'}", price)
        for eid, price, _relevance, extraction in entries
    ]
    vinted = FakeVinted(
        pages={(plan.queries[0], 1): summaries},
        details={s.id: make_detail(s, brand="Acme") for s in summaries},
    )
    llm = FakeLLM(
        plan=plan,
        extractions={eid: extraction for eid, _price, _relevance, extraction in entries},
        relevance={eid: relevance for eid, _price, relevance, _extraction in entries},
        specs_by_model=specs_by_model or {},
    )
    return vinted, llm


async def _model_hunt(settings, vinted, llm, cache=None, **overrides):
    return await run_hunt(
        "req", 500, HuntOptions(search_pages=1), None,
        settings=replace(settings, **overrides), vinted=vinted, llm=llm,
        cache=cache or HuntCache(settings.data_dir / "cache.sqlite3"),
    )


def _statuses(llm):
    return {item["item_id"]: item["specs_status"] for item in llm.rank_payloads[-1]}


async def test_every_distinct_model_is_researched_once_by_default(settings):
    entries = [
        ("a1", 300, 9, _model_extraction("a1", "Alpha")),
        ("a2", 310, 8, _model_extraction("a2", "Alpha")),
        ("b1", 320, 7, _model_extraction("b1", "Beta")),
        ("c1", 330, 6, _model_extraction("c1", "Gamma")),
        ("d1", 340, 5, _model_extraction("d1", "Delta")),
        ("e1", 350, 4, _model_extraction("e1", "Epsilon")),
        ("f1", 360, 4, _model_extraction("f1", "Zeta")),
    ]
    vinted, llm = _model_world(entries)
    result = await _model_hunt(settings, vinted, llm)
    assert sorted(llm.specs_calls) == sorted({"acme alpha", "acme beta", "acme gamma", "acme delta", "acme epsilon", "acme zeta"})
    assert result.stats["models_researched"] == 6
    assert result.stats["models_skipped_limit"] == 0
    assert set(_statuses(llm).values()) == {SPECS_RESEARCHED}


async def test_the_optional_limit_researches_the_best_models_first_and_marks_the_rest(settings):
    entries = [
        ("a1", 300, 9, _model_extraction("a1", "Alpha")),
        ("b1", 310, 8, _model_extraction("b1", "Beta")),
        ("c1", 320, 7, _model_extraction("c1", "Gamma")),
        ("a2", 330, 6, _model_extraction("a2", "Alpha")),
    ]
    vinted, llm = _model_world(entries)
    result = await _model_hunt(settings, vinted, llm, specs_max_per_hunt=2)
    assert sorted(llm.specs_calls) == ["acme alpha", "acme beta"]
    assert _statuses(llm) == {
        "a1": SPECS_RESEARCHED, "b1": SPECS_RESEARCHED, "c1": SPECS_LIMIT, "a2": SPECS_RESEARCHED,
    }
    assert result.stats["models_researched"] == 2
    assert result.stats["models_skipped_limit"] == 1


async def test_with_a_limit_ties_go_to_fewer_red_flags_then_the_lower_price(settings):
    entries = [
        ("t1", 200, 8, _model_extraction("t1", "Alpha", red_flags=["scratched screen", "dented corner"])),
        ("t2", 350, 8, _model_extraction("t2", "Beta")),
        ("t3", 260, 8, _model_extraction("t3", "Gamma")),
    ]
    vinted, llm = _model_world(entries)
    await _model_hunt(settings, vinted, llm, specs_max_per_hunt=1)
    assert llm.specs_calls == ["acme gamma"]  # no red flags beats two red flags, then 260 EUR beats 350 EUR


async def test_a_cached_model_does_not_use_up_the_research_limit(settings):
    cache = HuntCache(settings.data_dir / "cache.sqlite3")
    cache.put_specs(
        specs_cache_key("acme beta", _model_plan().spec_keys),
        ModelSpecs(model_key="acme beta", brand="Acme", model="Beta", specs={"resolution": "24 MP"}, variant_match="exact"),
    )
    entries = [
        ("a1", 300, 9, _model_extraction("a1", "Alpha")),
        ("b1", 310, 8, _model_extraction("b1", "Beta")),
    ]
    vinted, llm = _model_world(entries)
    result = await _model_hunt(settings, vinted, llm, cache=cache, specs_max_per_hunt=1)
    assert llm.specs_calls == ["acme alpha"]
    assert result.stats["models_cached_exact"] == 1
    assert result.stats["models_skipped_limit"] == 0
    assert _statuses(llm) == {"a1": SPECS_RESEARCHED, "b1": SPECS_RESEARCHED}


async def test_listings_that_already_fail_a_hard_requirement_are_not_researched(settings):
    failing = _model_extraction(
        "f1", "Broken",
        checks=[RequirementCheck(requirement="at least 20 MP", status="unmet", evidence="label reads 12 MP", hard_requirement_no=1)],
    )
    entries = [
        ("f1", 200, 9, failing),
        ("o1", 250, 8, _model_extraction("o1", "Other")),
    ]
    vinted, llm = _model_world(entries)
    result = await _model_hunt(settings, vinted, llm)
    assert llm.specs_calls == ["acme other"]
    assert _statuses(llm) == {"f1": SPECS_FAILED, "o1": SPECS_RESEARCHED}
    assert result.stats["models_skipped_failed"] == 1
    failed = next(entry for entry in result.ranked if entry.listing.id == "f1")
    assert failed.requirements_status == "failed"
    assert failed.ranking.score <= 30


async def test_listings_without_a_model_or_with_no_specs_found_say_so(settings):
    entries = [
        ("a1", 300, 9, _model_extraction("a1", "Alpha")),
        ("e1", 310, 8, _model_extraction("e1", "Empty")),
        ("u1", 320, 7, Extraction(item_id="u1", brand=None, model=None, model_confidence=0.1)),
    ]
    vinted, llm = _model_world(entries, specs_by_model={"Empty": {}})
    await _model_hunt(settings, vinted, llm)
    assert _statuses(llm) == {"a1": SPECS_RESEARCHED, "e1": SPECS_NOT_FOUND, "u1": SPECS_NOT_FOUND}


async def test_the_hunt_stats_and_log_carry_the_counters_and_the_usage_per_stage(settings, plan):
    vinted, llm = build_world(plan)
    llm.tracker.record("plan", "claude-haiku-5-5", SimpleNamespace(input_tokens=1000, output_tokens=100))
    events = []
    result = await run_hunt(
        "req", 50, HuntOptions(search_pages=2), events.append,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )
    for counter in HUNT_COUNTERS:
        assert counter in result.stats, counter
    messages = [event["message"] for event in events if event["type"] == "log"]
    usage = [m for m in messages if m.startswith("Usage by stage: ")]
    assert usage and "planner 1,000 in / 100 out" in usage[0] and "Total $" in usage[0]
    assert any(m.startswith("Specs: ") for m in messages)
