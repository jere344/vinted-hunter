"""Ranking: defect and product-type caps, verdict tiers, the new price, the triage tie-break, identification without a
brand, and the prompt rules of this round."""

import pytest
from helpers import FakeLLM, FakeVinted, make_detail, make_summary

from vinted_hunter.cache import HuntCache
from vinted_hunter.config import Settings
from vinted_hunter.models import (
    Extraction,
    HardRequirement,
    HuntOptions,
    IdentifyDraft,
    RankedListing,
    RequirementCheck,
    SearchPlan,
)
from vinted_hunter.pipeline import _Hunt, _identify, _noop, requirements_status, run_hunt
from vinted_hunter.prompts import (
    EXTRACT_SYSTEM,
    PLANNER_SYSTEM,
    RANK_SYSTEM,
    SPECS_DRAFT_SYSTEM,
    SPECS_RESEARCH_SYSTEM,
    TRIAGE_SYSTEM,
)

QUERY = "appareil photo reflex"


@pytest.fixture
def settings(tmp_path):
    return Settings(
        anthropic_api_key="test-key",
        data_dir=tmp_path / "data",
        vinted_min_delay_s=0,
        max_detailed=10,
        max_candidates=50,
        search_pages=1,
        max_photos=1,
    )


def _plan(hard=(), nice=(), product_type="", not_types=()):
    return SearchPlan(
        product_summary="reflex camera",
        queries=[QUERY],
        budget_eur=100,
        spec_keys=["resolution"],
        hard_requirements=list(hard),
        nice_to_haves=list(nice),
        product_type=product_type,
        not_product_types=list(not_types),
        exclude_keywords=[],
    )


def _world(listings, plan, extractions, scores=None, relevance=None, **llm_options):
    vinted = FakeVinted(
        pages={(QUERY, 1): listings},
        details={s.id: make_detail(s) for s in listings},
    )
    llm = FakeLLM(
        plan=plan,
        extractions=extractions,
        scores=scores or {},
        relevance=relevance if relevance is not None else {s.id: 9 for s in listings},
        **llm_options,
    )
    return vinted, llm


async def _hunt(settings, vinted, llm, budget=100, options=None):
    return await run_hunt(
        "req", budget, options or HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )


def _extraction(item_id, **fields):
    return Extraction(item_id=item_id, brand="Acme", model="X1", model_confidence=0.9, **fields)


# --- Defect caps -------------------------------------------------------------------


async def test_a_functional_defect_caps_the_score_at_40_and_the_verdict_at_ok(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    vinted, llm = _world(
        [listing], _plan(),
        {"a1": _extraction("a1", defect_severity="functional", defect_note="the shutter sticks")},
        scores={"a1": 95},
    )
    result = await _hunt(settings, vinted, llm)
    top = result.ranked[0]
    assert (top.ranking.score, top.ranking.verdict) == (40, "ok")
    assert top.extraction.defect_note == "the shutter sticks"
    payload = llm.rank_payloads[0][0]
    assert payload["defect_severity"] == "functional" and payload["defect_note"] == "the shutter sticks"


async def test_an_unusable_item_caps_the_score_at_20_and_avoids_it(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    vinted, llm = _world([listing], _plan(), {"a1": _extraction("a1", defect_severity="unusable")}, scores={"a1": 90})
    top = (await _hunt(settings, vinted, llm)).ranked[0]
    assert (top.ranking.score, top.ranking.verdict) == (20, "avoid")


async def test_cosmetic_wear_is_not_a_cap(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    vinted, llm = _world([listing], _plan(), {"a1": _extraction("a1", defect_severity="cosmetic")}, scores={"a1": 85})
    top = (await _hunt(settings, vinted, llm)).ranked[0]
    assert (top.ranking.score, top.ranking.verdict) == (85, "top_pick")


# --- Product type --------------------------------------------------------------------


async def test_the_wrong_product_type_is_capped_at_20_failed_and_not_researched(settings):
    listing = make_summary("a1", "Casque Acme X1 sans fil", 30.0)
    extraction = _extraction("a1", is_requested_product_type=False, product_type_evidence="over-ear headphones")
    vinted, llm = _world([listing], _plan(product_type="reflex camera"), {"a1": extraction}, scores={"a1": 90})
    result = await _hunt(settings, vinted, llm)
    top = result.ranked[0]
    assert (top.ranking.score, top.ranking.verdict) == (20, "avoid")
    assert top.requirements_status == "failed"
    assert top.ranking.meets_hard_requirements is False
    assert llm.specs_calls == []  # a wrong product is never researched
    assert llm.rank_payloads[0][0]["is_requested_product_type"] is False


async def test_a_wrong_product_type_is_neither_identified_nor_researched(settings):
    listing = make_summary("a1", "Casque Beta sans fil", 30.0)
    extraction = Extraction(item_id="a1", brand=None, model=None, model_confidence=0.1, is_requested_product_type=False)
    vinted, llm = _world([listing], _plan(product_type="reflex camera"), {"a1": extraction})
    result = await _hunt(settings, vinted, llm)
    assert llm.identify_calls == [] and llm.specs_calls == []
    assert llm.rank_payloads[0][0]["specs_status"] == "not researched (not the requested product type)"
    assert result.ranked[0].requirements_status == "failed"


def test_a_wrong_product_type_fails_the_requirements_whatever_they_are():
    plan = _plan()
    assert requirements_status(plan, Extraction(item_id="x", is_requested_product_type=False), None) == "failed"


# --- Verdict tiers and the unverified cap -------------------------------------------------


async def test_unverified_requirements_keep_the_verdict_at_ok_or_below(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    plan = _plan(hard=["at least 20 MP"])  # unbound: only the vision check can settle it, and there is none
    vinted, llm = _world([listing], plan, {"a1": _extraction("a1")}, scores={"a1": 90})
    top = (await _hunt(settings, vinted, llm)).ranked[0]
    assert top.requirements_status == "unverified"
    assert (top.ranking.score, top.ranking.verdict) == (90, "ok")


async def test_a_verified_listing_keeps_its_top_pick(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    plan = _plan(hard=["at least 20 MP"])
    check = RequirementCheck(requirement="at least 20 MP", status="met", evidence="label", hard_requirement_no=1)
    vinted, llm = _world([listing], plan, {"a1": _extraction("a1", requirement_checks=[check])}, scores={"a1": 90})
    top = (await _hunt(settings, vinted, llm)).ranked[0]
    assert top.requirements_status == "verified"
    assert top.ranking.verdict == "top_pick"


async def test_the_order_is_the_verdict_tier_first_then_the_score(settings):
    unverified = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)  # 85, capped to ok
    verified = make_summary("b1", "Appareil photo Acme X2 reflex", 46.0)  # 65, good
    plan = _plan(hard=["at least 20 MP"])
    check = RequirementCheck(requirement="at least 20 MP", status="met", evidence="label", hard_requirement_no=1)
    vinted, llm = _world(
        [unverified, verified], plan,
        {"a1": _extraction("a1"), "b1": _extraction("b1", requirement_checks=[check])},
        scores={"a1": 85, "b1": 65},
    )
    result = await _hunt(settings, vinted, llm)
    assert [e.listing.id for e in result.ranked] == ["b1", "a1"]
    assert [e.ranking.verdict for e in result.ranked] == ["good", "ok"]
    assert [e.rank for e in result.ranked] == [1, 2]


# --- New price ----------------------------------------------------------------------------


async def test_the_new_price_comes_from_the_web_and_the_discount_is_computed_in_code(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 60.0)  # total 60, shipping unknown
    vinted, llm = _world([listing], _plan(), {"a1": _extraction("a1")}, scores={"a1": 70}, new_prices={"X1": 120.0})
    top = (await _hunt(settings, vinted, llm)).ranked[0]
    assert top.new_price is not None
    assert top.new_price.eur == 120.0
    assert top.new_price.source == "https://shop.example/new"
    assert top.new_price.base_model is False
    assert top.new_price.discount_pct == 50  # 60 against 120: half the new price
    assert llm.rank_payloads[0][0]["specs"]["new_price_eur"] == 120.0


async def test_a_base_model_price_is_shown_but_labelled(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 60.0)
    vinted, llm = _world(
        [listing], _plan(), {"a1": _extraction("a1")}, scores={"a1": 70},
        new_prices={"X1": 120.0}, specs_variant="base_model",
    )
    result = await _hunt(settings, vinted, llm)
    assert result.ranked[0].new_price.base_model is True
    payload_specs = llm.rank_payloads[0][0]["specs"]
    assert payload_specs["new_price_eur"] == 120.0
    assert "specs" not in payload_specs  # the base model's spec values stay withheld


async def test_no_new_price_is_shown_without_a_web_price(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 60.0)
    vinted, llm = _world([listing], _plan(), {"a1": _extraction("a1")}, scores={"a1": 70})
    result = await _hunt(settings, vinted, llm)
    assert result.ranked[0].new_price is None
    assert llm.rank_payloads[0][0]["specs"]["new_price_eur"] is None


async def test_a_negative_discount_says_the_listing_costs_more_than_new(settings):
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 150.0)
    vinted, llm = _world([listing], _plan(), {"a1": _extraction("a1")}, scores={"a1": 70}, new_prices={"X1": 120.0})
    top = (await _hunt(settings, vinted, llm, budget=200)).ranked[0]
    assert top.new_price.discount_pct == -25


# --- Seller checks are gone -------------------------------------------------------------


def test_seller_checks_are_gone_from_the_schema_and_the_prompt_but_old_hunts_still_load():
    assert "seller_checks" not in RankedListing.model_fields
    old = RankedListing.model_validate(
        {"item_id": "a1", "score": 50, "verdict": "ok", "meets_hard_requirements": True, "headline": "h",
         "value_rationale": "r", "seller_checks": ["photo of the label"]}
    )
    assert not hasattr(old, "seller_checks")
    assert "seller_checks" not in RANK_SYSTEM


# --- Triage order ------------------------------------------------------------------------


async def test_a_nice_to_have_in_the_title_wins_a_relevance_tie_before_the_price(settings):
    cheap_plain = make_summary("b1", "Appareil photo reflex blanc", 30.0)
    dearer_colour = make_summary("a1", "Appareil photo reflex noir", 45.0)
    cheaper_colour = make_summary("c1", "Appareil photo reflex noir", 40.0)
    vinted, llm = _world(
        [cheap_plain, dearer_colour, cheaper_colour], _plan(nice=["noir"]),
        {}, relevance={"a1": 6, "b1": 6, "c1": 6},
    )
    result = await _hunt(settings, vinted, llm, options=HuntOptions(search_pages=1, max_detailed=1))
    assert [e.listing.id for e in result.ranked] == ["c1"]  # the cheaper of the two that name the colour
    rejected = {r.item_id: r.reason for r in result.rejected if r.stage == "triage"}
    assert "outside the top 1" in rejected["b1"] and "outside the top 1" in rejected["a1"]


async def test_the_lower_total_price_breaks_a_tie_with_no_nice_to_have(settings):
    pricier = make_summary("a1", "Appareil photo reflex", 45.0)
    cheaper = make_summary("b1", "Appareil photo reflex", 35.0)
    vinted, llm = _world([pricier, cheaper], _plan(), {}, relevance={"a1": 6, "b1": 6})
    result = await _hunt(settings, vinted, llm, options=HuntOptions(search_pages=1, max_detailed=1))
    assert [e.listing.id for e in result.ranked] == ["b1"]


# --- Identification without a brand ------------------------------------------------------------


async def test_a_listing_without_a_brand_is_a_web_identification_candidate(settings):
    title = "Appareil photo Beta B2 Full HD"
    listing = make_summary("u0", title, 25.0)
    extraction = Extraction(item_id="u0", brand=None, model=None, model_confidence=0.2, visual_description="white box")
    llm = FakeLLM(plan=_plan(), extractions={"u0": extraction},
                  identifications={title: IdentifyDraft(model="B2 Plus", model_confidence=0.8, evidence="catalogue")})
    details = {"u0": make_detail(listing)}
    extractions = {"u0": extraction}
    await _identify(extractions, details, settings, llm, HuntCache(settings.data_dir / "cache.sqlite3"), _Hunt(_noop))
    assert llm.identify_calls == [title]
    assert extractions["u0"].model == "B2 Plus"


async def test_a_brandless_identified_model_is_researched_by_its_name(settings):
    title = "Appareil photo Beta B2 Full HD"
    listing = make_summary("u0", title, 25.0)
    extraction = Extraction(item_id="u0", brand=None, model=None, model_confidence=0.2, visual_description="white box")
    vinted = FakeVinted(pages={(QUERY, 1): [listing]}, details={"u0": make_detail(listing)})
    llm = FakeLLM(plan=_plan(), extractions={"u0": extraction},
                  identifications={title: IdentifyDraft(model="B2 Plus", model_confidence=0.8, evidence="catalogue")})
    result = await _hunt(settings, vinted, llm)
    assert "b2 plus" in llm.specs_calls
    assert result.ranked[0].specs.model_key == "b2 plus"


async def test_a_doubtful_model_name_is_researched_without_its_qualifier(settings):
    title = "Appareil photo Beta B2 Full HD"
    listing = make_summary("u0", title, 25.0)
    extraction = Extraction(item_id="u0", brand="Beta", model="B2 (variant not confirmed)", model_confidence=0.8)
    vinted = FakeVinted(pages={(QUERY, 1): [listing]}, details={"u0": make_detail(listing)})
    llm = FakeLLM(plan=_plan(), extractions={"u0": extraction})
    await _hunt(settings, vinted, llm)
    assert llm.specs_calls == ["beta b2"]


# --- Prompt rules --------------------------------------------------------------------------------


def test_the_planner_prompt_asks_for_yes_no_comparators_and_the_product_type_fields():
    assert '"present"' in PLANNER_SYSTEM and '"absent"' in PLANNER_SYSTEM
    assert "not_product_types" in PLANNER_SYSTEM and "product_type" in PLANNER_SYSTEM
    assert "colors" in PLANNER_SYSTEM
    # The category pitfalls rule forbids the generic used-condition remarks.
    assert "never" in PLANNER_SYSTEM.lower() and "battery wears with age" in PLANNER_SYSTEM


def test_the_extraction_prompt_has_the_defect_levels_and_the_identity_rules():
    assert "defect_severity" in EXTRACT_SYSTEM
    for level in ('"none"', '"cosmetic"', '"functional"', '"unusable"'):
        assert level in EXTRACT_SYSTEM
    assert "Used or second-hand condition itself is NOT a defect" in EXTRACT_SYSTEM
    assert "The title and the description are the primary identity evidence" in EXTRACT_SYSTEM
    assert "variant not confirmed" in EXTRACT_SYSTEM  # the qualifier is named as a thing never to write in the model
    assert "is_requested_product_type" in EXTRACT_SYSTEM


def test_the_rank_prompt_never_treats_used_condition_or_review_counts_as_cons():
    assert "Few reviews is weak evidence" not in RANK_SYSTEM
    assert "used condition" in RANK_SYSTEM and "generic battery ageing" in RANK_SYSTEM
    assert "low number of seller reviews" in RANK_SYSTEM
    assert "new_price_eur" in RANK_SYSTEM and "typical_used_price_eur" in RANK_SYSTEM
    assert "defect_severity" in RANK_SYSTEM


def test_the_specs_prompts_fill_every_key_and_write_no_for_an_absent_feature():
    for prompt in (SPECS_RESEARCH_SYSTEM, SPECS_DRAFT_SYSTEM):
        assert 'write the value "no"' in prompt
    assert "new_price_eur" in SPECS_DRAFT_SYSTEM and "new_price_source" in SPECS_DRAFT_SYSTEM


def test_the_triage_prompt_knows_the_product_types_and_the_nice_to_have_tie_break():
    assert "product type" in TRIAGE_SYSTEM
    assert "nice-to-have" in TRIAGE_SYSTEM
    assert "relevance 0" in TRIAGE_SYSTEM


def test_the_hard_requirement_schema_accepts_present_and_absent():
    HardRequirement(text="anc", spec_key="anc", comparator="present", target=None)
    HardRequirement(text="no anc", spec_key="anc", comparator="absent", target=None)
    with pytest.raises(ValueError):
        HardRequirement(text="x", spec_key="anc", comparator="maybe", target=None)
