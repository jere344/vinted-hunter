"""Shipping awareness: landed price, the budget check after the item pages, the config toggle and the prompts."""

import pytest
from helpers import FakeLLM, FakeVinted, make_detail, make_summary

from vinted_hunter.cache import HuntCache
from vinted_hunter.config import Settings, load_settings
from vinted_hunter.models import HuntOptions, SearchPlan
from vinted_hunter.pipeline import _extract_text, run_hunt
from vinted_hunter.prompts import EXTRACT_SYSTEM, RANK_SYSTEM

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
        max_photos=2,
    )


def _plan(budget: float = 50) -> SearchPlan:
    return SearchPlan(product_summary="reflex camera", queries=[QUERY], budget_eur=budget, spec_keys=["resolution"])


# --- Landed price ---------------------------------------------------------------


def _detail(total: float = 45.0, **overrides):
    return make_detail(make_summary("s1", "Appareil photo", total=total), **overrides)


def test_landed_price_is_total_plus_shipping():
    detail = _detail(45.0, shipping_price_eur=5.19)
    assert detail.landed_price_eur == 50.19
    assert detail.shipping_known is True


def test_landed_price_rounds_to_cents():
    assert _detail(45.1, shipping_price_eur=5.19).landed_price_eur == 50.29


def test_free_shipping_adds_nothing_even_without_an_amount():
    assert _detail(45.0, shipping_is_free=True).landed_price_eur == 45.0
    assert _detail(45.0, shipping_is_free=True, shipping_price_eur=0.0).landed_price_eur == 45.0
    assert _detail(45.0, shipping_is_free=True).shipping_known is True


def test_pickup_only_adds_nothing_and_is_known():
    detail = _detail(45.0, pickup_only=True, shipping_price_eur=5.19)
    assert detail.landed_price_eur == 45.0
    assert detail.shipping_known is True


def test_unknown_shipping_gives_no_landed_price_and_keeps_the_total():
    detail = _detail(45.0)
    assert detail.landed_price_eur is None
    assert detail.shipping_known is False
    assert detail.total_price_eur == 45.0


def test_landed_price_is_none_without_a_total():
    summary = make_summary("s1", "Appareil photo", total=45.0)
    summary.total_price_eur = None
    detail = make_detail(summary, shipping_price_eur=5.19)
    assert detail.landed_price_eur is None


def test_landed_price_is_in_the_serialised_listing():
    dumped = _detail(45.0, shipping_price_eur=5.19).model_dump(mode="json")
    assert dumped["landed_price_eur"] == 50.19
    assert dumped["shipping_known"] is True


# --- Budget check after the item pages -------------------------------------------


def _world(tmp_path, details, budget: float = 50, *, include: bool = True):
    summaries = [make_summary(d.id, d.title, d.total_price_eur, d.url) for d in details]
    vinted = FakeVinted(pages={(QUERY, 1): summaries}, details={d.id: d for d in details})
    llm = FakeLLM(plan=_plan(budget))
    settings = Settings(
        anthropic_api_key="test-key",
        data_dir=tmp_path / "data",
        vinted_min_delay_s=0,
        max_detailed=10,
        max_candidates=50,
        search_pages=1,
        include_shipping_in_budget=include,
    )
    return vinted, llm, settings


async def _hunt(vinted, llm, settings, budget: float = 50):
    return await run_hunt(
        "req", budget, HuntOptions(search_pages=1), None,
        settings=settings, vinted=vinted, llm=llm, cache=HuntCache(settings.data_dir / "cache.sqlite3"),
    )


def _listing(item_id: str, total: float, **shipping):
    summary = make_summary(item_id, f"Appareil photo {item_id}", total=total)
    return make_detail(summary, **shipping)


async def test_listing_over_budget_once_shipping_is_added_is_rejected(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 45.0, shipping_price_eur=5.19)])
    result = await _hunt(vinted, llm, settings)
    assert result.ranked == []
    [rejected] = result.rejected
    assert rejected.item_id == "a1"
    assert rejected.stage == "shipping"
    assert rejected.reason == "over budget with shipping (45.00 € + 5.19 €)"
    assert result.stats["rejected_shipping"] == 1
    assert "a1" not in llm.extract_calls  # rejected before the paid vision step


async def test_landed_price_exactly_on_budget_is_kept(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 45.0, shipping_price_eur=5.0)])
    result = await _hunt(vinted, llm, settings)
    assert [entry.listing.id for entry in result.ranked] == ["a1"]
    assert result.rejected == []


async def test_listing_with_unknown_shipping_is_kept_and_marked(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 45.0)])
    result = await _hunt(vinted, llm, settings)
    assert [entry.listing.id for entry in result.ranked] == ["a1"]
    assert result.stats["shipping_unknown"] == 1
    payload = {item["item_id"]: item for item in llm.rank_payloads[0]}
    assert payload["a1"]["shipping_known"] is False
    assert payload["a1"]["landed_price_eur"] is None
    assert payload["a1"]["total_price_eur"] == 45.0


async def test_free_shipping_keeps_a_listing_at_its_total(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 49.5, shipping_is_free=True)])
    result = await _hunt(vinted, llm, settings)
    assert [entry.listing.id for entry in result.ranked] == ["a1"]
    payload = {item["item_id"]: item for item in llm.rank_payloads[0]}
    assert payload["a1"]["shipping_is_free"] is True
    assert payload["a1"]["landed_price_eur"] == 49.5


async def test_pickup_only_listing_has_no_shipping_in_the_payload(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 45.0, pickup_only=True, shipping_price_eur=5.19)])
    result = await _hunt(vinted, llm, settings)
    assert [entry.listing.id for entry in result.ranked] == ["a1"]
    payload = {item["item_id"]: item for item in llm.rank_payloads[0]}
    assert payload["a1"]["pickup_only"] is True
    assert payload["a1"]["shipping_price_eur"] is None
    assert payload["a1"]["landed_price_eur"] == 45.0


async def test_toggle_off_budgets_on_the_total_only(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 45.0, shipping_price_eur=5.19)], include=False)
    result = await _hunt(vinted, llm, settings)
    assert [entry.listing.id for entry in result.ranked] == ["a1"]
    assert result.rejected == []
    payload = {item["item_id"]: item for item in llm.rank_payloads[0]}
    # The shipping is still shown and still reaches the ranker, only the budget ignores it.
    assert payload["a1"]["shipping_price_eur"] == 5.19
    assert payload["a1"]["landed_price_eur"] == 50.19


async def test_rank_payload_carries_the_shipping_fields(tmp_path):
    vinted, llm, settings = _world(tmp_path, [_listing("a1", 40.0, shipping_price_eur=5.19, shipping_options_multiple=True)])
    await _hunt(vinted, llm, settings)
    item = {item["item_id"]: item for item in llm.rank_payloads[0]}["a1"]
    assert item["total_price_eur"] == 40.0
    assert item["shipping_price_eur"] == 5.19
    assert item["landed_price_eur"] == 45.19
    assert item["shipping_known"] is True
    assert item["shipping_is_free"] is None
    assert item["shipping_options_multiple"] is True
    assert item["pickup_only"] is None


# --- Prompts ---------------------------------------------------------------------


def test_extraction_text_shows_shipping_and_landed_cost():
    detail = _listing("a1", 40.0, shipping_price_eur=5.19)
    text = _extract_text(_plan(), detail, 1)
    assert "Total price incl. buyer fees: 40.00 EUR" in text
    assert "Shipping (Vinted default option, estimate): 5.19 EUR" in text
    assert "Landed cost: 45.19 EUR" in text


def test_extraction_text_says_when_shipping_is_unknown():
    text = _extract_text(_plan(), _listing("a1", 40.0), 1)
    assert "Shipping (Vinted default option, estimate): unknown" in text
    assert "Landed cost: unknown" in text


def test_ranker_prompt_uses_landed_cost_and_never_penalises_unknown_shipping():
    assert "landed_price_eur" in RANK_SYSTEM
    assert "shipping_known" in RANK_SYSTEM
    assert "never lower the score" in RANK_SYSTEM
    assert "pickup_only" in RANK_SYSTEM
    assert "remise en main propre" in RANK_SYSTEM


def test_extraction_prompt_never_makes_shipping_a_red_flag():
    assert "Shipping, delivery and pickup details are never red flags." in EXTRACT_SYSTEM


# --- Config ----------------------------------------------------------------------


def test_include_shipping_in_budget_is_on_by_default():
    assert load_settings(environ={}, env_file=None).include_shipping_in_budget is True


@pytest.mark.parametrize("value", ["false", "0", "no", "OFF"])
def test_include_shipping_in_budget_can_be_turned_off(value):
    settings = load_settings(environ={"INCLUDE_SHIPPING_IN_BUDGET": value}, env_file=None)
    assert settings.include_shipping_in_budget is False


def test_include_shipping_in_budget_rejects_other_values():
    with pytest.raises(ValueError, match="INCLUDE_SHIPPING_IN_BUDGET"):
        load_settings(environ={"INCLUDE_SHIPPING_IN_BUDGET": "maybe"}, env_file=None)
