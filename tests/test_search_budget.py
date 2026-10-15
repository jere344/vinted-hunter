"""Budget at search time: Vinted's buyer fee, the item price ceiling sent as price_to, and the colour filter plumbing."""

from dataclasses import dataclass, field
from decimal import Decimal

import pytest
from helpers import FakeLLM, FakeVinted, make_detail, make_summary

from vinted_hunter import vinted as vinted_module
from vinted_hunter.cache import HuntCache
from vinted_hunter.config import Settings
from vinted_hunter.models import HuntOptions, SearchPlan
from vinted_hunter.pipeline import _prepare_plan, run_hunt
from vinted_hunter.vinted import (
    buyer_fee_eur,
    color_ids_for,
    item_price_ceiling,
    search_url,
)

# (item price, service fee, total) as stored in the search results of data/cache.sqlite3 (40 distinct pairs,
# 2026-10-14). The fee fits price x 5 % + 0.70 EUR, rounded half up to the cent. Kept here so the tests need no cache.
OBSERVED_PAIRS = [
    (1.0, 0.75, 1.75), (1.99, 0.8, 2.79), (2.0, 0.8, 2.8), (2.3, 0.82, 3.12), (2.9, 0.85, 3.75), (3.0, 0.85, 3.85),
    (4.0, 0.9, 4.9), (4.3, 0.92, 5.22), (4.99, 0.95, 5.94), (5.0, 0.95, 5.95), (5.5, 0.98, 6.48), (5.99, 1.0, 6.99),
    (6.0, 1.0, 7.0), (6.5, 1.03, 7.53), (6.89, 1.04, 7.93), (7.0, 1.05, 8.05), (7.5, 1.08, 8.58), (7.9, 1.1, 9.0),
    (8.0, 1.1, 9.1), (8.5, 1.13, 9.63), (8.75, 1.14, 9.89), (9.0, 1.15, 10.15), (9.5, 1.18, 10.68), (9.9, 1.2, 11.1),
    (9.99, 1.2, 11.19), (10.0, 1.2, 11.2), (11.0, 1.25, 12.25), (11.5, 1.28, 12.78), (12.0, 1.3, 13.3),
    (12.5, 1.33, 13.83), (12.99, 1.35, 14.34), (13.0, 1.35, 14.35), (13.5, 1.38, 14.88), (13.99, 1.4, 15.39),
    (14.0, 1.4, 15.4), (14.5, 1.43, 15.93), (14.9, 1.45, 16.35), (14.97, 1.45, 16.42), (14.99, 1.45, 16.44),
    (15.0, 1.45, 16.45),
]


def test_the_observed_pairs_are_40_distinct_and_the_totals_add_up():
    assert len(OBSERVED_PAIRS) == 40
    assert all(abs(price + fee - total) < 0.005 for price, fee, total in OBSERVED_PAIRS)


@pytest.mark.parametrize("price, fee, _total", OBSERVED_PAIRS)
def test_buyer_fee_reproduces_every_observed_fee(price, fee, _total):
    assert buyer_fee_eur(price) == pytest.approx(fee, abs=1e-9)


def test_item_price_ceiling_keeps_every_listing_whose_total_is_within_budget():
    # Sweep budgets from 0.70 to 300 EUR in one-cent steps: no observed listing whose total fits may lie above the ceiling.
    violations = []
    for cents in range(70, 30001, 1):
        budget = cents / 100
        ceiling = item_price_ceiling(budget)
        for price, _fee, total in OBSERVED_PAIRS:
            if total <= budget + 1e-9 and price > ceiling + 1e-9:
                violations.append((budget, price, ceiling))
    assert violations == []


def test_item_price_ceiling_is_the_highest_price_within_budget_plus_slack():
    ceiling = item_price_ceiling(15)
    assert ceiling == 13.67  # 13.67 + 1.38 = 15.05, the slack being 0.05 EUR
    limit = Decimal("15.05")
    assert Decimal("13.67") + Decimal(str(buyer_fee_eur(13.67))) <= limit
    assert Decimal("13.68") + Decimal(str(buyer_fee_eur(13.68))) > limit


@pytest.mark.parametrize("budget", [15, 20, 50, 400])
def test_item_price_ceiling_is_below_the_budget_and_never_above_it_by_more_than_the_fee(budget):
    ceiling = item_price_ceiling(budget)
    assert ceiling < budget
    assert ceiling + buyer_fee_eur(ceiling) <= budget + 0.05 + 1e-9


def test_item_price_ceiling_is_zero_when_nothing_fits():
    assert item_price_ceiling(0.5) == 0.0


@dataclass
class RecordingVinted(FakeVinted):
    """A FakeVinted that records the price_to and colour ids of each search."""

    price_calls: list = field(default_factory=list)

    async def search(self, query, *, price_to, catalog_ids, page, color_ids=()):
        self.price_calls.append((query, price_to, tuple(color_ids)))
        return await super().search(query, price_to=price_to, catalog_ids=catalog_ids, page=page, color_ids=color_ids)


async def test_search_sends_the_item_price_ceiling_not_the_budget(tmp_path):
    settings = Settings(anthropic_api_key="test-key", data_dir=tmp_path / "data", vinted_min_delay_s=0,
                        max_candidates=50, max_detailed=5, max_photos=1)
    plan = SearchPlan(product_summary="p", queries=["appareil photo reflex"], budget_eur=15, spec_keys=["resolution"])
    listing = make_summary("a1", "Appareil photo reflex", 14.0)
    vinted = RecordingVinted(
        pages={("appareil photo reflex", 1): [listing]},
        details={"a1": make_detail(listing)},
    )
    llm = FakeLLM(plan=plan)
    await run_hunt("req", 15, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted, llm=llm,
                   cache=HuntCache(settings.data_dir / "cache.sqlite3"))
    assert vinted.price_calls == [("appareil photo reflex", item_price_ceiling(15), ())]


async def test_a_listing_over_the_total_budget_is_still_rejected_locally(tmp_path):
    # The search sends the item ceiling, and the exact check on the buyer's total still applies to what comes back.
    settings = Settings(anthropic_api_key="test-key", data_dir=tmp_path / "data", vinted_min_delay_s=0,
                        max_candidates=50, max_detailed=5, max_photos=1)
    plan = SearchPlan(product_summary="p", queries=["appareil photo reflex"], budget_eur=15, spec_keys=["resolution"])
    listing = make_summary("a1", "Appareil photo reflex", 14.98)
    listing.price_eur, listing.service_fee_eur = 13.60, 1.38  # total 14.98: within the budget
    over = make_summary("o1", "Appareil photo reflex trop cher", 15.2)  # total 15.20: over the budget
    vinted = FakeVinted(pages={("appareil photo reflex", 1): [listing, over]}, details={"a1": make_detail(listing)})
    result = await run_hunt("req", 15, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted,
                            llm=FakeLLM(plan=plan), cache=HuntCache(settings.data_dir / "cache.sqlite3"))
    assert {e.listing.id for e in result.ranked} == {"a1"}
    assert result.stats["search_over_budget"] == 1


def test_search_url_sends_colour_ids_as_a_repeated_filter():
    url = search_url("https://www.vinted.fr", "casque noir", price_to=13.67, catalog_ids=[], page=1, color_ids=[1, 7])
    assert "color_ids%5B%5D=1" in url and "color_ids%5B%5D=7" in url
    assert "search_text=casque+noir" in url and "price_to=13.67" in url


def test_search_url_without_colours_has_no_colour_filter():
    url = search_url("https://www.vinted.fr", "casque", price_to=10, catalog_ids=[], page=1)
    assert "color_ids" not in url


def test_colour_names_map_to_ids_accent_insensitively_and_without_duplicates():
    table = {"noir": 1, "blanc": 2}
    assert color_ids_for(["Noir", "blanc", "NOIR"], table) == [1, 2]
    assert color_ids_for(["gris"], table) == []
    assert color_ids_for([], table) == []


def test_the_colour_table_is_empty_until_its_ids_are_probed():
    # The ids are not in the stored data, so nothing is sent as a colour filter yet. The text keeps the colour.
    assert vinted_module.VINTED_COLOR_IDS == {}
    assert color_ids_for(["noir"]) == []


def test_the_planner_colours_are_mapped_by_code(monkeypatch):
    monkeypatch.setattr(vinted_module, "VINTED_COLOR_IDS", {"noir": 42})
    plan = SearchPlan(product_summary="p", queries=["casque noir"], budget_eur=20, colors=["Noir"], color_ids=[9])
    _prepare_plan(plan)
    assert plan.color_ids == [42]  # the planner's own value is replaced by the code's mapping


async def test_colour_ids_reach_the_search(tmp_path, monkeypatch):
    monkeypatch.setattr(vinted_module, "VINTED_COLOR_IDS", {"noir": 42})
    settings = Settings(anthropic_api_key="test-key", data_dir=tmp_path / "data", vinted_min_delay_s=0,
                        max_candidates=50, max_detailed=5, max_photos=1)
    plan = SearchPlan(product_summary="p", queries=["casque noir"], budget_eur=20, colors=["noir"], spec_keys=["x"])
    vinted = RecordingVinted(pages={}, details={})
    await run_hunt("req", 20, HuntOptions(search_pages=1), None, settings=settings, vinted=vinted,
                   llm=FakeLLM(plan=plan), cache=HuntCache(settings.data_dir / "cache.sqlite3"))
    assert vinted.price_calls == [("casque noir", item_price_ceiling(20), (42,))]


def test_the_default_detail_cap_is_60():
    assert Settings().max_detailed == 60

