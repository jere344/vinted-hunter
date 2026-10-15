"""Live smoke test against the real vinted.fr. Skipped by default.

Run with:  python -m pytest -m live -s
It makes a handful of requests (one search, one item page) and prints what it parsed.
"""

import pytest

from vinted_hunter.config import Settings
from vinted_hunter.vinted import VintedClient

pytestmark = pytest.mark.live


async def test_live_search_and_item_detail():
    client = VintedClient(Settings(vinted_min_delay_s=1.0))
    try:
        listings = await client.search("appareil photo reflex", price_to=50, catalog_ids=[], page=1)
        assert listings, "search returned no listings"
        in_budget = [s for s in listings if s.total_price_eur is not None and s.total_price_eur <= 50]
        print(f"\nsearch: {len(listings)} listings, {len(in_budget)} with total price <= 50 EUR")
        target = in_budget[0] if in_budget else listings[0]
        print(f"summary: id={target.id} price={target.price_eur} fee={target.service_fee_eur} "
              f"total={target.total_price_eur} url={target.url}")

        detail = await client.item_detail(target.url)
        print("detail:")
        for field in (
            "id", "title", "brand", "condition", "color", "upload_info", "seller_name",
            "seller_rating", "seller_review_count", "can_buy", "is_reserved",
        ):
            print(f"  {field}: {getattr(detail, field)!r}")
        description = (detail.description or "")[:200].replace("\n", " ")
        print(f"  description: {description!r}")
        print(f"  photo_urls: {len(detail.photo_urls)} ({detail.photo_urls[:1]})")

        assert detail.id == target.id
        assert detail.title
        assert detail.photo_urls
    finally:
        await client.close()
