"""Shared test doubles and builders. No network access anywhere in the test suite."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from vinted_hunter.llm import LLMError
from vinted_hunter.models import (
    Extraction,
    ExtractionDraft,
    ListingDetail,
    ListingSummary,
    ModelSpecs,
    RankedListing,
    RankingOutput,
    SearchPlan,
    SpecEntry,
    TriageBatch,
    TriageDecision,
)
from vinted_hunter.vinted import VintedError

PNG_1X1 = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
    b"\x00\x00\x00\rIDATx\x9cc\xf8\xcf\xc0\xf0\x1f\x00\x05\x05\x02\x00\xa3\x8f\xa9\x9d\x00\x00\x00\x00IEND\xaeB`\x82"
)


# --- Flight payload builders -------------------------------------------------


def flight_html(*payloads: str) -> str:
    """Wrap decoded flight text in self.__next_f.push chunks, as Next.js does."""
    scripts = "".join(f"<script>self.__next_f.push([1,{json.dumps(p)}])</script>" for p in payloads)
    return f"<!doctype html><html><body>{scripts}</body></html>"


def search_payload(products: list[dict[str, Any]]) -> str:
    items = [{"productItem": p} for p in products]
    pagination = {"current_page": 1, "per_page": 96, "total_entries": len(products), "total_pages": 1}
    return '2a:["$","div",null,{"items":{"items":' + json.dumps(items) + ',"pagination":' + json.dumps(pagination) + "}}]\n"


def product(
    item_id: int,
    title: str,
    *,
    price: float,
    fee: float = 0.0,
    total: float | None = None,
    url: str | None = None,
) -> dict[str, Any]:
    total = round(price + fee, 2) if total is None else total
    return {
        "id": item_id,
        "title": title,
        "url": url or f"/items/{item_id}-slug",
        "price": {"amount": f"{price:.2f}", "currency_code": "EUR"},
        "serviceFee": {"amount": f"{fee:.2f}", "currency_code": "EUR"},
        "totalItemPrice": {"amount": f"{total:.2f}", "currency_code": "EUR"},
        "photos": [{"url": f"https://images1.vinted.net/t/{item_id}/f800/a.webp", "is_main": True}],
        "thumbnailUrl": f"https://images1.vinted.net/t/{item_id}/150x150/a.webp",
        "favouriteCount": 2,
        "isPromoted": False,
        "user": {"id": 9, "isBusiness": False},
    }


def pricing_services(
    *, shipping: str | None = "5.19", is_free: bool = False, multiple: bool = True, pickup_only: bool = False
) -> dict[str, Any]:
    """The pricingServices block of an item page, in Vinted's shape. `shipping` is the amount of the default
    shipping option; None gives the "$undefined" marker Vinted sends when there is no price."""
    price: Any = {"amount": shipping, "currencyCode": "EUR"} if shipping is not None else "$undefined"
    return {
        "services": {
            "isPickUpOnly": pickup_only,
            "buyerProtection": {"finalPrice": {"amount": "2.85", "currencyCode": "EUR"}, "isFree": False},
            "shipping": {
                "type": "shipping",
                "originalPrice": price,
                "finalPrice": price,
                "discount": {"amount": "0", "currencyCode": "EUR"},
                "isFree": is_free,
                "areMultipleShippingOptionsAvailable": multiple,
                "deliveryType": "$undefined",
            },
        },
        "totalAmount": {"amount": "45.84", "currencyCode": "EUR"},
    }


def item_page_html(
    item_id: str = "111",
    title: str = "Appareil photo Acme X1 reflex",
    *,
    pricing: Any = None,
) -> str:
    """An item page. `pricing` (a pricingServices block, see pricing_services) is added when given."""
    item = {
        "brand_dto": {"id": "5", "title": "Acme", "slug": "acme"},
        "catalog_id": "1234",
        "can_buy": True,
        "currency": "EUR",
        "id": item_id,
        "is_reserved": False,
        "photos": [
            {
                "id": "p1",
                "is_main": True,
                "url": "https://images1.vinted.net/t/p1/f800/a.webp?s=1",
                "thumbnails": [{"type": "thumb150", "url": "https://images1.vinted.net/t/p1/150.webp", "width": 150}],
            },
            {
                "id": "p2",
                "is_main": False,
                "url": "https://images1.vinted.net/t/p2/f800/b.webp?s=2",
                "thumbnails": [],
            },
        ],
        "title": title,
    }
    description = 'Vendu avec la boîte d\'origine.\nCourroie incluse <3'
    attributes = [
        {"code": "brand", "data": {"id": "5", "title": "Marque", "value": "Acme"}, "type": "favouritable"},
        {"code": "status", "data": {"id": 50, "title": "État", "value": "Très bon état"}, "type": "faq"},
        {"code": "color", "data": {"title": "Couleur", "value": "Blanc"}, "type": "text"},
        {"code": "upload_date", "data": {"title": "Ajouté", "value": "Il y a 2 jours"}, "type": "text"},
    ]
    decoy_attributes = [
        {"code": "status", "data": {"id": 1, "title": "État", "value": "DECOY état"}, "type": "faq"},
    ]
    pieces = [
        '1a:["$","div",null,{"children":"loading"}]\n',
        # Decoys from another listing that appear before this item's own blocks.
        '{"data":{"description":' + json.dumps("DECOY description of another listing") + ',"item_id":"999"}}',
        '{"data":{"attributes":' + json.dumps(decoy_attributes) + ',"item_id":"999","section_title":"Détails"}}',
        '{"item":' + json.dumps(item) + ',"other":true}',
        '{"data":{"description":' + json.dumps(description) + f',"item_id":"{item_id}"}}}}',
        '{"data":{"attributes":' + json.dumps(attributes) + f',"item_id":"{item_id}","section_title":"Détails"}}}}',
        '{"badges":[],"business":false,"feedback_count":12,"feedback_reputation":0.97,'
        '"follow_action_visible":false,"is_favourite":false,"item_id":"111","name":"seller42","photo":{}}',
    ]
    if pricing is not None:
        pieces.append('{"pricingServices":' + json.dumps(pricing) + "}")
    return flight_html(*pieces)


# --- Fakes -------------------------------------------------------------------


@dataclass
class FakeVinted:
    """Serves canned search pages and item details. Records calls.

    search_errors: exception raised for every page of a query.
    search_exc: exception raised for every search.
    detail_errors: exceptions raised, in order, by the next item_detail calls.
    """

    pages: dict[tuple[str, int], list[ListingSummary]] = field(default_factory=dict)
    details: dict[str, ListingDetail] = field(default_factory=dict)
    failing_detail_urls: set[str] = field(default_factory=set)
    search_errors: dict[str, Exception] = field(default_factory=dict)
    search_exc: Exception | None = None
    detail_errors: list[Exception | None] = field(default_factory=list)
    failing_image_urls: set[str] = field(default_factory=set)
    search_calls: list[tuple[str, int]] = field(default_factory=list)
    catalog_calls: list[tuple[str, int, tuple[int, ...]]] = field(default_factory=list)
    empty_with_catalog: set[str] = field(default_factory=set)
    base_url: str = "https://www.vinted.fr"
    detail_calls: list[str] = field(default_factory=list)
    closed: bool = False

    async def search(self, query: str, *, price_to: float, catalog_ids: list[int], page: int, color_ids=()):
        self.search_calls.append((query, page))
        self.catalog_calls.append((query, page, tuple(catalog_ids)))
        if self.search_exc is not None:
            raise self.search_exc
        if query in self.search_errors:
            raise self.search_errors[query]
        if catalog_ids and query in self.empty_with_catalog:
            return []
        return list(self.pages.get((query, page), []))

    async def item_detail(self, url: str) -> ListingDetail:
        self.detail_calls.append(url)
        if self.detail_errors:
            outcome = self.detail_errors.pop(0)
            if outcome is not None:
                raise outcome
        if url in self.failing_detail_urls:
            raise VintedError(f"blocked {url}")
        for detail in self.details.values():
            if detail.url == url:
                return detail.model_copy()
        raise VintedError(f"no detail for {url}")

    async def download_image(self, url: str) -> tuple[bytes, str]:
        if url in self.failing_image_urls:
            raise VintedError(f"image unavailable {url}")
        return PNG_1X1, "image/png"

    async def close(self) -> None:
        self.closed = True


def make_validation_error() -> ValidationError:
    try:
        TriageBatch.model_validate({"decisions": 1})
    except ValidationError as exc:
        return exc
    raise AssertionError("expected a validation error")


DEFAULT_SPECS = {"resolution": "24 MP", "sensor size": "APS-C"}


@dataclass
class FakeLLM:
    """Answers each structured call based on the requested output model."""

    plan: SearchPlan
    extractions: dict[str, Extraction] = field(default_factory=dict)
    fail_extract: set[str] = field(default_factory=set)
    reject_triage: set[str] = field(default_factory=set)
    raise_triage_for: set[str] = field(default_factory=set)
    unmet: set[str] = field(default_factory=set)
    knowledge_only: bool = False
    specs_variant: str = "exact"  # variant_match of the web specs returned
    specs_by_model: dict[str, dict[str, str]] = field(default_factory=dict)  # model name -> spec values
    new_prices: dict[str, float] = field(default_factory=dict)  # model name -> new price in EUR, from the web
    spec_keys_seen: list[tuple[str, ...]] = field(default_factory=list)
    quota_in_triage: bool = False
    keep_flag_wrong: dict[str, int] = field(default_factory=dict)  # keep=false but this relevance
    identifications: dict[str, Any] = field(default_factory=dict)  # by listing title -> IdentifyDraft
    identify_calls: list[str] = field(default_factory=list)
    relevance: dict[str, int] = field(default_factory=dict)
    scores: dict[str, int] = field(default_factory=dict)
    fail_plan: bool = False
    fail_rank: bool = False
    specs_calls: list[str] = field(default_factory=list)
    extract_calls: list[str] = field(default_factory=list)
    rank_payloads: list[Any] = field(default_factory=list)
    configured: bool = True
    tracker: Any = None

    def __post_init__(self) -> None:
        from vinted_hunter.llm import UsageTracker

        self.tracker = UsageTracker()

    def ensure_configured(self) -> None:
        if not self.configured:
            from vinted_hunter.llm import NO_KEY_MESSAGE, LLMConfigError

            raise LLMConfigError(NO_KEY_MESSAGE)

    async def parse(self, *, stage, model, system, content, output_model, effort, cache=False, max_tokens=16000):
        if output_model is SearchPlan:
            if self.fail_plan:
                raise LLMError("planner exploded")
            return self.plan.model_copy(deep=True)

        if output_model is TriageBatch:
            decisions = []
            if self.quota_in_triage:
                from vinted_hunter.llm import LLMQuotaError

                raise LLMQuotaError("Claude subscription usage limit reached, try later")
            batch_ids = re.findall(r'<listing id="([^"]+)"', content)
            if any(lid in self.raise_triage_for for lid in batch_ids):
                raise make_validation_error()
            for listing_id, _title in re.findall(r'<listing id="([^"]+)"[^>]*>([^<]*)</listing>', content):
                rejected = listing_id in self.reject_triage
                if listing_id in self.keep_flag_wrong:
                    decisions.append(TriageDecision(
                        item_id=listing_id, keep=False,
                        relevance=self.keep_flag_wrong[listing_id], reason="fake triage (keep wrong)",
                    ))
                    continue
                decisions.append(
                    TriageDecision(
                        item_id=listing_id,
                        keep=not rejected,
                        relevance=0 if rejected else self.relevance.get(listing_id, 5),
                        reason="fake triage",
                    )
                )
            return TriageBatch(decisions=decisions)

        if output_model is ExtractionDraft:
            text = next(block["text"] for block in content if block.get("type") == "text")
            listing_id = re.search(r"Listing (\S+) \(untrusted", text).group(1)
            self.extract_calls.append(listing_id)
            if listing_id in self.fail_extract:
                raise LLMError("vision model overloaded")
            if listing_id not in self.extractions:
                return ExtractionDraft(model_confidence=0.0, identification_evidence="none")
            found = self.extractions[listing_id]
            return ExtractionDraft(**found.model_dump(exclude={"item_id"}))

        if output_model is RankingOutput:
            if self.fail_rank:
                raise LLMError("ranker exploded")
            block = re.search(r"<candidates>\n(.*)\n</candidates>", content, re.S).group(1)
            candidates = json.loads(block)
            self.rank_payloads.append(candidates)
            rankings = []
            for c in candidates:
                score = self.scores.get(c["item_id"], 50)
                rankings.append(
                    RankedListing(
                        item_id=c["item_id"],
                        score=score,
                        verdict="top_pick" if score >= 80 else "good" if score >= 60 else "ok",
                        meets_hard_requirements=c["item_id"] not in self.unmet,
                        headline=f"Headline {c['item_id']}",
                        value_rationale="fake rationale",
                        model_verdict="fake model verdict",
                        pros=["pro"],
                        cons=[],
                    )
                )
            return RankingOutput(rankings=rankings)

        raise AssertionError(f"unexpected output model {output_model}")

    async def identify_model(self, *, model, brand, title, visual_description, listing_gaps=()):
        from vinted_hunter.models import IdentifyDraft

        self.identify_calls.append(title)
        return self.identifications.get(title, IdentifyDraft(model=None, model_confidence=0.0, evidence="none"))

    async def research_specs(self, *, model, brand, model_name, model_key, spec_keys=()) -> ModelSpecs:
        self.specs_calls.append(model_key)
        self.spec_keys_seen.append(tuple(spec_keys))
        if self.knowledge_only:
            return ModelSpecs(
                brand=brand,
                model=model_name,
                specs=dict(DEFAULT_SPECS),
                summary="from memory",
                variant_match="exact",
                model_key=model_key,
                sources=[],
                lookup_mode="model_knowledge",
            )
        values = self.specs_by_model.get(model_name, DEFAULT_SPECS)
        return ModelSpecs(
            brand=brand,
            model=model_name,
            specs=dict(values),
            typical_used_price_eur=35.0,
            new_price_eur=self.new_prices.get(model_name),
            new_price_source="https://shop.example/new" if model_name in self.new_prices else None,
            variant_match=self.specs_variant,
            summary=f"{brand} {model_name} is a reflex camera.",
            model_key=model_key,
            sources=["https://example.com/spec"],
            lookup_mode="web",
        )


def make_summary(item_id: str, title: str, total: float, url: str | None = None) -> ListingSummary:
    return ListingSummary(
        id=item_id,
        title=title,
        url=url or f"https://www.vinted.fr/items/{item_id}-slug",
        price_eur=round(total - 2.0, 2),
        service_fee_eur=2.0,
        total_price_eur=total,
        photo_url=f"https://images1.vinted.net/t/{item_id}/f800/a.webp",
    )


def make_detail(summary: ListingSummary, **overrides: Any) -> ListingDetail:
    base = summary.model_dump()
    base.update(
        description="Appareil photo en bon état.",
        photo_urls=[f"https://images1.vinted.net/t/{summary.id}/f800/a.webp"],
        brand=None,
        condition="Très bon état",
        seller_rating=0.98,
        seller_review_count=20,
        can_buy=True,
        is_reserved=False,
    )
    base.update(overrides)
    return ListingDetail(**base)


def build_world(plan):
    a1 = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    b1 = make_summary("b1", "Appareil photo trop cher", 60.0)  # over budget
    c1 = make_summary("c1", "Objectif seul pour reflex", 5.0)  # excluded keyword
    d1 = make_summary("d1", "Appareil photo Beta B2 24 MP", 30.0)
    e1 = make_summary("e1", "Appareil photo Gamma G9", 40.0)  # extraction fails
    f1 = make_summary("f1", "Appareil photo Delta", 35.0)  # triage rejects
    g1 = make_summary("g1", "Appareil photo réservé", 20.0)  # reserved on the item page

    vinted = FakeVinted(
        pages={
            ("appareil photo reflex", 1): [a1, b1, c1, d1],
            ("appareil photo reflex", 2): [a1, e1, f1],  # a1 is a duplicate
            ("reflex 24 MP", 1): [d1, g1],  # d1 is a duplicate
            ("reflex 24 MP", 2): [],
        },
        details={s.id: make_detail(s, brand="Acme" if s.id == "a1" else None) for s in (a1, d1, e1, f1, g1)},
    )
    vinted.details["g1"] = make_detail(g1, is_reserved=True)
    llm = FakeLLM(
        plan=plan,
        extractions={
            "a1": Extraction(
                item_id="a1",
                brand="Acme",
                model="X1",
                model_confidence=0.9,
                identification_evidence="label reads X1",
                claimed_specs=[SpecEntry(name="resolution", value="24 MP")],
            ),
            "d1": Extraction(item_id="d1", model_confidence=0.2, identification_evidence="unclear"),
            "e1": Extraction(item_id="e1", brand="Gamma", model="G9", model_confidence=0.9),
        },
        fail_extract={"e1"},
        reject_triage={"f1"},
        relevance={"a1": 9, "d1": 4, "e1": 6, "g1": 5, "f1": 8},
        scores={"a1": 85, "d1": 40},
    )
    return vinted, llm
