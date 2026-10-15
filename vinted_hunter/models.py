"""Pydantic models for plans, listings, LLM outputs and hunt results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field, field_validator

# --- Planner -----------------------------------------------------------------


class HardRequirement(BaseModel):
    """A buyer's hard requirement. The text is checked by vision. When spec_key and comparator are set, a
    researched spec of the identified model can also settle it (see pipeline.spec_verdict)."""

    text: str = Field(description="The buyer's requirement, phrased as a checkable fact.")
    spec_key: str | None = Field(
        default=None,
        description="One of the spec_keys, when a researched spec of the model settles this requirement. Else null.",
    )
    comparator: Literal[">=", "<=", "==", "contains", "present", "absent"] | None = Field(
        default=None,
        description="How the spec value is compared with the target: present or absent for a yes/no feature "
        "(target null), contains for an enumerated value, >=, <= or == for a number. Null when spec_key is null.",
    )
    target: str | None = Field(
        default=None,
        description="The value to compare with, with its unit, e.g. '16 GB' or '30 x 20 cm'. Null when spec_key is "
        "null, and for present and absent.",
    )


class SearchPlan(BaseModel):
    product_summary: str = Field(description="One line: what product the buyer wants.")
    product_type: str = Field(
        default="",
        description="The specific product type, e.g. 'true wireless in-ear earbuds'. Listings of another type are wrong.",
    )
    not_product_types: list[str] = Field(
        default_factory=list,
        description="Product types that look similar but are wrong, e.g. ['over-ear headphones', 'casque', 'speaker'].",
    )
    spec_keys: list[str] = Field(
        default_factory=list,
        description="3 to 8 names of the specs that decide this product category, for example "
        "['ram', 'storage', 'cpu'] for a laptop, ['size', 'material'] for shoes, "
        "['sensor size', 'resolution', 'shutter count'] for a camera.",
    )
    risk_terms: list[str] = Field(
        default_factory=list,
        description="Optional words or short phrases that name a defect typical of this product category, "
        "for example 'batterie HS' or 'charniere cassee'. Matched accent-insensitively against red flags.",
    )
    queries: list[str] = Field(
        description="3 to 8 Vinted search strings, mostly French, including generic queries "
        "and specific good-value model names that fit the budget."
    )
    catalog_ids: list[int] = Field(
        default_factory=list,
        description="Vinted catalog ids to restrict the search to. Empty unless you are certain of the id.",
    )
    colors: list[str] = Field(
        default_factory=list,
        description="Colour names the buyer asked for, in French, e.g. ['noir']. Empty when no colour was named.",
    )
    color_ids: list[int] = Field(
        default_factory=list,
        description="Vinted colour ids, filled by the code from colors. The planner leaves this empty.",
    )
    budget_eur: float
    hard_requirements: list[HardRequirement] = Field(default_factory=list)
    nice_to_haves: list[str] = Field(default_factory=list)
    exclude_keywords: list[str] = Field(
        default_factory=list,
        description="Lowercase words that signal the wrong product (accessories, parts...).",
    )
    pitfalls: list[str] = Field(default_factory=list)
    triage_guidance: str = Field(default="", description="Guidance for the relevance triage.")

    @field_validator("hard_requirements", mode="before")
    @classmethod
    def _plain_strings_are_unbound_requirements(cls, value: Any) -> Any:
        """Older plans and some planner answers give requirements as plain text: no spec binding."""
        if isinstance(value, list):
            return [{"text": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("catalog_ids", "color_ids", "colors", "not_product_types", mode="before")
    @classmethod
    def _null_list_is_empty(cls, value: Any) -> Any:
        return [] if value is None else value


# --- Vinted listings ---------------------------------------------------------


class ListingSummary(BaseModel):
    id: str
    title: str
    url: str
    price_eur: float | None = None
    service_fee_eur: float | None = None
    total_price_eur: float | None = None
    photo_url: str | None = None
    favourite_count: int | None = None
    is_promoted: bool = False


class ListingDetail(ListingSummary):
    description: str | None = None
    photo_urls: list[str] = Field(default_factory=list)
    brand: str | None = None
    condition: str | None = None
    color: str | None = None
    upload_info: str | None = Field(default=None, description="Relative upload date as shown by Vinted.")
    seller_name: str | None = None
    seller_rating: float | None = Field(default=None, description="Feedback reputation, 0 to 1.")
    seller_review_count: int | None = None
    can_buy: bool | None = None
    is_reserved: bool | None = None
    dropped_fields: list[str] = Field(
        default_factory=list,
        description="Fields left empty because the page carried item-tagged data only for other listings.",
    )
    # Only on the item page, never on the search page: _with_summary merges the summary over the detail,
    # so shipping fields on ListingSummary would overwrite them with None.
    shipping_price_eur: float | None = Field(
        default=None,
        description="Vinted's default shipping option, paid by the buyer. An estimate: other options may cost more.",
    )
    shipping_is_free: bool | None = None
    shipping_options_multiple: bool | None = Field(
        default=None, description="True when Vinted offers several shipping options for this item."
    )
    pickup_only: bool | None = None

    @computed_field
    @property
    def shipping_known(self) -> bool:
        """True when the shipping cost is known: a price, free shipping, or pickup only (no shipping at all)."""
        return bool(self.pickup_only or self.shipping_is_free or self.shipping_price_eur is not None)

    @computed_field
    @property
    def landed_price_eur(self) -> float | None:
        """Total incl. buyer fees plus shipping. None when the shipping cost is unknown. Pickup only and free
        shipping add nothing."""
        total = self.total_price_eur
        if total is None:
            return None
        if self.pickup_only or self.shipping_is_free:
            return round(total, 2)
        if self.shipping_price_eur is None:
            return None
        return round(total + self.shipping_price_eur, 2)


# --- LLM outputs -------------------------------------------------------------


class TriageDecision(BaseModel):
    item_id: str
    keep: bool
    relevance: int = Field(description="0 to 10: how likely this is the right product at a good price.")
    reason: str


class TriageBatch(BaseModel):
    decisions: list[TriageDecision]


class SpecEntry(BaseModel):
    name: str
    value: str


class RequirementCheck(BaseModel):
    requirement: str
    status: Literal["met", "unmet", "unknown"]
    evidence: str = ""
    hard_requirement_no: int = Field(
        default=0,
        description="The number of the hard requirement this check answers, as listed. 0 for anything else.",
    )


class ExtractionDraft(BaseModel):
    """What the vision model fills in. The item_id is set by the pipeline, never by the model."""

    brand: str | None = None
    model: str | None = None
    model_confidence: float = Field(default=0.0, description="0 to 1.")
    identification_evidence: str = ""
    visual_description: str = Field(
        default="", description="One short line on the visible item (colour, shape, ports, label text) for a web search."
    )
    claimed_specs: list[SpecEntry] = Field(default_factory=list)
    condition_assessment: str = ""
    included_accessories: list[str] = Field(default_factory=list)
    is_requested_product_type: bool = Field(
        default=True,
        description="True when the item is the product type the buyer asked for; false for another product type.",
    )
    product_type_evidence: str = ""
    defect_severity: Literal["none", "cosmetic", "functional", "unusable"] = Field(
        default="none",
        description="none: no defect. cosmetic: wear only. functional: a part does not work properly. "
        "unusable: the item does not work at all. Used condition itself is not a defect.",
    )
    defect_note: str = Field(default="", description="A short explanation of the defect, or empty.")
    red_flags: list[str] = Field(
        default_factory=list, description="Only about the physical item and the transaction."
    )
    listing_gaps: list[str] = Field(
        default_factory=list,
        description="Information the listing does not provide, e.g. a spec not stated, a model number not shown, "
        "a battery health not given. Never a red flag.",
    )
    requirement_checks: list[RequirementCheck] = Field(default_factory=list)


class Extraction(ExtractionDraft):
    item_id: str


class _SpecsFields(BaseModel):
    brand: str | None = None
    model: str | None = None
    variant_match: Literal["exact", "base_model", "unknown"] = Field(
        default="unknown",
        description="exact: the facts describe this exact variant. base_model: only a base model or sibling. "
        "unknown: not clear.",
    )
    new_price_eur: float | None = Field(
        default=None, description="Current retail price of the new product in EUR, from a reputable shop. Null if unknown."
    )
    new_price_source: str | None = Field(default=None, description="Where the new price comes from, or null.")
    typical_used_price_eur: float | None = None
    price_notes: str = Field(
        default="",
        description="Where the prices came from (French or European sources), and whether any were converted from USD.",
    )
    known_issues: list[str] = Field(default_factory=list)
    summary: str = ""


class ModelSpecsDraft(_SpecsFields):
    """What the specs model returns. The pipeline adds model_key, sources and lookup_mode.

    The specs are a list of name and value pairs here, not a map: the structured-output schema of the
    API cannot express an open map, so the pipeline turns the list into ModelSpecs.specs.
    """

    specs: list[SpecEntry] = Field(
        default_factory=list,
        description="One entry per spec key named in the request: the name is the key exactly as given, the value "
        "is a string with its unit, e.g. 16 GB. Omit a key the notes do not state; never guess.",
    )


class ModelSpecs(_SpecsFields):
    specs: dict[str, str] = Field(
        default_factory=dict,
        description="Spec key to value string with its unit. Only exact web specs are used to settle a requirement.",
    )
    model_key: str = ""
    sources: list[str] = Field(default_factory=list)
    lookup_mode: Literal["web", "model_knowledge"] = "web"


def specs_dict(entries: Iterable[SpecEntry]) -> dict[str, str]:
    """Spec entries as a map. Empty names and values are dropped. A name given twice with different
    values keeps both values joined by " / ", so a comparison sees a conflict and fails closed."""
    values: dict[str, str] = {}
    for entry in entries:
        name = entry.name.strip()
        value = entry.value.strip()
        if not name or not value:
            continue
        if name in values and values[name] != value:
            values[name] = f"{values[name]} / {value}"
        else:
            values.setdefault(name, value)
    return values


class IdentifyDraft(BaseModel):
    """What the web identification step returns for a listing with an unclear model."""

    model: str | None = None
    model_confidence: float = Field(default=0.0, description="0 to 1.")
    evidence: str = ""


class RankedListing(BaseModel):
    item_id: str
    score: int = Field(description="0 to 100, higher is better value.")
    verdict: Literal["top_pick", "good", "ok", "avoid"]
    meets_hard_requirements: bool
    headline: str
    value_rationale: str
    model_verdict: str = Field(default="", description="What the researched specs say about the model's quality and value.")
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)


class RankingOutput(BaseModel):
    rankings: list[RankedListing]


# --- Results -----------------------------------------------------------------


class NewPrice(BaseModel):
    """The new price of the model, computed in code. Only web specs give one (see pipeline._new_price)."""

    eur: float
    source: str | None = None
    base_model: bool = Field(default=False, description="True when the price is for the base model, not the exact variant.")
    discount_pct: int | None = Field(
        default=None,
        description="How much cheaper this listing is than the new price, in percent, over the landed cost (the total "
        "when shipping is unknown). Negative when it costs more than new.",
    )


class RankedEntry(BaseModel):
    rank: int
    listing: ListingDetail
    extraction: Extraction | None = None
    specs: ModelSpecs | None = None
    new_price: NewPrice | None = None
    ranking: RankedListing
    # verified: every hard requirement confirmed. unverified: none failed, some unknown.
    # failed: at least one unmet (the score is capped). none: no hard requirements stated.
    # Computed in code.
    requirements_status: Literal["verified", "unverified", "failed", "none"] = "unverified"


class RejectedItem(BaseModel):
    item_id: str
    title: str
    url: str
    stage: str
    reason: str


class StageCost(BaseModel):
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_search_requests: int = 0
    usd: float = 0.0
    priced: bool = True


class CostReport(BaseModel):
    basis: str = "api_list_price"  # or "cli_reported": equivalent cost covered by a subscription
    priced: bool = True  # False when some call had no cost; total_usd is then a lower bound
    stages: dict[str, StageCost] = Field(default_factory=dict)
    total_usd: float = 0.0
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_web_search_requests: int = 0
    note: str = (
        "Estimate from token usage and the price table in config.py. "
        "Web search fees are not included."
    )


class HuntResult(BaseModel):
    hunt_id: str = ""
    request: str
    budget_eur: float
    created_at: str
    plan: SearchPlan
    ranked: list[RankedEntry] = Field(default_factory=list)
    rejected: list[RejectedItem] = Field(default_factory=list)
    stats: dict[str, int] = Field(default_factory=dict)
    cost: CostReport = Field(default_factory=CostReport)
    warnings: list[str] = Field(default_factory=list)


class HuntOptions(BaseModel):
    max_detailed: int | None = Field(default=None, ge=1, le=60)
    search_pages: int | None = Field(default=None, ge=1, le=5)
