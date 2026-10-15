"""Vinted access: HTML parsing of the Next.js flight data and a throttled HTTP client.

The catalog JSON endpoint is gone, so listings are read from the RSC payload embedded in
the HTML pages (`self.__next_f.push([1, "..."])` chunks).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import random
import re
import time
import unicodedata
from collections.abc import Callable, Iterable, Mapping, Sequence
from decimal import ROUND_FLOOR, ROUND_HALF_UP, Decimal
from typing import Any
from urllib.parse import urlencode, urljoin

from curl_cffi import requests as curl_requests
from curl_cffi.requests import RequestsError

from .config import Settings
from .models import ListingDetail, ListingSummary

log = logging.getLogger(__name__)

MAX_RETRIES = 3
RETRY_BASE_DELAY_S = 4.0
RETRY_STATUSES = {401, 403, 429}
MAX_IMAGE_BYTES = 5 * 1024 * 1024
IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}

_FLIGHT_RE = re.compile(r'self\.__next_f\.push\(\[1,("(?:[^"\\]|\\.)*")\]\)')
_ATTRIBUTE_RE = re.compile(r'\{"code":"(\w+)","data":(\{[^{}]*\})')
_DESCRIPTION_RE = re.compile(r'"description":"((?:[^"\\]|\\.)*)"')
_ITEM_ID_TAG_RE = re.compile(r'"item_id":"(\d+)"')
_ITEMS_ARRAY_OWNER_RE = re.compile(r'"items":\{"items":')
_ITEMS_OWNER_RE = re.compile(r'"items":\{')
PLUGIN_WINDOW = 60_000
PLUGIN_CANDIDATE_LIMIT = 100  # per item_id reference: bounds the walk back through {"data":{ openers
_JSON_DECODER = json.JSONDecoder()
_FEEDBACK_COUNT_RE = re.compile(r'"feedback_count":(\d+)')
_FEEDBACK_REPUTATION_RE = re.compile(r'"feedback_reputation":([0-9.]+)')


class VintedError(Exception):
    """Any failure talking to Vinted."""


class VintedBlocked(VintedError):
    """Vinted kept refusing (401/403/429) after retries."""


class VintedNotFound(VintedError):
    """The page does not exist (404). Usually a listing that was removed."""


class VintedParseError(VintedError):
    """A page did not contain the data we expected."""


# --- Pure parsing helpers ----------------------------------------------------


def flight_text(html: str) -> str:
    """Concatenate the decoded strings of every Next.js flight chunk in `html`."""
    parts: list[str] = []
    for match in _FLIGHT_RE.finditer(html):
        try:
            parts.append(json.loads(match.group(1)))
        except json.JSONDecodeError:
            log.debug("skipping undecodable flight chunk")
    return "".join(parts)


def extract_value(text: str, key: str) -> Any:
    """Parse the JSON array or object that starts right after `key`.

    `key` must end with '[' or '{', which is the opening bracket of the value.
    Returns None when the key is missing or the value is not valid JSON.
    """
    index = text.find(key)
    if index < 0:
        return None
    return _decode_value_at(text, index + len(key) - 1)


def _decode_value_at(text: str, start: int) -> Any:
    """Decode the JSON value starting at text[start]. None if it does not parse.

    Deeply nested or unbalanced input can exhaust the parser's recursion; that also means None.
    """
    try:
        value, _end = _JSON_DECODER.raw_decode(text, start)
    except (ValueError, RecursionError):
        return None
    return value


def _money(value: Any) -> float | None:
    if isinstance(value, dict):
        value = value.get("amount")
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(str(value))
    except ValueError:
        return None


def _absolute(base_url: str, url: str) -> str:
    return urljoin(base_url.rstrip("/") + "/", url)


def summary_from_product(product: dict[str, Any], base_url: str) -> ListingSummary | None:
    """Build a ListingSummary from a `productItem` object of the search payload."""
    item_id = product.get("id")
    title = product.get("title")
    url = product.get("url")
    if item_id is None or not isinstance(title, str) or not isinstance(url, str):
        return None
    price = _money(product.get("price"))
    fee = _money(product.get("serviceFee"))
    total = _money(product.get("totalItemPrice"))
    if total is None and price is not None:
        total = round(price + (fee or 0.0), 2)
    photos = product.get("photos")
    photo_url = None
    if isinstance(photos, list) and photos and isinstance(photos[0], dict):
        photo_url = photos[0].get("url")
    if not isinstance(photo_url, str):
        photo_url = product.get("thumbnailUrl") if isinstance(product.get("thumbnailUrl"), str) else None
    favourites = product.get("favouriteCount")
    return ListingSummary(
        id=str(item_id),
        title=title,
        url=_absolute(base_url, url),
        price_eur=price,
        service_fee_eur=fee,
        total_price_eur=total,
        photo_url=photo_url,
        favourite_count=favourites if isinstance(favourites, int) else None,
        is_promoted=bool(product.get("isPromoted")),
    )


def _is_zero_results_pagination(pagination: Any) -> bool:
    """Pagination of a results section that found nothing: total_entries 0, or the undefined shape
    Vinted sends for a page past the last one."""
    if pagination == "$undefined":
        return True
    return isinstance(pagination, dict) and pagination.get("total_entries") == 0


def _search_items(flight: str) -> list[Any]:
    """The items list of the results section, or [] for a genuine zero-result page.

    Only the section's own items and pagination count. Pagination elsewhere on the page is ignored.
    Raises VintedParseError when the section is missing, unreadable, or inconsistent.
    """
    match = _ITEMS_ARRAY_OWNER_RE.search(flight)
    if match is not None:
        section = _decode_value_at(flight, match.end() - len('"items":') - 1)
        if not isinstance(section, dict):
            raise VintedParseError("search results section is present but not readable")
        items = section.get("items")
        if not isinstance(items, list):
            raise VintedParseError("search results items value is not a list")
        if items:
            return items
        if _is_zero_results_pagination(section.get("pagination")):
            return []
        raise VintedParseError("search results are empty without a zero-results pagination")

    # No items array in the section: accept it only when the section itself reports zero results.
    first = _ITEMS_OWNER_RE.search(flight)
    if first is not None:
        section = _decode_value_at(flight, first.end() - 1)
        if isinstance(section, dict) and "items" not in section and _is_zero_results_pagination(
            section.get("pagination")
        ):
            return []
    raise VintedParseError("search page has no readable items array")


def parse_search_page(html: str, base_url: str) -> list[ListingSummary]:
    """Return the listings on one catalog results page.

    Raises VintedParseError when the items section is missing, unreadable, or inconsistent, which
    usually means Vinted changed its page format. A genuine zero-result page returns [].
    """
    raw = _search_items(flight_text(html))
    summaries: list[ListingSummary] = []
    for entry in raw:
        product = entry.get("productItem") if isinstance(entry, dict) else None
        if not isinstance(product, dict):
            continue
        try:
            summary = summary_from_product(product, base_url)
        except (TypeError, ValueError):
            log.warning("skipping malformed search item")
            continue
        if summary is not None:
            summaries.append(summary)
    return summaries


def _photo_full_url(photo: Any) -> str | None:
    if not isinstance(photo, dict):
        return None
    url = photo.get("url")
    if isinstance(url, str) and url.startswith("https://"):
        return url
    thumbs = photo.get("thumbnails")
    if isinstance(thumbs, list):
        candidates = [t for t in thumbs if isinstance(t, dict) and isinstance(t.get("url"), str)]
        if candidates:
            best = max(candidates, key=lambda t: t.get("width") or 0)
            return best["url"]
    return None


def _item_plugin_value(flight: str, item_id: str, key: str) -> Any:
    """Value of `key` in the plugin data object that belongs to `item_id`, or None.

    The description and attributes are not on the item object. They sit in plugin data objects
    shaped like {"data": {...,"item_id": "<id>"}}. We find each reference to our item_id, walk
    back through the nearby {"data":{ openers, and take the first object whose item_id is ours and
    which holds `key` anywhere in its fields. Position within the object does not matter.
    """
    needle = f'"item_id":"{item_id}"'
    seen: set[int] = set()
    for match in re.finditer(re.escape(needle), flight):
        low = max(0, match.start() - PLUGIN_WINDOW)
        start = flight.rfind('{"data":{', low, match.start())
        attempts = 0
        while start >= low and attempts < PLUGIN_CANDIDATE_LIMIT:
            attempts += 1
            if start not in seen:
                seen.add(start)
                inner = _decode_value_at(flight, start + len('{"data":'))
                if isinstance(inner, dict) and str(inner.get("item_id")) == item_id and key in inner:
                    return inner[key]
            start = flight.rfind('{"data":{', low, start)
    return None


def _item_ids_tagged(flight: str) -> set[str]:
    return set(_ITEM_ID_TAG_RE.findall(flight))


def _attribute_values(entries: Any) -> dict[str, str]:
    values: dict[str, str] = {}
    if not isinstance(entries, list):
        return values
    for entry in entries:
        if not isinstance(entry, dict) or not isinstance(entry.get("code"), str):
            continue
        data = entry.get("data")
        value = data.get("value") if isinstance(data, dict) else None
        if isinstance(value, str):
            values.setdefault(entry["code"], value)
    return values


def _attributes(flight: str) -> dict[str, str]:
    """Whole-page fallback: map attribute codes (status, upload_date, color, brand...) to values."""
    values: dict[str, str] = {}
    for match in _ATTRIBUTE_RE.finditer(flight):
        code = match.group(1)
        if code in values:
            continue
        try:
            data = json.loads(match.group(2))
        except json.JSONDecodeError:
            continue
        value = data.get("value")
        if isinstance(value, str):
            values[code] = value
    return values


def _seller(flight: str) -> dict[str, Any]:
    """Seller reputation from the object that holds `feedback_reputation`."""
    index = flight.find('"feedback_reputation":')
    if index < 0:
        return {}
    start = flight.rfind('{"badges"', 0, index)
    if start >= 0:
        seller = _decode_value_at(flight, start)
        if isinstance(seller, dict) and "feedback_reputation" in seller:
            return seller
    fallback: dict[str, Any] = {}
    window = flight[index : index + 400]
    count = _FEEDBACK_COUNT_RE.search(flight[max(0, index - 200) : index + 400])
    reputation = _FEEDBACK_REPUTATION_RE.search(window)
    if count:
        fallback["feedback_count"] = int(count.group(1))
    if reputation:
        fallback["feedback_reputation"] = float(reputation.group(1))
    return fallback


def _first_description(flight: str) -> str | None:
    """Whole-page fallback: the first "description" string in the payload."""
    match = _DESCRIPTION_RE.search(flight)
    if not match:
        return None
    try:
        value = json.loads('"' + match.group(1) + '"')
    except json.JSONDecodeError:
        return None
    return value.strip() or None


def _flag(value: Any) -> bool | None:
    return value if isinstance(value, bool) else None


def _pricing_shipping(flight: str) -> dict[str, Any]:
    """Shipping fields from the item page's pricingServices block: Vinted's default shipping option, the
    buyer's pickup-only flag, and whether several shipping options exist. Any missing or changed field is None."""
    fields: dict[str, Any] = {
        "shipping_price_eur": None,
        "shipping_is_free": None,
        "shipping_options_multiple": None,
        "pickup_only": None,
    }
    pricing = extract_value(flight, '"pricingServices":{')
    services = pricing.get("services") if isinstance(pricing, dict) else None
    if not isinstance(services, dict):
        return fields
    shipping = services.get("shipping")
    if isinstance(shipping, dict):
        price = _money(shipping.get("finalPrice"))
        if price is not None and math.isfinite(price) and price >= 0:
            fields["shipping_price_eur"] = price
        fields["shipping_is_free"] = _flag(shipping.get("isFree"))
        fields["shipping_options_multiple"] = _flag(shipping.get("areMultipleShippingOptionsAvailable"))
    fields["pickup_only"] = _flag(services.get("isPickUpOnly"))
    return fields


def parse_item_page(html: str, page_url: str) -> ListingDetail:
    """Extract listing details from an item page. Missing fields are left as None."""
    flight = flight_text(html)
    item = extract_value(flight, '"item":{')
    if not isinstance(item, dict):
        raise VintedParseError(f"no item object on {page_url}")
    item_id = item.get("id")
    title = item.get("title")
    if item_id is None or not isinstance(title, str):
        raise VintedParseError(f"item object without id/title on {page_url}")

    photos = item.get("photos")
    if not isinstance(photos, list):
        photos = extract_value(flight, '"photos":[')
    photo_urls = [u for u in (_photo_full_url(p) for p in (photos or [])) if u]

    item_id_text = str(item_id)
    description = _item_plugin_value(flight, item_id_text, "description")
    description = description.strip() if isinstance(description, str) and description.strip() else None
    attributes = _attribute_values(_item_plugin_value(flight, item_id_text, "attributes"))
    dropped: list[str] = []

    # The whole-page fallback is only safe when the page has no item-tagged data for other items.
    # Otherwise the first description or status on the page may belong to a different listing.
    if description is None or not attributes:
        other_items = _item_ids_tagged(flight) - {item_id_text}
        if other_items:
            if description is None:
                dropped.append("description")
            if not attributes:
                dropped.append("attributes")
            log.warning(
                "item %s: no item-tagged plugin data for this listing; page-wide fallback disabled (%s left empty)",
                item_id_text,
                ", ".join(dropped),
            )
        else:
            if description is None:
                description = _first_description(flight)
                if description is not None:
                    log.info("item %s: description taken from page-wide fallback", item_id_text)
            if not attributes:
                attributes = _attributes(flight)
                if attributes:
                    log.info("item %s: attributes taken from page-wide fallback", item_id_text)
    brand_dto = item.get("brand_dto")
    brand = None
    if isinstance(brand_dto, dict) and isinstance(brand_dto.get("title"), str) and brand_dto["title"]:
        brand = brand_dto["title"]
    brand = brand or attributes.get("brand")

    seller = _seller(flight)
    reputation = seller.get("feedback_reputation")
    review_count = seller.get("feedback_count")
    seller_name = seller.get("name")
    can_buy = item.get("can_buy")
    is_reserved = item.get("is_reserved")
    shipping = _pricing_shipping(flight)

    return ListingDetail(
        **shipping,
        id=str(item_id),
        title=title,
        url=page_url,
        photo_url=photo_urls[0] if photo_urls else None,
        description=description,
        photo_urls=photo_urls,
        brand=brand,
        condition=attributes.get("status"),
        color=attributes.get("color"),
        upload_info=attributes.get("upload_date"),
        seller_name=seller_name if isinstance(seller_name, str) else None,
        seller_rating=float(reputation) if isinstance(reputation, (int, float)) else None,
        seller_review_count=review_count if isinstance(review_count, int) else None,
        can_buy=can_buy if isinstance(can_buy, bool) else None,
        is_reserved=is_reserved if isinstance(is_reserved, bool) else None,
        dropped_fields=dropped,
    )


def detect_media_type(data: bytes, content_type: str | None = None) -> str:
    """Pick the image media type from magic bytes, falling back to the Content-Type header."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"GIF8"):
        return "image/gif"
    declared = (content_type or "").split(";")[0].strip().lower()
    if declared in IMAGE_TYPES:
        return declared
    raise VintedError(f"unsupported image type {declared or 'unknown'}")


# Vinted's buyer fee, fitted on the price and serviceFee pairs of the stored search results (40 distinct pairs,
# all within half a cent): fee = price x 5 % + 0.70 EUR, rounded half up to the cent. The buyer's total is
# price + fee. A change in Vinted's fee shows up as a search that misses listings; the slack below absorbs a
# small change, and the local total check in the pipeline still applies the exact budget.
BUYER_FEE_RATE = Decimal("0.05")
BUYER_FEE_FIXED_EUR = Decimal("0.70")
PRICE_TO_SLACK_EUR = Decimal("0.05")
_CENT = Decimal("0.01")


def buyer_fee_eur(price: float) -> float:
    """Estimated Vinted buyer fee for an item price, in EUR."""
    fee = Decimal(str(price)) * BUYER_FEE_RATE + BUYER_FEE_FIXED_EUR
    return float(fee.quantize(_CENT, rounding=ROUND_HALF_UP))


def item_price_ceiling(budget: float) -> float:
    """The highest item price (to the cent) whose estimated total, price plus buyer fee, is within the budget plus a
    small slack. It is sent as Vinted's price_to, which filters on the item price, not on the total. The slack means
    a listing whose true total is within the budget is never cut by rounding. Returns 0 when nothing fits."""
    limit = Decimal(str(budget)) + PRICE_TO_SLACK_EUR
    if limit < BUYER_FEE_FIXED_EUR:
        return 0.0

    def total(price: Decimal) -> Decimal:
        return price + (price * BUYER_FEE_RATE + BUYER_FEE_FIXED_EUR).quantize(_CENT, rounding=ROUND_HALF_UP)

    price = ((limit - BUYER_FEE_FIXED_EUR) / (1 + BUYER_FEE_RATE)).quantize(_CENT, rounding=ROUND_FLOOR)
    while price > 0 and total(price) > limit:
        price -= _CENT
    while total(price + _CENT) <= limit:
        price += _CENT
    return float(max(price, Decimal("0")))


# Vinted's colour filter (color_ids[]), by colour name: the key is the lowercase, accent-free name the planner gives.
# The ids are not in the data the app stores, so the table is empty: no colour is sent as a filter yet, and the colour
# stays in the search text. Filling it needs one live probe of Vinted's colour filter (the ids in its request).
VINTED_COLOR_IDS: dict[str, int] = {}


def color_ids_for(names: Iterable[str], table: Mapping[str, int] | None = None) -> list[int]:
    """Vinted colour ids for the colour names the planner gave, in order, without duplicates. Unmapped names are
    skipped: the search text still carries them."""
    mapping = VINTED_COLOR_IDS if table is None else table
    ids: list[int] = []
    for name in names:
        key = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)).strip().lower()
        color_id = mapping.get(key)
        if color_id is not None and color_id not in ids:
            ids.append(color_id)
    return ids


def search_url(
    base_url: str,
    query: str,
    *,
    price_to: float,
    catalog_ids: list[int],
    page: int,
    color_ids: Sequence[int] = (),
) -> str:
    pairs: list[tuple[str, str]] = [
        ("search_text", query),
        ("price_to", f"{price_to:g}"),
        ("page", str(page)),
        ("order", "relevance"),
    ]
    pairs.extend(("catalog[]", str(cid)) for cid in catalog_ids)
    pairs.extend(("color_ids[]", str(cid)) for cid in color_ids)
    return f"{base_url.rstrip('/')}/catalog?{urlencode(pairs)}"


# --- HTTP client -------------------------------------------------------------


def _default_session() -> Any:
    return curl_requests.AsyncSession(impersonate="chrome", timeout=30)


class VintedClient:
    """Serialised, throttled access to vinted.fr with browser impersonation."""

    def __init__(
        self,
        settings: Settings,
        *,
        session_factory: Callable[[], Any] = _default_session,
        sleep: Callable[[float], Any] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        jitter: Callable[[], float] = random.random,
    ) -> None:
        self._base = settings.vinted_base_url.rstrip("/")
        self._min_delay = settings.vinted_min_delay_s
        self._session_factory = session_factory
        self._sleep = sleep
        self._clock = clock
        self._jitter = jitter
        self._session: Any = None
        self._lock = asyncio.Lock()
        self._last_request_at: float | None = None

    @property
    def base_url(self) -> str:
        return self._base

    async def close(self) -> None:
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _send(self, url: str) -> Any:
        """One throttled GET. Caller holds the lock."""
        if self._last_request_at is not None:
            # The minimum delay varies by +/-30% per request so the traffic does not look machine-regular.
            delay = self._min_delay * (0.7 + 0.6 * self._jitter())
            wait = delay - (self._clock() - self._last_request_at)
            if wait > 0:
                await self._sleep(wait)
        try:
            response = await self._session.get(url)
        finally:
            self._last_request_at = self._clock()
        return response

    async def _warm(self) -> None:
        """Fetch the homepage to obtain cookies. Caller holds the lock."""
        if self._session is None:
            self._session = self._session_factory()
        try:
            await self._send(self._base + "/")
        except RequestsError as exc:
            log.warning("homepage warm-up failed: %s", exc)

    async def _fetch(self, url: str) -> Any:
        async with self._lock:
            if self._session is None:
                await self._warm()
            attempt = 0
            while True:
                try:
                    response = await self._send(url)
                except RequestsError as exc:
                    failure = f"network error: {exc}"
                else:
                    if response.status_code == 200:
                        return response
                    if response.status_code == 404:
                        raise VintedNotFound(f"HTTP 404 for {url}")
                    if response.status_code not in RETRY_STATUSES and response.status_code < 500:
                        raise VintedError(f"HTTP {response.status_code} for {url}")
                    failure = f"HTTP {response.status_code}"
                if attempt >= MAX_RETRIES:
                    raise VintedBlocked(f"{failure} after {attempt} retries: {url}")
                attempt += 1
                delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
                log.warning("%s, retry %d in %.0fs", failure, attempt, delay)
                await self._sleep(delay)
                await self._warm()

    async def search(
        self,
        query: str,
        *,
        price_to: float,
        catalog_ids: list[int],
        page: int,
        color_ids: Sequence[int] = (),
    ) -> list[ListingSummary]:
        url = search_url(
            self._base, query, price_to=price_to, catalog_ids=catalog_ids, page=page, color_ids=color_ids
        )
        response = await self._fetch(url)
        return parse_search_page(response.text, self._base)

    async def item_detail(self, url: str) -> ListingDetail:
        absolute = _absolute(self._base, url)
        response = await self._fetch(absolute)
        return parse_item_page(response.text, absolute)

    async def download_image(self, url: str) -> tuple[bytes, str]:
        response = await self._fetch(url)
        data = response.content
        if len(data) > MAX_IMAGE_BYTES:
            raise VintedError(f"image too large ({len(data)} bytes)")
        headers = getattr(response, "headers", {}) or {}
        media_type = detect_media_type(data, headers.get("content-type"))
        return data, media_type
