import json
import logging
import time
from pathlib import Path

import pytest
from helpers import PNG_1X1, flight_html, item_page_html, pricing_services, product, search_payload

from vinted_hunter.vinted import (
    VintedError,
    VintedParseError,
    _decode_value_at,
    _seller,
    detect_media_type,
    extract_value,
    flight_text,
    parse_item_page,
    parse_search_page,
    search_url,
)

BASE = "https://www.vinted.fr"


def test_flight_text_joins_chunks_and_decodes_escapes():
    html = (
        '<script>self.__next_f.push([1,"0:{\\"a\\":\\"x\\\\\\"y\\"}\\n"])</script>'
        '<script>self.__next_f.push([1,"1:[\\"b\\"]"])</script>'
    )
    text = flight_text(html)
    assert text == '0:{"a":"x\\"y"}\n1:["b"]'


def test_flight_text_skips_undecodable_chunks():
    html = '<script>self.__next_f.push([1,"ok "])</script><script>self.__next_f.push([1,"bad\\q"])</script>'
    assert flight_text(html) == "ok "


def test_extract_value_ignores_brackets_inside_strings():
    text = '"items":{"items":[{"t":"a ] [ } { \\" ]"},{"t":"b"}],"x":1}'
    assert extract_value(text, '"items":{"items":[') == [{"t": 'a ] [ } { " ]'}, {"t": "b"}]


def test_extract_value_missing_key_or_bad_json_returns_none():
    assert extract_value("nothing here", '"items":{"items":[') is None
    assert extract_value('"items":{"items":[1,', '"items":{"items":[') is None


def test_parse_search_page_reads_listings_and_skips_non_items():
    products = [
        product(101, "Appareil photo A", price=40.0, fee=2.5),
        product(102, "Appareil photo B", price=30.0, total=33.0),
    ]
    payload = search_payload(products) + '\n{"notAProduct":{"items":[{"x":1}]}}'
    html = flight_html(payload)
    listings = parse_search_page(html, BASE)
    assert [s.id for s in listings] == ["101", "102"]
    first = listings[0]
    assert first.url == "https://www.vinted.fr/items/101-slug"
    assert first.price_eur == 40.0
    assert first.service_fee_eur == 2.5
    assert first.total_price_eur == 42.5
    assert first.photo_url.endswith("/f800/a.webp")
    assert listings[1].total_price_eur == 33.0


def test_parse_search_page_computes_total_when_missing():
    p = product(7, "No total", price=10.0, fee=1.0)
    del p["totalItemPrice"]
    listings = parse_search_page(flight_html(search_payload([p])), BASE)
    assert listings[0].total_price_eur == 11.0


def test_parse_search_page_empty_items_array_is_a_valid_empty_result():
    assert parse_search_page(flight_html(search_payload([])), BASE) == []


def test_parse_search_page_without_items_array_raises_parse_error():
    with pytest.raises(VintedParseError):
        parse_search_page(flight_html("0:[]"), BASE)
    with pytest.raises(VintedParseError):
        parse_search_page("<html>maintenance</html>", BASE)


def test_parse_item_page_extracts_fields():
    detail = parse_item_page(item_page_html(), "https://www.vinted.fr/items/111-x")
    assert detail.id == "111"
    assert detail.title == "Appareil photo Acme X1 reflex"
    assert detail.brand == "Acme"
    assert detail.condition == "Très bon état"
    assert detail.upload_info == "Il y a 2 jours"
    assert detail.description.startswith('Vendu avec la boîte')
    assert detail.description.endswith("<3")
    assert detail.photo_urls == [
        "https://images1.vinted.net/t/p1/f800/a.webp?s=1",
        "https://images1.vinted.net/t/p2/f800/b.webp?s=2",
    ]
    assert detail.photo_url == detail.photo_urls[0]
    assert detail.seller_name == "seller42"
    assert detail.seller_rating == pytest.approx(0.97)
    assert detail.seller_review_count == 12
    assert detail.can_buy is True
    assert detail.is_reserved is False


def test_parse_item_page_tolerates_missing_optional_fields():
    html = flight_html('{"item":{"id":5,"title":"Bare","photos":[{"url":"http://insecure/x.jpg"}]}}')
    detail = parse_item_page(html, "https://www.vinted.fr/items/5-bare")
    assert detail.id == "5"
    assert detail.description is None
    assert detail.brand is None
    assert detail.seller_rating is None
    # Non-https photo URLs are not trusted.
    assert detail.photo_urls == []


def test_item_fields_are_anchored_to_the_item_id_not_position():
    # The fixture puts decoy description/attribute blocks (another item_id) before the item object.
    detail = parse_item_page(item_page_html("111"), "https://www.vinted.fr/items/111-x")
    assert detail.description.startswith('Vendu avec la boîte')
    assert "DECOY" not in detail.description
    assert detail.condition == "Très bon état"
    assert detail.color == "Blanc"
    assert detail.upload_info == "Il y a 2 jours"


def test_item_fields_fall_back_to_whole_page_when_no_anchored_block():
    html = flight_html(
        '{"item":{"id":5,"title":"Bare"}}',
        '{"description":"Only description"}',
        '{"code":"status","data":{"value":"Neuf"},"type":"faq"}',
    )
    detail = parse_item_page(html, "https://www.vinted.fr/items/5-bare")
    assert detail.description == "Only description"
    assert detail.condition == "Neuf"


def test_parse_item_page_without_item_raises():
    with pytest.raises(VintedParseError):
        parse_item_page(flight_html("0:nothing"), "https://www.vinted.fr/items/1-x")


def test_detect_media_type_by_magic_bytes_and_header():
    assert detect_media_type(PNG_1X1) == "image/png"
    assert detect_media_type(b"\xff\xd8\xff\xe0rest") == "image/jpeg"
    assert detect_media_type(b"RIFF\x00\x00\x00\x00WEBPVP8 ") == "image/webp"
    assert detect_media_type(b"nope", "image/webp; charset=x") == "image/webp"
    with pytest.raises(VintedError):
        detect_media_type(b"<html>", "text/html")


def test_search_url_encodes_query_and_repeats_catalog_ids():
    url = search_url(
        BASE, "appareil photo reflex", price_to=50, catalog_ids=[1234, 5], page=2
    )
    assert url.startswith("https://www.vinted.fr/catalog?")
    assert "search_text=appareil+photo+reflex" in url
    assert "price_to=50" in url
    assert "page=2" in url
    assert "order=relevance" in url
    assert "catalog%5B%5D=1234" in url and "catalog%5B%5D=5" in url


def test_search_payload_is_valid_json_inside_flight():
    text = flight_text(flight_html(search_payload([product(1, "t", price=1.0)])))
    assert '"items":{"items":[' in text
    json.loads(text.split('"items":{"items":', 1)[1].split(',"pagination"', 1)[0])


def test_plugin_key_is_found_anywhere_in_the_data_object():
    # The key comes after other fields, not first. Position inside the object must not matter.
    html = flight_html(
        '{"item":{"id":111,"title":"T"}}',
        '{"data":{"section":"sidebar","layout":{"x":1},"item_id":"111","description":"Anywhere desc"}}',
    )
    detail = parse_item_page(html, "https://www.vinted.fr/items/111-t")
    assert detail.description == "Anywhere desc"


def test_page_with_tags_only_for_other_items_disables_the_page_wide_fallback(caplog):
    html = flight_html(
        '{"item":{"id":111,"title":"T"}}',
        '{"data":{"description":"OTHER LISTING TEXT","item_id":"999"}}',
        '{"description":"BARE PAGE TEXT"}',
        '{"code":"status","data":{"value":"Neuf"},"type":"faq"}',
    )
    with caplog.at_level(logging.WARNING, logger="vinted_hunter.vinted"):
        detail = parse_item_page(html, "https://www.vinted.fr/items/111-t")
    assert detail.description is None
    assert detail.condition is None
    assert "OTHER LISTING TEXT" not in str(detail)
    assert any("page-wide fallback disabled" in r.message for r in caplog.records)


def test_page_without_any_item_tags_uses_the_fallback_and_logs(caplog):
    html = flight_html(
        '{"item":{"id":5,"title":"Bare"}}',
        '{"description":"Only description"}',
    )
    with caplog.at_level(logging.INFO, logger="vinted_hunter.vinted"):
        detail = parse_item_page(html, "https://www.vinted.fr/items/5-bare")
    assert detail.description == "Only description"
    assert any("description taken from page-wide fallback" in r.message for r in caplog.records)


def test_zero_result_page_with_total_entries_zero_is_a_valid_empty_result():
    # Shape seen live for a catalog with no matches: an empty items array plus pagination.
    payload = (
        '2a:["$","div",null,{"items":{"items":[],"pagination":{"current_page":1,'
        '"per_page":96,"time":1791913739,"total_entries":0,"total_pages":1}}}]'
    )
    assert parse_search_page(flight_html(payload), BASE) == []


def test_out_of_range_page_with_empty_items_is_a_valid_empty_result():
    # Shape seen live for a page past the last one: empty items, pagination undefined.
    payload = '2a:["$","div",null,{"items":{"items":[],"pagination":"$undefined","searchTrackingParams":{}}}]'
    assert parse_search_page(flight_html(payload), BASE) == []


def test_missing_items_key_without_zero_marker_still_raises():
    payload = '2a:["$","div",null,{"pagination":{"current_page":1,"total_entries":42,"total_pages":2}}]'
    with pytest.raises(VintedParseError):
        parse_search_page(flight_html(payload), BASE)


def test_zero_results_must_come_from_the_items_section_own_pagination():
    # The items section has no pagination, but an unrelated section elsewhere reports zero entries.
    payload = (
        '2a:["$","div",null,{"items":{"layout":"grid"}}]\n'
        '5a:["$","aside",null,{"pagination":{"current_page":1,"total_entries":0,"total_pages":1}}]'
    )
    with pytest.raises(VintedParseError):
        parse_search_page(flight_html(payload), BASE)


def test_items_array_present_but_undecodable_raises():
    payload = '2a:["$","div",null,{"items":{"items":[{"productItem":{"id":1,"title":"cut off"'
    with pytest.raises(VintedParseError):
        parse_search_page(flight_html(payload), BASE)


def test_items_value_that_is_not_a_list_raises():
    payload = '2a:["$","div",null,{"items":{"items":"oops","pagination":{"total_entries":0}}}]'
    with pytest.raises(VintedParseError):
        parse_search_page(flight_html(payload), BASE)


def test_empty_items_with_nonzero_pagination_raises():
    payload = (
        '2a:["$","div",null,{"items":{"items":[],"pagination":{"current_page":1,'
        '"total_entries":42,"total_pages":2}}}]'
    )
    with pytest.raises(VintedParseError):
        parse_search_page(flight_html(payload), BASE)


def test_absent_items_with_zero_pagination_in_its_own_section_is_empty():
    payload = '2a:["$","div",null,{"items":{"pagination":{"current_page":1,"total_entries":0,"total_pages":1}}}]'
    assert parse_search_page(flight_html(payload), BASE) == []


def test_seller_object_without_reputation_falls_through_to_the_fallback():
    flight = (
        '{"badges":[],"other":1}'
        '{"feedback_count":5,"feedback_reputation":0.5}'
    )
    seller = _seller(flight)
    assert seller == {"feedback_count": 5, "feedback_reputation": 0.5}


def test_decode_value_at_matches_json_on_valid_input():
    text = 'xx{"a":[1,2,{"b":"c}"}],"d":null}yy[true,"]"]'
    for start, expected in [
        (2, json.loads('{"a":[1,2,{"b":"c}"}],"d":null}')),
        (text.index("[true"), [True, "]"]),
    ]:
        assert _decode_value_at(text, start) == expected


def test_unbalanced_openers_parse_quickly():
    openers = '{"data":{' * 3000
    started = time.perf_counter()
    assert _decode_value_at(openers, 0) is None
    html = flight_html(openers + '"item_id":"111"' + '{"item":{"id":111,"title":"T"}}')
    detail = parse_item_page(html, "https://www.vinted.fr/items/111-t")
    elapsed = time.perf_counter() - started
    assert detail.id == "111"
    assert elapsed < 0.5, f"took {elapsed:.2f}s"


def test_page_scoped_to_other_items_reports_dropped_fields():
    html = flight_html(
        '{"item":{"id":111,"title":"T"}}',
        '{"data":{"description":"OTHER","item_id":"999"}}',
        '{"description":"BARE"}',
    )
    detail = parse_item_page(html, "https://www.vinted.fr/items/111-t")
    assert detail.description is None
    assert detail.dropped_fields == ["description", "attributes"]


def test_untagged_page_reports_no_dropped_fields():
    html = flight_html('{"item":{"id":5,"title":"Bare"}}', '{"description":"Only"}')
    detail = parse_item_page(html, "https://www.vinted.fr/items/5-bare")
    assert detail.dropped_fields == []


# --- Shipping (pricingServices) ----------------------------------------------

FIXTURE_ITEM_PAGE = Path(__file__).parent / "fixtures" / "item_page_pricing.html"


def test_item_page_fixture_reads_the_default_shipping_option():
    # A trimmed excerpt of a real item page: the pricingServices row and the item object, as captured.
    html = FIXTURE_ITEM_PAGE.read_text(encoding="utf-8")
    detail = parse_item_page(html, BASE + "/items/10347164722-videoprojecteur")
    assert detail.id == "10347164722"
    assert detail.shipping_price_eur == 5.19
    assert detail.shipping_is_free is False
    assert detail.shipping_options_multiple is True
    assert detail.pickup_only is False
    assert detail.shipping_known is True
    # The item page carries no price of its own: the total comes from the search page, so the landed price waits.
    assert detail.total_price_eur is None
    assert detail.landed_price_eur is None


def test_item_page_without_pricing_services_leaves_shipping_unknown():
    detail = parse_item_page(item_page_html(), BASE + "/items/111-appareil")
    assert detail.shipping_price_eur is None
    assert detail.shipping_is_free is None
    assert detail.shipping_options_multiple is None
    assert detail.pickup_only is None
    assert detail.shipping_known is False


def test_pricing_services_free_shipping_and_pickup_only_are_read():
    free = parse_item_page(item_page_html(pricing=pricing_services(shipping="0.00", is_free=True)), BASE + "/items/1-a")
    assert free.shipping_is_free is True
    assert free.shipping_price_eur == 0.0
    assert free.shipping_known is True

    pickup = parse_item_page(item_page_html(pricing=pricing_services(pickup_only=True)), BASE + "/items/1-a")
    assert pickup.pickup_only is True
    assert pickup.shipping_price_eur == 5.19  # kept as Vinted sent it; the pickup flag decides what applies
    assert pickup.shipping_known is True


def test_pricing_services_without_a_shipping_price_is_unknown_not_free():
    detail = parse_item_page(item_page_html(pricing=pricing_services(shipping=None)), BASE + "/items/1-a")
    assert detail.shipping_price_eur is None
    assert detail.shipping_is_free is False
    assert detail.shipping_known is False


@pytest.mark.parametrize(
    "pricing",
    [
        "not an object",
        {"services": "nope"},
        {"services": {"shipping": ["a", "list"], "isPickUpOnly": "yes"}},
        {"services": {"shipping": {"finalPrice": {"amount": "5,19 EUR"}, "isFree": "no", "areMultipleShippingOptionsAvailable": 1}}},
        {"services": {"shipping": {"finalPrice": {"amount": "NaN"}}}},
        {"services": {"shipping": {"finalPrice": {"amount": "-3"}}}},
        {"services": {"shipping": {"finalPrice": ["5.19"]}}},
        {"totalAmount": {"amount": "45.84"}},
    ],
)
def test_malformed_pricing_services_gives_unknown_shipping_and_never_raises(pricing):
    detail = parse_item_page(item_page_html(pricing=pricing), BASE + "/items/111-appareil")
    assert detail.shipping_price_eur is None
    assert detail.shipping_is_free is None
    assert detail.shipping_known is False
    assert detail.id == "111"


def test_truncated_pricing_services_gives_unknown_shipping():
    html = item_page_html() + (
        '<script>self.__next_f.push([1,' + json.dumps('{"pricingServices":{"services":{"shipping":{"finalPrice":{"amount":"5.1') + '])</script>'
    )
    detail = parse_item_page(html, BASE + "/items/111-appareil")
    assert detail.shipping_price_eur is None
    assert detail.shipping_known is False
