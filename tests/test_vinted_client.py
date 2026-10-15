from types import SimpleNamespace

import pytest
from curl_cffi.requests import RequestsError
from helpers import PNG_1X1, flight_html, item_page_html, product, search_payload

from vinted_hunter.config import Settings
from vinted_hunter.vinted import VintedBlocked, VintedClient, VintedNotFound


class FakeSession:
    """Returns queued responses per URL-independent call. Records every GET."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.urls = []
        self.closed = False

    async def get(self, url):
        self.urls.append(url)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def close(self):
        self.closed = True


def resp(status=200, text="", content=b"", headers=None):
    return SimpleNamespace(status_code=status, text=text, content=content, headers=headers or {})


class Clock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def make_client(session, clock, min_delay=1.0, jitter=0.5):
    # jitter=0.5 is the midpoint: the delay is exactly min_delay.
    settings = Settings(vinted_min_delay_s=min_delay)
    return VintedClient(
        settings,
        session_factory=lambda: session,
        sleep=clock.sleep,
        clock=clock,
        jitter=lambda: jitter,
    )


async def test_minimum_delay_varies_within_thirty_percent():
    for jitter, expected in ((0.0, 1.4), (1.0, 2.6)):
        session = FakeSession([resp(200, "home"), resp(200, flight_html(search_payload([]))), resp(200, flight_html(search_payload([])))])
        clock = Clock()
        client = make_client(session, clock, min_delay=2.0, jitter=jitter)
        await client.search("a", price_to=10, catalog_ids=[], page=1)
        await client.search("b", price_to=10, catalog_ids=[], page=1)
        # one wait before each search: the warm-up is the first request, so both searches are delayed
        assert clock.sleeps == [pytest.approx(expected), pytest.approx(expected)], (jitter, clock.sleeps)


async def test_search_warms_session_then_parses_results():
    search_html = flight_html(search_payload([product(1, "Appareil photo", price=20.0)]))
    session = FakeSession([resp(200, "home"), resp(200, search_html)])
    clock = Clock()
    client = make_client(session, clock)
    listings = await client.search("appareil photo", price_to=50, catalog_ids=[], page=1)
    assert [s.id for s in listings] == ["1"]
    assert session.urls[0] == "https://www.vinted.fr/"
    assert session.urls[1].startswith("https://www.vinted.fr/catalog?")


async def test_throttle_waits_for_min_delay_between_requests():
    session = FakeSession([resp(200, "home"), resp(200, flight_html(search_payload([]))), resp(200, flight_html(search_payload([])))])
    clock = Clock()
    client = make_client(session, clock, min_delay=2.0)
    await client.search("a", price_to=10, catalog_ids=[], page=1)
    await client.search("b", price_to=10, catalog_ids=[], page=1)
    # Each request after the first waits until 2s have elapsed since the previous one.
    assert clock.sleeps == [pytest.approx(2.0), pytest.approx(2.0)]


async def test_retries_on_429_then_succeeds_and_refreshes_cookies():
    detail_html = item_page_html("111")
    session = FakeSession(
        [
            resp(200, "home"),
            resp(429),
            resp(200, "home-again"),  # cookie refresh after the 429
            resp(200, detail_html),
        ]
    )
    clock = Clock()
    client = make_client(session, clock, min_delay=0)
    detail = await client.item_detail("/items/111-x")
    assert detail.id == "111"
    assert session.urls.count("https://www.vinted.fr/") == 2
    assert len(clock.sleeps) == 1  # one backoff before the retry


async def test_gives_up_after_three_retries():
    # Each attempt is followed by a homepage refresh, so responses alternate.
    session = FakeSession([resp(200, "home")] + [resp(403), resp(200, "home")] * 3 + [resp(403)])
    clock = Clock()
    client = make_client(session, clock, min_delay=0)
    with pytest.raises(VintedBlocked):
        await client.search("x", price_to=10, catalog_ids=[], page=1)
    assert len(clock.sleeps) == 3


async def test_not_found_is_an_error_without_retry():
    session = FakeSession([resp(200, "home"), resp(404)])
    clock = Clock()
    client = make_client(session, clock, min_delay=0)
    with pytest.raises(VintedNotFound, match="HTTP 404"):
        await client.item_detail("/items/1-x")
    assert clock.sleeps == []


async def test_network_error_is_retried():
    session = FakeSession(
        [resp(200, "home"), RequestsError("reset"), resp(200, "home-again"), resp(200, flight_html(search_payload([])))]
    )
    clock = Clock()
    client = make_client(session, clock, min_delay=0)
    listings = await client.search("x", price_to=10, catalog_ids=[], page=1)
    assert listings == []
    assert len(clock.sleeps) == 1


async def test_download_image_returns_bytes_and_media_type():
    session = FakeSession([resp(200, "home"), resp(200, content=PNG_1X1, headers={"content-type": "image/png"})])
    client = make_client(session, Clock(), min_delay=0)
    data, media_type = await client.download_image("https://images1.vinted.net/t/a/f800/x.png")
    assert data == PNG_1X1
    assert media_type == "image/png"


async def test_close_closes_session():
    session = FakeSession([resp(200, "home"), resp(200, flight_html(search_payload([])))])
    client = make_client(session, Clock(), min_delay=0)
    await client.search("x", price_to=10, catalog_ids=[], page=1)
    await client.close()
    assert session.closed
