import asyncio
import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient
from helpers import FakeLLM, FakeVinted, make_detail, make_summary

import vinted_hunter.app as app_module
from vinted_hunter.app import ALREADY_RUNNING_MESSAGE, MAX_IN_MEMORY_HUNTS, create_app
from vinted_hunter.config import Settings
from vinted_hunter.llm import NO_KEY_MESSAGE, UsageTracker
from vinted_hunter.models import Extraction, SearchPlan

LOCAL = "http://127.0.0.1"


@pytest.fixture
def settings(tmp_path):
    # The API backend keeps these tests off the real Claude CLI.
    return Settings(
        anthropic_api_key=None,
        data_dir=tmp_path / "data",
        vinted_min_delay_s=0,
        search_pages=1,
        max_detailed=5,
        llm_backend="api",
    )


def read_events(client, hunt_id):
    events = []
    with client.stream("GET", f"/api/hunts/{hunt_id}/events") as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line[len("data: ") :]))
    return events


def success_world():
    plan = SearchPlan(
        product_summary="reflex camera",
        queries=["appareil photo reflex"],
        budget_eur=50,
        hard_requirements=["at least 20 MP"],
    )
    listing = make_summary("a1", "Appareil photo Acme X1 reflex", 45.0)
    vinted = FakeVinted(
        pages={("appareil photo reflex", 1): [listing]},
        details={"a1": make_detail(listing, brand="Acme")},
    )
    llm = FakeLLM(
        plan=plan,
        extractions={"a1": Extraction(item_id="a1", brand="Acme", model="X1", model_confidence=0.9)},
        relevance={"a1": 9},
        scores={"a1": 90},
    )
    return vinted, llm


class HangingLLM:
    """Never answers the planner. Records whether the hunt was cancelled."""

    def __init__(self):
        self.cancelled = False
        self.tracker = UsageTracker()

    def ensure_configured(self):
        return None

    async def parse(self, **_kwargs):
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise


class FakeAnthropicClient:
    instances = 0

    def __init__(self, api_key):
        FakeAnthropicClient.instances += 1
        self.api_key = api_key
        self.closed = False

    async def close(self):
        self.closed = True


def test_index_and_health(settings):
    with TestClient(create_app(settings, vinted=FakeVinted()), base_url=LOCAL) as client:
        page = client.get("/")
        assert page.status_code == 200
        assert "<title>Vinted Hunter</title>" in page.text
        assert client.get("/static/app.js").status_code == 200
        assert client.get("/static/style.css").status_code == 200
        health = client.get("/api/health").json()
        assert health == {"ok": True, "backend": "api", "api_key_configured": False, "cli": None}


def test_untrusted_host_header_is_rejected(settings):
    with TestClient(create_app(settings, vinted=FakeVinted()), base_url="http://evil.example") as client:
        assert client.get("/api/health").status_code == 400


def test_post_without_key_streams_clear_error(settings):
    with TestClient(create_app(settings, vinted=FakeVinted()), base_url=LOCAL) as client:
        response = client.post("/api/hunts", json={"request": "reflex camera", "budget": 50})
        assert response.status_code == 200
        hunt_id = response.json()["hunt_id"]
        events = read_events(client, hunt_id)
        assert events[-1]["type"] == "error"
        assert events[-1]["message"] == NO_KEY_MESSAGE
        assert "ANTHROPIC_API_KEY" in events[-1]["message"]
        status = client.get(f"/api/hunts/{hunt_id}")
        assert status.status_code == 409
        assert "ANTHROPIC_API_KEY" in status.json()["message"]


def test_full_hunt_via_api_saves_and_replays(settings):
    vinted, llm = success_world()
    app = create_app(replace(settings, anthropic_api_key="test"), vinted=vinted, llm_factory=lambda s, t: llm)
    with TestClient(app, base_url=LOCAL) as client:
        response = client.post(
            "/api/hunts",
            json={"request": "reflex camera 24 MP", "budget": 50, "options": {"search_pages": 1}},
        )
        assert response.status_code == 200
        hunt_id = response.json()["hunt_id"]
        assert len(hunt_id) == 32

        events = read_events(client, hunt_id)
        assert events[-1]["type"] == "result"
        ranked = events[-1]["result"]["ranked"]
        assert ranked[0]["listing"]["id"] == "a1"
        assert ranked[0]["ranking"]["verdict"] == "ok"  # no requirement is confirmed: unverified caps at ok

        # A late reader gets the same history.
        assert read_events(client, hunt_id)[-1]["type"] == "result"

        result = client.get(f"/api/hunts/{hunt_id}")
        assert result.status_code == 200
        assert result.json()["hunt_id"] == hunt_id

        history = client.get("/api/hunts").json()
        assert history[0]["hunt_id"] == hunt_id
        assert history[0]["top_title"] == "Appareil photo Acme X1 reflex"

    # A fresh app instance (server restart) still serves the saved hunt.
    with TestClient(create_app(replace(settings, anthropic_api_key="test"), vinted=FakeVinted()), base_url=LOCAL) as client:
        assert client.get(f"/api/hunts/{hunt_id}").status_code == 200
        assert read_events(client, hunt_id)[-1]["type"] == "result"


def test_unknown_and_malformed_hunt_ids_are_404(settings):
    with TestClient(create_app(settings, vinted=FakeVinted()), base_url=LOCAL) as client:
        assert client.get("/api/hunts/" + "f" * 32).status_code == 404
        assert client.get("/api/hunts/not-a-hunt/events").status_code == 404
        assert client.get("/api/hunts/%2e%2e%2fsecret").status_code == 404


def test_request_validation(settings):
    with TestClient(create_app(settings, vinted=FakeVinted()), base_url=LOCAL) as client:
        assert client.post("/api/hunts", json={"request": "ab", "budget": 50}).status_code == 422
        assert client.post("/api/hunts", json={"request": "video", "budget": 0}).status_code == 422
        bad_options = {"request": "video", "budget": 50, "options": {"max_detailed": 500}}
        assert client.post("/api/hunts", json=bad_options).status_code == 422


def test_second_hunt_is_refused_while_one_runs_and_running_hunt_is_cancelled_on_shutdown(settings):
    llm = HangingLLM()
    app = create_app(replace(settings, anthropic_api_key="test"), vinted=FakeVinted(), llm_factory=lambda s, t: llm)
    with TestClient(app, base_url=LOCAL) as client:
        first = client.post("/api/hunts", json={"request": "reflex camera", "budget": 50})
        assert first.status_code == 200
        second = client.post("/api/hunts", json={"request": "another search", "budget": 30})
        assert second.status_code == 409
        assert second.json()["detail"] == ALREADY_RUNNING_MESSAGE
    # Leaving the context runs the lifespan shutdown, which must cancel the running hunt.
    assert llm.cancelled is True


def test_only_one_anthropic_client_is_created_and_it_is_closed(settings, monkeypatch):
    FakeAnthropicClient.instances = 0
    monkeypatch.setattr(app_module, "make_anthropic_client", lambda s: FakeAnthropicClient(s.anthropic_api_key))
    vinted, _ = success_world()
    app = create_app(replace(settings, anthropic_api_key="test"), vinted=vinted)
    created_clients = []
    with TestClient(app, base_url=LOCAL) as client:
        for _ in range(2):
            hunt_id = client.post("/api/hunts", json={"request": "reflex camera", "budget": 50}).json()["hunt_id"]
            events = read_events(client, hunt_id)
            assert events[-1]["type"] == "error"  # the stub client has no SDK surface; that is fine here
        created_clients.append(FakeAnthropicClient.instances)
    assert created_clients == [1]
    assert FakeAnthropicClient.instances == 1


def test_old_hunts_are_dropped_from_memory_but_still_served_from_disk(settings):
    vinted, llm = success_world()
    app = create_app(replace(settings, anthropic_api_key="test"), vinted=vinted, llm_factory=lambda s, t: llm)
    with TestClient(app, base_url=LOCAL) as client:
        ids = []
        for _ in range(MAX_IN_MEMORY_HUNTS + 2):
            hunt_id = client.post("/api/hunts", json={"request": "reflex camera", "budget": 50}).json()["hunt_id"]
            read_events(client, hunt_id)
            ids.append(hunt_id)
        assert len(app.state.hunt_records) <= MAX_IN_MEMORY_HUNTS
        assert ids[0] not in app.state.hunt_records
        assert client.get(f"/api/hunts/{ids[0]}").status_code == 200
