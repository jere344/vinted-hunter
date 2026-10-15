"""FastAPI app: hunt API, server-sent events, and the static UI."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .cache import HuntCache, list_hunts, load_hunt, save_hunt
from .claude_cli import build_llm, cli_status
from .config import Settings, load_settings
from .llm import LLM, UsageTracker, make_anthropic_client
from .models import HuntOptions, HuntResult
from .pipeline import run_hunt
from .vinted import VintedClient

log = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
HUNT_ID_RE = re.compile(r"^[0-9a-f]{32}$")
TERMINAL_EVENTS = {"result", "error"}
KEEPALIVE_S = 15.0
MAX_IN_MEMORY_HUNTS = 20
ALLOWED_HOSTS = ["127.0.0.1", "localhost"]
ALREADY_RUNNING_MESSAGE = "A hunt is already running. Wait for it to finish before starting another."

LLMFactory = Callable[[Settings, UsageTracker], Any]
HEALTH_CACHE_S = 60.0


class HuntRequest(BaseModel):
    request: str = Field(min_length=3, max_length=500)
    budget: float = Field(gt=0, le=100000)
    options: HuntOptions | None = None


@dataclass
class HuntRecord:
    hunt_id: str
    events: list[dict[str, Any]] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    terminal: bool = False
    done: bool = False
    result: dict[str, Any] | None = None

    @property
    def running(self) -> bool:
        return not (self.terminal or self.done)


def _sse(event: dict[str, Any]) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


async def _stream(record: HuntRecord) -> AsyncIterator[str]:
    index = 0
    while True:
        while index < len(record.events):
            event = record.events[index]
            index += 1
            yield _sse(event)
            if event.get("type") in TERMINAL_EVENTS:
                return
        if record.done:
            return
        waiter = record.changed
        try:
            await asyncio.wait_for(waiter.wait(), KEEPALIVE_S)
        except TimeoutError:
            yield ": keepalive\n\n"


def create_app(
    settings: Settings | None = None,
    *,
    vinted: VintedClient | None = None,
    llm_factory: LLMFactory | None = None,
) -> FastAPI:
    settings = settings or load_settings()
    vinted = vinted or VintedClient(settings)
    cache = HuntCache(settings.data_dir / "cache.sqlite3", specs_ttl_s=settings.specs_nonexact_ttl_hours * 3600)
    records: dict[str, HuntRecord] = {}
    tasks: set[asyncio.Task[None]] = set()
    anthropic_client: Any = None

    def shared_client() -> Any:
        """One AsyncAnthropic client for the whole app, created on first use when a key is set."""
        nonlocal anthropic_client
        if anthropic_client is None and settings.anthropic_api_key:
            anthropic_client = make_anthropic_client(settings)
        return anthropic_client

    # One CLI gate for the whole app: CLI_CONCURRENCY limits subprocesses across all calls.
    cli_gate = asyncio.Semaphore(settings.cli_concurrency)
    health_cache: dict[str, Any] = {"at": 0.0, "cli": None}

    def default_llm(s: Settings, tracker: UsageTracker) -> Any:
        if s.llm_backend == "api":
            return LLM(shared_client(), s, tracker)
        return build_llm(s, tracker, gate=cli_gate)

    make_llm: LLMFactory = llm_factory or default_llm

    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        running = list(tasks)
        for task in running:
            task.cancel()
        if running:
            await asyncio.gather(*running, return_exceptions=True)
        await vinted.close()
        if anthropic_client is not None:
            await anthropic_client.close()

    app = FastAPI(title="Vinted Hunter", version="0.1.0", lifespan=lifespan)
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=ALLOWED_HOSTS)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
    app.state.hunt_records = records

    def make_emit(record: HuntRecord) -> Callable[[dict[str, Any]], None]:
        def emit(event: dict[str, Any]) -> None:
            if event.get("type") == "result" and isinstance(event.get("result"), HuntResult):
                result: HuntResult = event["result"]
                try:
                    save_hunt(settings.data_dir, record.hunt_id, result)
                except OSError as exc:
                    log.error("could not save hunt %s: %s", record.hunt_id, exc)
                payload = result.model_dump(mode="json")
                record.result = payload
                event = {**event, "result": payload}
            if event.get("type") in TERMINAL_EVENTS:
                record.terminal = True
            record.events.append(event)
            record.changed.set()
            record.changed = asyncio.Event()

        return emit

    async def run_record(record: HuntRecord, body: HuntRequest) -> None:
        emit = make_emit(record)
        try:
            await run_hunt(
                body.request,
                body.budget,
                body.options,
                emit,
                settings=settings,
                vinted=vinted,
                llm=make_llm(settings, UsageTracker()),
                cache=cache,
                hunt_id=record.hunt_id,
            )
        except Exception as exc:  # run_hunt already emitted an error for known failures
            if not record.terminal:
                emit({"type": "error", "message": str(exc) or exc.__class__.__name__})
            log.info("hunt %s ended with error: %s", record.hunt_id, exc)
        finally:
            record.done = True
            record.changed.set()

    def replay_from_disk(hunt_id: str) -> HuntRecord | None:
        result = load_hunt(settings.data_dir, hunt_id)
        if result is None:
            return None
        payload = result.model_dump(mode="json")
        record = HuntRecord(hunt_id=hunt_id, done=True, terminal=True, result=payload)
        record.events.append({"type": "result", "result": payload})
        return record

    def lookup(hunt_id: str) -> HuntRecord | None:
        if not HUNT_ID_RE.match(hunt_id):
            return None
        return records.get(hunt_id) or replay_from_disk(hunt_id)

    def prune_records() -> None:
        """Keep at most MAX_IN_MEMORY_HUNTS records. Older ones are still served from disk."""
        while len(records) > MAX_IN_MEMORY_HUNTS:
            oldest_id, oldest = next(iter(records.items()))
            if oldest.running:
                break
            del records[oldest_id]

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health")
    async def health() -> dict[str, Any]:
        payload: dict[str, Any] = {
            "ok": True,
            "backend": settings.llm_backend,
            "api_key_configured": bool(settings.anthropic_api_key),
            "cli": None,
        }
        if settings.llm_backend == "claude-cli":
            loop = asyncio.get_running_loop()
            if loop.time() - health_cache["at"] > HEALTH_CACHE_S or health_cache["cli"] is None:
                health_cache["cli"] = await cli_status(settings)
                health_cache["at"] = loop.time()
            payload["cli"] = health_cache["cli"]
        return payload

    @app.post("/api/hunts")
    async def create_hunt(body: HuntRequest) -> dict[str, str]:
        if any(record.running for record in records.values()):
            raise HTTPException(status_code=409, detail=ALREADY_RUNNING_MESSAGE)
        hunt_id = uuid.uuid4().hex
        record = HuntRecord(hunt_id=hunt_id)
        records[hunt_id] = record
        prune_records()
        task = asyncio.create_task(run_record(record, body))
        tasks.add(task)
        task.add_done_callback(tasks.discard)
        return {"hunt_id": hunt_id}

    @app.get("/api/hunts")
    async def past_hunts() -> list[dict[str, Any]]:
        return list_hunts(settings.data_dir)

    @app.get("/api/hunts/{hunt_id}/events")
    async def hunt_events(hunt_id: str) -> StreamingResponse:
        record = lookup(hunt_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown hunt")
        return StreamingResponse(
            _stream(record),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.get("/api/hunts/{hunt_id}")
    async def hunt_result(hunt_id: str) -> Any:
        record = lookup(hunt_id)
        if record is None:
            raise HTTPException(status_code=404, detail="unknown hunt")
        if record.result is not None:
            return record.result
        if record.terminal:
            error = next((e for e in record.events if e.get("type") == "error"), {})
            return JSONResponse({"status": "error", "message": error.get("message", "")}, status_code=409)
        return JSONResponse({"status": "running"}, status_code=202)

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("vinted_hunter.app:app", host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
