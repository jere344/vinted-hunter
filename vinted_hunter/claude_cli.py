"""Claude Code CLI backend: runs `claude -p` headless on the user's subscription login.

Same interface as `llm.LLM` (ensure_configured, parse, research_specs, tracker), so the pipeline
does not care which backend it talks to.

Every flag below was checked against `claude --help` and small probe calls:
  -p --verbose                      print mode; stream-json output requires --verbose
  --input-format stream-json        one user message with text and image blocks, on stdin
  --output-format stream-json       the final line has type "result"
  --model <alias>                   haiku | sonnet | opus (the CLI resolves each to its current model)
  --effort <level>                  low | medium | high
  --system-prompt <text>            replaces the default Claude Code system prompt
  --json-schema <schema>            the parsed answer comes back as structured_output
  --tools "" | --tools WebSearch    no tools, or only WebSearch (specs research)
  --allowedTools WebSearch          grants WebSearch, which otherwise needs a permission prompt
  --strict-mcp-config               no MCP servers (no account connectors)
  --setting-sources ""              no project or local settings. Verified: the project's CLAUDE.md
                                    and MCP connectors are not loaded. Not verified: user-level memory
                                    (~/.claude/CLAUDE.md); the CLI help does not document whether it is skipped
  --no-session-persistence          nothing is saved to disk

Deliberately not used: --bare (the CLI says it never reads OAuth or keychain login, so it would
break the subscription login) and --dangerously-skip-permissions.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import shutil
import signal
import tempfile
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from . import prompts
from .config import Settings
from .llm import (
    DRAFT_EFFORT,
    LLM,
    RESEARCH_EFFORT,
    WEB_SEARCH_UNAVAILABLE_EVIDENCE,
    LLMConfigError,
    LLMError,
    LLMQuotaError,
    LLMRefusal,
    UsageTracker,
    _WebSearchUnavailable,
    identify_query,
    make_anthropic_client,
    spec_keys_line,
    text_block,
)
from .models import IdentifyDraft, ModelSpecs, ModelSpecsDraft, specs_dict

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

SCRUBBED_ENV_VARS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL")
NO_TOOLS: tuple[str, ...] = ()
WEB_TOOLS = ("WebSearch",)
MAX_ATTEMPTS = 2  # the first answer, plus one retry that feeds the validation error back
STATUS_TIMEOUT_S = 15.0
MAX_SOURCES = 8
RATE_LIMIT_RETRY_DELAYS_S = (5.0, 20.0)  # two retries with backoff before a rate limit becomes fatal

CLI_MISSING_MESSAGE = "Claude Code CLI not found ('{path}'). Install Claude Code and run `claude` once to log in."
NOT_LOGGED_IN_MESSAGE = "Not logged in to Claude Code. Run `claude` once to log in."
USAGE_LIMIT_MESSAGE = "Claude subscription usage limit reached, try later"
RATE_LIMIT_MESSAGE = "Claude is rate-limiting requests, try again in a few minutes"

# Subscription wording only. A generic "limit reached" or "quota" is not enough to call it a plan limit.
_USAGE_LIMIT_RE = re.compile(r"usage limit|5-hour limit|weekly limit|plan limit", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"rate limit|too many requests", re.IGNORECASE)
_NOT_LOGGED_IN_RE = re.compile(r"not logged in|please run /login", re.IGNORECASE)
_SOURCE_URL_RE = re.compile(r'"url":\s*"(https://[^"]+)"')


# --- Pure helpers (unit-tested) ----------------------------------------------


def cli_model_alias(model: str) -> str:
    """Map a model id to the CLI alias for its family. The CLI tracks the current model of each family."""
    for family in ("haiku", "sonnet", "opus"):
        if f"claude-{family}-" in model or model == family:
            return family
    return model


def build_cli_args(
    binary: str,
    *,
    model: str,
    system: str,
    effort: str | None,
    tools: Sequence[str],
    json_schema: dict[str, Any] | None,
) -> list[str]:
    args = [
        binary,
        "-p",
        "--verbose",
        "--input-format", "stream-json",
        "--output-format", "stream-json",
        "--model", cli_model_alias(model),
        "--system-prompt", system,
        "--tools", ",".join(tools),
        "--strict-mcp-config",
        "--setting-sources", "",
        "--no-session-persistence",
    ]
    if tools:
        args += ["--allowedTools", ",".join(tools)]
    if effort:
        args += ["--effort", effort]
    if json_schema is not None:
        args += ["--json-schema", json.dumps(json_schema, ensure_ascii=False)]
    return args


def stream_input(blocks: Sequence[dict[str, Any]]) -> bytes:
    """One stream-json user message holding the content blocks (text and images)."""
    message = {"type": "user", "message": {"role": "user", "content": list(blocks)}}
    # ASCII escapes: a lone surrogate or U+2028 in a title cannot break the encoding or the line.
    return (json.dumps(message, ensure_ascii=True) + "\n").encode("ascii")


def parse_events(stdout: str) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    # Split on "\n" only: str.splitlines also breaks on U+2028, U+0085 and friends, which can
    # appear inside a JSON string.
    for raw in stdout.split("\n"):
        line = raw[:-1] if raw.endswith("\r") else raw
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def final_result(events: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(events):
        if event.get("type") == "result":
            return event
    return None


def _classify_message(text: str, api_error_status: Any = None) -> LLMError | None:
    """Known failure wording, in priority order. Subscription plan wording is fatal even when the
    status is 429, so the plan being used up is not retried. Other 429s and rate-limit wording are
    transient and retried."""
    if _NOT_LOGGED_IN_RE.search(text):
        return LLMConfigError(NOT_LOGGED_IN_MESSAGE)
    if _USAGE_LIMIT_RE.search(text):
        return LLMQuotaError(USAGE_LIMIT_MESSAGE)
    if api_error_status == 429 or _RATE_LIMIT_RE.search(text):
        return _RateLimited(RATE_LIMIT_MESSAGE)
    return None


def classify_failure(result: dict[str, Any] | None, returncode: int | None, stderr: str) -> LLMError | None:
    """The error to raise for a finished call, or None if the call succeeded.

    A rate limit (429) is transient: _RateLimited, which _run retries before it becomes fatal. A plan
    or usage limit is fatal at once. Without a result line, stderr is checked the same way.
    """
    if result is None:
        known = _classify_message(stderr)
        if known is not None:
            return known
        tail = stderr.strip()[-300:]
        return LLMError(f"Claude CLI returned no result (exit {returncode})" + (f": {tail}" if tail else ""))
    text = str(result.get("result") or "")
    if result.get("is_error"):
        known = _classify_message(text, result.get("api_error_status"))
        if known is not None:
            return known
        return LLMError(f"Claude CLI error: {text[:300] or result.get('subtype') or 'unknown error'}")
    if not result.get("modelUsage") and not text.strip() and not result.get("structured_output"):
        return LLMError("Claude CLI returned an empty answer (is the model name recognised?)")
    return None


def usage_from_cli(result: dict[str, Any]) -> SimpleNamespace:
    usage = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    server = usage.get("server_tool_use") if isinstance(usage.get("server_tool_use"), dict) else {}
    return SimpleNamespace(
        input_tokens=int(usage.get("input_tokens") or 0),
        output_tokens=int(usage.get("output_tokens") or 0),
        cache_read_input_tokens=int(usage.get("cache_read_input_tokens") or 0),
        cache_creation_input_tokens=int(usage.get("cache_creation_input_tokens") or 0),
        web_search_requests=int(server.get("web_search_requests") or 0),
    )


def model_name_of(result: dict[str, Any], fallback: str) -> str:
    models = result.get("modelUsage")
    if isinstance(models, dict) and models:
        return str(next(iter(models)))
    return fallback


def structured_value(result: dict[str, Any]) -> Any:
    """The structured answer: structured_output if present, else JSON found in the result text."""
    if result.get("structured_output") is not None:
        return result["structured_output"]
    text = str(result.get("result") or "").strip()
    fenced = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    start, end = fenced.find("{"), fenced.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        return json.loads(fenced[start : end + 1])
    except json.JSONDecodeError:
        return None


def sources_from_events(events: Sequence[dict[str, Any]]) -> list[str]:
    """https URLs from web search tool results, in order, without duplicates."""
    urls: list[str] = []
    for event in events:
        if event.get("type") != "user":
            continue
        message = event.get("message") if isinstance(event.get("message"), dict) else {}
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            body = block.get("content")
            if isinstance(body, list):
                body = "\n".join(str(part.get("text", "")) for part in body if isinstance(part, dict))
            for url in _SOURCE_URL_RE.findall(str(body or "")):
                if url not in urls:
                    urls.append(url)
    return urls[:MAX_SOURCES]


def scrubbed_env(base: dict[str, str] | None = None) -> dict[str, str]:
    """The environment for a CLI call, without API credentials, so the subscription login is used."""
    source = dict(os.environ if base is None else base)
    for name in SCRUBBED_ENV_VARS:
        source.pop(name, None)
    return source


def resolve_binary(path: str) -> str | None:
    """Absolute path to the CLI, or None if it is not installed or not executable."""
    candidate = os.path.expanduser(path)
    if os.sep in candidate or (os.altsep and os.altsep in candidate):
        p = Path(candidate)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        return None
    return shutil.which(candidate)


async def _run_quick(binary: str, args: list[str]) -> str | None:
    """Run a short status command. Returns stdout, or None on any failure."""
    try:
        proc = await asyncio.create_subprocess_exec(
            binary,
            *args,
            stdin=asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=scrubbed_env(),
            start_new_session=True,
        )
    except OSError:
        return None
    try:
        out, _err = await asyncio.wait_for(proc.communicate(), timeout=STATUS_TIMEOUT_S)
    except TimeoutError:
        await _kill(proc)
        return None
    if proc.returncode not in (0, None):
        return None
    return out.decode("utf-8", "replace")


async def cli_status(settings: Settings) -> dict[str, Any]:
    """Whether the CLI is installed, its version, and whether the subscription login is present."""
    status: dict[str, Any] = {
        "found": False,
        "path": settings.claude_cli_path,
        "version": None,
        "logged_in": None,
        "subscription": None,
    }
    binary = resolve_binary(settings.claude_cli_path)
    if binary is None:
        return status
    status["found"] = True
    version = await _run_quick(binary, ["--version"])
    status["version"] = version.strip() if version else None
    auth = await _run_quick(binary, ["auth", "status"])
    try:
        data = json.loads(auth or "")
    except json.JSONDecodeError:
        data = {}
    if isinstance(data, dict) and "loggedIn" in data:
        status["logged_in"] = bool(data.get("loggedIn"))
        status["subscription"] = data.get("subscriptionType")
    return status


async def _kill(proc: asyncio.subprocess.Process) -> None:
    """Kill the whole process group the CLI was started in, so no child it spawned keeps running."""
    if proc.pid is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
    await proc.wait()


# --- Backend -----------------------------------------------------------------


class _RateLimited(LLMError):
    """A transient 429 from the service. _run retries it; it never reaches the pipeline."""


class ClaudeCliLLM:
    """Runs the local `claude` CLI headless. One call is one subprocess in a fresh temp directory."""

    def __init__(
        self,
        settings: Settings,
        tracker: UsageTracker | None = None,
        *,
        gate: asyncio.Semaphore | None = None,
        retry_delays: Sequence[float] = RATE_LIMIT_RETRY_DELAYS_S,
        sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
    ) -> None:
        self._cli = settings.claude_cli_path
        self._timeout = float(settings.cli_timeout_s)
        self._gate = gate or asyncio.Semaphore(settings.cli_concurrency)
        self._retry_delays = tuple(retry_delays)
        self._sleep = sleep
        self.tracker = tracker or UsageTracker()
        self.tracker.basis = "cli_reported"

    def ensure_configured(self) -> None:
        if resolve_binary(self._cli) is None:
            raise LLMConfigError(CLI_MISSING_MESSAGE.format(path=self._cli))

    async def parse(
        self,
        *,
        stage: str,
        model: str,
        system: str,
        content: str | list[dict[str, Any]],
        output_model: type[T],
        effort: str,
        cache: bool = False,
        max_tokens: int = 16000,
    ) -> T:
        """Structured call. One retry feeds a validation error back. Raises LLMError subclasses."""
        self.ensure_configured()
        blocks = [text_block(content)] if isinstance(content, str) else list(content)
        schema = output_model.model_json_schema()
        feedback: str | None = None
        for attempt in range(MAX_ATTEMPTS):
            attempt_blocks = blocks if feedback is None else blocks + [text_block(feedback)]
            result, _events = await self._run(
                stage, model, system, attempt_blocks, tools=NO_TOOLS, effort=effort, json_schema=schema
            )
            try:
                return output_model.model_validate(structured_value(result))
            except ValidationError as exc:
                log.warning("%s: answer failed validation (attempt %d)", stage, attempt + 1)
                feedback = (
                    "Your previous answer did not match the required JSON schema: "
                    f"{str(exc)[:800]}. Reply again with corrected JSON only."
                )
        raise LLMError(f"the {stage} output did not match the expected format")

    async def research_specs(
        self,
        *,
        model: str,
        brand: str,
        model_name: str,
        model_key: str,
        spec_keys: Sequence[str] = (),
    ) -> ModelSpecs:
        """Specs with WebSearch as the only tool. Falls back to model knowledge only when the CLI
        reports that WebSearch was refused. Raises LLMError subclasses."""
        self.ensure_configured()
        product = f"{brand} {model_name}".strip()
        keys = spec_keys_line(spec_keys)
        query = f"Product: {product}\n{keys}\nFind its technical specifications and typical EUR prices."
        try:
            notes, sources = await self._web_notes(model, query, prompts.SPECS_RESEARCH_SYSTEM)
            mode = "web"
        except _WebSearchUnavailable as exc:
            log.warning("web search refused by the CLI, using model knowledge: %s", exc)
            notes = await self._knowledge_notes(model, query)
            sources, mode = [], "model_knowledge"
        draft = await self.parse(
            stage="specs",
            model=model,
            system=prompts.SPECS_DRAFT_SYSTEM,
            content=f"Product: {product}\n{keys}\n<research_notes>\n{_escape(notes)}\n</research_notes>",
            output_model=ModelSpecsDraft,
            effort=DRAFT_EFFORT,
        )
        return ModelSpecs(
            **draft.model_dump(exclude={"specs"}),
            specs=specs_dict(draft.specs),
            model_key=model_key,
            sources=sources,
            lookup_mode=mode,
        )

    async def identify_model(
        self,
        *,
        model: str,
        brand: str,
        title: str,
        visual_description: str,
        listing_gaps: Sequence[str] = (),
    ) -> IdentifyDraft:
        """Web identification of a listing's model. Without WebSearch there is no identification."""
        self.ensure_configured()
        query = identify_query(brand, title, visual_description, listing_gaps)
        try:
            notes, _sources = await self._web_notes(model, query, prompts.IDENTIFY_RESEARCH_SYSTEM, stage="identify")
        except _WebSearchUnavailable as exc:
            log.warning("WebSearch refused, no web identification: %s", exc)
            return IdentifyDraft(model=None, model_confidence=0.0, evidence=WEB_SEARCH_UNAVAILABLE_EVIDENCE)
        return await self.parse(
            stage="identify",
            model=model,
            system=prompts.IDENTIFY_DRAFT_SYSTEM,
            content=(
                f"<listing>\nBrand: {_escape(brand)}\nTitle: {_escape(title)}\n</listing>\n"
                f"<research_notes>\n{_escape(notes)}\n</research_notes>"
            ),
            output_model=IdentifyDraft,
            effort=DRAFT_EFFORT,
        )

    async def _web_notes(self, model: str, query: str, system: str, stage: str = "specs") -> tuple[str, list[str]]:
        result, events = await self._run(
            stage,
            model,
            system,
            [text_block(query)],
            tools=WEB_TOOLS,
            effort=RESEARCH_EFFORT,
            json_schema=None,
        )
        denied = [d for d in result.get("permission_denials") or [] if isinstance(d, dict)]
        if any(d.get("tool_name") == "WebSearch" for d in denied):
            raise _WebSearchUnavailable("WebSearch was denied")
        notes = str(result.get("result") or "").strip()
        if not notes:
            raise LLMError("web research returned no notes")
        return notes, sources_from_events(events)

    async def _knowledge_notes(self, model: str, query: str) -> str:
        result, _events = await self._run(
            "specs",
            model,
            prompts.SPECS_KNOWLEDGE_SYSTEM,
            [text_block(query)],
            tools=NO_TOOLS,
            effort=RESEARCH_EFFORT,
            json_schema=None,
        )
        return str(result.get("result") or "").strip()

    async def _run(
        self,
        stage: str,
        model: str,
        system: str,
        blocks: Sequence[dict[str, Any]],
        *,
        tools: Sequence[str],
        effort: str | None,
        json_schema: dict[str, Any] | None,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        """One CLI call, with backoff retries on rate limits. Plan and usage limits are not retried."""
        binary = resolve_binary(self._cli)
        if binary is None:
            raise LLMConfigError(CLI_MISSING_MESSAGE.format(path=self._cli))
        args = build_cli_args(
            binary, model=model, system=system, effort=effort, tools=tools, json_schema=json_schema
        )
        payload = stream_input(blocks)
        attempt = 0
        while True:
            try:
                return await self._call_once(stage, model, args, payload)
            except _RateLimited as exc:
                if attempt >= len(self._retry_delays):
                    raise LLMQuotaError(RATE_LIMIT_MESSAGE) from exc
                delay = self._retry_delays[attempt]
                attempt += 1
                log.warning("%s: rate limited, retry %d in %gs", stage, attempt, delay)
                await self._sleep(delay)

    async def _call_once(
        self,
        stage: str,
        model: str,
        args: list[str],
        payload: bytes,
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        async with self._gate:
            workdir = tempfile.mkdtemp(prefix="vinted-hunter-cli-")
            try:
                stdout, stderr, returncode = await self._exec(args, payload, workdir)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        events = parse_events(stdout)
        result = final_result(events)
        error = classify_failure(result, returncode, stderr)
        # Usage is recorded for every attempt except a rate-limited one, which is retried and must not
        # be counted twice.
        if result is not None and not isinstance(error, _RateLimited):
            self._record(stage, model, result)
        if error is not None:
            raise error
        assert result is not None
        if result.get("stop_reason") == "refusal":
            raise LLMRefusal(f"the model refused the {stage} request")
        return result, events

    async def _exec(self, args: list[str], payload: bytes, workdir: str) -> tuple[str, str, int | None]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=scrubbed_env(),
                start_new_session=True,
            )
        except FileNotFoundError as exc:
            raise LLMConfigError(CLI_MISSING_MESSAGE.format(path=self._cli)) from exc
        try:
            out, err = await asyncio.wait_for(proc.communicate(payload), timeout=self._timeout)
        except TimeoutError as exc:
            await _kill(proc)
            raise LLMError(f"Claude CLI call timed out after {self._timeout:g}s") from exc
        except BaseException:
            await _kill(proc)  # cancelled or failed: never leave the child running
            raise
        return out.decode("utf-8", "replace"), err.decode("utf-8", "replace"), proc.returncode

    def _record(self, stage: str, model: str, result: dict[str, Any]) -> None:
        cost = result.get("total_cost_usd")
        usage = usage_from_cli(result)
        self.tracker.record(
            stage,
            model_name_of(result, model),
            usage,
            web_search_requests=usage.web_search_requests,
            # A missing cost is recorded as unknown, never as $0.
            reported_usd=float(cost) if isinstance(cost, (int, float)) and not isinstance(cost, bool) else None,
            from_cli=True,
        )


def _escape(text: str) -> str:
    return text.replace("<", "&lt;").replace(">", "&gt;")


# --- Backend selection -------------------------------------------------------


def build_llm(settings: Settings, tracker: UsageTracker | None = None, *, gate: asyncio.Semaphore | None = None):
    """The LLM backend chosen by LLM_BACKEND."""
    if settings.llm_backend == "api":
        return LLM(make_anthropic_client(settings), settings, tracker)
    return ClaudeCliLLM(settings, tracker, gate=gate)
