"""Thin wrapper around AsyncAnthropic: structured output, spec research, and usage/cost accounting."""

from __future__ import annotations

import base64
import html
import logging
import re
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel, ValidationError

from . import prompts
from .config import Settings, price_for_model
from .models import CostReport, IdentifyDraft, ModelSpecs, ModelSpecsDraft, StageCost, specs_dict

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

NO_KEY_MESSAGE = (
    "ANTHROPIC_API_KEY is not set. Add it to a .env file in the project root "
    "(copy .env.example), then restart vinted-hunter."
)
FALLBACK_BETA = "server-side-fallback-2026-07-01"
WEB_SEARCH_MAX_USES = 4  # the research prompts ask for at most 4 searches in total; keep the tool to match
RESEARCH_EFFORT = "medium"  # research is search-heavy and needs room to cross-check sources
DRAFT_EFFORT = "low"  # converting finished notes into fields is simple
# Evidence of the identify draft returned when web search is refused. It is not an answer, so the
# pipeline never caches it.
WEB_SEARCH_UNAVAILABLE_EVIDENCE = "web search unavailable"
# Models that accept the newer web search tool. Every other model gets the 2025 version.
_WEB_SEARCH_20260209_MODELS = {
    ("opus", "5-5"), ("opus", "5"), ("opus", "4-8"), ("opus", "4-7"), ("opus", "4-6"),
    ("sonnet", "5-5"), ("sonnet", "5"), ("sonnet", "4-6"),
}
_MODEL_ID_RE = re.compile(r"^claude-(opus|sonnet|haiku)-(\d+(?:-\d+)?)$")
_DATE_SUFFIX_RE = re.compile(r"-\d{8}$")


def web_search_tool(model: str) -> dict[str, Any]:
    """Web search tool definition for `model`.

    web_search_20260209 works only on some Opus and Sonnet versions. Every other model,
    including Haiku, gets web_search_20250305. A dated id such as claude-opus-5-20260101 is
    matched by its base name, so the date suffix cannot be read as part of the version.
    """
    base = _DATE_SUFFIX_RE.sub("", model)
    match = _MODEL_ID_RE.match(base)
    version = "web_search_20250305"
    if match and (match.group(1), match.group(2)) in _WEB_SEARCH_20260209_MODELS:
        version = "web_search_20260209"
    return {"type": version, "name": "web_search", "max_uses": WEB_SEARCH_MAX_USES}
PAUSE_TURN_LIMIT = 3
CACHE_CONTROL = {"type": "ephemeral"}


class LLMError(Exception):
    """A model call failed in a way the caller should report."""


class LLMFatal(LLMError):
    """Stops the whole hunt. The pipeline never turns it into a per-item rejection."""


class LLMConfigError(LLMFatal):
    """The LLM layer is not usable (no API key, CLI missing, not logged in)."""


class LLMQuotaError(LLMFatal):
    """The subscription or rate limit was hit. Continuing would only reject items one by one."""


class LLMRefusal(LLMError):
    """The model refused to answer."""


class _WebSearchUnavailable(Exception):
    """The web search tool was rejected (not enabled for this org, or unsupported)."""


_DISALLOWED_LABEL_CHARS = re.compile(r"[^\w \-._/+()]")


def _tag_safe(text: str) -> str:
    """Seller text inside a query: only the angle brackets are escaped, so it cannot open or close a tag.
    The query is sent to web search as plain text, so "&" stays as it is ("&amp;" would be noise there)."""
    return text.replace("<", "&lt;").replace(">", "&gt;")


def gaps_hint(listing_gaps: Sequence[str]) -> str:
    """A hint line for the identify query: what the listing does not show. Empty when there is none."""
    if not listing_gaps:
        return ""
    items = "; ".join(_tag_safe(" ".join(gap.split())[:120]) for gap in listing_gaps[:8])
    return f"Not shown in the listing (hints only): {items}\n"


def identify_query(brand: str, title: str, visual_description: str, listing_gaps: Sequence[str] = ()) -> str:
    """The web identification query. The listing text is seller data: it is kept inside <listing> tags, and
    its angle brackets are escaped so it cannot close the tags."""
    return (
        "<listing>\n"
        f"Brand: {_tag_safe(brand)}\n"
        f"Title: {_tag_safe(title)}\n"
        f"Visible features: {_tag_safe(visual_description or '(none)')}\n"
        f"{gaps_hint(listing_gaps)}"
        "</listing>\n"
        "Which product model is this?"
    )


def spec_keys_line(spec_keys: Sequence[str]) -> str:
    """The spec keys the research must fill, by exact name. Empty when the planner gave none. The keys come
    from the buyer's request, not from the seller, so they are not escaped."""
    keys = [" ".join(key.split())[:60] for key in spec_keys if key and key.strip()]
    if not keys:
        return "Spec keys: none given. Choose the specs that decide the value of this product category."
    return "Spec keys to fill (exact names): " + ", ".join(keys)


def sanitize_label(text: str | None, max_len: int = 60) -> str:
    """Make a brand or model name safe for a web-search query and a cache key.

    Keeps letters, digits, underscore, space and -._/+(). Collapses whitespace and truncates.
    Returns "" when nothing usable is left.
    """
    if not text:
        return ""
    spaced = re.sub(r"\s+", " ", text)
    cleaned = " ".join(_DISALLOWED_LABEL_CHARS.sub("", spaced).split())
    return cleaned[:max_len].strip()


def _model_id(response: Any, requested: str) -> str:
    """The model that actually answered, when the response reports it."""
    value = getattr(response, "model", None)
    return value if isinstance(value, str) and value else requested


def image_block(data: bytes, media_type: str) -> dict[str, Any]:
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": media_type,
            "data": base64.b64encode(data).decode("ascii"),
        },
    }


def text_block(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def supports_fallbacks(model: str) -> bool:
    """Refusal fallback applies to Opus and Sonnet only, not Haiku."""
    return "opus" in model or "sonnet" in model


def make_anthropic_client(settings: Settings) -> anthropic.AsyncAnthropic | None:
    if not settings.anthropic_api_key:
        return None
    return anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)


# --- Usage and cost ----------------------------------------------------------


@dataclass
class _Totals:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    web_search_requests: int = 0
    usd: float = 0.0
    priced: bool = True


def call_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
) -> float | None:
    """Estimated cost of one call in USD, or None if the model is not in the price table.

    The long-context rate applies when the call's prompt (input plus cached tokens) is above the
    model's threshold. Cache writes cost 1.25x the input rate and cache reads 0.1x.
    """
    price = price_for_model(model)
    if price is None:
        return None
    prompt_tokens = input_tokens + cache_read_tokens + cache_write_tokens
    long_prompt = prompt_tokens > price.long_context_threshold
    rate_in = price.long_input if long_prompt and price.long_input is not None else price.input
    rate_out = price.long_output if long_prompt and price.long_output is not None else price.output
    billed_input = input_tokens + 1.25 * cache_write_tokens + 0.1 * cache_read_tokens
    return (billed_input * rate_in + output_tokens * rate_out) / 1_000_000


class UsageTracker:
    """Accumulates token usage per (stage, model). Each call is priced on its own usage,
    so the long-context tier applies per call, not to the sum."""

    def __init__(self, basis: str = "api_list_price") -> None:
        self._totals: dict[tuple[str, str], _Totals] = {}
        # "api_list_price": cost is computed from tokens and the price table.
        # "cli_reported": cost is the equivalent the CLI reports; it is covered by the subscription.
        self.basis = basis

    def record(
        self,
        stage: str,
        model: str,
        usage: Any,
        web_search_requests: int = 0,
        reported_usd: float | None = None,
        from_cli: bool = False,
    ) -> None:
        input_tokens = _int(getattr(usage, "input_tokens", 0))
        output_tokens = _int(getattr(usage, "output_tokens", 0))
        cache_read = _int(getattr(usage, "cache_read_input_tokens", 0))
        cache_write = _int(getattr(usage, "cache_creation_input_tokens", 0))
        totals = self._totals.setdefault((stage, model), _Totals())
        totals.calls += 1
        totals.input_tokens += input_tokens
        totals.output_tokens += output_tokens
        totals.cache_read_tokens += cache_read
        totals.cache_write_tokens += cache_write
        totals.web_search_requests += web_search_requests
        if from_cli:
            # The CLI's own cost. A call without one is unpriced, never zero.
            if reported_usd is None:
                totals.priced = False
            else:
                totals.usd += reported_usd
            return
        if reported_usd is not None:
            totals.usd += reported_usd
            return
        cost = call_cost_usd(model, input_tokens, output_tokens, cache_read, cache_write)
        if cost is None:
            totals.priced = False
        else:
            totals.usd += cost

    def report(self) -> CostReport:
        stages: dict[str, StageCost] = {}
        for (stage, _model), t in self._totals.items():
            cost = stages.setdefault(stage, StageCost())
            cost.calls += t.calls
            cost.input_tokens += t.input_tokens
            cost.output_tokens += t.output_tokens
            cost.cache_read_tokens += t.cache_read_tokens
            cost.cache_write_tokens += t.cache_write_tokens
            cost.web_search_requests += t.web_search_requests
            cost.usd = round(cost.usd + t.usd, 6)
            cost.priced = cost.priced and t.priced
        report = CostReport(stages=stages, basis=self.basis)
        report.priced = all(s.priced for s in stages.values())
        report.total_usd = round(sum(s.usd for s in stages.values()), 6)
        report.total_calls = sum(s.calls for s in stages.values())
        report.total_input_tokens = sum(s.input_tokens for s in stages.values())
        report.total_output_tokens = sum(s.output_tokens for s in stages.values())
        report.total_web_search_requests = sum(s.web_search_requests for s in stages.values())
        return report


def _int(value: Any) -> int:
    return value if isinstance(value, int) else 0


@contextmanager
def _api_errors() -> Iterator[None]:
    """Translate SDK and parsing failures into LLMError. LLMError passes through unchanged."""
    try:
        yield
    except LLMError:
        raise
    except anthropic.RateLimitError as exc:
        raise LLMError("Anthropic rate limit reached. Wait a moment and try again.") from exc
    except anthropic.APIConnectionError as exc:
        raise LLMError("Could not reach the Anthropic API.") from exc
    except anthropic.APIStatusError as exc:
        raise LLMError(f"Anthropic API error {exc.status_code}: {exc.message}") from exc
    except ValidationError as exc:
        raise LLMError("the model output did not match the expected format") from exc
    except Exception as exc:  # e.g. a missing attribute on a client stub; never let it escape raw
        raise LLMError(f"unexpected error calling the model: {exc.__class__.__name__}") from exc


# --- LLM facade --------------------------------------------------------------


class LLM:
    def __init__(
        self,
        client: anthropic.AsyncAnthropic | None,
        settings: Settings,
        tracker: UsageTracker | None = None,
    ) -> None:
        self._client = client
        self._settings = settings
        self.tracker = tracker or UsageTracker()

    def ensure_configured(self) -> None:
        if self._client is None:
            raise LLMConfigError(NO_KEY_MESSAGE)

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
        """Structured call. Raises LLMRefusal or LLMError."""
        self.ensure_configured()
        messages = [{"role": "user", "content": content}]
        with _api_errors():
            if supports_fallbacks(model):
                kwargs: dict[str, Any] = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                    "output_format": output_model,
                    "output_config": {"effort": effort},
                    "betas": [FALLBACK_BETA],
                    "fallbacks": "default",
                }
                if cache:
                    kwargs["cache_control"] = CACHE_CONTROL
                response = await self._client.beta.messages.parse(**kwargs)
            else:
                kwargs = {
                    "model": model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": messages,
                    "output_format": output_model,
                    "output_config": {"effort": effort},
                }
                if cache:
                    kwargs["extra_body"] = {"cache_control": CACHE_CONTROL}
                response = await self._client.messages.parse(**kwargs)

        self.tracker.record(stage, _model_id(response, model), response.usage)
        if response.stop_reason == "refusal":
            raise LLMRefusal(f"the model refused the {stage} request")
        if response.parsed_output is None:
            raise LLMError(f"the {stage} output did not match the expected format")
        return response.parsed_output

    async def identify_model(
        self,
        *,
        model: str,
        brand: str,
        title: str,
        visual_description: str,
        listing_gaps: Sequence[str] = (),
    ) -> IdentifyDraft:
        """Web identification of a listing's model. Without web search there is no identification."""
        self.ensure_configured()
        query = identify_query(brand, title, visual_description, listing_gaps)
        with _api_errors():
            try:
                notes, _sources = await self._web_research(
                    model, query, prompts.IDENTIFY_RESEARCH_SYSTEM, stage="identify"
                )
            except _WebSearchUnavailable as exc:
                log.warning("web search unavailable, no web identification: %s", exc)
                return IdentifyDraft(model=None, model_confidence=0.0, evidence=WEB_SEARCH_UNAVAILABLE_EVIDENCE)
            return await self.parse(
                stage="identify",
                model=model,
                system=prompts.IDENTIFY_DRAFT_SYSTEM,
                content=(
                    f"<listing>\nBrand: {html.escape(brand, quote=False)}\n"
                    f"Title: {html.escape(title, quote=False)}\n</listing>\n"
                    f"<research_notes>\n{html.escape(notes, quote=False)}\n</research_notes>"
                ),
                output_model=IdentifyDraft,
                effort=DRAFT_EFFORT,
            )

    async def research_specs(
        self,
        *,
        model: str,
        brand: str,
        model_name: str,
        model_key: str,
        spec_keys: Sequence[str] = (),
    ) -> ModelSpecs:
        """Look up specs with web search, falling back to model knowledge. Raises LLMError."""
        self.ensure_configured()
        product = f"{brand} {model_name}".strip()
        keys = spec_keys_line(spec_keys)
        query = f"Product: {product}\n{keys}\nFind its technical specifications and typical EUR prices."
        with _api_errors():
            try:
                notes, sources = await self._web_research(model, query, prompts.SPECS_RESEARCH_SYSTEM)
                mode = "web"
            except _WebSearchUnavailable as exc:
                log.warning("web search unavailable, using model knowledge: %s", exc)
                notes = await self._knowledge_notes(model, query)
                sources, mode = [], "model_knowledge"
            draft = await self.parse(
                stage="specs",
                model=model,
                system=prompts.SPECS_DRAFT_SYSTEM,
                content=(
                    f"Product: {product}\n{keys}\n"
                    f"<research_notes>\n{html.escape(notes, quote=False)}\n</research_notes>"
                ),
                output_model=ModelSpecsDraft,
                effort=DRAFT_EFFORT,
            )
        return ModelSpecs(
            **draft.model_dump(exclude={"specs"}),
            specs=specs_dict(draft.specs),
            model_key=model_key,
            sources=sources[:8],
            lookup_mode=mode,
        )

    async def _web_research(self, model: str, query: str, system: str, stage: str = "specs") -> tuple[str, list[str]]:
        messages: list[dict[str, Any]] = [{"role": "user", "content": query}]
        sources: list[str] = []
        response = None
        for _ in range(PAUSE_TURN_LIMIT + 1):
            try:
                response = await self._client.messages.create(
                    model=model,
                    max_tokens=8000,
                    system=system,
                    messages=messages,
                    tools=[web_search_tool(model)],
                    output_config={"effort": RESEARCH_EFFORT},
                )
            except anthropic.PermissionDeniedError as exc:
                raise _WebSearchUnavailable(str(exc)) from exc
            except anthropic.BadRequestError as exc:
                if not _names_web_search(exc):
                    raise  # some other bad request (bad parameter, size, ...): report it, do not hide it
                raise _WebSearchUnavailable(str(exc)) from exc
            self._record_response(stage, model, response)
            sources.extend(_collect_sources(response.content))
            if response.stop_reason == "refusal":
                raise LLMRefusal("the model refused the specs lookup")
            if response.stop_reason != "pause_turn":
                break
            messages.append(
                {
                    "role": "assistant",
                    "content": [b.model_dump(mode="json", exclude_none=True) for b in response.content],
                }
            )
        notes = _text_of(response.content) if response is not None else ""
        if not notes.strip():
            raise LLMError("web research returned no notes")
        return notes, _dedupe(sources)

    async def _knowledge_notes(self, model: str, query: str) -> str:
        response = await self._client.messages.create(
            model=model,
            max_tokens=4000,
            system=prompts.SPECS_KNOWLEDGE_SYSTEM,
            messages=[{"role": "user", "content": query}],
            output_config={"effort": RESEARCH_EFFORT},
        )
        self._record_response("specs", model, response)
        if response.stop_reason == "refusal":
            raise LLMRefusal("the model refused the specs lookup")
        return _text_of(response.content)

    def _record_response(self, stage: str, model: str, response: Any) -> None:
        server_tool = getattr(response.usage, "server_tool_use", None)
        searches = _int(getattr(server_tool, "web_search_requests", 0)) if server_tool else 0
        self.tracker.record(stage, _model_id(response, model), response.usage, web_search_requests=searches)


_WEB_SEARCH_MENTION_RE = re.compile(r"web[_ ]search|tools\[", re.IGNORECASE)


def _names_web_search(exc: anthropic.APIStatusError) -> bool:
    """True when a 400 is plausibly about the web search tool itself."""
    return bool(_WEB_SEARCH_MENTION_RE.search(str(exc)))


def _text_of(content: list[Any]) -> str:
    return "".join(getattr(b, "text", "") for b in content if getattr(b, "type", "") == "text")


def _collect_sources(content: list[Any]) -> list[str]:
    urls: list[str] = []
    for block in content:
        if getattr(block, "type", "") != "web_search_tool_result":
            continue
        results = getattr(block, "content", None)
        if not isinstance(results, list):
            continue
        for result in results:
            url = getattr(result, "url", None)
            if url is None and isinstance(result, dict):
                url = result.get("url")
            if isinstance(url, str) and url.startswith("https://"):
                urls.append(url)
    return urls


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out
