"""Settings loaded from the environment and an optional .env file.

Real environment variables win over values in .env.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = PROJECT_ROOT / ".env"

# Haiku keeps every stage cheap. Opus 5.5 is the quality upgrade for RANK_MODEL and PLANNER_MODEL.
DEFAULT_MODEL = "claude-haiku-5-5"

# "claude-cli" runs the local Claude Code CLI on the user's subscription login (default).
# "api" calls the Anthropic API with ANTHROPIC_API_KEY.
LLM_BACKENDS = ("claude-cli", "api")
DEFAULT_LLM_BACKEND = "claude-cli"


@dataclass(frozen=True)
class ModelPrice:
    """$ per million tokens. The long_* rates apply to a single call whose prompt exceeds the threshold."""

    input: float
    output: float
    long_input: float | None = None
    long_output: float | None = None
    long_context_threshold: int = 100_000


# Matched by substring of the model id, so dated ids such as claude-haiku-5-5-20260101 also match.
MODEL_PRICES: dict[str, ModelPrice] = {
    "opus-5-5": ModelPrice(input=4.0, output=20.0),
    "sonnet-5-5": ModelPrice(input=2.0, output=10.0),
    "haiku-5-5": ModelPrice(input=0.10, output=0.50, long_input=0.50, long_output=2.50),
}


@dataclass(frozen=True)
class Settings:
    anthropic_api_key: str | None = None
    planner_model: str = DEFAULT_MODEL
    triage_model: str = DEFAULT_MODEL
    extract_model: str = DEFAULT_MODEL
    specs_model: str = DEFAULT_MODEL
    rank_model: str = DEFAULT_MODEL
    vinted_base_url: str = "https://www.vinted.fr"
    search_pages: int = 1
    max_candidates: int = 300
    max_detailed: int = 60  # listings whose item pages are read; the most relevant first
    max_photos: int = 6  # the first N photos of a listing, in Vinted's order (the main photo first)
    vinted_min_delay_s: float = 2.5
    data_dir: Path = PROJECT_ROOT / "data"
    llm_backend: str = DEFAULT_LLM_BACKEND
    claude_cli_path: str = "claude"
    cli_concurrency: int = 3
    cli_timeout_s: float = 180.0
    include_shipping_in_budget: bool = True
    # 0 researches every distinct model of a hunt (the default). A positive value is an opt-in cap on the
    # number of models researched per hunt; the best listings are researched first.
    specs_max_per_hunt: int = 0
    # Lifetime of web specs that are not exact-variant, and of "no specs found" answers.
    specs_nonexact_ttl_hours: float = 72.0

    def __post_init__(self) -> None:
        for name in ("search_pages", "max_candidates", "max_detailed", "max_photos", "cli_concurrency"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.vinted_min_delay_s < 0:
            raise ValueError("vinted_min_delay_s must be >= 0")
        if self.cli_timeout_s <= 0:
            raise ValueError("cli_timeout_s must be > 0")
        if self.specs_max_per_hunt < 0:
            raise ValueError("SPECS_MAX_PER_HUNT must be >= 0 (0 means unlimited)")
        if self.specs_nonexact_ttl_hours < 0:
            raise ValueError("SPECS_NONEXACT_TTL_HOURS must be >= 0")
        if self.llm_backend not in LLM_BACKENDS:
            raise ValueError(f"LLM_BACKEND must be one of {', '.join(LLM_BACKENDS)}, not {self.llm_backend!r}")


def parse_dotenv(text: str) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export ") :].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key] = value
    return values


def _bool_setting(name: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in ("1", "true", "yes", "on"):
        return True
    if lowered in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"{name} must be true or false, not {value!r}")


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (PROJECT_ROOT / path)


def load_settings(
    environ: Mapping[str, str] | None = None,
    env_file: Path | None = DEFAULT_ENV_FILE,
) -> Settings:
    """Build Settings from `environ` (defaults to os.environ) and `env_file`."""
    dotenv: dict[str, str] = {}
    if env_file is not None and env_file.is_file():
        dotenv = parse_dotenv(env_file.read_text(encoding="utf-8"))
    env = dict(os.environ if environ is None else environ)
    merged = {**dotenv, **{k: v for k, v in env.items() if v != ""}}

    def get(name: str, default: str) -> str:
        return merged.get(name, default) or default

    defaults = Settings()
    return Settings(
        anthropic_api_key=merged.get("ANTHROPIC_API_KEY") or None,
        planner_model=get("PLANNER_MODEL", DEFAULT_MODEL),
        triage_model=get("TRIAGE_MODEL", DEFAULT_MODEL),
        extract_model=get("EXTRACT_MODEL", DEFAULT_MODEL),
        specs_model=get("SPECS_MODEL", DEFAULT_MODEL),
        rank_model=get("RANK_MODEL", DEFAULT_MODEL),
        vinted_base_url=get("VINTED_BASE_URL", defaults.vinted_base_url).rstrip("/"),
        search_pages=int(get("SEARCH_PAGES", str(defaults.search_pages))),
        max_candidates=int(get("MAX_CANDIDATES", str(defaults.max_candidates))),
        max_detailed=int(get("MAX_DETAILED", str(defaults.max_detailed))),
        max_photos=int(get("MAX_PHOTOS", str(defaults.max_photos))),
        vinted_min_delay_s=float(get("VINTED_MIN_DELAY_S", str(defaults.vinted_min_delay_s))),
        data_dir=_resolve_path(get("DATA_DIR", "./data")),
        llm_backend=get("LLM_BACKEND", defaults.llm_backend).strip().lower(),
        claude_cli_path=get("CLAUDE_CLI_PATH", defaults.claude_cli_path),
        cli_concurrency=int(get("CLI_CONCURRENCY", str(defaults.cli_concurrency))),
        cli_timeout_s=float(get("CLI_TIMEOUT_S", str(defaults.cli_timeout_s))),
        include_shipping_in_budget=_bool_setting(
            "INCLUDE_SHIPPING_IN_BUDGET", get("INCLUDE_SHIPPING_IN_BUDGET", "true")
        ),
        specs_max_per_hunt=int(get("SPECS_MAX_PER_HUNT", str(defaults.specs_max_per_hunt))),
        specs_nonexact_ttl_hours=float(get("SPECS_NONEXACT_TTL_HOURS", str(defaults.specs_nonexact_ttl_hours))),
    )


def price_for_model(model: str) -> ModelPrice | None:
    """Price table entry for a model id, or None if the model is unknown."""
    for key, price in MODEL_PRICES.items():
        if key in model:
            return price
    return None
