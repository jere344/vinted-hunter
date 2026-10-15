"""Hunt orchestrator: plan -> search -> triage -> details -> extract -> specs -> rank.

Progress is reported through `emit(event_dict)`. Per-item failures become entries in
`rejected`. Warnings are collected in `HuntResult.warnings`. Only planner and ranker failures,
configuration errors, and two Vinted-level failures stop the hunt: Vinted blocking every search,
and a search page format that no longer parses.
"""

from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import re
import unicodedata
from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime
from typing import Any

from .cache import HuntCache
from .claude_cli import build_llm
from .config import Settings, load_settings
from .llm import (
    LLM,
    WEB_SEARCH_UNAVAILABLE_EVIDENCE,
    LLMError,
    LLMFatal,
    LLMRefusal,
    image_block,
    sanitize_label,
    text_block,
)
from .models import (
    CostReport,
    Extraction,
    ExtractionDraft,
    HardRequirement,
    HuntOptions,
    HuntResult,
    IdentifyDraft,
    ListingDetail,
    ListingSummary,
    ModelSpecs,
    NewPrice,
    RankedEntry,
    RankedListing,
    RankingOutput,
    RejectedItem,
    RequirementCheck,
    SearchPlan,
    TriageBatch,
    TriageDecision,
)
from .prompts import EXTRACT_SYSTEM, PLANNER_SYSTEM, PROMPT_VERSION, RANK_SYSTEM, TRIAGE_SYSTEM
from .vinted import (
    VintedBlocked,
    VintedClient,
    VintedError,
    VintedNotFound,
    VintedParseError,
    color_ids_for,
    item_price_ceiling,
)

log = logging.getLogger(__name__)

Emit = Callable[[dict[str, Any]], None]

BLOCKED_MESSAGE = "Vinted is blocking requests, try again later"
PARSE_CHANGED_MESSAGE = "Vinted page format changed, parser needs updating"
BREAKER_THRESHOLD = 3
TRIAGE_BATCH_SIZE = 25
TRIAGE_CONCURRENCY = 3
DETAIL_CONCURRENCY = 2
EXTRACT_CONCURRENCY = 4
SPECS_CONCURRENCY = 3
IDENTIFY_MIN_CONFIDENCE = 0.5
TRIAGE_MIN_RELEVANCE = 3
MAX_PLANNER_QUERIES = 5  # each query is a Vinted request per page; keep the request volume low
IDENTIFY_ACCEPT_CONFIDENCE = 0.6  # a web identification at or above this is used for the specs lookup
IDENTIFY_MAX_PER_HUNT = 15  # bounds the web identification calls per hunt (cache hits are free); most relevant first
IDENTIFY_SHARE_MIN_CONFIDENCE = 0.7  # a high-confidence identification is also reused for the same brand and title
SPECS_RESEARCHED = "researched"
SPECS_NOT_FOUND = "not found"
SPECS_LIMIT = "not researched (limit)"
SPECS_FAILED = "not researched (already failed a hard requirement)"
SPECS_WRONG_TYPE = "not researched (not the requested product type)"
# Counters that are always present in the hunt stats, even when zero.
HUNT_COUNTERS = (
    "models_researched",
    "models_cached_exact",
    "models_cached_nonexact",
    "models_skipped_limit",
    "models_skipped_failed",
    "identify_calls",
    "identify_cache_hits",
    "identify_shared",
)
SPEC_FILL_MIN_CONFIDENCE = 0.7  # researched specs may fill a vision "unknown" check (0.7 is a confirmed identification)
SPEC_OVERRIDE_MIN_CONFIDENCE = 0.85  # researched specs may turn an explicit vision "met" into "unmet" (never the reverse)
MAX_SPEC_KEYS = 8  # the planner's spec keys, bounded so one lookup stays cheap
MAX_RISK_TERMS = 12  # the planner's risk terms, bounded like the spec keys
MAX_RISK_TERM_LENGTH = 40  # characters per risk term; a longer term is a sentence, not a defect name
MAX_PRODUCT_TYPE_LENGTH = 120
MAX_NOT_PRODUCT_TYPES = 6
MAX_COLORS = 4
HARD_REQUIREMENT_SCORE_CAP = 30
FUNCTIONAL_DEFECT_SCORE_CAP = 40  # a part does not work properly: the score is capped, and the verdict at most "ok"
UNUSABLE_DEFECT_SCORE_CAP = 20  # the item does not work at all: avoid
WRONG_PRODUCT_TYPE_SCORE_CAP = 20  # not the product type the buyer asked for: avoid
# Verdicts from the best to the worst. A cap can only lower a verdict.
VERDICT_ORDER = ("top_pick", "good", "ok", "avoid")
DESCRIPTION_LIMIT = 3000


class HuntError(Exception):
    """The hunt cannot continue. The message is shown to the user as-is."""


def _noop(_event: dict[str, Any]) -> None:
    return None


class _Hunt:
    """Per-run bookkeeping: rejected items, counters, warnings, and event emission."""

    def __init__(self, emit: Emit) -> None:
        self._emit = emit
        self.rejected: list[RejectedItem] = []
        self.stats: dict[str, int] = {}
        self.warnings: list[str] = []

    def event(self, event: dict[str, Any]) -> None:
        self._emit(event)

    def stage(self, stage: str, status: str, detail: str = "") -> None:
        self._emit({"type": "stage", "stage": stage, "status": status, "detail": detail})

    def progress(self, stage: str, done: int, total: int) -> None:
        self._emit({"type": "progress", "stage": stage, "done": done, "total": total})

    def log(self, message: str) -> None:
        log.info("%s", message)
        self._emit({"type": "log", "message": message})

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
        log.warning("%s", message)
        self._emit({"type": "log", "message": f"Warning: {message}"})

    def count(self, key: str, amount: int = 1) -> None:
        self.stats[key] = self.stats.get(key, 0) + amount

    def reject(self, stage: str, listing: ListingSummary, reason: str) -> None:
        self.count(f"rejected_{stage}")
        self.rejected.append(
            RejectedItem(item_id=listing.id, title=listing.title, url=listing.url, stage=stage, reason=reason)
        )


class _Breaker:
    """Stops further Vinted requests for the rest of a stage.

    Trips on VintedBlocked, or after BREAKER_THRESHOLD failures in a row. Missing pages (404) and
    parse errors are per-item problems, not signs of blocking, so they do not count.
    """

    def __init__(self, stage: str, hunt: _Hunt) -> None:
        self.stage = stage
        self._hunt = hunt
        self._consecutive = 0
        self.tripped = False
        self.blocked = False

    def allow(self) -> bool:
        return not self.tripped

    def success(self) -> None:
        self._consecutive = 0

    def failure(self, exc: VintedError) -> None:
        if isinstance(exc, (VintedNotFound, VintedParseError)):
            return
        if isinstance(exc, VintedBlocked):
            self._trip(f"Vinted blocked requests during {self.stage}")
            return
        self._consecutive += 1
        if self._consecutive >= BREAKER_THRESHOLD:
            self._trip(f"{BREAKER_THRESHOLD} failed {self.stage} requests in a row")

    def _trip(self, reason: str) -> None:
        self.blocked = True
        if self.tripped:
            return
        self.tripped = True
        self._hunt.warn(f"{reason}; the remaining {self.stage} requests were skipped.")


def _describe(exc: BaseException) -> str:
    return str(exc) or exc.__class__.__name__


def _escape(text: str) -> str:
    """Escape seller text so it cannot close or open our prompt tags."""
    return html.escape(text, quote=False)


def _money(value: float | None) -> str:
    return "unknown" if value is None else f"{value:.2f}"


# A doubt written into a model name ("EH-TW650 (variant not confirmed)", "... - probably the 2nd gen"). The doubt
# belongs in model_confidence, so the qualifier is removed before the name is used for a search or a cache key.
_MODEL_QUALIFIER_RE = re.compile(
    r"\s*\((?:[^()]*?)(?:confirm|verif|variant|probabl|possib|likely|unsure|guess|inferred|approx|estim)[^()]*\)",
    re.IGNORECASE,
)
_MODEL_DOUBT_TAIL_RE = re.compile(
    r"\s*[-–,]\s*(?:variant|probabl|possib|likely|unconfirmed|not confirmed|unverified|not verified|approx).*$",
    re.IGNORECASE,
)


def clean_model_label(model: str | None) -> str:
    """A model name without qualifiers of doubt, sanitised. The doubt itself is carried by model_confidence."""
    if not model:
        return ""
    text = _MODEL_QUALIFIER_RE.sub("", model)
    text = _MODEL_DOUBT_TAIL_RE.sub("", text)
    return sanitize_label(text.replace("?", ""))


def _specs_labels(extraction: Extraction) -> tuple[str, str] | None:
    """Sanitised (brand, model) for a confident identification, or None if no lookup should run. The brand may be
    empty: a listing whose title names only the model is still researched by its model."""
    if extraction.model_confidence < IDENTIFY_MIN_CONFIDENCE:
        return None
    model = clean_model_label(extraction.model)
    if not model:
        return None
    return sanitize_label(extraction.brand), model


def model_key(brand: str, model: str) -> str:
    """Normalised cache key for a brand and model: lowercase ASCII words joined by spaces."""
    decomposed = unicodedata.normalize("NFKD", f"{sanitize_label(brand)} {sanitize_label(model)}")
    ascii_text = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.lower()))


def _specs_key(extraction: Extraction) -> str | None:
    labels = _specs_labels(extraction)
    if labels is None:
        return None
    return model_key(*labels) or None


# A red flag is moved to listing_gaps only when it names a specification (a generic spec word, or a word of
# the plan's spec keys), says that the information is missing or unconfirmed, AND names no item or
# transaction risk. Failing closed: any risk term keeps the flag a red flag, even when it also mentions a
# spec. Matches normalised text (lowercase, accents removed), so the patterns are written without accents.
_GENERIC_SPEC_WORDS = (
    "spec",
    "specs",
    "caracteristique",
    "caracteristiques",
    "reference",
    "references",
    "modele",
    "model",
    "taille",
    "size",
    "capacite",
    "capacity",
    "version",
    "annee",
    "year",
    "storage",
    "memoire",
    "ram",
    "resolution",
    "luminosite",
    "brightness",
    "contraste",
    "contrast",
    "duree de vie",
)
_SPEC_STOPWORDS = frozenset({"the", "and", "for", "with", "per", "les", "des", "une", "pour", "avec"})
_ABSENCE_RE = re.compile(
    r"\b(?:not|no|non|pas|aucun|aucune|sans|missing|absent|unknown|unclear|unspecified|inconnu\w*|manquant\w*)\b"
)
# A spec claim that is doubted or unconfirmed ("battery claim cannot be verified") is a gap too.
# No leading \b before the verif stem: "unverified" and "non verifiee" start with a prefix.
_UNCONFIRMED_RE = re.compile(
    r"\b(?:\w*verif\w*|doubt\w*|unconfirmed|not confirmed|rests? on|claims?)\b"
)
# Accessories that a listing can lack: "box missing", "boite manquante", "charger not included".
_MISSING_WORD = r"(?:missing|manquant\w*|absent\w*|not included|non fourni\w*)"
_ACCESSORY_NOUN = (
    r"(?:box|boites?|charger|chargers|chargeurs?|cables?|remotes?|telecommandes?|bags?|sacs?|caps?|bouchons?"
    r"|cases?|etuis?|straps?|sangles?|batter(?:y|ies|ie)|lens|lenses|objectifs?|accessor\w*|accessoires?)"
)
_ARTICLE = r"(?:(?:the|a|an|le|la|les|un|une|du|de|des|d|l)\s*['’]?\s*)?"
# Item and transaction risks: counterfeits, damage, wear, missing parts, defects, faults, inconsistencies.
# These are generic. Category-specific defects come from the plan's risk_terms (see split_spec_gaps).
# Wear is matched on specific terms only. "uses" is not a wear term, and accented "usé" is checked on
# the raw flag by _WORN_ACCENTED_RE, because accent removal turns "usé" into the English "use".
# A bare "used" or "usage" is not a wear term ("used model unknown" stays a gap).
_ITEM_RISK_RE = re.compile(
    r"\b(?:contrefa\w*|counterfeit\w*|fakes?|faux|authentic\w*"
    r"|cass\w*|broken|break\w*|abim\w*|damag\w*|dommag\w*|rayu\w*|rayee?s?|scratch\w*"
    r"|usure\w*|wear\w*|worn|fissur\w*|crack\w*|burn\w*|grill\w*|stain\w*|tach\w*"
    r"|hs|hors service|panne\w*|faulty|defective|defaut\w*|defect\w*"
    r"|not working|doesn[’']?t work\w*|does not work|untested|non test\w*"
    r"|ne (?:fonctionn\w*|marche\w*|demarr\w*|s[’']?allum\w*)|pas fonctionnel\w*|non fonctionnel\w*"
    r"|(?:missing|manquant\w*|absent\w*) (?:parts?|pieces?|elements?|accessor\w*)"
    r"|(?:pieces?|elements?|accessoires?) (?:manquant\w*|absent\w*)"
    rf"|{_MISSING_WORD} {_ARTICLE}{_ACCESSORY_NOUN}"
    rf"|{_ARTICLE}{_ACCESSORY_NOUN} (?:(?:is|are|est|sont) )?{_MISSING_WORD}"
    r"|pour (?:pieces?|parts?)|for parts|spare parts"
    r"|heavily used|well used|sold as used|bien utilis\w*|usagee\b|utilisee\b|etat d[’' ]?usage"
    # Physical defects, EN and FR. A spec word next to one of these is still an item risk.
    r"|bad sectors?|secteurs? defectueux\w*|dead pixels?|pixels? morts?|ghosting|image persistante\w*"
    r"|dent(?:s|ed)?|bossel\w*|cabosse\w*|bent|tord\w*|plie\w*|bruis\w*|chip(?:p\w*|s)?|ebreche\w*|peel\w*"
    r"|gonfl(?:e|ee|es|ement)|swell\w*|swoll\w*|leak\w*|fuites?|rust(?:y|ed|ing|s)?|rouill\w*"
    r"|does(?:n[’']?t| not) hold\w*(?: a| the)? charge|ne tient pas (?:la |sa )?charge"
    r"|won[’']?t (?:turn|charge|boot|power|start)\w*|ne charge\b"
    r"|overheat\w*|surchauff\w*"
    r"|bruit\w* (?:anormal\w*|etrange\w*|suspect\w*|inhabituel\w*)|(?:abnormal|strange|unusual|weird|grinding|rattling|loud) noises?"
    r"|loose(?! fit)\w*|lache\w*|(?:has|a) (?:some |un peu de |du )?(?:play|jeu)\b"
    r"|sticky|collante?\b|yellow(?:ed|ing)|jauni\w*|moisi\w*|mouldy|moldy|mould|mold|smell\w*|odeurs?|odou?r\w*|stink\w*"
    r"|contradict\w*|incoheren\w*|incoherent|inconsisten\w*|mismatch\w*|ne correspond\w*)\b"
)
# Accented wear words, checked on the raw flag: "usé", "usagé", "utilisé" and their feminine and plural forms.
_WORN_ACCENTED_RE = re.compile(r"\b(?:us|usag|utilis)(?:é|ée|és|ées)\b", re.IGNORECASE)
# A photo or label that shows, or contradicts, what the text claims is an item risk, whatever the spec word:
# "seller claims 16 GB RAM, photo shows an 8 GB sticker". Matched on normalised text (accent-free, lowercase),
# so the patterns are written without accents. A showing verb after a negation ("not", "ne", "n't", "no") is a
# plain absence, not a risk, and so is a photo noun after "no photo", "pas de photo" or "aucune photo".
_PHOTO_WORD = r"(?:photos?|images?|pictures?|pics?|labels?|stickers?|etiquettes?|autocollants?|captures?|screenshots?)"
_NEGATED_PHOTO_WORD = rf"(?<!\bno )(?<!\bpas de )(?<!\baucune )(?<!\baucun ){_PHOTO_WORD}"
_SHOWING_WORDS = r"(?:showing|shows?|montr(?:e|ent)|indiqu(?:e|ent))"
_SHOWING_VERB = rf"(?:(?<!\bnot )(?<!\bne )(?<!\bno )(?<!n[’']t )(?<!n[’'])(?:{_SHOWING_WORDS}))"
# "no photo shows an 8 GB sticker": the negated subject and its verb are removed before the conflict check,
# so the sticker that follows the verb is not read as a photo that shows something.
_NEGATION_WORD = r"(?:no|aucune?|aucun|pas de)"
_NEGATED_SHOWING_RE = re.compile(
    rf"\b(?:{_NEGATION_WORD}\s+{_PHOTO_WORD}\s+(?:ne\s+)?{_SHOWING_WORDS}"
    rf"|{_SHOWING_WORDS}\s+{_NEGATION_WORD}\s+{_PHOTO_WORD})\b"
)
_CONTRADICTION_WORD = (
    r"(?:contradict\w*|differ\w*|mismatch\w*|does not match|does(?:n[’']?t| not) match|ne correspond\w*"
    r"|inconsisten\w*|incoheren\w*|incoherent)"
)
_PHOTO_CONFLICT_RE = re.compile(
    rf"\b{_NEGATED_PHOTO_WORD}\b[^;.!?]*\b(?:{_SHOWING_VERB}|{_CONTRADICTION_WORD})\b"
    rf"|\b(?:{_SHOWING_VERB}|{_CONTRADICTION_WORD})\b[^;.!?]*\b{_NEGATED_PHOTO_WORD}\b"
)


def _photo_conflict(text: str) -> bool:
    """True when a photo, label or sticker shows or contradicts something, ignoring negated showings."""
    return bool(_PHOTO_CONFLICT_RE.search(_NEGATED_SHOWING_RE.sub(" ", text)))
# A claim (or an announcement) next to a photo, label or sticker word is an item risk as well.
_CLAIM_WORD = r"(?:claims?|annonce\w*|pretend\w*)"
_CLAIM_PHOTO_RE = re.compile(
    rf"\b{_CLAIM_WORD}\b.*\b{_NEGATED_PHOTO_WORD}\b|\b{_NEGATED_PHOTO_WORD}\b.*\b{_CLAIM_WORD}\b"
)


def _phrase_alternation(terms: Iterable[str]) -> str:
    """Accent-insensitive, whole-word alternation of the terms. Matches nothing when there are none."""
    phrases = sorted({normalize_text(t) for t in terms if normalize_text(t)}, key=len, reverse=True)
    if not phrases:
        return "(?!)"
    return r"(?<!\w)(?:" + "|".join(re.escape(p) for p in phrases) + r")(?!\w)"


def _spec_term_pattern(spec_keys: Iterable[str]) -> re.Pattern[str]:
    """The words that make a flag a spec mention: the generic spec words plus the words of the plan's spec keys."""
    words: set[str] = set(_GENERIC_SPEC_WORDS)
    for key in spec_keys:
        words.update(
            word
            for word in re.findall(r"[a-z0-9]+", normalize_text(key))
            if len(word) >= 3 and word not in _SPEC_STOPWORDS
        )
    return re.compile(_phrase_alternation(words))


# Clauses of a flag: split at , ; : / "but", "mais" and " - ". A dot or comma between two digits is a decimal
# or a thousands separator, not a split ("RAM 16.5 GB not stated" stays one clause).
_CLAUSE_BREAK_RE = re.compile(r"(?<!\d)[.,]|[.,](?!\d)|[;:/]|\s(?:but|mais)\s|\s-\s")


def _has_spec_absence_clause(text: str, spec_re: re.Pattern[str]) -> bool:
    """True when one clause names a spec AND says it is absent or unconfirmed. A spec word and an absence in
    different clauses ("storage has bad sectors, size not stated") do not make a gap."""
    return any(
        spec_re.search(clause) and (_ABSENCE_RE.search(clause) or _UNCONFIRMED_RE.search(clause))
        for clause in _CLAUSE_BREAK_RE.split(text)
        if clause.strip()
    )


def split_spec_gaps(
    red_flags: Iterable[str],
    extra_risk_terms: Iterable[str] = (),
    spec_keys: Iterable[str] = (),
) -> tuple[list[str], list[str]]:
    """Split red flags into (item flags, spec and information gaps), keeping the order of each.

    extra_risk_terms are the plan's category defects (for example "batterie HS"). A flag that names one
    stays a red flag. spec_keys are the plan's spec names, which make a flag a spec mention.

    A flag is a gap only when one clause names a spec with its absence or doubt, and no part of the flag
    names an item risk. The risk check covers the whole flag, so a defect in any clause keeps it red.
    """
    risk_terms = list(extra_risk_terms)
    spec_re = _spec_term_pattern(spec_keys)
    risk_re = re.compile(_phrase_alternation(risk_terms))
    flags: list[str] = []
    gaps: list[str] = []
    for flag in red_flags:
        text = normalize_text(flag)
        is_gap = (
            _has_spec_absence_clause(text, spec_re)
            and not _ITEM_RISK_RE.search(text)
            and not _photo_conflict(text)
            and not _CLAIM_PHOTO_RE.search(text)
            and not risk_re.search(text)
            and not _WORN_ACCENTED_RE.search(unicodedata.normalize("NFC", flag))
        )
        if is_gap:
            gaps.append(flag)
        else:
            flags.append(flag)
    return flags, gaps


def _identify_query_key(query: str, listing_gaps: Iterable[str] = ()) -> str:
    """Cache key for an identification answer: the prompt version, the normalised search query, and the
    listing gaps, which the identify prompt is given as hints.

    The version is part of the key, so a prompt change invalidates answers given by the old prompt.
    """
    gaps = sorted({normalize_text(gap) for gap in listing_gaps if normalize_text(gap)})
    payload = json.dumps(
        {"prompt_version": PROMPT_VERSION, "query": normalize_text(query), "gaps": gaps},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def specs_cache_key(model_key_value: str, spec_keys: Iterable[str]) -> str:
    """Cache key for specs: the model key plus a hash of the prompt version and the normalised spec-key set.
    Specs researched for other spec keys do not answer this plan's question, so they are not reused. The
    prompt version is hashed in too, so a prompt change makes the answers given by the old prompt misses."""
    names = sorted({normalize_text(key) for key in spec_keys if normalize_text(key)})
    payload = json.dumps({"prompt_version": PROMPT_VERSION, "spec_keys": names}, ensure_ascii=False, sort_keys=True)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"{model_key_value}|{digest}"


def _extraction_digest(system: str, plan: SearchPlan, detail: ListingDetail, photo_urls: Sequence[str]) -> str:
    """Cache key for an extraction: the exact system prompt, the exact user text, and the photos.

    Hashing what is actually sent means any change to the prompt, the plan, or a listing field
    that the text uses (price, upload date, seller rating) produces a new key. The plan's risk terms
    and spec keys decide how red flags are split into gaps, so they are part of the key too.
    """
    payload = json.dumps(
        {
            "prompt_version": PROMPT_VERSION,
            "system": system,
            "text": _extract_text(plan, detail, len(photo_urls)),
            "photos": list(photo_urls),
            "risk_terms": plan.risk_terms,
            "spec_keys": plan.spec_keys,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --- Spec comparison ---------------------------------------------------------
#
# A hard requirement bound to a spec is compared with the researched value of that spec. A value is read as
# measures: a number with a unit ("16 GB", "1.2 kg", "144 Hz"), dimensions ("1920x1080", "30 x 20 cm"), or a
# resolution name ("1080p", "4K", "Full HD", which stand for their pixel dimensions). Units convert only
# inside their own family (storage, length, mass, ...). Anything unparseable, several distinct values, or
# a unit from another family gives "unknown": the comparison fails closed.

_NUMBER = r"\d+(?:\.\d+)?"
# unit -> (family, factor to the family's base unit). Units are lowercase and accent-free.
_UNITS: dict[str, tuple[str, float]] = {
    "mb": ("data", 0.001),
    "mo": ("data", 0.001),
    "gb": ("data", 1.0),
    "go": ("data", 1.0),
    "tb": ("data", 1000.0),
    "teraoctet": ("data", 1000.0),
    "teraoctets": ("data", 1000.0),
    "mhz": ("frequency", 1.0),
    "ghz": ("frequency", 1000.0),
    "hz": ("rate", 1.0),
    "g": ("mass", 0.001),
    "kg": ("mass", 1.0),
    "lb": ("mass", 0.45359237),
    "lbs": ("mass", 0.45359237),
    "mm": ("length", 0.001),
    "cm": ("length", 0.01),
    "m": ("length", 1.0),
    "in": ("length", 0.0254),
    "inch": ("length", 0.0254),
    "inches": ("length", 0.0254),
    "pouce": ("length", 0.0254),
    "pouces": ("length", 0.0254),
    '"': ("length", 0.0254),
    "mah": ("charge", 1.0),
    "wh": ("energy", 1.0),
    "w": ("power", 1.0),
    "lm": ("light", 1.0),
    "lumen": ("light", 1.0),
    "lumens": ("light", 1.0),
    "mp": ("pixels", 1.0),
    "px": ("pixels", 1.0),
    "pixel": ("pixels", 1.0),
    "pixels": ("pixels", 1.0),
}
# Other spellings of a unit, compared as the spelling in _UNITS: "512 Go" is "512 GB" for "contains".
_UNIT_SPELLING = {
    "go": "gb",
    "mo": "mb",
    "teraoctet": "tb",
    "teraoctets": "tb",
    "inch": "in",
    "inches": "in",
    "pouce": "in",
    "pouces": "in",
    "lbs": "lb",
}
# Dimensions "WxH" or "WxHxD", optionally with a unit. Without a unit they are pixels.
_DIMS_RE = re.compile(
    rf"(?<![\w.])({_NUMBER})\s*x\s*({_NUMBER})(?:\s*x\s*({_NUMBER}))?"
    r'(?:\s*(px|pixels?|mm|cm|m|in|inch|inches|pouces?|"))?(?!\w)'
)
_QUANTITY_RE = re.compile(rf'(?<![\w.])({_NUMBER})(?:\s*("|[a-z]+))?(?!\w)')
_RESOLUTION_NAME_RE = re.compile(
    r"(?<![a-z0-9])(?:(720|1080|1440|2160)p(?:/?\d{2,3}(?:hz|fps)?)?(?![a-z])|(4k|2k|uhd|qhd|fhd|fullhd)(?![a-z0-9]))"
)
_P_RESOLUTIONS = {"720": (1280, 720), "1080": (1920, 1080), "1440": (2560, 1440), "2160": (3840, 2160)}
_RESOLUTION_NAMES = {
    "4k": (3840, 2160),
    "uhd": (3840, 2160),
    "2k": (2560, 1440),
    "qhd": (2560, 1440),
    "fhd": (1920, 1080),
    "fullhd": (1920, 1080),
}
_EPSILON = 1e-9

Measure = tuple[str | None, tuple[float, ...]]
# A value that cannot be read as one measure (a range, or a thousands group whose unit is ambiguous).
_AMBIGUOUS: Measure = ("ambiguous", ())
# Exactly three digits after a dot or comma are a thousands separator ("5.000 mAh") only before a unit whose
# fractional part of three digits is implausible. Any other such group ("1,200 kg") is ambiguous.
_THOUSANDS_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:[.,]\d{3})+(?![.,]?\d)")
_THOUSANDS_UNITS = frozenset({"mah", "hz", "w", "px", "pixel", "pixels", "lm", "lumen", "lumens"})
# Two numbers joined as a range ("16 to 32 GB", "16-32 GB", "16 a 32 GB") name several values, never one.
_RANGE_RE = re.compile(r"(?<![\w.,])\d+(?:[.,]\d+)?(?:\s*[-–]\s*|\s+(?:to|a)\s+)\d+(?![.,]?\d)")


def _thousands(text: str) -> tuple[str, bool]:
    """Joins the thousands groups that a plausible unit follows. Returns the text and whether an ambiguous group is left."""
    ambiguous = False

    def join(match: re.Match[str]) -> str:
        nonlocal ambiguous
        following = re.match(r"\s*([a-z]+)", match.string[match.end():])
        if following and following.group(1) in _THOUSANDS_UNITS:
            return re.sub(r"[.,]", "", match.group(0))
        ambiguous = True
        return match.group(0)

    return _THOUSANDS_RE.sub(join, text), ambiguous


# A space-separated thousands group: "1 920 px" is 1920, but "16 512 GB" or "1 500 g" is ambiguous.
_SPACED_GROUP_RE = re.compile(r"(?<![\d.,])\d{1,3}(?:\s\d{3}(?!\d))+")


def _join_spaced_group(match: re.Match[str]) -> str:
    """Joins a spaced thousands group only before a thousands-safe unit. Any other group stays spaced."""
    following = re.match(r"\s*([a-z]+)", match.string[match.end():])
    if following and following.group(1) in _THOUSANDS_UNITS:
        return match.group(0).replace(" ", "")
    return match.group(0)


def _measure_text(text: str) -> str:
    """Normalised text for measuring: "×" and "*" are x, a spaced thousands group before a safe unit is one
    number, and a decimal comma (one digit after it) is a point. Other commas are list separators."""
    text = normalize_text(text).replace("×", "x").replace("*", "x").replace("full hd", "fullhd")
    text = _SPACED_GROUP_RE.sub(_join_spaced_group, text)
    return re.sub(r"(?<=\d),(?=\d(?!\d))", ".", text)


def _measures(text: str) -> set[Measure]:
    """Every measure in the text, as (family, values in the family's base unit). Family None is a bare number.
    A range or an ambiguous thousands group adds _AMBIGUOUS, which makes the comparison unknown."""
    text, ambiguous = _thousands(_measure_text(text))
    # A spaced group that no safe unit joined is ambiguous: it may be one number or several.
    ambiguous = ambiguous or bool(_SPACED_GROUP_RE.search(text))
    found: set[Measure] = {_AMBIGUOUS} if ambiguous or _RANGE_RE.search(text) else set()
    for match in _DIMS_RE.finditer(text):
        values = tuple(float(g) for g in match.group(1, 2, 3) if g is not None)
        unit = match.group(4)
        if unit is None or unit in ("px", "pixel", "pixels"):
            family, factor = "pixels", 1.0
        else:
            family, factor = _UNITS[unit]
        found.add((family, tuple(value * factor for value in values)))
    text = _DIMS_RE.sub(" ", text)
    for match in _RESOLUTION_NAME_RE.finditer(text):
        dims = _P_RESOLUTIONS[match.group(1)] if match.group(1) else _RESOLUTION_NAMES[match.group(2)]
        found.add(("pixels", tuple(float(value) for value in dims)))
    text = _RESOLUTION_NAME_RE.sub(" ", text)
    for match in _QUANTITY_RE.finditer(text):
        number = float(match.group(1))
        unit = match.group(2)
        if unit is None:
            family, factor = None, 1.0
        elif unit in _UNITS:
            family, factor = _UNITS[unit]
        else:
            family, factor = f"?{unit}", 1.0  # an unknown unit only ever matches the same unknown unit
        found.add((family, (number * factor,)))
    return found


def _compare(actual: Measure, comparator: str, target: Measure) -> str:
    family, values = actual
    target_family, target_values = target
    if family != target_family or len(values) != len(target_values):
        return "unknown"
    if family == "length":
        # A physical size has no orientation: "20 x 30 cm" is the same size as "30 x 20 cm". Pixels keep their order.
        values = tuple(sorted(values, reverse=True))
        target_values = tuple(sorted(target_values, reverse=True))
    pairs = list(zip(values, target_values, strict=True))
    if comparator == ">=":
        holds = all(a >= b - _EPSILON for a, b in pairs)
    elif comparator == "<=":
        holds = all(a <= b + _EPSILON for a, b in pairs)
    elif comparator == "==":
        holds = all(abs(a - b) <= _EPSILON for a, b in pairs)
    else:
        return "unknown"
    return "met" if holds else "unmet"


# "contains" compares words and numbers. A piece is (kind, text, prefix): kind "n" is a number and "a" a word;
# prefix is the letters glued in front of a number ("ddr" in "ddr4"), so "ddr4" is never the number 4.
_CONTAINS_PIECE_RE = re.compile(r"\d+(?:\.\d+)?|[a-z]+")
_CONTAINS_ALTERNATIVE_RE = re.compile(r"\s*/\s*|\s*;\s*|\s*,\s*|\s+(?:or|ou)\s+")
_USBC_RE = re.compile(r"\busb(?: type)? c\b|\btype c\b")
Piece = tuple[str, str, str]


# Short forms a listing or a spec may use for the same thing ("ANC" for "active noise cancellation"). Both sides are
# expanded before they are compared, so an abbreviation never looks like a different value.
_ABBREVIATIONS = (
    (re.compile(r"(?<![a-z0-9])anc(?![a-z0-9])"), "active noise cancellation"),
    (re.compile(r"(?<![a-z0-9])bt(?![a-z0-9])"), "bluetooth"),
    (re.compile(r"(?<![a-z0-9])wi fi(?![a-z0-9])"), "wifi"),
    (re.compile(r"(?<![a-z0-9])ssd(?![a-z0-9])"), "solid state drive"),
    (re.compile(r"(?<![a-z0-9])hdd(?![a-z0-9])"), "hard disk drive"),
)


def _expand_abbreviations(text: str) -> str:
    for pattern, long_form in _ABBREVIATIONS:
        text = pattern.sub(long_form, text)
    return text


def _contains_text(text: str) -> str:
    """Canonical text for "contains": lowercase, accent-free, hyphens and spaces alike, one spelling of USB-C,
    abbreviations expanded, decimal commas as points and thousands commas removed. A number between two numbers
    ("41-42") is a range."""
    text = normalize_text(text).replace("×", "x")
    text = re.sub(r"(?<=\d),(?=\d{3}(?!\d))", "", text)
    text = re.sub(r"(?<=\d),(?=\d(?!\d))", ".", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " / ", text)
    text = _USBC_RE.sub("usbc", text.replace("-", " "))
    return _expand_abbreviations(text)


# Yes/no values. A negative value says the feature is absent; a value that says nothing is unknown.
_NEGATIVE_VALUE_RE = re.compile(
    r"^(?:no|none|non|not supported|unsupported|not available|sans|n/a|absent|aucun|aucune|not included|pas de)"
    r"(?=$|[\s,;:(/])"
)
_UNKNOWN_VALUES = frozenset(
    {"", "-", "--", "?", "unknown", "inconnu", "inconnue", "not stated", "not specified", "unspecified",
     "non renseigne", "non renseignee", "non precise", "non precisee"}
)


def _value_alternatives(value: str) -> list[str]:
    """The normalised parts of a spec value. Duplicate spec values are joined by " / " (see models.specs_dict)."""
    return [normalize_text(part).strip(" .!()[]") for part in value.split(" / ")]


def _negative_value(part: str) -> bool:
    return bool(_NEGATIVE_VALUE_RE.match(part))


def _alternatives(text: str) -> list[str]:
    """The alternatives of a value: items of a list, duplicate spec values joined by " / ", or ranges."""
    return [part.strip() for part in _CONTAINS_ALTERNATIVE_RE.split(_contains_text(text)) if part.strip()]


def _pieces(text: str) -> list[Piece]:
    pieces: list[Piece] = []
    for word in text.split():
        for match in _CONTAINS_PIECE_RE.finditer(word):
            token = match.group()
            if token[0].isdigit():
                prefix = re.search(r"[a-z]*$", word[: match.start()]).group()
                number = token.rstrip("0").rstrip(".") if "." in token else token  # "5.0" is "5"
                pieces.append(("n", number, prefix))
            else:
                pieces.append(("a", _UNIT_SPELLING.get(token, token), ""))
    return pieces


def _has_run(found: list[Piece], wanted: list[Piece]) -> bool:
    return any(found[i : i + len(wanted)] == wanted for i in range(len(found) - len(wanted) + 1))


def _contains_verdict(value: str, target: str) -> str:
    """"met" when the target's words and numbers appear in the value. A value with several alternatives (a list,
    duplicate spec values joined by " / ", a range) is met only when every alternative contains the target.
    "unmet" only for an explicit negative value ("no"), or for one clean value (at most one number) that shares no
    word or number with the target, when the target has no short word that could be an abbreviation. Otherwise
    unknown: prefer unknown to unmet. Abbreviations are expanded first, so "ANC" meets "active noise cancellation"."""
    target_parts = _alternatives(target)
    value_parts = _alternatives(value)
    if len(target_parts) != 1 or not value_parts:
        return "unknown"
    wanted = _pieces(target_parts[0])
    if not wanted:
        return "unknown"
    if all(_negative_value(part) for part in _value_alternatives(value)):
        return "unmet"
    if len(value_parts) > 1:
        return "met" if all(_has_run(_pieces(part), wanted) for part in value_parts) else "unknown"
    found = _pieces(value_parts[0])
    if _has_run(found, wanted):
        return "met"
    # A short word of the target ("usb", "eu", "ddr") can be an abbreviation or a family name of the value: unknown.
    if any(piece[0] == "a" and piece[1] not in _UNITS and len(piece[1]) <= 3 for piece in wanted):
        return "unknown"
    # A range or several numbers in one value ("EU 36 to 46") can hide the target: not a clean value.
    if sum(1 for piece in found if piece[0] == "n") > 1:
        return "unknown"
    # A unit word alone never makes two values alike: "16 GB" and "32 GB" share only "gb". But two units
    # from different families ("32 kg" against "16 GB") are not comparable, so that is unknown.
    found_units = {piece[1] for piece in found if piece[0] == "a" and piece[1] in _UNITS}
    wanted_units = {piece[1] for piece in wanted if piece[0] == "a" and piece[1] in _UNITS}
    if found_units and wanted_units and not {_UNITS[u][0] for u in found_units} & {_UNITS[u][0] for u in wanted_units}:
        return "unknown"
    # Shared numbers and shared words make the value unknown, except unit words (numbers are never units).
    shared = {piece[1] for piece in found} & {piece[1] for piece in wanted}
    if shared - set(_UNITS):
        return "unknown"
    if {piece[0] for piece in wanted} - {piece[0] for piece in found}:
        return "unknown"
    return "unmet"


def _spec_value(specs: dict[str, str], key: str) -> str | None:
    """The one value of a spec key, matched accent-insensitively. None when absent or when two keys disagree."""
    wanted = normalize_text(key)
    values = {value.strip() for name, value in specs.items() if normalize_text(name) == wanted and value and value.strip()}
    return values.pop() if len(values) == 1 else None


def _presence_verdict(value: str, comparator: str) -> str:
    """"present" or "absent" against a yes/no spec value. A value is present when it is non-empty and not a negative
    ("no", "none", "non", "not supported", "not available", "sans", "n/a", "absent"). Alternatives that disagree
    give unknown, so a conflict fails closed."""
    parts = _value_alternatives(value)
    if any(part in _UNKNOWN_VALUES for part in parts):
        return "unknown"
    negatives = [_negative_value(part) for part in parts]
    if all(negatives):
        is_present = False
    elif not any(negatives):
        is_present = True
    else:
        return "unknown"
    wanted_present = comparator == "present"
    return "met" if is_present == wanted_present else "unmet"


def spec_verdict(specs: ModelSpecs | None, requirement: HardRequirement) -> str:
    """"met", "unmet" or "unknown": a researched spec compared with a bound hard requirement.

    Only the caller decides whether the specs may count (exact web specs). Unknown whenever the requirement
    is unbound, the spec is missing or conflicting, or the value and the target do not parse into the same
    unit family. "present" and "absent" compare a yes/no value and need no target.
    """
    if specs is None or not requirement.spec_key or not requirement.comparator:
        return "unknown"
    value = _spec_value(specs.specs, requirement.spec_key)
    if value is None:
        return "unknown"
    if requirement.comparator in ("present", "absent"):
        return _presence_verdict(value, requirement.comparator)
    if not requirement.target:
        return "unknown"
    if requirement.comparator == "contains":
        return _contains_verdict(value, requirement.target)
    target = _measures(requirement.target)
    actual = _measures(value)
    if _AMBIGUOUS in target or _AMBIGUOUS in actual or len(target) != 1 or len(actual) != 1:
        return "unknown"
    return _compare(next(iter(actual)), requirement.comparator, next(iter(target)))


def _is_exact_web(specs: ModelSpecs | None) -> bool:
    """True for web specs of the exact variant: the only specs that count as evidence about a listing."""
    return specs is not None and specs.lookup_mode == "web" and specs.variant_match == "exact"


def _web_price_eligible(specs: ModelSpecs | None) -> bool:
    """True for a web new price: web specs of the exact variant, or of the base model (labelled as such). A price
    from model knowledge, or for an unknown variant, is never shown."""
    return specs is not None and specs.lookup_mode == "web" and specs.variant_match in ("exact", "base_model")


def _new_price(specs: ModelSpecs | None, detail: ListingDetail) -> NewPrice | None:
    """The new price of the model and how much cheaper this listing is than it, computed here and never by the
    model. The listing's landed cost is compared with the new price, or its total when the shipping is unknown."""
    if not _web_price_eligible(specs) or specs is None or specs.new_price_eur is None or specs.new_price_eur <= 0:
        return None
    basis = detail.landed_price_eur if detail.landed_price_eur is not None else detail.total_price_eur
    discount = None if basis is None else round((1 - basis / specs.new_price_eur) * 100)
    return NewPrice(
        eur=round(specs.new_price_eur, 2),
        source=specs.new_price_source,
        base_model=specs.variant_match == "base_model",
        discount_pct=discount,
    )


def _at_most(verdict: str, ceiling: str) -> str:
    """The verdict, but no better than the ceiling. A verdict that is already worse stays as it is."""
    if VERDICT_ORDER.index(verdict) < VERDICT_ORDER.index(ceiling):
        return ceiling
    return verdict


def _apply_caps(ranking: RankedListing, status: str, extraction: Extraction) -> None:
    """The code's caps on a ranker answer, applied in place. The ranker's own flags never lift a cap.

    - the wrong product type (20), an unusable item (20) and a failed hard requirement (30): the score is capped
      and the verdict is "avoid";
    - a functional defect: score capped at FUNCTIONAL_DEFECT_SCORE_CAP, verdict at most "ok";
    - "unverified" requirements: verdict at most "ok", because something is still not confirmed.
    """
    if not extraction.is_requested_product_type:
        ranking.score = min(ranking.score, WRONG_PRODUCT_TYPE_SCORE_CAP)
        ranking.verdict = "avoid"
    elif extraction.defect_severity == "unusable":
        ranking.score = min(ranking.score, UNUSABLE_DEFECT_SCORE_CAP)
        ranking.verdict = "avoid"
    elif status == "failed":
        ranking.score = min(ranking.score, HARD_REQUIREMENT_SCORE_CAP)
        ranking.verdict = "avoid"
    if extraction.defect_severity == "functional":
        ranking.score = min(ranking.score, FUNCTIONAL_DEFECT_SCORE_CAP)
        ranking.verdict = _at_most(ranking.verdict, "ok")
    if status == "unverified":
        ranking.verdict = _at_most(ranking.verdict, "ok")


def _nice_hits(title: str, nice_to_haves: Iterable[str]) -> int:
    """How many nice-to-haves the title names: every significant word of a nice-to-have appears in the title
    (accent-insensitive). A cheap tie-breaker for the triage order, for example a colour word."""
    title_words = set(re.findall(r"[a-z0-9]+", normalize_text(title)))
    hits = 0
    for nice in nice_to_haves:
        words = [w for w in re.findall(r"[a-z0-9]+", normalize_text(nice)) if len(w) >= 2 and w not in _NICE_STOPWORDS]
        if words and all(word in title_words for word in words):
            hits += 1
    return hits


_NICE_STOPWORDS = _SPEC_STOPWORDS | {"and", "the", "avec", "sans", "et", "ou", "or", "de", "du", "en", "au", "aux"}


def _answers_by_number(checks: Iterable[RequirementCheck], count: int) -> dict[int, str]:
    """One status per hard requirement number 1..count, from every check that names that number.

    unmet if any entry is unmet, met only if every entry is met, otherwise unknown (no entry counts as unknown).
    Entries with a number outside 1..count are ignored, so they never affect a listing.
    """
    by_number: dict[int, list[str]] = {}
    for check in checks:
        if 1 <= check.hard_requirement_no <= count:
            by_number.setdefault(check.hard_requirement_no, []).append(check.status)
    answers: dict[int, str] = {}
    for number in range(1, count + 1):
        statuses = by_number.get(number, [])
        if "unmet" in statuses:
            answers[number] = "unmet"
        elif statuses and all(status == "met" for status in statuses):
            answers[number] = "met"
        else:
            answers[number] = "unknown"
    return answers


def _requirement_statuses(plan: SearchPlan, extraction: Extraction, specs: ModelSpecs | None) -> list[str]:
    """The final status of each hard requirement, in order: the vision answer for its number, filled or
    overruled by the spec rule described in requirements_status. The ranker and the status both use this."""
    hard = plan.hard_requirements
    exact_web = _is_exact_web(specs)
    fill_ok = extraction.model_confidence >= SPEC_FILL_MIN_CONFIDENCE
    override_ok = extraction.model_confidence >= SPEC_OVERRIDE_MIN_CONFIDENCE
    answers = _answers_by_number(extraction.requirement_checks, len(hard))
    statuses: list[str] = []
    for number, requirement in enumerate(hard, start=1):
        status = answers[number]
        verdict = spec_verdict(specs, requirement) if exact_web else "unknown"
        if verdict == "met":
            if status == "unknown" and fill_ok:
                status = "met"
        elif verdict == "unmet":
            if status == "unknown" and fill_ok:
                status = "unmet"
            elif status == "met" and override_ok:
                status = "unmet"
        statuses.append(status)
    return statuses


def requirements_status(plan: SearchPlan, extraction: Extraction, specs: ModelSpecs | None) -> str:
    """verified, unverified, failed, or none (no hard requirements), from the vision checks and the specs.

    Each hard requirement is answered by the vision check whose hard_requirement_no is its number.
    Checks with no valid number are ignored, so they can never cap a listing. Then:
    - failed: any hard requirement is unmet.
    - verified: every hard requirement is met.
    - unverified: nothing is unmet, but some hard requirement is unknown or has no check.
    - none: the buyer stated no hard requirements.

    Spec rule. Only web specs for the exact variant count (model-knowledge specs, and base-model specs,
    never change a status), and only for a requirement bound to a spec (spec_verdict). Then:
    - a vision "unknown" or missing answer is filled by the spec, met or unmet, when model_confidence >= 0.7;
    - an explicit vision "met" is overridden to "unmet" by a spec that says the model cannot satisfy it,
      only when model_confidence >= 0.85;
    - an explicit vision "unmet" is never turned into "met". The listing may be a non-default
      configuration, which the specs of the default model do not describe.
    A spec verdict of "unknown" (or a conflict) leaves the vision answer as it is.

    A listing that is not the requested product type (is_requested_product_type false) is "failed", whatever the
    requirements: it is the wrong product. The ranker caps it at WRONG_PRODUCT_TYPE_SCORE_CAP.
    """
    if not extraction.is_requested_product_type:
        return "failed"
    if not plan.hard_requirements:
        return "none"
    statuses = _requirement_statuses(plan, extraction, specs)
    if "unmet" in statuses:
        return "failed"
    if all(status == "met" for status in statuses):
        return "verified"
    return "unverified"


def _prepare_plan(plan: SearchPlan) -> None:
    """Clean the planner's spec keys, risk terms and requirement bindings, in place.

    A requirement bound to a key that is not among spec_keys loses its binding, so only the vision check
    can settle it. The bound key is replaced by its canonical spelling from spec_keys.
    """
    keys: dict[str, str] = {}
    for key in plan.spec_keys:
        name = " ".join(key.split())
        if name:
            keys.setdefault(normalize_text(name), name)
    plan.spec_keys = list(keys.values())[:MAX_SPEC_KEYS]
    canonical = {normalize_text(key): key for key in plan.spec_keys}
    for requirement in plan.hard_requirements:
        bound = canonical.get(normalize_text(requirement.spec_key or ""))
        if bound is None:
            requirement.spec_key = None
            requirement.comparator = None
            requirement.target = None
        else:
            requirement.spec_key = bound
    # A term longer than MAX_RISK_TERM_LENGTH is a sentence, not a defect name: it is dropped, never cut mid-word.
    terms = (" ".join(term.split()) for term in plan.risk_terms)
    plan.risk_terms = list(dict.fromkeys(t for t in terms if t and len(t) <= MAX_RISK_TERM_LENGTH))[:MAX_RISK_TERMS]
    plan.product_type = " ".join(plan.product_type.split())[:MAX_PRODUCT_TYPE_LENGTH]
    plan.not_product_types = _clean_phrases(plan.not_product_types, MAX_NOT_PRODUCT_TYPES)
    plan.colors = _clean_phrases(plan.colors, MAX_COLORS)
    # The colour filter ids come from the code's table, never from the planner (see vinted.VINTED_COLOR_IDS).
    plan.color_ids = color_ids_for(plan.colors)


def _clean_phrases(items: Iterable[str], limit: int) -> list[str]:
    """Whitespace-collapsed, non-empty, distinct phrases, at most `limit`."""
    cleaned = (" ".join(item.split()) for item in items if isinstance(item, str))
    return list(dict.fromkeys(item for item in cleaned if item))[:limit]


def _bullets(items: Iterable[str]) -> str:
    return "\n".join(f"- {_escape(item)}" for item in items) or "- (none)"


def _hunt_context(plan: SearchPlan) -> str:
    return (
        f"Buyer is looking for: {_escape(plan.product_summary)}\n"
        f"Product type: {_escape(plan.product_type or '(not given)')}\n"
        f"Not this product type (relevance 0):\n{_bullets(plan.not_product_types)}\n"
        f"Hard requirements:\n{_bullets(r.text for r in plan.hard_requirements)}\n"
        f"Nice to have:\n{_bullets(plan.nice_to_haves)}\n"
        f"Pitfalls:\n{_bullets(plan.pitfalls)}"
    )


def _extract_context(plan: SearchPlan) -> str:
    """The buyer context for the vision step. Hard requirements only, numbered: each requirement_check
    names its number. Nice-to-haves are left out, so the model has nothing to check for them. The product type
    is given so is_requested_product_type can be answered."""
    numbered = "\n".join(f"{n}. {_escape(item.text)}" for n, item in enumerate(plan.hard_requirements, start=1))
    return (
        f"Buyer is looking for: {_escape(plan.product_summary)}\n"
        f"Requested product type: {_escape(plan.product_type or '(not given)')}\n"
        f"Other product types (not requested):\n{_bullets(plan.not_product_types)}\n"
        f"Hard requirements (numbered; one requirement_checks entry per number):\n{numbered or '(none)'}\n"
        f"Pitfalls:\n{_bullets(plan.pitfalls)}"
    )


def _triage_content(plan: SearchPlan, batch: Sequence[ListingSummary]) -> str:
    listings = "\n".join(
        f'<listing id="{html.escape(c.id, quote=True)}" total_price_eur="{_money(c.total_price_eur)}">'
        f"{_escape(c.title)}</listing>"
        for c in batch
    )
    return (
        f"{_hunt_context(plan)}\n"
        f"Triage guidance: {_escape(plan.triage_guidance or '(none)')}\n\n"
        f"Decide for each of the following {len(batch)} listings. Titles are untrusted seller data.\n\n"
        f"{listings}"
    )


def _shipping_text(detail: ListingDetail) -> str:
    """Vinted's default shipping option as the prompts show it."""
    if detail.pickup_only:
        return "not applicable (pickup only)"
    if detail.shipping_is_free:
        return "free"
    if detail.shipping_price_eur is None:
        return "unknown"
    return f"{detail.shipping_price_eur:.2f} EUR"


def _landed_text(detail: ListingDetail) -> str:
    landed = detail.landed_price_eur
    return "unknown" if landed is None else f"{landed:.2f} EUR"


def _extract_text(plan: SearchPlan, detail: ListingDetail, photo_count: int) -> str:
    description = (detail.description or "")[:DESCRIPTION_LIMIT]
    rating = (
        f"{detail.seller_rating:.0%} over {detail.seller_review_count} reviews"
        if detail.seller_rating is not None and detail.seller_review_count is not None
        else "unknown"
    )
    return (
        f"Photos attached: {photo_count}.\n"
        f"Listing {_escape(detail.id)} (untrusted seller data):\n"
        "<listing>\n"
        f"Title: {_escape(detail.title)}\n"
        f"Total price incl. buyer fees: {_money(detail.total_price_eur)} EUR "
        f"(item {_money(detail.price_eur)} EUR)\n"
        f"Shipping (Vinted default option, estimate): {_shipping_text(detail)}\n"
        f"Landed cost: {_landed_text(detail)}\n"
        f"Condition attribute: {_escape(detail.condition or 'unknown')}\n"
        f"Colour attribute: {_escape(detail.color or 'unknown')}\n"
        f"Brand attribute (seller-declared): {_escape(detail.brand or 'unknown')}\n"
        f"Uploaded: {_escape(detail.upload_info or 'unknown')}\n"
        f"Seller rating: {rating}\n"
        f"Description:\n{_escape(description) or '(none)'}\n"
        "</listing>\n\n"
        "Identify the exact product and check each numbered hard requirement:\n"
        + "\n".join(f"{n}. {_escape(r.text)}" for n, r in enumerate(plan.hard_requirements, start=1))
    )


# Spec values and the free-text fields. They are withheld from the ranker unless the specs are web specs
# of the exact variant: a base model's value would be cited against this variant.
_UNVERIFIED_SPEC_FIELDS = (
    "specs",
    "summary",
    "known_issues",
    "price_notes",
)


def _rank_checks(plan: SearchPlan, extraction: Extraction, specs: ModelSpecs | None) -> list[dict[str, Any]]:
    """The requirement checks the ranker sees: one entry per hard requirement, carrying the final status that
    requirements_status counts (after the spec rule), so the ranker never reads a status the status overrules."""
    vision = _answers_by_number(extraction.requirement_checks, len(plan.hard_requirements))
    final = _requirement_statuses(plan, extraction, specs)
    entries: list[dict[str, Any]] = []
    for number, requirement in enumerate(plan.hard_requirements, start=1):
        status = final[number - 1]
        if status != vision[number]:
            evidence = "Decided by the researched specs of this exact variant, not by the listing or the photos."
        else:
            evidence = next(
                (
                    c.evidence
                    for c in extraction.requirement_checks
                    if c.hard_requirement_no == number and c.status == status
                ),
                "",
            )
        entries.append(
            {"hard_requirement_no": number, "requirement": requirement.text, "status": status, "evidence": evidence}
        )
    return entries


def _rank_item(
    detail: ListingDetail,
    extraction: Extraction,
    specs: ModelSpecs | None,
    identified: bool,
    requirement_checks: list[dict[str, Any]],
    *,
    specs_status: str,
) -> dict[str, Any]:
    if specs is not None:
        specs_payload: Any = specs.model_dump(exclude={"sources"})
        if not _is_exact_web(specs):
            for name in _UNVERIFIED_SPEC_FIELDS:
                specs_payload.pop(name, None)
            specs_payload["note"] = (
                "These specs are for a base or unknown model, not confirmed for this exact variant "
                f"(variant_match={specs.variant_match}, lookup_mode={specs.lookup_mode}). The spec values and "
                "free-text notes were withheld and must not be used as evidence for or against the buyer's "
                "requirements."
            )
        if not _web_price_eligible(specs):
            # A new price is only a web price, for the exact variant or the base model (labelled as such).
            specs_payload["new_price_eur"] = None
            specs_payload["new_price_source"] = None
    elif identified:
        specs_payload = "lookup failed or unavailable"
    else:
        specs_payload = "unidentified"
    return {
        "item_id": detail.id,
        "title": detail.title,
        "total_price_eur": detail.total_price_eur,
        "item_price_eur": detail.price_eur,
        "shipping_price_eur": None if detail.pickup_only else detail.shipping_price_eur,
        "shipping_is_free": detail.shipping_is_free,
        "shipping_known": detail.shipping_known,
        "shipping_options_multiple": detail.shipping_options_multiple,
        "pickup_only": detail.pickup_only,
        "landed_price_eur": detail.landed_price_eur,
        "condition": detail.condition,
        "defect_severity": extraction.defect_severity,
        "defect_note": extraction.defect_note,
        "is_requested_product_type": extraction.is_requested_product_type,
        "upload_info": detail.upload_info,
        "seller_rating": detail.seller_rating,
        "seller_review_count": detail.seller_review_count,
        "identification": {
            "brand": extraction.brand,
            "model": extraction.model,
            "confidence": extraction.model_confidence,
            "evidence": extraction.identification_evidence,
        },
        "claimed_specs": [s.model_dump() for s in extraction.claimed_specs],
        "included_accessories": extraction.included_accessories,
        "red_flags": extraction.red_flags,
        "info_not_in_listing": extraction.listing_gaps,
        "requirement_checks": requirement_checks,
        "condition_assessment": extraction.condition_assessment,
        "specs_status": specs_status,
        "specs": specs_payload,
    }


def _json_for_prompt(payload: Any) -> str:
    # JSON never needs a raw '<' or '>' outside strings, so escaping them keeps tags intact.
    text = json.dumps(payload, ensure_ascii=False, indent=1)
    return text.replace("<", "\\u003c").replace(">", "\\u003e")


def normalize_text(text: str) -> str:
    """Accent-insensitive, case-insensitive form of text, with single spaces."""
    decomposed = unicodedata.normalize("NFKD", text)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return " ".join(stripped.casefold().split())


def _matches_excluded(title: str, keywords: Iterable[str]) -> str | None:
    """The first exclude keyword that occurs in the title as a whole word or phrase, else None.

    "support" does not match "supporté", and "coque seule" matches "Coque seule pour tablette".
    """
    haystack = normalize_text(title)
    for keyword in keywords:
        needle = normalize_text(keyword)
        if not needle:
            continue
        if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
            return keyword
    return None


async def _gather_all(coroutines: Iterable[Any]) -> list[Any]:
    """asyncio.gather that never leaves siblings running. If one child raises (for example an
    LLMFatal that aborts the hunt), the rest are cancelled and awaited before the error propagates."""
    tasks = [asyncio.ensure_future(c) for c in coroutines]
    try:
        return list(await asyncio.gather(*tasks))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


def _plural_listings(count: int) -> str:
    return "1 listing" if count == 1 else f"{count} listings"


def _warn_dropped_fields(hunt: _Hunt, details: Sequence[ListingDetail]) -> None:
    """One aggregated warning per field: item pages whose own tagged data was missing."""
    no_description = sum(1 for d in details if "description" in d.dropped_fields)
    no_attributes = sum(1 for d in details if "attributes" in d.dropped_fields)
    if no_description:
        hunt.warn(
            f"{_plural_listings(no_description)} had missing descriptions: "
            "the item page carried data only for other listings."
        )
    if no_attributes:
        hunt.warn(
            f"{_plural_listings(no_attributes)} had missing condition, colour and upload date: "
            "the item page carried data only for other listings."
        )


def _with_summary(detail: ListingDetail, summary: ListingSummary) -> ListingDetail:
    """Search-time fields (price, fee, url) are fresher than the item page, so they win."""
    merged = detail.model_copy(update=summary.model_dump(exclude={"photo_url"}))
    merged.photo_url = detail.photo_url or summary.photo_url
    return merged


def _shipping_budget_check(
    details: Sequence[ListingDetail],
    budget: float,
    settings: Settings,
    hunt: _Hunt,
) -> list[ListingDetail]:
    """Reject a listing whose landed price (total incl. fees plus Vinted's default shipping) is over the budget.

    A listing whose shipping cost is unknown stays in the hunt and is counted as "shipping unknown". Off when
    INCLUDE_SHIPPING_IN_BUDGET is false. The search stage still filters on the total price alone.
    """
    if not settings.include_shipping_in_budget:
        return list(details)
    kept: list[ListingDetail] = []
    for detail in details:
        if not detail.shipping_known:
            hunt.count("shipping_unknown")
            kept.append(detail)
            continue
        landed = detail.landed_price_eur
        if landed is not None and landed > budget + 1e-9:
            shipping = detail.shipping_price_eur or 0.0
            hunt.reject(
                "shipping",
                detail,
                f"over budget with shipping ({detail.total_price_eur:.2f} € + {shipping:.2f} €)",
            )
            continue
        kept.append(detail)
    return kept


_STAGE_ORDER = ("plan", "triage", "extract", "identify", "specs", "rank")
_STAGE_LABELS = {"plan": "planner"}


def _usage_summary(cost: CostReport) -> str:
    """One log line: tokens and equivalent cost per stage, in pipeline order."""
    names = [n for n in _STAGE_ORDER if n in cost.stages] + [n for n in cost.stages if n not in _STAGE_ORDER]
    parts = []
    for name in names:
        stage = cost.stages[name]
        tokens_in = stage.input_tokens + stage.cache_read_tokens + stage.cache_write_tokens
        label = _STAGE_LABELS.get(name, name)
        parts.append(f"{label} {tokens_in:,} in / {stage.output_tokens:,} out ${stage.usd:.4f}")
    basis = "equivalent API cost, covered by your subscription" if cost.basis == "cli_reported" else "estimated API cost"
    partial = "" if cost.priced else " (some calls were not priced: a lower bound)"
    return (
        f"Usage by stage: {'; '.join(parts) or 'no calls'}. "
        f"Total ${cost.total_usd:.4f} over {cost.total_calls} calls, {basis}{partial}."
    )


def _research_summary(stats: dict[str, int]) -> str:
    """One log line: the specs and identification counters of the hunt."""

    def count(key: str) -> int:
        return stats.get(key, 0)

    return (
        f"Specs: {count('models_researched')} models researched, {count('models_cached_exact')} cached exact, "
        f"{count('models_cached_nonexact')} cached non-exact, {count('models_skipped_limit')} skipped by the limit, "
        f"{count('models_skipped_failed')} skipped (already failed a hard requirement). "
        f"Identify: {count('identify_calls')} web calls, {count('identify_cache_hits')} cache hits, "
        f"{count('identify_shared')} listings reused a call in this hunt."
    )


# --- Public entry point ------------------------------------------------------


async def run_hunt(
    request: str,
    budget: float,
    options: HuntOptions | None = None,
    emit: Emit | None = None,
    *,
    settings: Settings | None = None,
    vinted: VintedClient | None = None,
    llm: LLM | None = None,
    cache: HuntCache | None = None,
    hunt_id: str = "",
) -> HuntResult:
    """Run a full hunt. Emits progress events, then a 'result' event. Raises on fatal errors.

    On fatal errors an 'error' event is emitted before the exception propagates.
    """
    settings = settings or load_settings()
    hunt = _Hunt(emit or _noop)
    owns_vinted = vinted is None
    vinted = vinted or VintedClient(settings)
    cache = cache or HuntCache(
        settings.data_dir / "cache.sqlite3", specs_ttl_s=settings.specs_nonexact_ttl_hours * 3600
    )
    llm = llm or build_llm(settings)
    options = options or HuntOptions()
    max_detailed = options.max_detailed or settings.max_detailed
    search_pages = options.search_pages or settings.search_pages
    try:
        llm.ensure_configured()
        return await _run(
            request=request,
            budget=budget,
            max_detailed=max_detailed,
            search_pages=search_pages,
            settings=settings,
            vinted=vinted,
            llm=llm,
            cache=cache,
            hunt=hunt,
            hunt_id=hunt_id,
        )
    except Exception as exc:
        hunt.event({"type": "error", "message": _describe(exc)})
        raise
    finally:
        if owns_vinted:
            await vinted.close()


async def _run(
    *,
    request: str,
    budget: float,
    max_detailed: int,
    search_pages: int,
    settings: Settings,
    vinted: VintedClient,
    llm: LLM,
    cache: HuntCache,
    hunt: _Hunt,
    hunt_id: str,
) -> HuntResult:
    for counter in HUNT_COUNTERS:
        hunt.count(counter, 0)

    # 1. Plan
    hunt.stage("plan", "start")
    plan = await llm.parse(
        stage="plan",
        model=settings.planner_model,
        system=PLANNER_SYSTEM,
        content=(
            f"Buyer request: {request}\n"
            f"Budget: {budget:g} EUR. The total price including buyer fees must stay within this budget."
        ),
        output_model=SearchPlan,
        effort="medium",
    )
    plan.budget_eur = budget
    _prepare_plan(plan)
    plan.queries = list(dict.fromkeys(q.strip() for q in plan.queries if q.strip()))[:MAX_PLANNER_QUERIES]
    if not plan.queries:
        raise LLMError("the planner returned no search queries")
    hunt.stage("plan", "done", f"{len(plan.queries)} queries")
    hunt.log(f"Plan: {plan.product_summary} ({len(plan.queries)} queries)")

    # 2. Search
    hunt.stage("search", "start")
    candidates = await _search(plan, budget, search_pages, settings, vinted, cache, hunt)
    hunt.stage("search", "done", f"{len(candidates)} candidates")

    # 3. Triage
    hunt.stage("triage", "start", f"{len(candidates)} listings")
    decisions = await _triage(plan, candidates, settings, llm, hunt)
    scored: list[tuple[ListingSummary, TriageDecision]] = []
    for candidate in candidates:
        decision = decisions.get(candidate.id)
        if isinstance(decision, str):
            hunt.reject("triage", candidate, decision)
        elif decision is None:
            hunt.reject("triage", candidate, "triage: no decision")
        elif decision.relevance < TRIAGE_MIN_RELEVANCE:
            # Selection is by relevance alone. The model's keep flag is not consulted.
            hunt.reject("triage", candidate, f"triage: relevance {decision.relevance} (below {TRIAGE_MIN_RELEVANCE}): {decision.reason}")
        else:
            scored.append((candidate, decision))
    # Relevance first. Ties go to the listing whose title names more nice-to-haves, then to the lower total price.
    scored.sort(
        key=lambda pair: (
            -pair[1].relevance,
            -_nice_hits(pair[0].title, plan.nice_to_haves),
            pair[0].total_price_eur if pair[0].total_price_eur is not None else float("inf"),
        )
    )
    selected = [candidate for candidate, _ in scored[:max_detailed]]
    for candidate, decision in scored[max_detailed:]:
        hunt.reject("triage", candidate, f"relevance {decision.relevance}, outside the top {max_detailed} kept")
    hunt.count("triage_kept", len(scored))
    hunt.stage("triage", "done", f"{len(scored)} kept, {len(selected)} selected")
    relevance = {candidate.id: decision.relevance for candidate, decision in scored}

    # 4. Details (always a live fetch: availability must be current)
    hunt.stage("details", "start", f"{len(selected)} listings")
    details = await _details(selected, vinted, hunt)
    details_list = [d for d in details if d is not None]
    _warn_dropped_fields(hunt, details_list)
    hunt.count("details", len(details_list))
    hunt.stage("details", "done", f"{len(details_list)} available")
    details_list = _shipping_budget_check(details_list, budget, settings, hunt)

    # 5. Extract (vision)
    hunt.stage("extract", "start", f"{len(details_list)} listings")
    extractions = await _extract(details_list, plan, settings, vinted, llm, cache, hunt)
    hunt.count("extracted", len(extractions))
    hunt.stage("extract", "done", f"{len(extractions)} identified or inspected")

    # 6. Specs
    hunt.stage("specs", "start")
    specs_by_key, specs_status = await _specs(
        extractions, {d.id: d for d in details_list}, plan, relevance, settings, llm, cache, hunt
    )
    hunt.count("specs_resolved", len(specs_by_key))
    hunt.stage("specs", "done", f"{len(specs_by_key)} models")

    # 7. Rank
    hunt.stage("rank", "start")
    ranked = await _rank(plan, details_list, extractions, specs_by_key, specs_status, settings, llm, hunt)
    hunt.count("ranked", len(ranked))
    hunt.stage("rank", "done", f"{len(ranked)} ranked")

    cost = llm.tracker.report()
    hunt.log(_usage_summary(cost))
    hunt.log(_research_summary(hunt.stats))
    result = HuntResult(
        hunt_id=hunt_id,
        request=request,
        budget_eur=budget,
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        plan=plan,
        ranked=ranked,
        rejected=hunt.rejected,
        stats=dict(hunt.stats),
        cost=cost,
        warnings=hunt.warnings,
    )
    hunt.event({"type": "result", "result": result})
    return result


# --- Stage 2: search ---------------------------------------------------------


async def _fetch_search(
    vinted: VintedClient,
    cache: HuntCache,
    query: str,
    page: int,
    price_to: float,
    catalog_ids: list[int],
    color_ids: list[int],
    hunt: _Hunt,
) -> list[ListingSummary]:
    """One Vinted search request, served from the cache when the same search ran in the last 30
    minutes. Only successful results are stored, so an error is never replayed from the cache."""
    colors = ",".join(map(str, color_ids))
    cache_key = f"{vinted.base_url}|{query}|{price_to:g}|{','.join(map(str, catalog_ids))}|{colors}|{page}"
    cached = cache.get_search_page(cache_key)
    if cached is not None:
        hunt.count("search_cache_hits")
        return cached
    listings = await vinted.search(
        query, price_to=price_to, catalog_ids=catalog_ids, page=page, color_ids=color_ids
    )
    cache.put_search_page(cache_key, listings)
    return listings


async def _search_page(
    vinted: VintedClient,
    cache: HuntCache,
    query: str,
    page: int,
    price_to: float,
    catalog_ids: list[int],
    color_ids: list[int],
    hunt: _Hunt,
) -> list[ListingSummary]:
    """One results page. A catalog-filtered search that comes back empty is retried once without
    the filter. Raises the VintedError subclasses for the caller to classify."""
    listings = await _fetch_search(vinted, cache, query, page, price_to, catalog_ids, color_ids, hunt)
    if not listings and catalog_ids:
        hunt.log(f"catalog filter returned nothing for '{query}' page {page}; retrying without it")
        listings = await _fetch_search(vinted, cache, query, page, price_to, [], color_ids, hunt)
    return listings


def _interleave(per_query: dict[str, list[ListingSummary]]) -> list[tuple[str, ListingSummary]]:
    """Round-robin across queries: the first result of each query, then the second of each, and so
    on. Each query keeps Vinted's order, so no single query can fill the candidate cap alone."""
    lists = list(per_query.items())
    depth = max((len(items) for _, items in lists), default=0)
    ordered: list[tuple[str, ListingSummary]] = []
    for index in range(depth):
        for query, items in lists:
            if index < len(items):
                ordered.append((query, items[index]))
    return ordered


async def _search(
    plan: SearchPlan,
    budget: float,
    search_pages: int,
    settings: Settings,
    vinted: VintedClient,
    cache: HuntCache,
    hunt: _Hunt,
) -> list[ListingSummary]:
    breaker = _Breaker("search", hunt)
    per_query: dict[str, list[ListingSummary]] = {query: [] for query in plan.queries}
    requests = [(query, page) for query in plan.queries for page in range(1, search_pages + 1)]
    # Vinted's price_to filters the item price, while the budget applies to the buyer's total (item plus fee).
    # The ceiling is the highest item price that can still have a total within the budget.
    price_to = item_price_ceiling(budget)
    hunt.log(f"Search price ceiling {price_to:.2f} EUR for a total budget of {budget:g} EUR")
    attempted = 0
    parse_failures = 0
    done = 0
    for query, page in requests:
        done += 1
        hunt.progress("search", done, len(requests))
        if not breaker.allow():
            continue
        attempted += 1
        try:
            listings = await _search_page(
                vinted, cache, query, page, price_to, plan.catalog_ids, plan.color_ids, hunt
            )
        except VintedParseError as exc:
            parse_failures += 1
            hunt.warn(f"search '{query}' page {page} could not be read: {exc}")
            listings = []
        except VintedError as exc:
            breaker.failure(exc)
            hunt.log(f"search '{query}' page {page} failed: {exc}")
            listings = []
        else:
            breaker.success()
        hunt.count("search_raw", len(listings))
        per_query[query].extend(listings)

    if attempted and parse_failures == attempted:
        raise HuntError(PARSE_CHANGED_MESSAGE)
    if breaker.blocked and hunt.stats.get("search_raw", 0) == 0:
        raise HuntError(BLOCKED_MESSAGE)

    eligible: list[tuple[str, ListingSummary]] = []
    seen_ids: set[str] = set()
    for query, listing in _interleave(per_query):
        if listing.id in seen_ids:
            hunt.count("search_duplicate")
            continue
        seen_ids.add(listing.id)
        if listing.total_price_eur is None:
            hunt.reject("search", listing, "no price shown")
            continue
        if listing.total_price_eur > budget + 1e-9:
            hunt.count("search_over_budget")
            continue
        excluded = _matches_excluded(listing.title, plan.exclude_keywords)
        if excluded is not None:
            hunt.reject("search", listing, f"title matches excluded keyword '{excluded}'")
            continue
        eligible.append((query, listing))

    # The cap keeps the first N in round-robin order. Listings past it are counted, not listed.
    kept = eligible[: settings.max_candidates]
    hunt.count("search_over_cap", len(eligible) - len(kept))
    candidates = [listing for _, listing in kept]
    for query, _listing in kept:
        hunt.count(f"candidates from: {query}")
    hunt.count("candidates", len(candidates))
    hunt.log(
        f"Search: {hunt.stats.get('search_raw', 0)} listings, "
        f"{hunt.stats.get('search_over_budget', 0)} over budget, "
        f"{len(candidates)} candidates ({hunt.stats.get('search_over_cap', 0)} past the cap)"
    )
    return candidates


# --- Stage 3: triage ---------------------------------------------------------


async def _triage(
    plan: SearchPlan,
    candidates: Sequence[ListingSummary],
    settings: Settings,
    llm: LLM,
    hunt: _Hunt,
) -> dict[str, TriageDecision | str]:
    """Map listing id to its decision, or to a failure reason string.

    Each batch handles every error inside its own body, so one failing batch rejects only its own
    listings and never raises into gather.
    """
    results: dict[str, TriageDecision | str] = {}
    batches = [candidates[i : i + TRIAGE_BATCH_SIZE] for i in range(0, len(candidates), TRIAGE_BATCH_SIZE)]
    semaphore = asyncio.Semaphore(TRIAGE_CONCURRENCY)
    done = 0

    async def run_batch(batch: Sequence[ListingSummary]) -> None:
        nonlocal done
        async with semaphore:
            try:
                output = await llm.parse(
                    stage="triage",
                    model=settings.triage_model,
                    system=TRIAGE_SYSTEM,
                    content=_triage_content(plan, batch),
                    output_model=TriageBatch,
                    effort="low",
                    cache=True,
                )
                by_id = {d.item_id: d for d in output.decisions}
                for listing in batch:
                    results[listing.id] = by_id.get(listing.id) or "no triage decision returned"
            except LLMFatal:
                raise
            except Exception as exc:
                if not isinstance(exc, LLMError):
                    log.exception("unexpected error in a triage batch")
                hunt.log(f"triage batch of {len(batch)} failed: {_describe(exc)}")
                for listing in batch:
                    results[listing.id] = f"triage failed: {_describe(exc)}"
            finally:
                done += 1
                hunt.progress("triage", done, len(batches))

    await _gather_all(run_batch(batch) for batch in batches)
    return results


# --- Stage 4: details --------------------------------------------------------


async def _details(
    selected: Sequence[ListingSummary],
    vinted: VintedClient,
    hunt: _Hunt,
) -> list[ListingDetail]:
    """Live item pages for the selected listings. Returns the available ones.

    Raises HuntError when listings were selected but none could be loaded, so an empty result is
    never reported as a success.
    """
    semaphore = asyncio.Semaphore(DETAIL_CONCURRENCY)
    breaker = _Breaker("details", hunt)
    done = 0

    async def load(summary: ListingSummary) -> ListingDetail | None:
        nonlocal done
        detail: ListingDetail | None = None
        async with semaphore:
            try:
                if not breaker.allow():
                    hunt.reject("details", summary, "skipped: Vinted is blocking requests")
                else:
                    try:
                        fetched = await vinted.item_detail(summary.url)
                    except VintedError as exc:
                        breaker.failure(exc)
                        hunt.reject("details", summary, f"could not load item page: {exc}")
                    else:
                        breaker.success()
                        merged = _with_summary(fetched, summary)
                        if merged.can_buy is False or merged.is_reserved is True:
                            hunt.reject("details", summary, "not available (sold or reserved)")
                        else:
                            detail = merged
            except Exception as exc:  # unexpected: reject this item, never raise into gather
                log.exception("unexpected error loading item %s", summary.id)
                hunt.reject("details", summary, f"could not load item page: {_describe(exc)}")
                detail = None
            finally:
                done += 1
                hunt.progress("details", done, len(selected))
        return detail

    results = list(await _gather_all(load(s) for s in selected))
    loaded = [d for d in results if d is not None]
    if selected and not loaded:
        if breaker.blocked:
            raise HuntError(BLOCKED_MESSAGE)
        raise HuntError(
            f"Could not load any of the {len(selected)} selected listings from Vinted: "
            "every item page failed or was unavailable."
        )
    return loaded


# --- Stage 5: extract --------------------------------------------------------


async def _extract(
    details: Sequence[ListingDetail],
    plan: SearchPlan,
    settings: Settings,
    vinted: VintedClient,
    llm: LLM,
    cache: HuntCache,
    hunt: _Hunt,
) -> dict[str, Extraction]:
    semaphore = asyncio.Semaphore(EXTRACT_CONCURRENCY)
    photo_breaker = _Breaker("photo download", hunt)
    extractions: dict[str, Extraction] = {}
    done = 0
    system = f"{EXTRACT_SYSTEM}\n{_extract_context(plan)}"

    async def run_one(detail: ListingDetail) -> None:
        nonlocal done
        async with semaphore:
            try:
                extractions[detail.id] = await _extract_one(
                    detail, plan, system, settings, vinted, llm, cache, hunt, photo_breaker
                )
            except LLMFatal:
                raise
            except LLMRefusal:
                hunt.reject("extract", detail, "the model refused to analyse this listing")
            except LLMError as exc:
                hunt.reject("extract", detail, f"analysis failed: {exc}")
            except Exception as exc:  # per-item failure: reject, never raise into gather
                if not isinstance(exc, VintedError):
                    log.exception("unexpected error extracting item %s", detail.id)
                hunt.reject("extract", detail, f"analysis failed: {_describe(exc)}")
            finally:
                done += 1
                hunt.progress("extract", done, len(details))

    await _gather_all(run_one(d) for d in details)
    return extractions


async def _extract_one(
    detail: ListingDetail,
    plan: SearchPlan,
    system: str,
    settings: Settings,
    vinted: VintedClient,
    llm: LLM,
    cache: HuntCache,
    hunt: _Hunt,
    photo_breaker: _Breaker,
) -> Extraction:
    photo_urls = detail.photo_urls[: settings.max_photos] or (
        [detail.photo_url] if detail.photo_url else []
    )
    digest = _extraction_digest(system, plan, detail, photo_urls)
    cached = cache.get_extraction(detail.id, settings.extract_model, digest)
    if cached is not None:
        return cached.model_copy(update={"item_id": detail.id})

    blocks: list[dict[str, Any]] = []
    attached = 0
    for url in photo_urls:
        if not photo_breaker.allow():
            break
        try:
            data, media_type = await vinted.download_image(url)
        except VintedError as exc:
            photo_breaker.failure(exc)
            hunt.log(f"photo skipped for item {detail.id}: {exc}")
            continue
        photo_breaker.success()
        blocks.append(image_block(data, media_type))
        attached += 1
    blocks.append(text_block(_extract_text(plan, detail, attached)))

    draft = await llm.parse(
        stage="extract",
        model=settings.extract_model,
        system=system,
        content=blocks,
        output_model=ExtractionDraft,
        effort="medium",
        cache=True,
    )
    flags, moved = split_spec_gaps(draft.red_flags, plan.risk_terms, plan.spec_keys)
    gaps = list(dict.fromkeys([*draft.listing_gaps, *moved]))
    extraction = Extraction(
        item_id=detail.id, **draft.model_copy(update={"red_flags": flags, "listing_gaps": gaps}).model_dump()
    )
    if attached == len(photo_urls):
        cache.put_extraction(extraction, settings.extract_model, digest)
    else:
        # Missing photos mean the answer was made from less evidence. Do not cache it, so a
        # later run with the photos available gets a proper analysis.
        hunt.log(f"item {detail.id}: {len(photo_urls) - attached} photo(s) missing, result not cached")
    return extraction


# --- Stage 6: identify and specs ---------------------------------------------


def _identify_query(brand: str, title: str, visual_description: str) -> str:
    return f"{brand} {title} {visual_description}"


def _identify_title_key(brand: str, title: str) -> str:
    """Cache key for an identification by normalised brand and title only, with no listing gaps. It is the
    fallback for a listing whose brand and title were identified before. Only high-confidence answers are
    stored under it."""
    payload = json.dumps(
        {"prompt_version": PROMPT_VERSION, "brand": normalize_text(brand), "title": normalize_text(title)},
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cached_identify(cache: HuntCache, extraction: Extraction, title: str) -> IdentifyDraft | None:
    """An earlier identification for this listing: the full key (brand, title, visible features and gaps) first,
    then the gaps-free brand and title key. A refusal is never an answer."""
    brand = extraction.brand or ""
    query_key = _identify_query_key(_identify_query(brand, title, extraction.visual_description), extraction.listing_gaps)
    for key in (query_key, _identify_title_key(brand, title)):
        draft = cache.get_identify(key)
        if draft is not None and draft.evidence != WEB_SEARCH_UNAVAILABLE_EVIDENCE:
            return draft
    return None


def _apply_identify(extractions: dict[str, Extraction], eid: str, draft: IdentifyDraft, hunt: _Hunt) -> None:
    extraction = extractions[eid]
    model = clean_model_label(draft.model)
    if model and draft.model_confidence >= IDENTIFY_ACCEPT_CONFIDENCE:
        evidence = f"{extraction.identification_evidence} | web: {draft.evidence}".strip(" |")
        extractions[eid] = extraction.model_copy(
            update={
                "model": model,
                "model_confidence": draft.model_confidence,
                "identification_evidence": evidence,
            }
        )
        hunt.count("identified_on_web")
    else:
        hunt.log(f"web identification for item {eid} was inconclusive")


async def _identify(
    extractions: dict[str, Extraction],
    details: dict[str, ListingDetail],
    settings: Settings,
    llm: LLM,
    cache: HuntCache,
    hunt: _Hunt,
) -> None:
    """Web identification for listings whose model is unclear. Updates the extractions in place.

    Every listing whose model is missing or below IDENTIFY_ACCEPT_CONFIDENCE is a candidate, with or without a
    brand. Candidates with the same normalised brand and title share one identification, taken from the most
    relevant of them (in-hunt dedupe). Each group first looks in the cache (see _cached_identify); only the
    groups that miss are searched, and at most IDENTIFY_MAX_PER_HUNT of those, in the order of `details`, which
    is the triage relevance order. A web-search refusal is never cached, so a later run tries again.
    """
    groups: dict[str, list[str]] = {}
    for eid in details:
        extraction = extractions.get(eid)
        if extraction is None or not extraction.is_requested_product_type:
            continue  # a wrong product type is never researched, so its model is not needed
        if extraction.model and extraction.model_confidence >= IDENTIFY_ACCEPT_CONFIDENCE:
            continue
        groups.setdefault(_identify_title_key(extraction.brand or "", details[eid].title), []).append(eid)
    if not groups:
        return

    answers: dict[str, IdentifyDraft] = {}
    misses: list[str] = []
    for group_key, eids in groups.items():
        cached = _cached_identify(cache, extractions[eids[0]], details[eids[0]].title)
        if cached is None:
            misses.append(group_key)
        else:
            answers[group_key] = cached
            hunt.count("identify_cache_hits")
    misses = misses[:IDENTIFY_MAX_PER_HUNT]
    semaphore = asyncio.Semaphore(SPECS_CONCURRENCY)

    async def identify(group_key: str) -> None:
        rep = groups[group_key][0]
        extraction = extractions[rep]
        title = details[rep].title
        brand = extraction.brand or ""
        async with semaphore:
            try:
                draft = await llm.identify_model(
                    model=settings.specs_model,
                    brand=brand,
                    title=title,
                    visual_description=extraction.visual_description,
                    listing_gaps=extraction.listing_gaps,
                )
            except LLMFatal:
                raise
            except Exception as exc:  # identification is optional: keep the listing as it is
                if not isinstance(exc, LLMError):
                    log.exception("unexpected error identifying item %s", rep)
                hunt.log(f"web identification failed for item {rep}: {_describe(exc)}")
                return
            hunt.count("identify_calls")
            if draft.evidence != WEB_SEARCH_UNAVAILABLE_EVIDENCE:
                query_key = _identify_query_key(
                    _identify_query(brand, title, extraction.visual_description), extraction.listing_gaps
                )
                cache.put_identify(query_key, draft)
                if draft.model and draft.model_confidence >= IDENTIFY_SHARE_MIN_CONFIDENCE:
                    cache.put_identify(_identify_title_key(brand, title), draft)
        answers[group_key] = draft

    await _gather_all(identify(group_key) for group_key in misses)

    for group_key, eids in groups.items():
        draft = answers.get(group_key)
        if draft is None:
            continue
        if len(eids) > 1:
            hunt.count("identify_shared", len(eids) - 1)
        for eid in eids:
            _apply_identify(extractions, eid, draft, hunt)


def _research_order(
    plan: SearchPlan,
    extractions: dict[str, Extraction],
    details: dict[str, ListingDetail],
    relevance: dict[str, int],
    limit: int,
) -> tuple[dict[str, tuple[str, str]], set[str], set[str]]:
    """The model keys to research, as (brand, model) labels, plus the listings and model keys that are skipped
    because a listing already fails a hard requirement from its photos (a vision "unmet"; specs cannot change that).

    The order matters only when a limit is set: then the most relevant listings come first (relevance from
    triage, then fewer red flags, then the lower landed or total price, then the original order). Without a
    limit every distinct model is researched once, in the order of `details`.
    """
    ranked: list[tuple[tuple[Any, ...], str, tuple[str, str]]] = []
    skipped: set[str] = set()
    failed_keys: set[str] = set()
    for index, eid in enumerate(details):
        extraction = extractions.get(eid)
        if extraction is None:
            continue
        key = _specs_key(extraction)
        if requirements_status(plan, extraction, None) == "failed":
            skipped.add(eid)
            if key:
                failed_keys.add(key)
            continue
        labels = _specs_labels(extraction)
        if key is None or labels is None:
            continue
        if limit:
            detail = details[eid]
            price = detail.landed_price_eur if detail.landed_price_eur is not None else detail.total_price_eur
            sort_key = (
                -relevance.get(eid, 0),
                len(extraction.red_flags),
                price if price is not None else float("inf"),
                index,
            )
            ranked.append((sort_key, key, labels))
        else:
            ranked.append(((index,), key, labels))
    if limit:
        ranked.sort(key=lambda entry: entry[0])
    labels_by_key: dict[str, tuple[str, str]] = {}
    for _, key, labels in ranked:
        labels_by_key.setdefault(key, labels)
    return labels_by_key, skipped, failed_keys


def _store_specs(
    cache: HuntCache,
    cache_key: str,
    specs: ModelSpecs,
    brand: str,
    model_name: str,
    hunt: _Hunt,
) -> None:
    """Store a web answer. Exact-variant specs with values never expire, and the other web answers (non-exact,
    or no specs found) expire after SPECS_NONEXACT_TTL_HOURS (see HuntCache.get_specs). A model-knowledge answer
    is not research, so it is never stored."""
    if specs.lookup_mode != "web":
        hunt.warn(
            f"Specs for {brand} {model_name} come from model knowledge "
            "(web search unavailable); verify before buying."
        )
        return
    cache.put_specs(cache_key, specs)


def _specs_status(specs: ModelSpecs | None, failed: bool, limited: bool) -> str:
    if specs is not None and specs.specs:
        return SPECS_RESEARCHED
    if failed:
        return SPECS_FAILED
    if limited:
        return SPECS_LIMIT
    return SPECS_NOT_FOUND


async def _specs(
    extractions: dict[str, Extraction],
    details: dict[str, ListingDetail],
    plan: SearchPlan,
    relevance: dict[str, int],
    settings: Settings,
    llm: LLM,
    cache: HuntCache,
    hunt: _Hunt,
) -> tuple[dict[str, ModelSpecs], dict[str, str]]:
    """Specs per model key, and the specs status of each listing (SPECS_* constants).

    Every distinct model of the hunt is researched once, unless the cache has it: exact web specs with values
    within their lifetime, or other web answers within SPECS_NONEXACT_TTL_HOURS. A cache hit never uses the
    research budget. When SPECS_MAX_PER_HUNT is set, at most that many models are researched, and the rest are
    marked "not researched (limit)". Listings that already fail a hard requirement are not researched.
    """
    await _identify(extractions, details, settings, llm, cache, hunt)
    limit = settings.specs_max_per_hunt
    labels_by_key, skipped, failed_keys = _research_order(plan, extractions, details, relevance, limit)

    specs_by_key: dict[str, ModelSpecs] = {}
    to_research: list[str] = []
    limited: set[str] = set()
    for key in labels_by_key:
        cached = cache.get_specs(specs_cache_key(key, plan.spec_keys))
        if cached is not None:
            specs_by_key[key] = cached
            hunt.count("models_cached_exact" if _is_exact_web(cached) else "models_cached_nonexact")
        elif limit and len(to_research) >= limit:
            limited.add(key)
        else:
            to_research.append(key)
    hunt.count("models_skipped_limit", len(limited))
    hunt.count("models_skipped_failed", len(failed_keys - set(labels_by_key)))

    semaphore = asyncio.Semaphore(SPECS_CONCURRENCY)
    done = 0

    async def research(key: str) -> None:
        nonlocal done
        brand, model_name = labels_by_key[key]
        async with semaphore:
            try:
                hunt.count("models_researched")
                try:
                    specs = await llm.research_specs(
                        model=settings.specs_model,
                        brand=brand,
                        model_name=model_name,
                        model_key=key,
                        spec_keys=plan.spec_keys,
                    )
                except LLMFatal:
                    raise
                except Exception as exc:  # specs are optional enrichment
                    if not isinstance(exc, LLMError):
                        log.exception("unexpected error looking up %s", key)
                    hunt.log(f"specs lookup failed for {brand} {model_name}: {_describe(exc)}")
                else:
                    specs_by_key[key] = specs
                    _store_specs(cache, specs_cache_key(key, plan.spec_keys), specs, brand, model_name, hunt)
            except LLMFatal:
                raise
            except Exception as exc:  # e.g. a cache write failure: keep the specs we have, warn
                log.exception("unexpected error handling specs for %s", key)
                hunt.warn(f"Specs for {brand} {model_name} could not be stored: {_describe(exc)}")
            finally:
                done += 1
                hunt.progress("specs", done, len(to_research))

    await _gather_all(research(key) for key in to_research)

    status: dict[str, str] = {}
    for eid, extraction in extractions.items():
        key = _specs_key(extraction)
        specs = specs_by_key.get(key) if key else None
        if eid in skipped and not extraction.is_requested_product_type:
            status[eid] = SPECS_WRONG_TYPE
        else:
            status[eid] = _specs_status(specs, eid in skipped, bool(key) and key in limited)
    return specs_by_key, status


# --- Stage 7: rank -----------------------------------------------------------


async def _rank(
    plan: SearchPlan,
    details: Sequence[ListingDetail],
    extractions: dict[str, Extraction],
    specs_by_key: dict[str, ModelSpecs],
    specs_status: dict[str, str],
    settings: Settings,
    llm: LLM,
    hunt: _Hunt,
) -> list[RankedEntry]:
    to_rank = [(d, extractions[d.id]) for d in details if d.id in extractions]
    if not to_rank:
        return []

    statuses: dict[str, str] = {}
    specs_for: dict[str, ModelSpecs | None] = {}
    payload = []
    for detail, extraction in to_rank:
        key = _specs_key(extraction)
        specs = specs_by_key.get(key) if key else None
        specs_for[detail.id] = specs
        statuses[detail.id] = requirements_status(plan, extraction, specs)
        item = _rank_item(
            detail,
            extraction,
            specs,
            identified=key is not None,
            requirement_checks=_rank_checks(plan, extraction, specs),
            specs_status=specs_status.get(detail.id, SPECS_NOT_FOUND),
        )
        item["requirements_status"] = statuses[detail.id]
        payload.append(item)
    plan_payload = plan.model_dump(exclude={"queries", "catalog_ids", "color_ids"})
    content = (
        f"Buyer plan:\n{_json_for_prompt(plan_payload)}\n\n"
        f"Candidates ({len(payload)}):\n<candidates>\n{_json_for_prompt(payload)}\n</candidates>"
    )
    output = await llm.parse(
        stage="rank",
        model=settings.rank_model,
        system=RANK_SYSTEM,
        content=content,
        output_model=RankingOutput,
        effort="high",
    )

    by_id = {d.id: (d, e) for d, e in to_rank}
    ranked_ids: set[str] = set()
    ordered = []
    for ranking in output.rankings:
        if ranking.item_id not in by_id or ranking.item_id in ranked_ids:
            continue
        ranked_ids.add(ranking.item_id)
        detail, extraction = by_id[ranking.item_id]
        ranking.score = max(0, min(100, ranking.score))
        status = statuses[detail.id]
        # The code's status decides the caps, not the ranker's flag.
        ranking.meets_hard_requirements = status != "failed"
        _apply_caps(ranking, status, extraction)
        ordered.append((ranking, detail, extraction, specs_for[detail.id], status))
    for detail, _ in to_rank:
        if detail.id not in ranked_ids:
            hunt.reject("rank", detail, "not ranked by the model")

    # The verdict tier first (top_pick, good, ok, avoid), then the score. The caps above are already applied.
    ordered.sort(key=lambda entry: (VERDICT_ORDER.index(entry[0].verdict), -entry[0].score))
    return [
        RankedEntry(
            rank=index,
            listing=detail,
            extraction=extraction,
            specs=specs,
            new_price=_new_price(specs, detail),
            ranking=ranking,
            requirements_status=status,
        )
        for index, (ranking, detail, extraction, specs, status) in enumerate(ordered, start=1)
    ]


