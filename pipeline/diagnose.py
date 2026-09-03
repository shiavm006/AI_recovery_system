from __future__ import annotations

import json
import logging
import os
import re
import sys
import time
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Protocol

import requests
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import Diagnosis, FailureEvent, RootCause

load_dotenv(_ROOT / ".env")

log = logging.getLogger(__name__)

Sig = tuple[str, str, str, str, str, int, int, float, float]
CacheKey = tuple[str, str, str, str, str, str, str, str, str]

TECHNICAL_REASONS: frozenset[str] = frozenset(
    {
        "payment_timed_out",
        "request_timed_out",
        "bank_technical_error",
        "bank_not_available",
        "gateway_technical_error",
        "issuer_technical_error",
    }
)

# Unique Razorpay reasons only. Shared reasons (payment_failed,
# debit_instrument_blocked) stay None so the LLM layer can handle them.
RULE_BY_REASON: dict[str, RootCause] = {
    "insufficient_funds": RootCause.INSUFFICIENT_FUNDS,
    "debit_declined": RootCause.INSUFFICIENT_FUNDS,
    "debit_instrument_inactive": RootCause.DEAD_MANDATE,
    "bank_account_invalid": RootCause.DEAD_MANDATE,
    "mandate_creation_failed": RootCause.DEAD_MANDATE,
    "card_declined": RootCause.HARD_DECLINE,
    "payment_declined": RootCause.HARD_DECLINE,
    "transaction_limit_exceeded": RootCause.HARD_DECLINE,
    "bank_technical_error": RootCause.ISSUER_DOWNTIME,
    "bank_not_available": RootCause.ISSUER_DOWNTIME,
    "gateway_technical_error": RootCause.ISSUER_DOWNTIME,
    "issuer_technical_error": RootCause.ISSUER_DOWNTIME,
    "payment_timed_out": RootCause.NETWORK_TIMEOUT,
    "request_timed_out": RootCause.NETWORK_TIMEOUT,
    "payment_risk_check_failed": RootCause.RISK_BLOCK,
}

_outage_counts: dict[str, int] = {}
_cache: dict[CacheKey, tuple[RootCause, float, list[str]]] = {}

# Issuer values that cannot support bank-outage correlation. "unknown" is a
# missing identity; "psp:…" is a payer-app handle from a VPA, not a bank.
UNKNOWN_ISSUER = "unknown"
PSP_ISSUER_PREFIX = "psp:"

# Sliding window and absolute-cluster floor for fleet outage detection.
OUTAGE_WINDOW_MINUTES = 15
OUTAGE_MIN_EVENTS = 8
OUTAGE_CONCENTRATION = 0.7
# CHOSEN, not derived: how far above the issuer's own recent technical rate
# the in-window mix must climb before concentration alone is enough. Without
# this, a bank that is 35% of volume trips the detector at its normal failure
# rate whenever nine of its timeouts land in one window. 3× is a round number
# that clears the generator's normal-rate noise and still catches the outage
# windows the simulator plants; replace with a calibrated multiple from
# production baselines when those exist.
OUTAGE_ELEVATION_FACTOR = 3.0

_stats: dict = {
    "llm_calls": 0,
    "cache_hits": 0,
    "fallbacks": 0,
    "llm_errors": [],
    "unique_signatures": 0,
    "unique_cache_keys": 0,
    "fallback_reasons": Counter(),
}


def _is_bank_issuer(issuer: str) -> bool:
    """Whether ``issuer`` names an issuing bank we can correlate on."""
    return bool(issuer) and issuer != UNKNOWN_ISSUER and not issuer.startswith(
        PSP_ISSUER_PREFIX
    )


def _is_correlatable(event: FailureEvent) -> bool:
    """Whether this failure may enter the outage-correlation input.

    Re-presentations (``retries_used > 0``) are excluded: a retry storm we
    caused produces exactly the technical-cluster signature the detector
    reads as an issuer outage. The field does not distinguish our retries
    from the merchant's, so both are dropped — safer than treating a storm
    we started as evidence the bank is down.
    """
    return (
        _is_bank_issuer(event.issuer)
        and event.retries_used == 0
    )
_gemini_client = None
# The most recent provider failure, so a fallback can name the actual cause
# rather than the category it fell into.
_last_error: str | None = None


class _LlmItem(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cause: RootCause
    confidence: float
    evidence: list[str] = Field(default_factory=list)

    @field_validator("confidence")
    @classmethod
    def confidence_unit(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be 0–1")
        return value


class _LlmResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    items: list[_LlmItem]


def _is_technical(event: FailureEvent) -> bool:
    reason = event.error_reason
    if reason in TECHNICAL_REASONS:
        return True
    if "timed_out" in reason or "timeout" in reason:
        return True
    return "technical_error" in reason


def _sig(event: FailureEvent) -> Sig:
    return (
        event.error_code,
        event.error_source,
        event.error_step,
        event.error_reason,
        event.method,
        event.days_since_mandate_created,
        event.day_of_month,
        event.issuer_recent_failure_rate,
        event.amount_vs_customer_avg,
    )


def _cache_key(event: FailureEvent) -> CacheKey:
    days = event.days_since_mandate_created
    day = event.day_of_month
    rate = event.issuer_recent_failure_rate
    amt = event.amount_vs_customer_avg
    return (
        event.error_code,
        event.error_source,
        event.error_step,
        event.error_reason,
        event.method,
        "<30" if days < 30 else "30–90" if days < 90 else "90–180" if days < 180 else "180+",
        "1–5" if day <= 5 else "6–20" if day <= 20 else "21–25" if day <= 25 else "26–31",
        "<0.05" if rate < 0.05 else "0.05–0.15" if rate < 0.15 else "0.15+",
        "<0.8" if amt < 0.8 else "0.8–1.3" if amt < 1.3 else "1.3+",
    )


_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# Single source for which model each provider calls, so run_config reports the
# model that was actually used rather than a second copy of the defaults.
_MODEL_ENV = {
    "anthropic": ("ANTHROPIC_MODEL", "claude-sonnet-4-5"),
    "openai": ("OPENAI_MODEL", "gpt-4o-mini"),
    "gemini": ("GEMINI_MODEL", "gemini-2.0-flash"),
}


DEFAULT_BATCH_DELAY_SECONDS = 2.0


def _provider_name() -> str:
    raw = os.getenv("NAKAD_LLM_PROVIDER", "anthropic").strip().lower()
    return raw or "anthropic"


def _batch_delay_seconds() -> float:
    """Pause between sequential LLM batches, so a run paces itself.

    Six back-to-back batches spend a whole minute's token allowance in a few
    seconds and then sit through the 429 backoff anyway; a small fixed gap is
    cheaper than the retries it avoids.
    """
    raw = os.getenv("LLM_BATCH_DELAY_SECONDS", "").strip()
    if not raw:
        return DEFAULT_BATCH_DELAY_SECONDS
    try:
        return max(0.0, float(raw))
    except ValueError:
        log.warning(
            "ignoring LLM_BATCH_DELAY_SECONDS=%r, using %.1fs",
            raw,
            DEFAULT_BATCH_DELAY_SECONDS,
        )
        return DEFAULT_BATCH_DELAY_SECONDS


def _key_for(provider: str) -> str:
    env_name = _KEY_ENV.get(provider)
    if not env_name:
        return ""
    return os.getenv(env_name, "").strip()


def _model_name(provider: str) -> str:
    """The model this provider would actually call, defaults included."""
    env_name, default = _MODEL_ENV.get(provider, ("", ""))
    return os.getenv(env_name, default).strip() or default if env_name else ""


def run_config(diagnoses: list[Diagnosis]) -> dict:
    """Provenance to travel with anything computed from these diagnoses.

    Every headline number here is downstream of the diagnosis layer, so a run
    with the provider off still produces real-looking figures that mean
    something entirely different. Reporting the provider next to the numbers
    is the only thing that stops the two being confused.
    """
    provider = _provider_name()
    unknown = sum(diagnosis.cause is RootCause.UNKNOWN for diagnosis in diagnoses)
    # A fallback still carries the method of the layer that failed it, which
    # would report "llm" for events no LLM ever saw. Counting those separately
    # keeps by_method a record of work done rather than work attempted.
    by_method = Counter(
        "undiagnosed" if diagnosis.cause is RootCause.UNKNOWN else diagnosis.method
        for diagnosis in diagnoses
    )
    return {
        "provider": provider,
        "model": _model_name(provider),
        "events": len(diagnoses),
        "by_method": {name: by_method[name] for name in sorted(by_method)},
        "unknown": unknown,
        "degraded": bool(unknown),
    }


def _note_error(exc: BaseException) -> str:
    """Record a provider failure and surface it immediately.

    Accumulating these in stats alone means a degraded run is only visible to
    whoever reads the returned dict. Logging at the point of failure puts the
    specific error in front of the operator while the run is happening.
    """
    global _last_error
    detail = f"{type(exc).__name__}: {exc}"
    _last_error = detail
    _stats["llm_errors"].append(detail)
    log.error("llm provider failure: %s", detail)
    return detail


class LlmProvider(Protocol):
    def __call__(self, events: list[FailureEvent]) -> list[_LlmItem] | None: ...


class OutageReport:
    """Result of one fleet pass over a set of failures.

    ``correlation_possible`` is True when at least one sliding window held
    enough correlatable technical failures to run the concentration test.
    False means the layer had nothing to work with — a single live event, or
    sixty dripped across four hours — and callers must say so in evidence
    rather than silently omitting the layer.
    """

    __slots__ = ("outages", "correlation_possible", "window_minutes", "min_events")

    def __init__(
        self,
        outages: dict[str, tuple[datetime, datetime]],
        correlation_possible: bool,
        window_minutes: int = OUTAGE_WINDOW_MINUTES,
        min_events: int = OUTAGE_MIN_EVENTS,
    ) -> None:
        self.outages = outages
        self.correlation_possible = correlation_possible
        self.window_minutes = window_minutes
        self.min_events = min_events

    def insufficient_evidence(self) -> str:
        return (
            f"fleet_correlation=skipped: fewer than {self.min_events} correlatable "
            f"technical failures in any {self.window_minutes}-minute window "
            f"(slow drips outside that horizon are out of scope)"
        )


def _issuer_elevated(
    issuer: str,
    window_events: list[FailureEvent],
    elevation_factor: float,
) -> bool:
    """Whether this issuer's in-window mix is above its own recent baseline.

    Among the issuer's failures in the window, the technical share is compared
    to the mean of ``issuer_recent_failure_rate`` on those same events. The
    batch only contains failures, so this is a mix shift, not a volume rate —
    exactly the signal that separates "bank is 35% of traffic" from "bank is
    suddenly all timeouts."
    """
    theirs = [event for event in window_events if event.issuer == issuer]
    if not theirs:
        return False
    tech = sum(1 for event in theirs if _is_technical(event))
    observed = tech / len(theirs)
    baseline = sum(event.issuer_recent_failure_rate for event in theirs) / len(theirs)
    return observed >= baseline * elevation_factor


def detect_outages(
    events: list[FailureEvent],
    window_minutes: int = OUTAGE_WINDOW_MINUTES,
    min_events: int = OUTAGE_MIN_EVENTS,
    concentration: float = OUTAGE_CONCENTRATION,
    elevation_factor: float = OUTAGE_ELEVATION_FACTOR,
) -> OutageReport:
    """Flag issuer outages from a set of failures before per-event diagnosis.

    A single timeout looks like a customer problem. Only a cluster of
    technical-flavoured failures on one issuer, across distinct payment_ids,
    reveals a bank outage. That correlation does not exist until enough
    events are seen together — a batch, or a live buffer of recent traffic.

    Two gates, both required:

    * Absolute concentration: the dominant issuer owns at least
      ``concentration`` of the technical failures in the window.
    * Elevation: that issuer's in-window technical mix is at least
      ``elevation_factor`` times its own ``issuer_recent_failure_rate``.
      Concentration alone fires on high-volume banks at their normal rate.

    Scope: only bursts inside ``window_minutes``. Sixty failures dripped
    evenly over four hours never co-occur in one window and are out of
    scope — ``correlation_possible`` will be False and the caller must say
    so, not pretend the layer ran.

    Events whose issuer is missing (``unknown``) or a PSP handle (``psp:…``),
    and events with ``retries_used > 0`` (re-presentations), are excluded
    from the correlation input.
    """
    global _outage_counts
    _outage_counts = {}
    empty = OutageReport({}, False, window_minutes, min_events)
    if not events:
        return empty

    ordered = sorted(events, key=lambda event: event.occurred_at)
    found: dict[str, tuple[datetime, datetime]] = {}
    right = 0
    n = len(ordered)
    delta = timedelta(minutes=window_minutes)
    any_dense_window = False

    for left in range(n):
        limit = ordered[left].occurred_at + delta
        while right < n and ordered[right].occurred_at <= limit:
            right += 1
        window_events = ordered[left:right]
        by_payment: dict[str, FailureEvent] = {}
        for event in window_events:
            if (
                _is_technical(event)
                and _is_correlatable(event)
                and event.payment_id not in by_payment
            ):
                by_payment[event.payment_id] = event
        cluster = list(by_payment.values())
        if len(cluster) < min_events:
            continue
        any_dense_window = True
        counts = Counter(event.issuer for event in cluster)
        issuer, top = counts.most_common(1)[0]
        if top / len(cluster) < concentration:
            continue
        if not _issuer_elevated(issuer, window_events, elevation_factor):
            continue
        issuer_events = [event for event in cluster if event.issuer == issuer]
        start = min(event.occurred_at for event in issuer_events)
        end = max(event.occurred_at for event in issuer_events)
        if issuer in found:
            prev_s, prev_e = found[issuer]
            start = min(start, prev_s)
            end = max(end, prev_e)
            top = max(_outage_counts[issuer], top)
        found[issuer] = (start, end)
        _outage_counts[issuer] = top

    return OutageReport(found, any_dense_window, window_minutes, min_events)


def diagnose_by_rule(
    event: FailureEvent,
    outages: dict[str, tuple[datetime, datetime]] | OutageReport,
) -> Diagnosis | None:
    windows = outages.outages if isinstance(outages, OutageReport) else outages
    # Belt-and-braces: even if an old outage map somehow named a non-bank
    # issuer, do not promote a missing identity into a confident finding.
    if _is_bank_issuer(event.issuer):
        window = windows.get(event.issuer)
        if window is not None:
            start, end = window
            if start <= event.occurred_at <= end:
                count = _outage_counts.get(event.issuer, 0)
                return Diagnosis(
                    event_id=event.event_id,
                    cause=RootCause.ISSUER_DOWNTIME,
                    confidence=0.95,
                    evidence=[
                        f"issuer={event.issuer}",
                        f"window={start.isoformat()}..{end.isoformat()}",
                        f"correlated_failures={count}",
                    ],
                    method="fleet",
                )

    cause = RULE_BY_REASON.get(event.error_reason)
    if cause is None:
        return None
    return Diagnosis(
        event_id=event.event_id,
        cause=cause,
        confidence=0.9,
        evidence=[
            f"error_reason={event.error_reason}",
            f"error_source={event.error_source}",
            f"error_code={event.error_code}",
        ],
        method="rule",
    )


def _with_fleet_note(diagnosis: Diagnosis, report: OutageReport) -> Diagnosis:
    """Attach an explicit skip reason when the fleet layer had nothing to work with."""
    if report.correlation_possible or diagnosis.method == "fleet":
        return diagnosis
    return diagnosis.model_copy(
        update={"evidence": list(diagnosis.evidence) + [report.insufficient_evidence()]}
    )


def _fallback(event: FailureEvent, why: str) -> Diagnosis:
    """Record that no diagnosis was obtained instead of inventing one.

    This used to return INSUFFICIENT_FUNDS at confidence 0.0. That is both the
    most common real cause and the highest-scoring one in allocate, so a failed
    LLM call did not merely lose information — it manufactured plausible demand
    for the retry budget. On a 500-event batch with the provider off it inflated
    INSUFFICIENT_FUNDS from ~192 to 321, and nothing in the output said so.

    ``why`` is the specific failure, not a category, so the evidence on the
    event distinguishes a missing package from a 401 from a malformed response.
    """
    _stats["fallbacks"] += 1
    _stats["fallback_reasons"][why] += 1
    return Diagnosis(
        event_id=event.event_id,
        cause=RootCause.UNKNOWN,
        confidence=0.0,
        evidence=[f"no diagnosis: {why}"],
        method="llm",
    )


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[-1]
        if text.endswith("```"):
            text = text[: -3]
    return text.strip()


# UNKNOWN records that the pipeline failed to get an answer, which is not a
# verdict a classifier is entitled to reach, so it is not offered as an option.
# The response schema still admits it — the enum is shared — and allocate
# suppresses it either way.
CLASSIFIABLE_CAUSES: tuple[RootCause, ...] = tuple(
    cause for cause in RootCause if cause is not RootCause.UNKNOWN
)


def _classify_prompt(events: list[FailureEvent]) -> str:
    causes = ", ".join(member.value for member in CLASSIFIABLE_CAUSES)
    catalog = [
        {
            "index": i,
            "error_code": event.error_code,
            "error_source": event.error_source,
            "error_step": event.error_step,
            "error_reason": event.error_reason,
            "method": event.method,
            "days_since_mandate_created": event.days_since_mandate_created,
            "day_of_month": event.day_of_month,
            "issuer_recent_failure_rate": event.issuer_recent_failure_rate,
            "amount_vs_customer_avg": event.amount_vs_customer_avg,
        }
        for i, event in enumerate(events)
    ]
    return (
        "Classify each payment failure signature into exactly one RootCause "
        f"({causes}). Return JSON: "
        '{"items":[{"cause":"...","confidence":0.0,"evidence":["..."]}]} '
        "with one item per input, same order. confidence is in [0,1]. "
        f"Input: {json.dumps(catalog)}"
    )


def _parse_items(text: str, n: int) -> list[_LlmItem] | None:
    try:
        parsed = _LlmResponse.model_validate_json(_strip_fences(text))
    except (json.JSONDecodeError, ValidationError) as exc:
        _note_error(exc)
        return None
    # Items are positional, so a long response still answers every input in
    # order and the surplus is discardable. A short one leaves inputs
    # unanswered, and silently mapping the wrong item onto them would be worse
    # than UNKNOWN.
    if len(parsed.items) > n:
        log.warning("openai returned %d items for %d inputs, truncating", len(parsed.items), n)
        return parsed.items[:n]
    if len(parsed.items) < n:
        _note_error(ValueError(f"item count {len(parsed.items)} != {n}"))
        return None
    return parsed.items


def _call_anthropic(events: list[FailureEvent]) -> list[_LlmItem] | None:
    try:
        from anthropic import Anthropic
    except ImportError as exc:
        _note_error(exc)
        return None
    key = _key_for("anthropic")
    if not key:
        _note_error(RuntimeError("no ANTHROPIC_API_KEY"))
        return None
    try:
        workspace = os.getenv("ANTHROPIC_WORKSPACE_ID", "").strip()
        headers = {"anthropic-workspace-id": workspace} if workspace else None
        message = Anthropic(api_key=key, default_headers=headers).messages.create(
            model=_model_name("anthropic"),
            max_tokens=4096,
            system="You output only valid JSON matching the requested schema.",
            messages=[{"role": "user", "content": _classify_prompt(events)}],
        )
        text = "".join(
            block.text for block in message.content if getattr(block, "type", None) == "text"
        )
    except Exception as exc:
        _note_error(exc)
        return None
    return _parse_items(text, len(events))


OPENAI_MAX_ATTEMPTS = 3
RATE_LIMIT_DEFAULT_WAIT_SECONDS = 10.0
# A per-minute 429 clears in seconds; a daily-quota 429 asks for the rest of
# the day (1435s observed). Waiting that out stalls the whole run for one
# batch, so past this ceiling we degrade the batch instead of blocking.
OPENAI_MAX_WAIT_SECONDS = 60.0

# Groq names the delay in the error body: "Please try again in 7.66s", or
# "in 2m59.56s" once the daily bucket is involved, or "in 500ms".
_RETRY_AFTER_RE = re.compile(r"try again in\s+(?:(\d+)m)?([\d.]+)(m?s)\b", re.IGNORECASE)


def _rate_limit_wait(response: requests.Response) -> float:
    """Seconds to wait before retrying a 429.

    Prefers the standard Retry-After header and falls back to the delay named
    in the error body, since Groq sets both. An unparseable body gets a fixed
    delay rather than a retry storm.
    """
    header = response.headers.get("retry-after", "")
    try:
        return float(header)
    except ValueError:
        pass
    match = _RETRY_AFTER_RE.search(response.text or "")
    if match is None:
        return RATE_LIMIT_DEFAULT_WAIT_SECONDS
    minutes, value, unit = match.groups()
    seconds = float(value) / 1000 if unit.lower() == "ms" else float(value)
    return float(minutes or 0) * 60 + seconds


def _call_openai(events: list[FailureEvent]) -> list[_LlmItem] | None:
    key = _key_for("openai")
    if not key:
        _note_error(RuntimeError("no OPENAI_API_KEY"))
        return None
    model = _model_name("openai")
    url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1/chat/completions")
    try:
        mode = _openai_structured_mode()
    except ValueError as exc:
        _note_error(exc)
        return None
    body = {
        "model": model,
        "temperature": 0,
        "response_format": _openai_response_format(mode),
        "messages": [
            {"role": "system", "content": _openai_system_prompt(mode)},
            {"role": "user", "content": _classify_prompt(events)},
        ],
    }
    # requests, not urllib: Cloudflare's browser integrity check rejects the
    # default Python-urllib user agent with error 1010, and the identical
    # request through curl or requests returns 200.
    for attempt in range(1, OPENAI_MAX_ATTEMPTS + 1):
        try:
            response = requests.post(
                url,
                json=body,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {key}",
                },
                timeout=30,
            )
        except requests.RequestException as exc:
            log.error(
                "openai request failed: url=%s model=%s mode=%s", url, model, mode
            )
            _note_error(exc)
            return None

        # A 429 on the free tier is a tokens-per-minute ceiling, not a dead
        # request: waiting the named delay makes the same call succeed. Only
        # the final attempt is allowed to fall through to the fallback.
        if response.status_code == 429 and attempt < OPENAI_MAX_ATTEMPTS:
            wait = _rate_limit_wait(response)
            if wait > OPENAI_MAX_WAIT_SECONDS:
                log.error(
                    "openai rate limited beyond retry ceiling: url=%s model=%s "
                    "wanted=%.0fs ceiling=%.0fs body=%s",
                    url,
                    model,
                    wait,
                    OPENAI_MAX_WAIT_SECONDS,
                    response.text or "(empty)",
                )
                _note_error(
                    RuntimeError(
                        f"429 rate limited: asked for {wait:.0f}s, above the "
                        f"{OPENAI_MAX_WAIT_SECONDS:.0f}s retry ceiling"
                    )
                )
                return None
            log.warning(
                "openai rate limited: url=%s model=%s attempt=%d/%d waiting=%.2fs",
                url,
                model,
                attempt,
                OPENAI_MAX_ATTEMPTS,
                wait,
            )
            time.sleep(wait)
            continue
        break

    if not response.ok:
        # The status line alone is useless: 403 Forbidden covers a bad model,
        # a rate limit, and a rejected parameter. The body is the only way to
        # tell them apart.
        response_body = response.text or ""
        log.error(
            "openai request failed: url=%s model=%s mode=%s status=%s body=%s",
            url,
            model,
            mode,
            response.status_code,
            response_body or "(empty)",
        )
        detail = f"{response.status_code} {response.reason}"
        if response_body:
            detail = f"{detail}: {response_body}"
        _note_error(RuntimeError(detail))
        return None

    try:
        content = response.json()["choices"][0]["message"]["content"]
    except (ValueError, KeyError, IndexError, TypeError) as exc:
        log.error("openai request failed: url=%s model=%s mode=%s", url, model, mode)
        _note_error(exc)
        return None
    # Same Pydantic validation either mode: an invalid cause or malformed
    # shape falls through to UNKNOWN via the shared fallback path.
    return _parse_items(content, len(events))


# Groq's OpenAI-compatible endpoint rejects json_schema; OpenAI accepts both.
# The mode only changes how the shape is requested — response validation and
# the RootCause enum constraint are identical either way.
OPENAI_STRUCTURED_JSON_SCHEMA = "json_schema"
OPENAI_STRUCTURED_JSON_OBJECT = "json_object"
OPENAI_STRUCTURED_MODES = frozenset(
    {OPENAI_STRUCTURED_JSON_SCHEMA, OPENAI_STRUCTURED_JSON_OBJECT}
)


def _openai_structured_mode() -> str:
    mode = os.getenv("OPENAI_STRUCTURED_MODE", OPENAI_STRUCTURED_JSON_SCHEMA).strip().lower()
    if mode not in OPENAI_STRUCTURED_MODES:
        raise ValueError(
            f"OPENAI_STRUCTURED_MODE must be one of "
            f"{sorted(OPENAI_STRUCTURED_MODES)}, got {mode!r}"
        )
    return mode


def _openai_json_schema() -> dict:
    """Strict schema whose cause enum matches CLASSIFIABLE_CAUSES.

    Built by hand rather than from ``_LlmResponse.model_json_schema()`` so
    UNKNOWN is not offered — that enum member exists for fallbacks, not as a
    classifier verdict.
    """
    causes = [cause.value for cause in CLASSIFIABLE_CAUSES]
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "cause": {"type": "string", "enum": causes},
                        "confidence": {"type": "number"},
                        "evidence": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                    },
                    "required": ["cause", "confidence", "evidence"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }


def _openai_response_format(mode: str) -> dict:
    if mode == OPENAI_STRUCTURED_JSON_OBJECT:
        return {"type": "json_object"}
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "root_cause_batch",
            "strict": True,
            "schema": _openai_json_schema(),
        },
    }


def _openai_system_prompt(mode: str) -> str:
    causes = ", ".join(cause.value for cause in CLASSIFIABLE_CAUSES)
    shape = (
        '{"items":[{"cause":"<RootCause>","confidence":0.0,"evidence":["..."]}]} '
        f"where cause is one of [{causes}], confidence is in [0,1], "
        "and items has one entry per input in the same order."
    )
    if mode == OPENAI_STRUCTURED_JSON_OBJECT:
        # Groq only guarantees JSON object shape; the enum constraint has to
        # live in the prompt and is re-enforced by Pydantic after the call.
        return f"You output only valid JSON matching this exact shape: {shape}"
    return "You output only valid JSON matching the requested schema."


def _close_gemini_client() -> None:
    global _gemini_client
    client = _gemini_client
    _gemini_client = None
    close = getattr(client, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            pass


def _call_gemini(events: list[FailureEvent]) -> list[_LlmItem] | None:
    global _gemini_client
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        _note_error(exc)
        return None
    key = _key_for("gemini")
    if not key:
        _note_error(RuntimeError("no GEMINI_API_KEY"))
        return None
    try:
        if _gemini_client is None:
            _gemini_client = genai.Client(api_key=key)
        response = _gemini_client.models.generate_content(
            model=_model_name("gemini"),
            contents=_classify_prompt(events),
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=_LlmResponse,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    except Exception as exc:
        _note_error(exc)
        return None
    parsed = getattr(response, "parsed", None)
    if parsed is not None:
        try:
            body = parsed if isinstance(parsed, _LlmResponse) else _LlmResponse.model_validate(parsed)
        except ValidationError as exc:
            _note_error(exc)
            return None
        if len(body.items) != len(events):
            _note_error(ValueError(f"item count {len(body.items)} != {len(events)}"))
            return None
        return body.items
    return _parse_items(getattr(response, "text", None) or "", len(events))


PROVIDERS: dict[str, LlmProvider] = {
    "anthropic": _call_anthropic,
    "openai": _call_openai,
    "gemini": _call_gemini,
}


def _call_llm(events: list[FailureEvent]) -> tuple[list[_LlmItem] | None, str]:
    """Call the provider and report why it failed, if it did.

    The reason travels with the result so the fallback can put the actual
    error on the event instead of a category name; every provider path already
    routes its exception through :func:`_note_error`.
    """
    global _last_error
    _last_error = None
    name = _provider_name()
    provider = PROVIDERS.get(name)
    if provider is None:
        return None, _note_error(RuntimeError(f"unknown NAKAD_LLM_PROVIDER={name!r}"))
    items = provider(events)
    return items, _last_error or "provider returned no usable items"


def diagnose_by_llm(events: list[FailureEvent]) -> list[Diagnosis]:
    if not events:
        return []

    name = _provider_name()
    key_found = bool(_key_for(name))

    by_key: dict[CacheKey, list[int]] = {}
    order: list[CacheKey] = []
    raw: set[Sig] = set()
    for index, event in enumerate(events):
        raw.add(_sig(event))
        key = _cache_key(event)
        if key not in by_key:
            by_key[key] = []
            order.append(key)
        by_key[key].append(index)
    _stats["unique_signatures"] = len(raw)
    _stats["unique_cache_keys"] = len(order)

    if name == "none":
        detail = _note_error(RuntimeError("NAKAD_LLM_PROVIDER=none"))
        return [_fallback(event, detail) for event in events]
    if name not in PROVIDERS:
        detail = _note_error(RuntimeError(f"unknown NAKAD_LLM_PROVIDER={name!r}"))
        return [_fallback(event, detail) for event in events]
    if not key_found:
        env_name = _KEY_ENV.get(name, "API_KEY")
        detail = _note_error(RuntimeError(f"no {env_name} in environment"))
        return [_fallback(event, detail) for event in events]

    diagnoses: list[Diagnosis | None] = [None] * len(events)
    uncached: list[CacheKey] = []
    for key in order:
        cached = _cache.get(key)
        indices = by_key[key]
        if cached is not None:
            cause, confidence, evidence = cached
            for index in indices:
                _stats["cache_hits"] += 1
                diagnoses[index] = Diagnosis(
                    event_id=events[index].event_id,
                    cause=cause,
                    confidence=confidence,
                    evidence=list(evidence),
                    method="llm",
                )
            continue
        uncached.append(key)

    batch_delay = _batch_delay_seconds()
    try:
        for offset in range(0, len(uncached), 20):
            if offset and batch_delay:
                time.sleep(batch_delay)
            batch_keys = uncached[offset : offset + 20]
            _stats["llm_calls"] += 1
            reps = [events[by_key[key][0]] for key in batch_keys]
            items, reason = _call_llm(reps)
            for position, key in enumerate(batch_keys):
                indices = by_key[key]
                if items is None:
                    for index in indices:
                        diagnoses[index] = _fallback(events[index], reason)
                    continue
                try:
                    item = items[position]
                except IndexError as exc:
                    detail = _note_error(exc)
                    for index in indices:
                        diagnoses[index] = _fallback(events[index], detail)
                    continue
                _cache[key] = (item.cause, item.confidence, list(item.evidence))
                for hit, index in enumerate(indices):
                    if hit > 0:
                        _stats["cache_hits"] += 1
                    diagnoses[index] = Diagnosis(
                        event_id=events[index].event_id,
                        cause=item.cause,
                        confidence=item.confidence,
                        evidence=list(item.evidence),
                        method="llm",
                    )
    finally:
        if name == "gemini":
            _close_gemini_client()

    return [
        item
        if item is not None
        else _fallback(events[i], "no result returned for this event")
        for i, item in enumerate(diagnoses)
    ]


def diagnose_batch(
    events: list[FailureEvent],
    correlation_events: list[FailureEvent] | None = None,
) -> tuple[list[Diagnosis], dict]:
    """Diagnose ``events``, correlating outages over ``correlation_events``.

    ``correlation_events`` defaults to ``events``. The live path passes a
    rolling buffer of recent traffic so a single webhook is not correlated
    against itself alone — ``min_events`` is 8 and one event can never trip it.
    """
    global _cache
    _cache = {}
    _stats["llm_calls"] = 0
    _stats["cache_hits"] = 0
    _stats["fallbacks"] = 0
    _stats["llm_errors"] = []
    _stats["unique_signatures"] = 0
    _stats["unique_cache_keys"] = 0
    _stats["fallback_reasons"] = Counter()

    name = _provider_name()
    log.info("LLM provider selected: %s; key found: %s", name, bool(_key_for(name)))

    report = detect_outages(
        correlation_events if correlation_events is not None else events
    )
    diagnoses: list[Diagnosis | None] = [None] * len(events)
    leftover_idx: list[int] = []
    for index, event in enumerate(events):
        result = diagnose_by_rule(event, report)
        if result is None:
            leftover_idx.append(index)
        else:
            diagnoses[index] = _with_fleet_note(result, report)

    leftover = diagnose_by_llm([events[index] for index in leftover_idx])
    for index, result in zip(leftover_idx, leftover):
        diagnoses[index] = _with_fleet_note(result, report)

    filled = [item for item in diagnoses if item is not None]
    if _stats["fallbacks"]:
        log.error(
            "degraded run: %d of %d events have no diagnosis (cause=UNKNOWN); "
            "distinct reasons: %s",
            _stats["fallbacks"],
            len(events),
            "; ".join(
                f"{reason} (x{count})"
                for reason, count in _stats["fallback_reasons"].most_common()
            ),
        )

    by_method = Counter(item.method for item in filled)
    stats = {
        "by_method": dict(by_method),
        "llm_calls": _stats["llm_calls"],
        "cache_hits": _stats["cache_hits"],
        "fallbacks": _stats["fallbacks"],
        "unique_signatures": _stats["unique_signatures"],
        "unique_cache_keys": _stats["unique_cache_keys"],
        "llm_errors": list(_stats["llm_errors"]),
        "fallback_reasons": dict(_stats["fallback_reasons"]),
        "fleet_correlation_possible": report.correlation_possible,
        "fleet_outages": sorted(report.outages),
    }
    return filled, stats


def _load_frozen_batch(path: Path) -> list[FailureEvent]:
    import pandas as pd

    frame = pd.read_parquet(path)
    events: list[FailureEvent] = []
    for row in frame.to_dict(orient="records"):
        if row.get("subscription_id") is not None and not isinstance(
            row["subscription_id"], str
        ):
            row["subscription_id"] = None
        events.append(FailureEvent.model_validate(row))
    return events


def _print_ambiguous_true_cause(events: list[FailureEvent]) -> None:
    """Grading-only dump. diagnose_batch must not call this."""
    report = detect_outages(events)
    leftover = [
        event for event in events if diagnose_by_rule(event, report) is None
    ]
    groups: dict[Sig, Counter[str]] = {}
    order: list[Sig] = []
    for event in leftover:
        key = _sig(event)
        if key not in groups:
            groups[key] = Counter()
            order.append(key)
        label = event.true_cause.value if event.true_cause is not None else "None"
        groups[key][label] += 1
    print("ambiguous signatures vs true_cause:")
    for key in order:
        counts = groups[key]
        kind = "mixed" if len(counts) > 1 else "pure"
        print(f"  n={sum(counts.values())} {kind}  {dict(counts)}  {key}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frozen = _ROOT / "data" / "batch_seed42.parquet"
    if not frozen.exists():
        from generator.generate import freeze

        freeze(str(_ROOT / "data"), seed=42)
    events = _load_frozen_batch(frozen)
    _print_ambiguous_true_cause(events)
    _, stats = diagnose_batch(events)
    print(stats)
