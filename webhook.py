"""Live Razorpay webhook ingress. Test mode only.

Receives signed webhooks, maps them onto :class:`FailureEvent`, and runs the
same diagnose → allocate → govern path as the batch orchestrator. Writes to a
separate ledger so the demo trail is not mixed with the frozen batch run.

Never re-serialise the body for signature verification. Razorpay signs the
exact bytes it sends; ``json.loads`` then ``json.dumps`` changes whitespace
and key order and the HMAC will not match even though the payload is logically
the same. Read ``await request.body()``, verify that, then parse once.

This path allocates one event at a time, which is a degenerate case of the
batch policy rather than an equivalent of it: with a single candidate there is
no ranking, only an admission decision against whatever budget is left. That
decision is worth nothing unless the budget persists between requests, so the
remaining allowance and the per-mandate attempt count both live in
``LiveBudget``, in the same database file as the ledger. See allocate.py's
module docstring.

Fleet outage correlation also needs more than one event. Each webhook is
recorded into ``RecentFailures`` and diagnosed against the last
``OUTAGE_WINDOW_MINUTES`` of traffic; without that buffer the layer pitched as
requiring batch context silently never fires in production.

Bank identity is resolved from a vendored copy of Razorpay's IFSC bank-code
map (``data/reference/ifsc-banknames.json``), never from ifsc.razorpay.com at
runtime. The dataset is Razorpay's own, MIT-licensed code with a public-domain
dataset, sourced from RBI NEFT/RTGS lists and the NPCI ACH live-banks list —
see ``data/reference/ifsc-banknames.SOURCE.txt``.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, ConfigDict, ValidationError
from razorpay.errors import SignatureVerificationError
from razorpay.utility import Utility

from ledger import Ledger
from models import Diagnosis, FailureEvent, GateDecision, ProposedAction
from pipeline import govern
from pipeline.allocate import (
    CONTACT_POOL,
    DEFAULT_CONTACT_BUDGET,
    DEFAULT_RETRY_BUDGET,
    RETRY_POOL,
    allocate,
    budget_pool,
    index_score,
)
from pipeline.diagnose import (
    OUTAGE_WINDOW_MINUTES,
    PSP_ISSUER_PREFIX,
    UNKNOWN_ISSUER,
    diagnose_batch,
)
from pipeline.govern import RAIL_ACTIONS, to_ist
from run import (
    ENTRY_DIAGNOSED,
    ENTRY_GATED,
    ENTRY_INGESTED,
    ENTRY_PROPOSED,
    gate_context_for,
)

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

log = logging.getLogger(__name__)

LIVE_LEDGER_PATH = os.environ.get("LIVE_LEDGER_PATH", str(_ROOT / "data" / "ledger_live.db"))
WEBHOOK_SECRET_ENV = "RAZORPAY_WEBHOOK_SECRET"

FAILURE_EVENTS = frozenset({"payment.failed", "subscription.pending"})
ACK_LOG_EVENTS = frozenset({"subscription.halted", "subscription.charged"})
DOWNTIME_PREFIX = "payment.downtime."

# Fields Razorpay does not send on a typical failure webhook. Each default is
# named in provenance so a judge can see what was inferred rather than observed.
DEFAULT_DAYS_SINCE_MANDATE = 45
DEFAULT_ISSUER_FAILURE_RATE = 0.05
DEFAULT_AMOUNT_VS_AVG = 1.0
DEFAULT_RETRIES_USED = 0
DEFAULT_ISSUER = UNKNOWN_ISSUER

BANKNAMES_PATH = _ROOT / "data" / "reference" / "ifsc-banknames.json"

# Bank-issued VPA handles where the handle itself names the issuing bank,
# mapped to the IFSC four-character bank code. PSP handles (ybl, paytm, axl, …)
# are deliberately absent: those name the payer's app.
BANK_VPA_HANDLES = {
    "oksbi": "SBIN",
    "okhdfcbank": "HDFC",
    "okicici": "ICIC",
    "okaxis": "UTIB",
}

_banknames: dict[str, str] | None = None


def _load_banknames() -> dict[str, str]:
    """Load the vendored IFSC bank-code → name map. No network."""
    global _banknames
    if _banknames is None:
        _banknames = json.loads(BANKNAMES_PATH.read_text())
    return _banknames


def bank_name(code: str) -> str | None:
    """Resolve a four-character IFSC bank code to its name, or None if unknown."""
    return _load_banknames().get(code.upper())


def _ifsc_from_payment(payment: dict[str, Any]) -> str | None:
    """Pull a raw IFSC string out of the payment entity, if present."""
    for key in ("ifsc", "bank_ifsc"):
        value = payment.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    account = payment.get("bank_account")
    if isinstance(account, dict):
        value = account.get("ifsc")
        if isinstance(value, str) and value.strip():
            return value.strip()
    notes = payment.get("notes")
    if isinstance(notes, dict):
        value = notes.get("ifsc") or notes.get("bank_ifsc")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _resolve_ifsc(raw: str) -> tuple[str, str] | None:
    """Return (bank_code, bank_name) from an IFSC, or None if the code is unknown.

    Does not raise on an unknown prefix — the caller falls through to the next
    identity path instead of failing the webhook.
    """
    code = raw.strip().upper()[:4]
    if len(code) < 4 or not code.isalpha():
        return None
    name = bank_name(code)
    if name is None:
        return None
    return code, name

METHOD_MAP = {
    "upi": "upi_autopay",
    "card": "card",
    "nach": "nach",
    "emandate": "nach",
}

# A payload we cannot parse or map will fail identically on every redelivery,
# so retrying it burns Razorpay's 24-hour retry budget to no purpose. Anything
# else — a locked database, a provider timeout, a bug — may well succeed next
# time and must be left for redelivery instead of being acknowledged away.
UNPROCESSABLE_ERRORS = (ValueError, KeyError, TypeError, ValidationError)

OUTCOME_PROCESSED = "processed"
OUTCOME_ACKNOWLEDGED = "acknowledged"
OUTCOME_UNPROCESSABLE = "unprocessable"

# Razorpay stops retrying after 24 hours, so a row older than that can never
# match a redelivery and is only taking up space.
DEDUPE_RETENTION = timedelta(hours=48)
PRUNE_EVERY = 500

# Budget the live path may spend inside BUDGET_WINDOW. The batch numbers are
# per 500-event batch; reused here as a rolling daily allowance so a live event
# competes against what has actually been spent rather than an empty pool.
BUDGET_WINDOW = timedelta(hours=24)

# Where the attempt count for the NPCI cap came from.
SOURCE_LEDGER = "ledger"
SOURCE_PAYLOAD = "payload"

_live_feed: deque[LiveRecord] = deque(maxlen=20)
_ledger: Ledger | None = None
_seen: SeenEvents | None = None
_budget: LiveBudget | None = None
_recent: RecentFailures | None = None


class LiveRecord(BaseModel):
    """What GET /live returns — one row per failure webhook processed."""

    model_config = ConfigDict(frozen=True)

    razorpay_event_id: str
    razorpay_event_type: str
    event_id: str
    payment_id: str
    amount_paise: int
    method: str
    received_at: datetime
    provenance: dict[str, str]
    diagnosis: Diagnosis | None = None
    action: ProposedAction | None = None
    gate: GateDecision | None = None
    pipeline_error: str | None = None


class SeenEvents:
    """Webhook dedupe keyed by Razorpay's event id, on disk.

    Razorpay delivers at least once, so the same id can arrive minutes later
    during a retry or hours later after a redeploy. An in-memory set forgets
    everything on restart — reprocessing a whole retry burst — and grows
    without bound in a long-lived process. A table does neither.

    Shares the live ledger's database file: one artifact to ship, and the
    dedupe record cannot end up on a different disk from the trail it guards.
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the connection is cached across requests and
        # a threaded server may not hand every one to the thread that opened it.
        # Safe because sqlite3.threadsafety is 3 (serialized), which serialises
        # access to a single connection.
        self._conn = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        # Ledger.append holds this file's write lock under BEGIN IMMEDIATE.
        # Wait for it rather than raising "database is locked" mid-webhook.
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS seen_events (
              event_id TEXT PRIMARY KEY,
              seen_at TEXT NOT NULL,
              outcome TEXT NOT NULL
            )
            """
        )
        self._marks = 0
        self.prune()

    def __contains__(self, event_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM seen_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return row is not None

    def outcome_of(self, event_id: str) -> str | None:
        row = self._conn.execute(
            "SELECT outcome FROM seen_events WHERE event_id = ?", (event_id,)
        ).fetchone()
        return row[0] if row else None

    def mark(self, event_id: str, outcome: str) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO seen_events (event_id, seen_at, outcome) "
            "VALUES (?, ?, ?)",
            (event_id, datetime.now(timezone.utc).isoformat(), outcome),
        )
        self._marks += 1
        if self._marks % PRUNE_EVERY == 0:
            self.prune()

    def prune(self, older_than: timedelta = DEDUPE_RETENTION) -> int:
        """Drop rows past Razorpay's retry horizon. Returns the number removed."""
        cutoff = (datetime.now(timezone.utc) - older_than).isoformat()
        cursor = self._conn.execute(
            "DELETE FROM seen_events WHERE seen_at < ?", (cutoff,)
        )
        return cursor.rowcount

    def close(self) -> None:
        self._conn.close()


def mandate_cycle(event: FailureEvent) -> str:
    """The mandate cycle this failure falls in.

    ponytail: the IST calendar month, not the real cycle. NPCI's cap is per
    mandate cycle, whose boundaries come from the subscription's billing anchor
    — a mandate charged on the 20th runs the 20th to the 19th, not the 1st to
    the 31st. The webhook does not carry the anchor. Consequence: a failure
    either side of a month boundary looks like a fresh cycle when it is not, so
    the cap can be over-granted once per mandate per month. Upgrade path is to
    read the anchor from the Subscriptions API and bucket from there.
    """
    return to_ist(event.occurred_at).strftime("%Y-%m")


class LiveBudget:
    """Durable scarcity for the live path, in the ledger's database file.

    Without this every webhook is allocated against a full, untouched budget,
    so ten thousand webhooks produce ten thousand approvals and the constraint
    the whole policy is built around does not exist outside the batch run.

    Two pieces of state:

    * ``mandate_attempts`` — attempts *this system* has placed, per mandate per
      cycle. This is the trustworthy source for the NPCI cap. The alternative,
      ``payment.notes.retries_used``, is a merchant-controlled free-text field,
      which means the one genuinely binding rule in the gate would be enforced
      against input the merchant can set to zero.
    * ``budget_spend`` — one row per unit actually placed, so a rolling window
      can be counted. Rows, not a counter, because a counter cannot expire.
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS mandate_attempts (
              subscription_id TEXT NOT NULL,
              cycle TEXT NOT NULL,
              attempts INTEGER NOT NULL,
              updated_at TEXT NOT NULL,
              PRIMARY KEY (subscription_id, cycle)
            )
            """
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS budget_spend (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              pool TEXT NOT NULL,
              event_id TEXT NOT NULL,
              spent_at TEXT NOT NULL
            )
            """
        )

    def attempts_for(self, subscription_id: str | None, cycle: str) -> int | None:
        """Attempts recorded against this mandate cycle, or None if we have none."""
        if not subscription_id:
            return None
        row = self._conn.execute(
            "SELECT attempts FROM mandate_attempts "
            "WHERE subscription_id = ? AND cycle = ?",
            (subscription_id, cycle),
        ).fetchone()
        return int(row[0]) if row else None

    def record_attempt(self, subscription_id: str | None, cycle: str) -> int:
        """Count one rail attempt against the mandate. Returns the new total."""
        if not subscription_id:
            return 0
        self._conn.execute(
            """
            INSERT INTO mandate_attempts (subscription_id, cycle, attempts, updated_at)
            VALUES (?, ?, 1, ?)
            ON CONFLICT(subscription_id, cycle) DO UPDATE SET
              attempts = attempts + 1, updated_at = excluded.updated_at
            """,
            (subscription_id, cycle, datetime.now(timezone.utc).isoformat()),
        )
        return self.attempts_for(subscription_id, cycle) or 0

    def spent(self, pool: str, window: timedelta = BUDGET_WINDOW) -> int:
        cutoff = (datetime.now(timezone.utc) - window).isoformat()
        row = self._conn.execute(
            "SELECT COUNT(*) FROM budget_spend WHERE pool = ? AND spent_at >= ?",
            (pool, cutoff),
        ).fetchone()
        return int(row[0])

    def remaining(self, pool: str, limit: int) -> int:
        return max(0, limit - self.spent(pool))

    def record_spend(self, pool: str, event_id: str) -> None:
        self._conn.execute(
            "INSERT INTO budget_spend (pool, event_id, spent_at) VALUES (?, ?, ?)",
            (pool, event_id, datetime.now(timezone.utc).isoformat()),
        )

    def prune(self, older_than: timedelta = BUDGET_WINDOW) -> int:
        """Drop spend rows that have aged out of the window."""
        cutoff = (datetime.now(timezone.utc) - older_than).isoformat()
        return self._conn.execute(
            "DELETE FROM budget_spend WHERE spent_at < ?", (cutoff,)
        ).rowcount

    def close(self) -> None:
        self._conn.close()


class RecentFailures:
    """Rolling window of live failures for fleet outage correlation.

    ``diagnose_batch([event])`` can never trip ``min_events=8``, so without a
    buffer the layer pitched as requiring batch context does not exist in
    production. Each webhook is correlated against the last
    ``OUTAGE_WINDOW_MINUTES`` of traffic stored here, in the same SQLite file
    as the ledger.
    """

    def __init__(self, db_path: str) -> None:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(
            db_path, isolation_level=None, check_same_thread=False
        )
        self._conn.execute("PRAGMA busy_timeout = 5000")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS recent_failures (
              event_id TEXT PRIMARY KEY,
              occurred_at TEXT NOT NULL,
              payload TEXT NOT NULL
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS recent_failures_occurred_at "
            "ON recent_failures (occurred_at)"
        )

    def record(self, event: FailureEvent) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO recent_failures "
            "(event_id, occurred_at, payload) VALUES (?, ?, ?)",
            (
                event.event_id,
                event.occurred_at.isoformat(),
                event.model_dump_json(),
            ),
        )
        self.prune()

    def recent(
        self, window_minutes: int = OUTAGE_WINDOW_MINUTES
    ) -> list[FailureEvent]:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=window_minutes)
        ).isoformat()
        rows = self._conn.execute(
            "SELECT payload FROM recent_failures WHERE occurred_at >= ? "
            "ORDER BY occurred_at",
            (cutoff,),
        ).fetchall()
        return [FailureEvent.model_validate_json(row[0]) for row in rows]

    def prune(self, window_minutes: int = OUTAGE_WINDOW_MINUTES) -> int:
        # Keep a little past the window so a clock skew at the edge does not
        # drop the event that just arrived.
        cutoff = (
            datetime.now(timezone.utc) - timedelta(minutes=window_minutes * 2)
        ).isoformat()
        return self._conn.execute(
            "DELETE FROM recent_failures WHERE occurred_at < ?", (cutoff,)
        ).rowcount

    def close(self) -> None:
        self._conn.close()


def reset_live_state() -> None:
    """Drop cached handles and the feed. For tests only.

    Does not delete the dedupe, budget, or recent-failure tables: surviving a
    restart is the point, and a test that wants a clean slate should point
    LIVE_LEDGER_PATH at a new file.
    """
    global _ledger, _seen, _budget, _recent
    _live_feed.clear()
    _ledger = None
    if _seen is not None:
        _seen.close()
    _seen = None
    if _budget is not None:
        _budget.close()
    _budget = None
    if _recent is not None:
        _recent.close()
    _recent = None


def _ledger_instance() -> Ledger:
    global _ledger
    if _ledger is None:
        Path(LIVE_LEDGER_PATH).parent.mkdir(parents=True, exist_ok=True)
        _ledger = Ledger(LIVE_LEDGER_PATH)
    return _ledger


def _seen_events() -> SeenEvents:
    global _seen
    if _seen is None:
        _seen = SeenEvents(LIVE_LEDGER_PATH)
    return _seen


def _live_budget() -> LiveBudget:
    global _budget
    if _budget is None:
        _budget = LiveBudget(LIVE_LEDGER_PATH)
    return _budget


def _recent_failures() -> RecentFailures:
    global _recent
    if _recent is None:
        _recent = RecentFailures(LIVE_LEDGER_PATH)
    return _recent


def verify_webhook_signature(raw_body: bytes, signature: str, secret: str) -> None:
    """Verify ``raw_body`` against Razorpay's HMAC.

    The SDK's ``verify_webhook_signature`` accepts ``str`` only — passing
    ``bytes`` raises inside the library. Decoding UTF-8 is equivalent to
    signing the raw bytes for JSON payloads; what must never happen is
    ``json.dumps(json.loads(raw_body))``.
    """
    Utility().verify_webhook_signature(raw_body.decode("utf-8"), signature, secret)


def _entity(payload: dict[str, Any], key: str) -> dict[str, Any] | None:
    block = payload.get(key)
    if not isinstance(block, dict):
        return None
    entity = block.get("entity")
    return entity if isinstance(entity, dict) else None


def _parse_timestamp(value: Any) -> datetime:
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    if isinstance(value, str):
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _map_method(raw: str | None, provenance: dict[str, str]) -> str:
    if not raw:
        provenance["method"] = "missing in payload; default upi_autopay"
        return "upi_autopay"
    mapped = METHOD_MAP.get(raw.lower(), raw)
    if mapped != raw:
        provenance["method"] = f"mapped Razorpay {raw!r} → {mapped!r}"
    return mapped


def _map_issuer(payment: dict[str, Any], provenance: dict[str, str]) -> str:
    """Resolve issuing-bank identity for outage correlation.

    Order: IFSC (vendored bank-code map) → unambiguous bank VPA handle →
    PSP-tagged VPA handle → unknown. Each path records the raw input in
    provenance. An unknown IFSC prefix falls through rather than raising.
    """
    prior: str | None = None
    raw_ifsc = _ifsc_from_payment(payment)
    if raw_ifsc is not None:
        resolved = _resolve_ifsc(raw_ifsc)
        if resolved is not None:
            code, name = resolved
            provenance["issuer"] = (
                f"IFSC {raw_ifsc!r} → bank code {code} ({name}) "
                f"via vendored ifsc-banknames.json"
            )
            return code
        prior = (
            f"IFSC {raw_ifsc!r} has unknown bank code "
            f"{raw_ifsc.strip().upper()[:4]!r}; falling through"
        )

    def note(message: str) -> None:
        provenance["issuer"] = f"{prior}; {message}" if prior else message

    vpa = payment.get("vpa")
    if isinstance(vpa, str) and "@" in vpa:
        handle = vpa.split("@", 1)[1].strip().lower()
        if handle in BANK_VPA_HANDLES:
            code = BANK_VPA_HANDLES[handle]
            name = bank_name(code) or code
            note(
                f"VPA handle {handle!r} (from {vpa!r}) maps unambiguously to "
                f"bank code {code} ({name})"
            )
            return code
        # A PSP handle names the payer's app (PhonePe, Paytm, …), not the
        # issuing bank. Storing it as psp:… keeps the handle visible for the
        # LLM while keeping detect_outages from correlating on it.
        tagged = f"{PSP_ISSUER_PREFIX}{handle}"
        note(
            f"VPA handle {handle!r} (from {vpa!r}) is a PSP, not an issuing bank; "
            f"stored as {tagged!r} and excluded from bank-outage correlation"
        )
        return tagged

    note(
        f"no IFSC or VPA on payment entity; default {DEFAULT_ISSUER!r} — "
        "excluded from bank-outage correlation"
    )
    return DEFAULT_ISSUER


def failure_event_from_webhook(
    event_type: str, body: dict[str, Any]
) -> tuple[FailureEvent, dict[str, str]]:
    """Map a Razorpay webhook body onto :class:`FailureEvent`.

    ``true_cause`` is always ``None`` — live traffic has no grader label.
    Discriminating features absent from the payload carry documented defaults
  in the returned provenance dict.
    """
    provenance: dict[str, str] = {"true_cause": "live webhook; no ground-truth label"}
    payload = body.get("payload") or {}
    payment = _entity(payload, "payment")
    subscription = _entity(payload, "subscription")

    if payment is None and subscription is not None:
        payment = {
            "id": f"pending_{subscription.get('id', 'sub')}",
            "amount": subscription.get("plan_amount") or subscription.get("amount") or 0,
            "method": subscription.get("payment_method") or "upi",
            "error_code": "BAD_REQUEST_ERROR",
            "error_source": "issuer_bank",
            "error_step": "payment_authorization",
            "error_reason": "payment_failed",
            "created_at": body.get("created_at") or subscription.get("created_at"),
        }
        provenance["payment"] = (
            f"{event_type} carried subscription only; synthesised payment fields "
            "for pipeline compatibility"
        )

    if payment is None:
        raise ValueError(f"{event_type} payload has no payment or subscription entity")

    occurred_at = _parse_timestamp(payment.get("created_at") or body.get("created_at"))
    method = _map_method(payment.get("method"), provenance)
    issuer = _map_issuer(payment, provenance)

    retries = payment.get("notes", {}) if isinstance(payment.get("notes"), dict) else {}
    retries_used = retries.get("retries_used", DEFAULT_RETRIES_USED)
    if retries_used == DEFAULT_RETRIES_USED:
        provenance["retries_used"] = (
            f"not in payment.notes; default {DEFAULT_RETRIES_USED}"
        )

    provenance["days_since_mandate_created"] = (
        f"not in webhook payload; default {DEFAULT_DAYS_SINCE_MANDATE}"
    )
    provenance["issuer_recent_failure_rate"] = (
        f"not in webhook payload; default {DEFAULT_ISSUER_FAILURE_RATE}"
    )
    provenance["amount_vs_customer_avg"] = (
        f"not in webhook payload; default {DEFAULT_AMOUNT_VS_AVG}"
    )

    subscription_id = None
    if subscription and subscription.get("id"):
        subscription_id = subscription["id"]
    elif isinstance(payment.get("subscription_id"), str):
        subscription_id = payment["subscription_id"]

    event = FailureEvent(
        event_id=payment["id"],
        payment_id=payment["id"],
        subscription_id=subscription_id,
        amount_paise=int(payment.get("amount") or 0),
        method=method,
        issuer=issuer,
        error_code=str(payment.get("error_code") or "BAD_REQUEST_ERROR"),
        error_source=str(payment.get("error_source") or "issuer_bank"),
        error_step=str(payment.get("error_step") or "payment_authorization"),
        error_reason=str(payment.get("error_reason") or "payment_failed"),
        occurred_at=occurred_at,
        retries_used=int(retries_used),
        days_since_mandate_created=DEFAULT_DAYS_SINCE_MANDATE,
        # IST, not UTC: allocate's PAYDAY_DAYS models Indian salary credits, so
        # the day that matters is the local one. A 06:00 IST failure on the 6th
        # is still the 5th in UTC, which would swap PAYDAY_UPLIFT for
        # MID_MONTH_PENALTY — a 0.45 swing on INSUFFICIENT_FUNDS.
        day_of_month=to_ist(occurred_at).day,
        issuer_recent_failure_rate=DEFAULT_ISSUER_FAILURE_RATE,
        amount_vs_customer_avg=DEFAULT_AMOUNT_VS_AVG,
        true_cause=None,
    )
    return event, provenance


def write_live_trace(
    ledger: Ledger,
    event: FailureEvent,
    provenance: dict[str, str],
    razorpay_event_type: str,
    diagnosis: Diagnosis,
    action: ProposedAction,
    decision: GateDecision,
    score: float,
    attempts_source: str = SOURCE_PAYLOAD,
) -> None:
    """Four ledger entries per event, matching the batch orchestrator."""
    ledger.append(
        event.event_id,
        ENTRY_INGESTED,
        {
            "source": "razorpay_webhook",
            "razorpay_event_type": razorpay_event_type,
            "payment_id": event.payment_id,
            "subscription_id": event.subscription_id,
            "amount_paise": event.amount_paise,
            "method": event.method,
            "issuer": event.issuer,
            "error_code": event.error_code,
            "error_reason": event.error_reason,
            "retries_used": event.retries_used,
            # Which count the NPCI cap was enforced against. "payload" means a
            # merchant-controlled field; "ledger" means our own attempt record.
            "retries_used_source": attempts_source,
            "occurred_at": event.occurred_at,
            "provenance": provenance,
        },
    )
    ledger.append(
        event.event_id,
        ENTRY_DIAGNOSED,
        {
            "cause": diagnosis.cause.value,
            "method": diagnosis.method,
            "confidence": diagnosis.confidence,
            "evidence": diagnosis.evidence,
        },
    )
    ledger.append(
        event.event_id,
        ENTRY_PROPOSED,
        {
            "action": action.action.value,
            "index_score": score,
            "allocation_pass": 1,
            "rationale": action.rationale,
            "expected_recovery_paise": action.expected_recovery_paise,
            "scheduled_for": action.scheduled_for,
        },
    )
    ledger.append(
        event.event_id,
        ENTRY_GATED,
        {
            "approved": decision.approved,
            "blocked_by": decision.blocked_by,
            "reason": decision.reason,
            "rule_ids_passed": decision.rule_ids_passed,
        },
    )


def process_failure_event(
    event: FailureEvent,
    provenance: dict[str, str],
    razorpay_event_id: str,
    razorpay_event_type: str,
) -> LiveRecord:
    """Run one event through diagnose → allocate → govern and append the ledger."""
    received_at = datetime.now(timezone.utc)
    try:
        budget = _live_budget()
        cycle = mandate_cycle(event)

        # Our own count wins over payment.notes, which the merchant controls.
        # Falling back to the payload is only for a mandate we have never seen;
        # from the first attempt we place, we are the authority.
        claimed = event.retries_used
        recorded = budget.attempts_for(event.subscription_id, cycle)
        attempts_source = SOURCE_PAYLOAD if recorded is None else SOURCE_LEDGER
        if recorded is None:
            provenance["retries_used"] = (
                f"{claimed} from payment.notes (merchant-controlled); no attempt "
                f"recorded for {event.subscription_id} in cycle {cycle} yet"
            )
        else:
            event = event.model_copy(update={"retries_used": recorded})
            provenance["retries_used"] = (
                f"{recorded} from our own attempt record for cycle {cycle}; "
                f"payment.notes claimed {claimed}"
            )
            if recorded != claimed:
                log.info(
                    "mandate %s cycle %s: enforcing cap against recorded %d, "
                    "not payload %d",
                    event.subscription_id,
                    cycle,
                    recorded,
                    claimed,
                )

        # Correlate against recent live traffic, not against this event alone —
        # min_events is 8 and a single webhook can never trip the fleet layer.
        buffer = _recent_failures()
        buffer.record(event)
        diagnoses, _ = diagnose_batch([event], correlation_events=buffer.recent())
        diagnosis = diagnoses[0]
        # Remaining budget, not the full allowance: see allocate's module
        # docstring on why one event against a fresh pool decides nothing.
        actions, _ = allocate(
            [event],
            diagnoses,
            budget.remaining(RETRY_POOL, DEFAULT_RETRY_BUDGET),
            budget.remaining(CONTACT_POOL, DEFAULT_CONTACT_BUDGET),
        )
        action = actions[0]
        decision = govern.evaluate(action, gate_context_for(event, action))
        score, _ = index_score(event, diagnosis)

        # Spend is recorded only for an action that cleared the gate, matching
        # the batch path where a blocked action's budget is reclaimed.
        pool = budget_pool(action.action)
        if decision.approved and pool is not None:
            budget.record_spend(pool, event.event_id)
            if action.action in RAIL_ACTIONS:
                budget.record_attempt(event.subscription_id, cycle)

        write_live_trace(
            _ledger_instance(),
            event,
            provenance,
            razorpay_event_type,
            diagnosis,
            action,
            decision,
            score,
            attempts_source,
        )
        record = LiveRecord(
            razorpay_event_id=razorpay_event_id,
            razorpay_event_type=razorpay_event_type,
            event_id=event.event_id,
            payment_id=event.payment_id,
            amount_paise=event.amount_paise,
            method=event.method,
            received_at=received_at,
            provenance=provenance,
            diagnosis=diagnosis,
            action=action,
            gate=decision,
        )
    except Exception as exc:
        log.exception("pipeline failed for %s", event.event_id)
        _live_feed.appendleft(
            LiveRecord(
                razorpay_event_id=razorpay_event_id,
                razorpay_event_type=razorpay_event_type,
                event_id=event.event_id,
                payment_id=event.payment_id,
                amount_paise=event.amount_paise,
                method=event.method,
                received_at=received_at,
                provenance=provenance,
                pipeline_error=str(exc),
            )
        )
        # Re-raised so the caller can decide whether Razorpay should redeliver.
        # Swallowing here returned 200 on a transient fault, which told Razorpay
        # the event was handled and dropped it for good.
        raise
    _live_feed.appendleft(record)
    return record


def _should_process(event_type: str) -> str | None:
    """Return 'failure', 'ack', or 'ignore'."""
    if event_type in FAILURE_EVENTS:
        return "failure"
    if event_type in ACK_LOG_EVENTS or event_type.startswith(DOWNTIME_PREFIX):
        return "ack"
    return "ignore"


app = FastAPI(title="nakad live webhook")


@app.post("/webhook")
async def webhook(
    request: Request,
    x_razorpay_signature: str = Header(alias="X-Razorpay-Signature"),
    x_razorpay_event_id: str = Header(alias="X-Razorpay-Event-Id"),
) -> dict[str, str]:
    raw_body = await request.body()

    secret = os.environ.get(WEBHOOK_SECRET_ENV)
    if not secret:
        log.error("%s is not set", WEBHOOK_SECRET_ENV)
        raise HTTPException(status_code=500, detail="webhook secret not configured")

    try:
        verify_webhook_signature(raw_body, x_razorpay_signature, secret)
    except SignatureVerificationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    seen = _seen_events()
    if x_razorpay_event_id in seen:
        log.info(
            "duplicate webhook %s (already %s) — acknowledging without reprocessing",
            x_razorpay_event_id,
            seen.outcome_of(x_razorpay_event_id),
        )
        return {"status": "duplicate"}

    # Marked seen only on the way out. Marking on the way in meant a transient
    # fault dropped the event twice over: Razorpay was told 200 and stopped
    # retrying, and any redelivery that did arrive was rejected as a duplicate.
    event_type = ""
    try:
        body = json.loads(raw_body)
        event_type = str(body.get("event") or "")
        route = _should_process(event_type)

        if route == "ack":
            outcome, result = OUTCOME_ACKNOWLEDGED, {
                "status": "acknowledged",
                "event": event_type,
            }
        elif route == "ignore":
            outcome, result = OUTCOME_ACKNOWLEDGED, {
                "status": "ignored",
                "event": event_type,
            }
        else:
            event, provenance = failure_event_from_webhook(event_type, body)
            process_failure_event(event, provenance, x_razorpay_event_id, event_type)
            outcome, result = OUTCOME_PROCESSED, {
                "status": "processed",
                "event": event_type,
                "event_id": event.event_id,
            }
    except UNPROCESSABLE_ERRORS as exc:
        seen.mark(x_razorpay_event_id, OUTCOME_UNPROCESSABLE)
        log.warning(
            "unprocessable webhook %s (%s): %s — marked seen and acknowledged, "
            "a redelivery of the same bytes would fail identically",
            x_razorpay_event_id,
            event_type or "(unparsed)",
            exc,
        )
        return {"status": "accepted", "error": str(exc)}
    except Exception as exc:
        log.exception(
            "transient failure on webhook %s (%s) — left unseen and returning 500 "
            "so Razorpay redelivers",
            x_razorpay_event_id,
            event_type or "(unparsed)",
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    seen.mark(x_razorpay_event_id, outcome)
    log.info(
        "%s webhook %s (%s) — marked seen", outcome, x_razorpay_event_id, event_type
    )
    return result


@app.get("/live")
def live() -> list[dict[str, Any]]:
    """Last 20 failure webhooks that entered the pipeline."""
    return [record.model_dump(mode="json") for record in list(_live_feed)]
