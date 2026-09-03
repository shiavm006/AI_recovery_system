from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RootCause(str, Enum):
    INSUFFICIENT_FUNDS = "INSUFFICIENT_FUNDS"
    ISSUER_DOWNTIME = "ISSUER_DOWNTIME"
    DEAD_MANDATE = "DEAD_MANDATE"
    NETWORK_TIMEOUT = "NETWORK_TIMEOUT"
    HARD_DECLINE = "HARD_DECLINE"
    RISK_BLOCK = "RISK_BLOCK"
    # Not a cause: the absence of one. Recorded when the pipeline could not
    # obtain a diagnosis, so a degraded run shows up as its own category
    # instead of being absorbed into whichever real cause was most plausible.
    UNKNOWN = "UNKNOWN"


class ActionType(str, Enum):
    RETRY = "RETRY"
    RAIL_SWITCH = "RAIL_SWITCH"
    MANDATE_REPRESENT = "MANDATE_REPRESENT"
    PAYMENT_LINK = "PAYMENT_LINK"
    NUDGE = "NUDGE"
    SUPPRESS = "SUPPRESS"


class FailureEvent(BaseModel):
    """A single payment failure observed from Razorpay or synthetic data.

    ``true_cause`` is only populated in synthetic data and is used for grading;
    the pipeline must never read it.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    payment_id: str
    subscription_id: str | None
    amount_paise: int
    method: str
    issuer: str
    error_code: str
    error_source: str
    error_step: str
    error_reason: str
    occurred_at: datetime
    retries_used: int = 0
    days_since_mandate_created: int = Field(ge=0)
    day_of_month: int = Field(ge=1, le=31)
    issuer_recent_failure_rate: float = Field(ge=0.0, le=1.0)
    amount_vs_customer_avg: float = Field(ge=0.0)
    true_cause: RootCause | None = None

    @field_validator("amount_paise")
    @classmethod
    def amount_must_be_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("amount_paise must be non-negative")
        return value


class Diagnosis(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    cause: RootCause
    confidence: float
    evidence: list[str]
    method: Literal["rule", "llm", "fleet"]

    @field_validator("confidence")
    @classmethod
    def confidence_must_be_unit_interval(cls, value: float) -> float:
        if not 0.0 <= value <= 1.0:
            raise ValueError("confidence must be between 0 and 1 inclusive")
        return value


class ProposedAction(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    action: ActionType
    rationale: str
    expected_recovery_paise: int
    scheduled_for: datetime | None = None


class GateDecision(BaseModel):
    model_config = ConfigDict(frozen=True)

    event_id: str
    approved: bool
    rule_ids_passed: list[str]
    blocked_by: str | None = None
    reason: str


class BatchResult(BaseModel):
    """What one policy did to one batch, and what it recovered."""

    model_config = ConfigDict(frozen=True)

    policy: Literal["agent", "control"]
    events_processed: int
    actions_by_type: dict[str, int]
    blocked_by_rule: dict[str, int]
    gross_recovered_paise: int
    # Split by payment method because chargeback exposure is card-only: UPI
    # Autopay and NACH disputes run through different mechanisms entirely.
    recovered_by_method: dict[str, int]
    # Budget spent means budget actually placed: an action the gate refused is
    # reclaimed and re-offered, so this never counts an attempt that never ran.
    retry_budget_spent: int
    contact_budget_spent: int
    allocation_passes: int
    suppressed_for_budget: int
    suppressed_for_low_value: int
    suppressed_for_low_confidence: int
    suppressed_for_no_headroom: int
    suppressed_for_no_window: int
    # Events the pipeline could not diagnose at all. Reported on its own so a
    # degraded run is visible rather than buried in the cause distribution.
    unknown_diagnoses: int
    suppressed_for_no_diagnosis: int
    ledger_path: str
    # Provider, model and diagnosis-method mix behind these numbers. A run with
    # the LLM off still returns a plausible gross_recovered_paise, so the state
    # that produced it travels with the result rather than beside it.
    run_config: dict


class LedgerEntry(BaseModel):
    model_config = ConfigDict(frozen=True)

    seq: int
    timestamp: datetime
    event_id: str
    entry_type: str
    payload: dict
    prev_hash: str
    entry_hash: str
