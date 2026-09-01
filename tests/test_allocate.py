from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ActionType, Diagnosis, FailureEvent, RootCause
from pipeline.allocate import (
    CONTACT_POOL,
    DEFAULT_CONTACT_BUDGET,
    DEFAULT_RETRY_BUDGET,
    MIN_ACTIONABLE_CONFIDENCE,
    RETRY_POOL,
    SUPPRESSED_FOR_BUDGET,
    SUPPRESSED_FOR_LOW_CONFIDENCE,
    SUPPRESSED_FOR_LOW_VALUE,
    allocate,
    index_score,
)
from pipeline.govern import CONTACT_ACTIONS, RAIL_ACTIONS

PAYDAY = 28
MID_MONTH = 15


def _event(event_id: str, amount_paise: int, day_of_month: int) -> FailureEvent:
    return FailureEvent(
        event_id=event_id,
        payment_id=f"pay_{event_id}",
        subscription_id=f"sub_{event_id}",
        amount_paise=amount_paise,
        method="upi_autopay",
        issuer="hdfc",
        error_code="BAD_REQUEST_ERROR",
        error_source="issuer_bank",
        error_step="payment_authorization",
        error_reason="payment_failed",
        occurred_at=datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc),
        days_since_mandate_created=40,
        day_of_month=day_of_month,
        issuer_recent_failure_rate=0.05,
        amount_vs_customer_avg=1.0,
    )


def _diagnosis(
    event_id: str,
    cause: RootCause,
    confidence: float = 0.9,
    method: str = "rule",
) -> Diagnosis:
    return Diagnosis(
        event_id=event_id,
        cause=cause,
        confidence=confidence,
        evidence=["test"],
        method=method,
    )


def test_hard_decline_suppresses_at_any_amount():
    for amount in (100, 99_900, 50_000_000):
        event = _event("e1", amount, PAYDAY)
        diagnosis = _diagnosis("e1", RootCause.HARD_DECLINE)
        actions, stats = allocate(
            [event], [diagnosis], retry_budget=10, contact_budget=10
        )
        assert actions[0].action is ActionType.SUPPRESS
        assert actions[0].rationale.startswith(SUPPRESSED_FOR_LOW_VALUE)
        assert actions[0].expected_recovery_paise == 0
        assert stats["retry_budget_spent"] == 0
        assert stats["contact_budget_spent"] == 0


def test_high_value_payday_outranks_low_value_mid_month():
    rich = _event("rich", 99_900, PAYDAY)
    poor = _event("poor", 14_900, MID_MONTH)
    diagnoses = [
        _diagnosis("rich", RootCause.INSUFFICIENT_FUNDS),
        _diagnosis("poor", RootCause.INSUFFICIENT_FUNDS),
    ]

    rich_score, _ = index_score(rich, diagnoses[0])
    poor_score, _ = index_score(poor, diagnoses[1])
    assert rich_score > poor_score

    actions, _ = allocate([poor, rich], diagnoses, retry_budget=1, contact_budget=1)
    assert [action.event_id for action in actions] == ["rich", "poor"]
    assert actions[0].action is ActionType.RETRY


def test_retry_budget_is_never_exceeded():
    events = [_event(f"e{i}", 99_900, PAYDAY) for i in range(10)]
    diagnoses = [_diagnosis(f"e{i}", RootCause.INSUFFICIENT_FUNDS) for i in range(10)]

    for budget in (0, 1, 3, 10, 25):
        actions, stats = allocate(
            events, diagnoses, retry_budget=budget, contact_budget=0
        )
        attempts = [a for a in actions if a.action in RAIL_ACTIONS]
        assert len(attempts) <= budget
        assert stats["retry_budget_spent"] == len(attempts)
        assert stats["retry_budget_spent"] <= budget
        assert stats["suppressed_for_budget"] == len(events) - len(attempts)


def test_contact_budget_is_never_exceeded():
    # Mid-month balance failures map to PAYMENT_LINK, so they draw on the
    # contact pool and leave the retry pool untouched.
    events = [_event(f"e{i}", 99_900, MID_MONTH) for i in range(10)]
    diagnoses = [_diagnosis(f"e{i}", RootCause.INSUFFICIENT_FUNDS) for i in range(10)]

    for budget in (0, 1, 4, 10, 25):
        actions, stats = allocate(
            events, diagnoses, retry_budget=99, contact_budget=budget
        )
        contacts = [a for a in actions if a.action in CONTACT_ACTIONS]
        assert len(contacts) <= budget
        assert stats["contact_budget_spent"] == len(contacts)
        assert stats["retry_budget_spent"] == 0
        assert stats["suppressed_for_budget"] == len(events) - len(contacts)


def test_each_budget_cuts_independently():
    payday = [_event(f"pay{i}", 99_900, PAYDAY) for i in range(3)]
    mid = [_event(f"mid{i}", 99_900, MID_MONTH) for i in range(3)]
    events = payday + mid
    diagnoses = [
        _diagnosis(event.event_id, RootCause.INSUFFICIENT_FUNDS) for event in events
    ]

    actions, stats = allocate(events, diagnoses, retry_budget=1, contact_budget=2)
    assert stats["retry_budget_spent"] == 1
    assert stats["contact_budget_spent"] == 2
    assert stats["retry_cut_index_score"] is not None
    assert stats["contact_cut_index_score"] is not None

    exhausted = [
        a.rationale for a in actions if a.rationale.startswith(SUPPRESSED_FOR_BUDGET)
    ]
    assert sum(RETRY_POOL in reason for reason in exhausted) == 2
    assert sum(CONTACT_POOL in reason for reason in exhausted) == 1


def test_every_event_appears_exactly_once():
    causes = list(RootCause)
    events = [_event(f"e{i}", 10_000 * (i + 1), MID_MONTH) for i in range(len(causes))]
    diagnoses = [_diagnosis(f"e{i}", cause) for i, cause in enumerate(causes)]

    actions, stats = allocate(events, diagnoses, retry_budget=1, contact_budget=1)
    assert len(actions) == len(events)
    assert sorted(a.event_id for a in actions) == sorted(e.event_id for e in events)
    assert sum(stats["by_action"].values()) == len(events)


def test_zero_confidence_fallback_never_wins_budget():
    # diagnose.py's fallback fabricates INSUFFICIENT_FUNDS at confidence 0.0,
    # which is otherwise the top-scoring cause. Budget is deliberately ample:
    # the guess must lose on confidence, not on scarcity.
    events = [_event(f"e{i}", 99_900, PAYDAY) for i in range(5)]
    diagnoses = [
        _diagnosis(f"e{i}", RootCause.INSUFFICIENT_FUNDS, confidence=0.0, method="llm")
        for i in range(5)
    ]

    actions, stats = allocate(events, diagnoses, retry_budget=99, contact_budget=99)
    assert stats["retry_budget_spent"] == 0
    assert stats["contact_budget_spent"] == 0
    assert stats["suppressed_for_low_confidence"] == 5
    assert all(a.action is ActionType.SUPPRESS for a in actions)
    assert all(a.rationale.startswith(SUPPRESSED_FOR_LOW_CONFIDENCE) for a in actions)


def test_believed_diagnosis_outranks_guess_and_takes_the_budget():
    events = [_event("sure", 99_900, PAYDAY), _event("guess", 99_900, PAYDAY)]
    diagnoses = [
        _diagnosis("sure", RootCause.INSUFFICIENT_FUNDS, confidence=0.9),
        _diagnosis("guess", RootCause.INSUFFICIENT_FUNDS, confidence=0.0, method="llm"),
    ]

    sure_score, _ = index_score(events[0], diagnoses[0])
    guess_score, _ = index_score(events[1], diagnoses[1])
    assert sure_score > guess_score

    actions, stats = allocate(events, diagnoses, retry_budget=1, contact_budget=0)
    by_id = {action.event_id: action for action in actions}
    assert by_id["sure"].action is ActionType.RETRY
    assert by_id["guess"].rationale.startswith(SUPPRESSED_FOR_LOW_CONFIDENCE)
    assert stats["retry_budget_spent"] == 1


def test_three_suppression_rationales_are_distinguishable():
    events = [
        _event("win", 99_900, PAYDAY),
        _event("lose", 49_900, PAYDAY),
        _event("dead", 99_900, PAYDAY),
        _event("unsure", 99_900, PAYDAY),
    ]
    diagnoses = [
        _diagnosis("win", RootCause.INSUFFICIENT_FUNDS),
        _diagnosis("lose", RootCause.INSUFFICIENT_FUNDS),
        _diagnosis("dead", RootCause.HARD_DECLINE),
        _diagnosis(
            "unsure",
            RootCause.INSUFFICIENT_FUNDS,
            confidence=MIN_ACTIONABLE_CONFIDENCE - 0.01,
            method="llm",
        ),
    ]
    actions, stats = allocate(events, diagnoses, retry_budget=1, contact_budget=0)
    by_id = {action.event_id: action for action in actions}
    prefixes = {
        action.event_id: action.rationale.split(":")[0] for action in actions
    }
    assert by_id["win"].action is ActionType.RETRY
    assert prefixes["lose"] == SUPPRESSED_FOR_BUDGET
    assert prefixes["dead"] == SUPPRESSED_FOR_LOW_VALUE
    assert prefixes["unsure"] == SUPPRESSED_FOR_LOW_CONFIDENCE
    assert len({prefixes["lose"], prefixes["dead"], prefixes["unsure"]}) == 3
    assert stats["suppressed_for_budget"] == 1
    assert stats["suppressed_for_low_value"] == 1
    assert stats["suppressed_for_low_confidence"] == 1
    assert stats["retry_cut_index_score"] == pytest.approx(
        index_score(events[0], diagnoses[0])[0]
    )


def test_confidence_floor_is_inclusive_at_the_boundary():
    at_floor = _event("at", 99_900, PAYDAY)
    diagnosis = _diagnosis(
        "at", RootCause.INSUFFICIENT_FUNDS, confidence=MIN_ACTIONABLE_CONFIDENCE
    )
    actions, stats = allocate([at_floor], [diagnosis], retry_budget=1, contact_budget=0)
    assert actions[0].action is ActionType.RETRY
    assert stats["suppressed_for_low_confidence"] == 0


def test_omitted_budgets_fall_back_to_the_house_throttles():
    wanted = DEFAULT_RETRY_BUDGET + DEFAULT_CONTACT_BUDGET + 10
    payday = [_event(f"pay{i}", 99_900, PAYDAY) for i in range(wanted)]
    mid = [_event(f"mid{i}", 99_900, MID_MONTH) for i in range(wanted)]
    events = payday + mid
    diagnoses = [
        _diagnosis(event.event_id, RootCause.INSUFFICIENT_FUNDS) for event in events
    ]

    _, stats = allocate(events, diagnoses)
    assert stats["retry_budget_spent"] == DEFAULT_RETRY_BUDGET
    assert stats["contact_budget_spent"] == DEFAULT_CONTACT_BUDGET


def test_breakdown_contributions_sum_to_score():
    for cause in RootCause:
        for day in (PAYDAY, MID_MONTH):
            for confidence in (0.0, 0.5, 0.9, 1.0):
                event = _event("e1", 99_900, day)
                diagnosis = _diagnosis("e1", cause, confidence=confidence)
                score, breakdown = index_score(event, diagnosis)
                assert sum(breakdown["contributions"].values()) == pytest.approx(score)
                assert breakdown["score"] == pytest.approx(score)
