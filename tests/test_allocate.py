from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pipeline.allocate as allocate_module
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
    SUPPRESSED_FOR_NO_DIAGNOSIS,
    SUPPRESSED_FOR_NO_HEADROOM,
    SUPPRESSED_FOR_NO_WINDOW,
    allocate,
    index_score,
    intervention_for,
    next_permitted_time,
)
from pipeline.govern import (
    CONTACT_ACTIONS,
    IST,
    NPCI_RETRY_CAP,
    RAIL_ACTIONS,
    GateContext,
    in_contact_hours,
    in_peak_rail_window,
)

PAYDAY = 28
MID_MONTH = 15


def _event(
    event_id: str,
    amount_paise: int,
    day_of_month: int,
    retries_used: int = 0,
    occurred_at: datetime | None = None,
) -> FailureEvent:
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
        occurred_at=occurred_at or datetime(2026, 9, 1, 9, 0, tzinfo=timezone.utc),
        retries_used=retries_used,
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


def test_no_rail_action_is_proposed_without_npci_headroom():
    # govern r01 would refuse these, and a refused proposal still occupies a
    # slot in the ranking, so the allocator must not offer one at all.
    for cause in (RootCause.NETWORK_TIMEOUT, RootCause.ISSUER_DOWNTIME, RootCause.DEAD_MANDATE):
        event = _event("capped", 99_900, PAYDAY, retries_used=NPCI_RETRY_CAP)
        diagnosis = _diagnosis("capped", cause)
        assert intervention_for(event, diagnosis) is ActionType.SUPPRESS

        actions, stats = allocate([event], [diagnosis], retry_budget=10, contact_budget=10)
        assert actions[0].action is ActionType.SUPPRESS
        assert actions[0].rationale.startswith(SUPPRESSED_FOR_NO_HEADROOM)
        assert stats["retry_budget_spent"] == 0
        assert stats["suppressed_for_no_headroom"] == 1


def test_headroom_screen_leaves_contact_actions_alone():
    # A spent mandate blocks the rails, not the customer's ability to pay a
    # link, so the contact path must survive the screen.
    event = _event("capped", 99_900, MID_MONTH, retries_used=NPCI_RETRY_CAP)
    diagnosis = _diagnosis("capped", RootCause.INSUFFICIENT_FUNDS)
    assert intervention_for(event, diagnosis) is ActionType.PAYMENT_LINK

    actions, stats = allocate([event], [diagnosis], retry_budget=0, contact_budget=1)
    assert actions[0].action is ActionType.PAYMENT_LINK
    assert stats["contact_budget_spent"] == 1


def test_headroom_boundary_keeps_the_last_attempt():
    below = _event("below", 99_900, PAYDAY, retries_used=NPCI_RETRY_CAP - 1)
    diagnosis = _diagnosis("below", RootCause.NETWORK_TIMEOUT)
    assert intervention_for(below, diagnosis) is ActionType.RETRY


def _ctx(retries_used: int = 0) -> GateContext:
    return GateContext(
        now=datetime(2026, 9, 1, 12, 0, tzinfo=IST),
        retries_used=retries_used,
        contacts_this_cycle=0,
        last_notice_sent_at=None,
        stop_requested=False,
        promise_to_pay=False,
        dispute_open=False,
        consent_logged=True,
        channel=None,
    )


def _at(hour: int, minute: int = 0, day: int = 1) -> datetime:
    return datetime(2026, 9, day, hour, minute, tzinfo=IST)


@pytest.mark.parametrize(
    ("earliest", "expected"),
    [
        (_at(11, 0), _at(13, 0)),  # midday freeze lifts at 13:00
        (_at(10, 0), _at(13, 0)),  # first minute of the freeze
        (_at(12, 59), _at(13, 0)),
        (_at(18, 0), _at(21, 31)),  # evening freeze lifts a minute after 21:30
        (_at(21, 30), _at(21, 31)),
    ],
)
def test_a_frozen_rail_is_booked_for_the_moment_the_freeze_lifts(earliest, expected):
    assert next_permitted_time(ActionType.RETRY, earliest, _ctx()) == expected


@pytest.mark.parametrize("earliest", [_at(2, 0), _at(9, 59), _at(13, 0), _at(21, 31)])
def test_a_rail_already_in_the_clear_is_not_moved(earliest):
    assert next_permitted_time(ActionType.RETRY, earliest, _ctx()) == earliest


@pytest.mark.parametrize(
    ("earliest", "expected"),
    [
        (_at(6, 0), _at(8, 0)),  # before the day opens
        (_at(7, 59), _at(8, 0)),
        (_at(19, 0), _at(8, 0, day=2)),  # first minute past close, so tomorrow
        (_at(23, 59), _at(8, 0, day=2)),
        (_at(0, 30), _at(8, 0)),
    ],
)
def test_contact_outside_hours_waits_for_the_next_opening(earliest, expected):
    assert next_permitted_time(ActionType.PAYMENT_LINK, earliest, _ctx()) == expected


def test_contact_is_allowed_during_the_rail_freeze():
    # The two windows are independent: the peak freeze is about the rails, and
    # a payment link at noon is fine.
    noon = _at(12, 0)
    assert next_permitted_time(ActionType.PAYMENT_LINK, noon, _ctx()) == noon


def test_no_hour_helps_a_mandate_at_the_cap():
    # r01 refuses these at every moment, so there is no time to find and the
    # caller must suppress rather than schedule.
    assert next_permitted_time(ActionType.RETRY, _at(14, 0), _ctx(NPCI_RETRY_CAP)) is None
    # The cap is a rail limit, so contact is still schedulable.
    assert next_permitted_time(
        ActionType.PAYMENT_LINK, _at(14, 0), _ctx(NPCI_RETRY_CAP)
    ) == _at(14, 0)


def test_search_gives_up_at_the_horizon(monkeypatch):
    monkeypatch.setattr(allocate_module, "SCHEDULING_HORIZON_HOURS", 0)
    assert next_permitted_time(ActionType.RETRY, _at(11, 0), _ctx()) is None


def test_unschedulable_action_suppresses_instead_of_being_proposed(monkeypatch):
    monkeypatch.setattr(allocate_module, "SCHEDULING_HORIZON_HOURS", 0)
    # A payday balance failure retries at T+24h, landing back inside the
    # midday freeze, and with no horizon there is nowhere to move it to.
    event = _event("boxed", 99_900, PAYDAY, occurred_at=_at(11, 0))
    diagnosis = _diagnosis("boxed", RootCause.INSUFFICIENT_FUNDS)

    actions, stats = allocate([event], [diagnosis], retry_budget=5, contact_budget=5)
    assert actions[0].action is ActionType.SUPPRESS
    assert actions[0].rationale.startswith(SUPPRESSED_FOR_NO_WINDOW)
    assert stats["retry_budget_spent"] == 0
    assert stats["suppressed_for_no_window"] == 1


def test_no_allocated_action_is_ever_booked_into_a_forbidden_window():
    # The property that matters: whatever the allocator emits, the gate's
    # clock rules will not refuse it.
    events, diagnoses = [], []
    for hour in range(24):
        for index, cause in enumerate(RootCause):
            event_id = f"h{hour}c{index}"
            events.append(
                _event(
                    event_id,
                    99_900,
                    PAYDAY if index % 2 else MID_MONTH,
                    occurred_at=_at(hour, 17),
                )
            )
            diagnoses.append(_diagnosis(event_id, cause))

    actions, _ = allocate(events, diagnoses, retry_budget=999, contact_budget=999)
    scheduled = [a for a in actions if a.action is not ActionType.SUPPRESS]
    assert scheduled, "expected the allocator to schedule something"
    for action in scheduled:
        assert action.scheduled_for is not None
        if action.action in RAIL_ACTIONS:
            assert not in_peak_rail_window(action.scheduled_for)
        if action.action in CONTACT_ACTIONS:
            assert in_contact_hours(action.scheduled_for)


def test_unknown_always_suppresses_whatever_the_amount_or_budget():
    for amount in (100, 99_900, 50_000_000):
        event = _event("undiagnosed", amount, PAYDAY)
        diagnosis = _diagnosis("undiagnosed", RootCause.UNKNOWN, confidence=0.0)
        assert intervention_for(event, diagnosis) is ActionType.SUPPRESS

        actions, stats = allocate(
            [event], [diagnosis], retry_budget=99, contact_budget=99
        )
        assert actions[0].action is ActionType.SUPPRESS
        assert actions[0].rationale.startswith(SUPPRESSED_FOR_NO_DIAGNOSIS)
        assert actions[0].expected_recovery_paise == 0
        assert stats["retry_budget_spent"] == 0
        assert stats["contact_budget_spent"] == 0
        assert stats["suppressed_for_no_diagnosis"] == 1


def test_unknown_suppresses_even_when_the_confidence_floor_would_not_catch_it():
    # The belt-and-braces claim: every UNKNOWN this pipeline emits carries
    # confidence 0.0, so the floor already excludes it. The cause mapping is
    # what covers an UNKNOWN that arrives believed, which the floor would pass.
    event = _event("confident-nonsense", 99_900, PAYDAY)
    diagnosis = _diagnosis("confident-nonsense", RootCause.UNKNOWN, confidence=1.0)
    assert diagnosis.confidence >= MIN_ACTIONABLE_CONFIDENCE

    actions, stats = allocate([event], [diagnosis], retry_budget=99, contact_budget=99)
    assert actions[0].action is ActionType.SUPPRESS
    assert actions[0].rationale.startswith(SUPPRESSED_FOR_NO_DIAGNOSIS)
    assert stats["suppressed_for_low_confidence"] == 0
    assert stats["retry_budget_spent"] == 0


def test_unknown_scores_zero_and_ranks_below_every_real_cause():
    event = _event("e1", 99_900, PAYDAY)
    unknown_score, breakdown = index_score(
        event, _diagnosis("e1", RootCause.UNKNOWN, confidence=1.0)
    )
    assert unknown_score == 0.0
    assert breakdown["expected_recovery_paise"] == 0
    for cause in RootCause:
        if cause is RootCause.UNKNOWN:
            continue
        real, _ = index_score(event, _diagnosis("e1", cause, confidence=1.0))
        assert real >= unknown_score


def test_undiagnosed_events_never_displace_diagnosed_ones():
    # The failure the old fallback caused: fabricated diagnoses competing for,
    # and winning, a budget that should have gone to real work.
    real = [_event(f"real{i}", 10_000, PAYDAY) for i in range(3)]
    unknown = [_event(f"unknown{i}", 99_900, PAYDAY) for i in range(20)]
    diagnoses = [
        _diagnosis(event.event_id, RootCause.NETWORK_TIMEOUT) for event in real
    ] + [_diagnosis(event.event_id, RootCause.UNKNOWN, confidence=0.0) for event in unknown]

    actions, stats = allocate(real + unknown, diagnoses, retry_budget=3, contact_budget=0)
    granted = {a.event_id for a in actions if a.action is not ActionType.SUPPRESS}
    assert granted == {"real0", "real1", "real2"}
    assert stats["retry_budget_spent"] == 3
    assert stats["suppressed_for_no_diagnosis"] == len(unknown)


def test_breakdown_contributions_sum_to_score():
    for cause in RootCause:
        for day in (PAYDAY, MID_MONTH):
            for confidence in (0.0, 0.5, 0.9, 1.0):
                event = _event("e1", 99_900, day)
                diagnosis = _diagnosis("e1", cause, confidence=confidence)
                score, breakdown = index_score(event, diagnosis)
                assert sum(breakdown["contributions"].values()) == pytest.approx(score)
                assert breakdown["score"] == pytest.approx(score)
