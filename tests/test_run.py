from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ledger import Ledger
from models import ActionType, Diagnosis, FailureEvent, ProposedAction, RootCause
from pipeline.govern import CONTACT_ACTIONS, NPCI_RETRY_CAP, RAIL_ACTIONS
from run import (
    AGENT,
    CONTACT_CHANNEL,
    CONTROL,
    TRACE_ENTRY_TYPES,
    gate_context_for,
    run_batch,
    simulate_outcomes,
)

# 09:00 IST on the 28th: inside contact hours, outside the peak rail freeze,
# and near payday, so a proposed action is not blocked for reasons unrelated
# to the assertion under test.
OCCURRED = datetime(2026, 9, 28, 3, 30, tzinfo=timezone.utc)

RAIL_ACTION_NAMES = {action.value for action in RAIL_ACTIONS}
CONTACT_ACTION_NAMES = {action.value for action in CONTACT_ACTIONS}


@pytest.fixture(autouse=True)
def no_llm(monkeypatch):
    monkeypatch.setenv("NAKAD_LLM_PROVIDER", "none")


def _event(index: int, cause: RootCause = RootCause.INSUFFICIENT_FUNDS) -> FailureEvent:
    return FailureEvent(
        event_id=f"e{index}",
        payment_id=f"pay_{index}",
        subscription_id=f"sub_{index}",
        amount_paise=10_000 * (index + 1),
        method="upi_autopay",
        issuer="hdfc",
        error_code="BAD_REQUEST_ERROR",
        error_source="issuer_bank",
        error_step="payment_authorization",
        error_reason="payment_failed",
        occurred_at=OCCURRED,
        retries_used=0,
        days_since_mandate_created=40,
        day_of_month=28,
        issuer_recent_failure_rate=0.05,
        amount_vs_customer_avg=1.0,
        true_cause=cause,
    )


def _action(event_id: str, action: ActionType = ActionType.RETRY) -> ProposedAction:
    return ProposedAction(
        event_id=event_id,
        action=action,
        rationale="test",
        expected_recovery_paise=0,
        scheduled_for=OCCURRED + timedelta(hours=24),
    )


def _diagnosis(event_id: str, cause: RootCause) -> Diagnosis:
    return Diagnosis(
        event_id=event_id, cause=cause, confidence=0.9, evidence=["test"], method="rule"
    )


def test_gate_time_is_when_the_action_fires_not_wall_clock():
    event = _event(0)
    action = _action("e0")
    assert gate_context_for(event, action).now == action.scheduled_for
    # A suppression has no schedule, so it falls back to the failure time.
    suppressed = ProposedAction(
        event_id="e0",
        action=ActionType.SUPPRESS,
        rationale="test",
        expected_recovery_paise=0,
    )
    assert gate_context_for(event, suppressed).now == event.occurred_at


def test_contact_channel_is_set_only_for_contact_actions():
    event = _event(0)
    for action_type in ActionType:
        ctx = gate_context_for(event, _action("e0", action_type))
        expected = CONTACT_CHANNEL if action_type in CONTACT_ACTIONS else None
        assert ctx.channel == expected


def test_consent_is_derived_from_the_mandate_not_assumed():
    with_mandate = _event(0)
    assert gate_context_for(with_mandate, _action("e0")).consent_logged is True

    one_off = with_mandate.model_copy(update={"subscription_id": None})
    assert gate_context_for(one_off, _action("e0")).consent_logged is False


def test_simulation_is_reproducible_and_seed_dependent():
    events = [_event(i) for i in range(40)]
    actions = [_action(f"e{i}") for i in range(40)]
    diagnoses = [_diagnosis(f"e{i}", RootCause.INSUFFICIENT_FUNDS) for i in range(40)]

    first = simulate_outcomes(actions, events, diagnoses, seed=42)
    assert first == simulate_outcomes(actions, events, diagnoses, seed=42)
    assert first != simulate_outcomes(actions, events, diagnoses, seed=7)


def test_both_arms_face_the_same_coin_for_the_same_event():
    # Common random numbers: the draw is keyed by event_id, so a difference
    # between policies is selection, never luck.
    events = [_event(i) for i in range(20)]
    diagnoses = [_diagnosis(f"e{i}", RootCause.INSUFFICIENT_FUNDS) for i in range(20)]

    everyone = simulate_outcomes([_action(e.event_id) for e in events], events, diagnoses, 42)
    subset_ids = ["e3", "e11", "e17"]
    subset = simulate_outcomes([_action(i) for i in subset_ids], events, diagnoses, 42)
    assert all(everyone[i] == subset[i] for i in subset_ids)


def test_unacted_events_recover_nothing():
    events = [_event(i) for i in range(10)]
    diagnoses = [_diagnosis(f"e{i}", RootCause.NETWORK_TIMEOUT) for i in range(10)]
    suppressed = [_action(f"e{i}", ActionType.SUPPRESS) for i in range(10)]

    assert not any(simulate_outcomes(suppressed, events, diagnoses, 42).values())
    # An event absent from the action list (blocked at the gate) is also False.
    assert not any(simulate_outcomes([], events, diagnoses, 42).values())


def test_control_never_exceeds_its_budget_and_never_contacts(tmp_path):
    events = [_event(i) for i in range(30)]
    for budget in (0, 1, 12, 30, 99):
        result = run_batch(
            events,
            policy=CONTROL,
            ledger_path=str(tmp_path / f"c{budget}.db"),
            retry_budget=budget,
            contact_budget=99,
        )
        granted = min(budget, len(events))
        assert result.actions_by_type.get("RETRY", 0) == granted
        assert result.retry_budget_spent == granted
        assert result.contact_budget_spent == 0
        assert set(result.actions_by_type) <= {"RETRY", "SUPPRESS"}
        assert result.suppressed_for_budget == len(events) - granted


def test_every_event_gets_the_full_four_entry_trace(tmp_path):
    events = [_event(i) for i in range(12)]
    for policy in (AGENT, CONTROL):
        path = tmp_path / f"{policy}.db"
        result = run_batch(events, policy=policy, ledger_path=str(path))
        assert result.events_processed == len(events)

        entries = Ledger(str(path)).read_all()
        assert len(entries) == 4 * len(events)
        for index, event in enumerate(events):
            window = entries[4 * index : 4 * index + 4]
            assert [entry.entry_type for entry in window] == list(TRACE_ENTRY_TYPES)
            assert {entry.event_id for entry in window} == {event.event_id}

        check = Ledger(str(path)).verify()
        assert check.ok, f"{policy} ledger {check.failure} at seq {check.seq}"


def test_blocked_actions_do_not_consume_budget(tmp_path):
    # The first five mandates are at the NPCI cap, so control proposes retries
    # r01 refuses. Without reclamation the whole budget vanishes into those
    # five and nothing is placed at all.
    capped = [
        _event(i).model_copy(update={"retries_used": NPCI_RETRY_CAP}) for i in range(5)
    ]
    fresh = [_event(i) for i in range(5, 25)]
    events = capped + fresh

    result = run_batch(
        events,
        policy=CONTROL,
        ledger_path=str(tmp_path / "reclaim.db"),
        retry_budget=5,
        contact_budget=0,
    )
    assert result.retry_budget_spent == 5
    assert result.allocation_passes > 1
    assert result.blocked_by_rule.get("r01_rail_cap") == len(capped)


def test_budget_placed_equals_actions_that_cleared_the_gate(tmp_path):
    # The single number a reader will check: what we say we spent has to be
    # what actually ran, not what we planned before the gate had its say.
    capped = [
        _event(i).model_copy(update={"retries_used": NPCI_RETRY_CAP}) for i in range(8)
    ]
    events = capped + [_event(i) for i in range(8, 40)]

    for policy in (AGENT, CONTROL):
        path = tmp_path / f"{policy}_spend.db"
        result = run_batch(
            events,
            policy=policy,
            ledger_path=str(path),
            retry_budget=6,
            contact_budget=3,
        )
        gated = [e for e in Ledger(str(path)).read_all() if e.entry_type == "gated"]
        proposed = {
            e.event_id: e.payload["action"]
            for e in Ledger(str(path)).read_all()
            if e.entry_type == "proposed"
        }
        approved_rails = sum(
            entry.payload["approved"] and proposed[entry.event_id] in RAIL_ACTION_NAMES
            for entry in gated
        )
        approved_contacts = sum(
            entry.payload["approved"]
            and proposed[entry.event_id] in CONTACT_ACTION_NAMES
            for entry in gated
        )
        assert result.retry_budget_spent == approved_rails <= 6
        assert result.contact_budget_spent == approved_contacts <= 3


def test_agent_never_proposes_a_rail_action_it_has_no_headroom_for(tmp_path):
    path = tmp_path / "screen.db"
    # A reason the rule layer resolves, so these get a real diagnosis and reach
    # the headroom screen rather than stopping at UNKNOWN with no provider.
    events = [
        _event(i).model_copy(
            update={
                "retries_used": NPCI_RETRY_CAP,
                "error_reason": "insufficient_funds",
            }
        )
        for i in range(10)
    ]
    result = run_batch(events, policy=AGENT, ledger_path=str(path), retry_budget=10)
    assert result.unknown_diagnoses == 0
    assert result.retry_budget_spent == 0
    assert "r01_rail_cap" not in result.blocked_by_rule
    assert result.suppressed_for_no_headroom == len(events)


def test_a_degraded_run_is_reported_as_undiagnosed(tmp_path):
    # With no provider, the ambiguous reasons cannot be diagnosed. They must
    # show up as UNKNOWN rather than being absorbed into a real cause.
    path = tmp_path / "degraded.db"
    events = [_event(i) for i in range(10)]
    result = run_batch(events, policy=AGENT, ledger_path=str(path), retry_budget=10)

    assert result.unknown_diagnoses == len(events)
    assert result.suppressed_for_no_diagnosis == len(events)
    assert result.retry_budget_spent == 0
    assert result.gross_recovered_paise == 0

    diagnosed = [
        entry for entry in Ledger(str(path)).read_all() if entry.entry_type == "diagnosed"
    ]
    assert {entry.payload["cause"] for entry in diagnosed} == {"UNKNOWN"}
    assert all("no diagnosis" in entry.payload["evidence"][0] for entry in diagnosed)


def test_control_records_that_it_did_not_diagnose(tmp_path):
    path = tmp_path / "control.db"
    run_batch([_event(0)], policy=CONTROL, ledger_path=str(path))
    diagnosed = [e for e in Ledger(str(path)).read_all() if e.entry_type == "diagnosed"]
    assert diagnosed[0].payload["method"] == "none"
    assert diagnosed[0].payload["cause"] is None


def test_run_batch_rejects_an_unknown_policy(tmp_path):
    with pytest.raises(ValueError, match="policy must be one of"):
        run_batch([_event(0)], policy="whatever", ledger_path=str(tmp_path / "x.db"))
