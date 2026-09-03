from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval import (
    HISTORY_EXPOSURE_PAISE,
    bayes_ceiling,
    compare,
    diagnosis_report,
    observable_signature,
    reserve_report,
)
from generator.generate import VALUATION_MONTH, generate_history
from models import BatchResult, Diagnosis, FailureEvent, RootCause

OCCURRED = datetime(2026, 9, 28, 3, 30, tzinfo=timezone.utc)


def _event(index: int, cause: RootCause, reason: str = "insufficient_funds"):
    return FailureEvent(
        event_id=f"e{index}",
        payment_id=f"pay_{index}",
        subscription_id=f"sub_{index}",
        amount_paise=10_000,
        method="upi_autopay",
        issuer="hdfc",
        error_code="BAD_REQUEST_ERROR",
        error_source="issuer_bank",
        error_step="payment_authorization",
        error_reason=reason,
        occurred_at=OCCURRED,
        retries_used=0,
        days_since_mandate_created=40,
        day_of_month=28,
        issuer_recent_failure_rate=0.05,
        amount_vs_customer_avg=1.0,
        true_cause=cause,
    )


def _diagnosis(event: FailureEvent, cause: RootCause) -> Diagnosis:
    return Diagnosis(
        event_id=event.event_id,
        cause=cause,
        confidence=0.9,
        evidence=["test"],
        method="rule",
    )


def _result(policy: str, **overrides) -> BatchResult:
    base = dict(
        policy=policy,
        events_processed=10,
        actions_by_type={"RETRY": 4, "SUPPRESS": 6},
        blocked_by_rule={"r02_predebit_notice": 1},
        gross_recovered_paise=100_000,
        recovered_by_method={"card": 40_000, "upi_autopay": 60_000},
        retry_budget_spent=4,
        contact_budget_spent=0,
        allocation_passes=1,
        suppressed_for_budget=6,
        suppressed_for_low_value=0,
        suppressed_for_low_confidence=0,
        suppressed_for_no_headroom=0,
        suppressed_for_no_window=0,
        unknown_diagnoses=0,
        suppressed_for_no_diagnosis=0,
        ledger_path="/tmp/x.db",
        run_config={
            "provider": "openai",
            "model": "test-model",
            "events": 0,
            "by_method": {},
            "unknown": 0,
            "degraded": False,
        },
    )
    return BatchResult(**{**base, **overrides})


def test_an_unambiguous_signature_set_has_a_ceiling_of_one():
    events = [
        _event(0, RootCause.INSUFFICIENT_FUNDS, "insufficient_funds"),
        _event(1, RootCause.DEAD_MANDATE, "debit_instrument_inactive"),
    ]
    assert bayes_ceiling(events) == (1.0, 2)


def test_one_signature_split_evenly_between_two_causes_caps_at_half():
    events = [
        _event(0, RootCause.INSUFFICIENT_FUNDS, "payment_failed"),
        _event(1, RootCause.DEAD_MANDATE, "payment_failed"),
    ]
    ceiling, groups = bayes_ceiling(events)
    assert groups == 1
    assert ceiling == 0.5


def test_the_ceiling_is_the_modal_share_not_the_group_count():
    # Three of one cause and one of another under a single signature: the best
    # any model can do is answer with the majority and lose the odd one out.
    events = [_event(i, RootCause.INSUFFICIENT_FUNDS, "payment_failed") for i in range(3)]
    events.append(_event(3, RootCause.DEAD_MANDATE, "payment_failed"))
    assert bayes_ceiling(events) == (0.75, 1)


def test_signature_ignores_fields_the_error_code_does_not_carry():
    left = _event(0, RootCause.INSUFFICIENT_FUNDS)
    right = _event(1, RootCause.DEAD_MANDATE)
    right = right.model_copy(update={"amount_paise": 999_999, "day_of_month": 3})
    assert observable_signature(left) == observable_signature(right)


def test_a_perfect_classifier_reaches_the_ceiling_and_no_further():
    events = [
        _event(0, RootCause.INSUFFICIENT_FUNDS, "payment_failed"),
        _event(1, RootCause.DEAD_MANDATE, "payment_failed"),
    ]
    # Both answered INSUFFICIENT_FUNDS: one right, one wrong, which is the most
    # this signature allows.
    diagnoses = [_diagnosis(e, RootCause.INSUFFICIENT_FUNDS) for e in events]
    report = diagnosis_report(events, diagnoses)
    assert report["accuracy_all_events"] == 0.5
    assert report["bayes_ceiling"] == 0.5
    assert report["share_of_ceiling"] == 1.0


def test_unknown_counts_against_overall_accuracy_but_not_answered_accuracy():
    events = [_event(i, RootCause.INSUFFICIENT_FUNDS) for i in range(4)]
    diagnoses = [
        _diagnosis(events[0], RootCause.INSUFFICIENT_FUNDS),
        _diagnosis(events[1], RootCause.INSUFFICIENT_FUNDS),
        _diagnosis(events[2], RootCause.UNKNOWN),
        _diagnosis(events[3], RootCause.UNKNOWN),
    ]
    report = diagnosis_report(events, diagnoses)
    assert report["accuracy_all_events"] == 0.5
    assert report["accuracy_when_answered"] == 1.0
    assert report["undiagnosed"] == 2


def test_a_missing_diagnosis_is_treated_as_undiagnosed_not_dropped():
    events = [_event(i, RootCause.INSUFFICIENT_FUNDS) for i in range(3)]
    report = diagnosis_report(events, [_diagnosis(events[0], RootCause.INSUFFICIENT_FUNDS)])
    assert report["events"] == 3
    assert report["undiagnosed"] == 2


def test_confusion_matrix_totals_match_the_event_count():
    events = [_event(i, RootCause.INSUFFICIENT_FUNDS) for i in range(5)]
    diagnoses = [_diagnosis(e, RootCause.ISSUER_DOWNTIME) for e in events]
    report = diagnosis_report(events, diagnoses)
    assert sum(sum(row) for row in report["confusion_matrix"]) == 5
    # Every answer wrong, so the class we claimed has zero precision.
    assert report["per_class"]["ISSUER_DOWNTIME"]["precision"] == 0.0
    assert report["per_class"]["INSUFFICIENT_FUNDS"]["recall"] == 0.0

@pytest.fixture(scope="module")
def full_history() -> pd.DataFrame:
    return generate_history(seed=42, valuation_cutoff=None)


def test_ultimate_exceeds_reported_because_disputes_are_still_arriving(full_history):
    report = reserve_report(full_history, batch_recovered_paise=1_000_000)
    assert report["ultimate_paise"] > report["reported_paise"]
    assert report["ultimate_dispute_rate"] > report["reported_dispute_rate"]


def test_clawback_is_the_ultimate_rate_applied_to_the_recovered_volume(full_history):
    recovered = 1_000_000
    report = reserve_report(full_history, batch_recovered_paise=recovered)
    assert report["projected_clawback_paise"] == round(
        recovered * report["ultimate_dispute_rate"]
    )
    assert (
        report["net_recovered_paise"]
        == recovered - report["projected_clawback_paise"]
    )


def test_the_projection_lands_within_an_order_of_magnitude_of_seeded_truth(full_history):
    report = reserve_report(full_history, batch_recovered_paise=1_000_000)
    assert report["held_out_disputes"] > 0
    assert report["total_true_ibnr_paise"] > 0
    # Chain ladder over-reserves the youngest cohort on thin data; the check is
    # that the model is in the right ballpark, not that it is unbiased.
    assert 0.5 < report["total_projected_ibnr_paise"] / report["total_true_ibnr_paise"] < 2.0


def test_an_observed_only_history_still_reserves_but_cannot_be_graded():
    observed = generate_history(seed=42, valuation_cutoff=VALUATION_MONTH)
    report = reserve_report(observed, batch_recovered_paise=1_000_000)
    assert report["held_out_disputes"] == 0
    assert report["total_true_ibnr_paise"] == 0
    assert report["total_projected_ibnr_paise"] > 0


def test_the_seeded_dispute_rate_is_recovered_from_the_triangle(full_history):
    # generate_history disputes 2.5% of payments; the ultimate rate should find
    # its way back to roughly that, which is what makes the exposure constant
    # trustworthy rather than decorative.
    report = reserve_report(full_history, batch_recovered_paise=0)
    assert report["exposure_paise"] == HISTORY_EXPOSURE_PAISE
    assert 0.020 < report["ultimate_dispute_rate"] < 0.030


# -- comparison ------------------------------------------------------------


def test_lift_is_reported_absolutely_and_as_a_ratio():
    agent = _result("agent", gross_recovered_paise=300_000)
    control = _result("control", gross_recovered_paise=100_000)
    report = compare(agent, control)
    assert report["gross"]["absolute"] == 200_000
    assert report["gross"]["ratio"] == 3.0


def test_without_a_dispute_rate_net_equals_gross_and_says_so():
    report = compare(_result("agent"), _result("control"))
    assert report["net_is_modelled"] is False
    assert report["net"]["agent"] == report["gross"]["agent"]


def test_clawback_is_charged_on_card_volume_only():
    agent = _result(
        "agent",
        gross_recovered_paise=100_000,
        recovered_by_method={"card": 40_000, "upi_autopay": 60_000},
    )
    report = compare(agent, _result("control"), ultimate_dispute_rate=0.10)
    # 10% of the 40,000 card volume, not of the 100,000 total.
    assert report["agent_detail"]["projected_clawback_paise"] == 4_000
    assert report["net"]["agent"] == 96_000


def test_executed_actions_exclude_the_ones_the_gate_refused():
    result = _result(
        "agent",
        actions_by_type={"RETRY": 10, "SUPPRESS": 5},
        blocked_by_rule={"r02_predebit_notice": 3, "r04_whatsapp_policy": 1},
    )
    report = compare(result, _result("control"))
    assert report["agent_detail"]["actions_executed"] == 6
    assert report["agent_detail"]["actions_blocked"] == 4


def test_recovery_per_retry_does_not_divide_by_a_budget_never_spent():
    idle = _result("control", retry_budget_spent=0, gross_recovered_paise=0)
    report = compare(_result("agent"), idle)
    assert report["control_detail"]["recovery_per_retry_paise"] is None
    assert report["recovery_per_retry"]["absolute"] is None
    assert report["recovery_per_retry"]["ratio"] is None
    # Gross is still comparable: the control genuinely recovered nothing.
    assert report["gross"]["absolute"] == 100_000
    assert report["gross"]["ratio"] is None
