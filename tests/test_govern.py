from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models import ActionType, ProposedAction
from pipeline.govern import (
    IST,
    R01_RAIL_CAP,
    R02_PREDEBIT_NOTICE,
    R03_DLT_TEMPLATE,
    R04_WHATSAPP_POLICY,
    R05_CONSENT,
    R06_QUIET_HOURS,
    R07_HALT,
    RULE_STANDING,
    RULES,
    BINDING,
    CONTRACT,
    VOLUNTARY,
    GateContext,
    evaluate,
    r01_rail_cap,
    r02_predebit_notice,
    r03_dlt_template,
    r04_whatsapp_policy,
    r05_consent,
    r06_quiet_hours,
    r07_halt,
)

NINE_AM = datetime(2026, 9, 1, 9, 0, tzinfo=IST)


def _action(kind: ActionType = ActionType.RETRY) -> ProposedAction:
    return ProposedAction(
        event_id="evt_1",
        action=kind,
        rationale="test",
        expected_recovery_paise=14900,
    )


def _ctx(**overrides: object) -> GateContext:
    values: dict[str, object] = {
        "now": NINE_AM,
        "retries_used": 0,
        "contacts_this_cycle": 0,
        "last_notice_sent_at": NINE_AM - timedelta(hours=48),
        "stop_requested": False,
        "promise_to_pay": False,
        "dispute_open": False,
        "consent_logged": True,
        "channel": None,
    }
    values.update(overrides)
    return GateContext.model_validate(values)


def _at(hour: int, minute: int) -> datetime:
    return datetime(2026, 9, 1, hour, minute, tzinfo=IST)


def _assert_block(ok: bool, reason: str, *needles: str) -> None:
    assert ok is False
    assert reason.strip()
    haystack = reason.lower()
    for needle in needles:
        assert needle.lower() in haystack, f"{needle!r} not in {reason!r}"


def test_r01_retries_used_2_pass() -> None:
    ok, _ = r01_rail_cap(_action(), _ctx(retries_used=2))
    assert ok


def test_r01_retries_used_3_block() -> None:
    ok, reason = r01_rail_cap(_action(), _ctx(retries_used=3))
    _assert_block(ok, reason, "3", "retr")


def test_r01_retries_used_4_block() -> None:
    ok, reason = r01_rail_cap(_action(), _ctx(retries_used=4))
    _assert_block(ok, reason, "4", "retr")


@pytest.mark.parametrize(
    ("hour", "minute", "should_pass"),
    [
        (9, 59, True),
        (10, 0, False),
        (13, 0, True),
        (16, 59, True),
        (17, 0, False),
        (21, 30, False),
        (21, 31, True),
    ],
    ids=["09:59", "10:00", "13:00", "16:59", "17:00", "21:30", "21:31"],
)
def test_r01_peak_window_boundaries(
    hour: int, minute: int, should_pass: bool
) -> None:
    ok, reason = r01_rail_cap(_action(), _ctx(now=_at(hour, minute)))
    if should_pass:
        assert ok
        return
    _assert_block(ok, reason, "peak")


def test_r02_pass_notice_older_than_24h() -> None:
    ok, _ = r02_predebit_notice(
        _action(ActionType.MANDATE_REPRESENT),
        _ctx(last_notice_sent_at=NINE_AM - timedelta(hours=24)),
    )
    assert ok


def test_r02_block_fresh_notice() -> None:
    ok, reason = r02_predebit_notice(
        _action(ActionType.MANDATE_REPRESENT),
        _ctx(last_notice_sent_at=NINE_AM - timedelta(hours=23)),
    )
    _assert_block(ok, reason, "notice", "24")


def test_r03_pass_non_dlt_channel() -> None:
    ok, _ = r03_dlt_template(_action(ActionType.NUDGE), _ctx(channel="email"))
    assert ok


def test_r03_block_sms_without_template() -> None:
    ok, reason = r03_dlt_template(_action(ActionType.NUDGE), _ctx(channel="sms"))
    _assert_block(ok, reason, "template")


def test_r04_pass_whatsapp_with_consent() -> None:
    ok, _ = r04_whatsapp_policy(
        _action(ActionType.NUDGE), _ctx(channel="whatsapp", consent_logged=True)
    )
    assert ok


def test_r04_block_whatsapp_without_consent() -> None:
    ok, reason = r04_whatsapp_policy(
        _action(ActionType.NUDGE), _ctx(channel="whatsapp", consent_logged=False)
    )
    _assert_block(ok, reason, "consent")


def test_r05_pass_nudge_with_consent() -> None:
    ok, _ = r05_consent(_action(ActionType.NUDGE), _ctx(consent_logged=True))
    assert ok


def test_r05_block_nudge_without_consent() -> None:
    ok, reason = r05_consent(_action(ActionType.NUDGE), _ctx(consent_logged=False))
    _assert_block(ok, reason, "consent")


@pytest.mark.parametrize(
    ("hour", "minute", "should_pass"),
    [
        (7, 59, False),
        (8, 0, True),
        (18, 59, True),
        (19, 0, False),
    ],
    ids=["07:59", "08:00", "18:59", "19:00"],
)
def test_r06_contact_hour_boundaries(
    hour: int, minute: int, should_pass: bool
) -> None:
    ok, reason = r06_quiet_hours(
        _action(ActionType.NUDGE), _ctx(now=_at(hour, minute))
    )
    if should_pass:
        assert ok
        return
    _assert_block(ok, reason, "08:00", "18:59")


def test_r07_pass_no_halt_flags() -> None:
    ok, _ = r07_halt(_action(), _ctx())
    assert ok


def test_r07_block_stop_requested() -> None:
    ok, reason = r07_halt(_action(), _ctx(stop_requested=True))
    _assert_block(ok, reason, "stop")


def test_blocked_action_names_rule_id() -> None:
    decision = evaluate(_action(), _ctx(retries_used=3))
    assert decision.approved is False
    assert decision.blocked_by == R01_RAIL_CAP
    _assert_block(False, decision.reason, "3", "retr")


def test_suppress_always_passes() -> None:
    decision = evaluate(
        _action(ActionType.SUPPRESS),
        _ctx(
            retries_used=3,
            stop_requested=True,
            promise_to_pay=True,
            dispute_open=True,
            consent_logged=False,
            channel="sms",
            now=datetime(2026, 9, 1, 23, 0, tzinfo=IST),
        ),
    )
    assert decision.approved is True
    assert decision.blocked_by is None


def test_blocked_by_lowest_rule_id() -> None:
    decision = evaluate(
        _action(ActionType.RETRY),
        _ctx(retries_used=3, stop_requested=True, dispute_open=True),
    )
    assert decision.blocked_by == R01_RAIL_CAP
    assert decision.blocked_by not in (R07_HALT, R02_PREDEBIT_NOTICE)
    _assert_block(False, decision.reason, "retr")


def test_evaluate_blocks_sms_with_r03() -> None:
    decision = evaluate(_action(ActionType.NUDGE), _ctx(channel="sms"))
    assert decision.blocked_by == R03_DLT_TEMPLATE
    _assert_block(False, decision.reason, "template")


def test_evaluate_blocks_whatsapp_without_consent_with_r04() -> None:
    decision = evaluate(
        _action(ActionType.NUDGE), _ctx(channel="whatsapp", consent_logged=False)
    )
    assert decision.blocked_by == R04_WHATSAPP_POLICY
    _assert_block(False, decision.reason, "consent")


def test_evaluate_blocks_nudge_without_consent_with_r05() -> None:
    decision = evaluate(_action(ActionType.NUDGE), _ctx(consent_logged=False))
    assert decision.blocked_by == R05_CONSENT
    _assert_block(False, decision.reason, "consent")


def test_evaluate_blocks_quiet_hours_with_r06() -> None:
    decision = evaluate(
        _action(ActionType.NUDGE),
        _ctx(now=datetime(2026, 9, 1, 20, 0, tzinfo=IST)),
    )
    assert decision.blocked_by == R06_QUIET_HOURS
    _assert_block(False, decision.reason, "08:00", "18:59")


def test_passing_decision_lists_all_rule_ids() -> None:
    decision = evaluate(_action(), _ctx())
    assert decision.approved is True
    assert decision.blocked_by is None
    assert decision.rule_ids_passed == list(RULES)
    assert decision.rule_ids_passed != []


def test_every_rule_declares_its_standing() -> None:
    # The console publishes this table, so a rule added without a standing
    # would be shown to an audience with no claim about what backs it.
    assert set(RULE_STANDING) == set(RULES)
    for rule_id, (standing, source) in RULE_STANDING.items():
        assert standing in {BINDING, CONTRACT, VOLUNTARY}, rule_id
        assert source.strip(), rule_id


def test_the_rules_we_impose_on_ourselves_are_not_claimed_as_law() -> None:
    # Overstating standing is the failure that matters: r06's contact hours and
    # r07's promise-to-pay restraint bind lenders and nobody respectively.
    assert RULE_STANDING[R06_QUIET_HOURS][0] == VOLUNTARY
    assert RULE_STANDING[R07_HALT][0] == VOLUNTARY
    assert RULE_STANDING[R04_WHATSAPP_POLICY][0] == CONTRACT
