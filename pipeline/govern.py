"""Policy gate. Deterministic — no LLM, no diagnose.py."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ConfigDict

from models import ActionType, GateDecision, ProposedAction

IST = ZoneInfo("Asia/Kolkata")

RAIL_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.RETRY, ActionType.MANDATE_REPRESENT}
)
CONTACT_ACTIONS: frozenset[ActionType] = frozenset(
    {ActionType.NUDGE, ActionType.PAYMENT_LINK}
)
DLT_CHANNELS: frozenset[str] = frozenset({"sms", "voice"})

R01_RAIL_CAP = "r01_rail_cap"
R01_RAIL_CAP_DESC = "NPCI: one execution plus three retries; freeze rails in peak IST windows."
R02_PREDEBIT_NOTICE = "r02_predebit_notice"
R02_PREDEBIT_NOTICE_DESC = "Mandate represent requires a pre-debit notice at least 24 hours prior."
R03_DLT_TEMPLATE = "r03_dlt_template"
R03_DLT_TEMPLATE_DESC = "SMS/voice contact requires a registered TRAI DLT template."
R04_WHATSAPP_POLICY = "r04_whatsapp_policy"
R04_WHATSAPP_POLICY_DESC = "WhatsApp contact requires logged consent (Meta policy, not TRAI DLT)."
R05_CONSENT = "r05_consent"
R05_CONSENT_DESC = "Nudge and payment-link contact require logged consent."
R06_QUIET_HOURS = "r06_quiet_hours"
R06_QUIET_HOURS_DESC = "Contact only 08:00–18:59 IST inclusive (voluntary RBI-style recovery-agent hours)."
R07_HALT = "r07_halt"
R07_HALT_DESC = "Halt all action on stop request, promise-to-pay, or open dispute."

RULES: dict[str, str] = {
    R01_RAIL_CAP: R01_RAIL_CAP_DESC,
    R02_PREDEBIT_NOTICE: R02_PREDEBIT_NOTICE_DESC,
    R03_DLT_TEMPLATE: R03_DLT_TEMPLATE_DESC,
    R04_WHATSAPP_POLICY: R04_WHATSAPP_POLICY_DESC,
    R05_CONSENT: R05_CONSENT_DESC,
    R06_QUIET_HOURS: R06_QUIET_HOURS_DESC,
    R07_HALT: R07_HALT_DESC,
}


class GateContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    now: datetime
    retries_used: int
    contacts_this_cycle: int
    last_notice_sent_at: datetime | None
    stop_requested: bool
    promise_to_pay: bool
    dispute_open: bool
    consent_logged: bool
    channel: str | None


def _ist(stamp: datetime) -> datetime:
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=IST)
    return stamp.astimezone(IST)


def _minutes(stamp: datetime) -> int:
    local = _ist(stamp)
    return local.hour * 60 + local.minute


def _in_peak_window(stamp: datetime) -> bool:
    minute = _minutes(stamp)
    return (10 * 60 <= minute < 13 * 60) or (17 * 60 <= minute <= 21 * 60 + 30)


def r01_rail_cap(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block RETRY/MANDATE_REPRESENT at NPCI retry cap or in peak IST windows."""
    if action.action not in RAIL_ACTIONS:
        return True, "not a rail action"
    if ctx.retries_used >= 3:
        return False, f"retries_used={ctx.retries_used} (>= 3; NPCI: one execution plus three retries)"
    if _in_peak_window(ctx.now):
        return False, "peak window 10:00–13:00 or 17:00–21:30 IST"
    return True, "rail cap ok"


def r02_predebit_notice(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block MANDATE_REPRESENT unless a notice was sent at least 24 hours ago."""
    if action.action is not ActionType.MANDATE_REPRESENT:
        return True, "not mandate represent"
    if ctx.last_notice_sent_at is None:
        return False, "no pre-debit notice on file"
    if _ist(ctx.now) - _ist(ctx.last_notice_sent_at) < timedelta(hours=24):
        return False, "pre-debit notice younger than 24 hours"
    return True, "pre-debit notice aged in"


def r03_dlt_template(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block SMS/voice when no registered DLT template is indicated.

    GateContext has no template field, so SMS/voice always fail this rule.
    """
    if ctx.channel not in DLT_CHANNELS:
        return True, "not an SMS/voice channel"
    return False, "no registered DLT template indicated"


def r04_whatsapp_policy(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block WhatsApp unless consent is logged.

    WhatsApp is governed by Meta's business messaging policy, not TRAI DLT.
    """
    if ctx.channel != "whatsapp":
        return True, "not whatsapp"
    if not ctx.consent_logged:
        return False, "WhatsApp requires logged consent (Meta policy)"
    return True, "whatsapp consent logged"


def r05_consent(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block NUDGE and PAYMENT_LINK unless consent is logged."""
    if action.action not in CONTACT_ACTIONS:
        return True, "not a contact action"
    if not ctx.consent_logged:
        return False, "contact requires logged consent"
    return True, "consent logged"


def r06_quiet_hours(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block contact actions outside 08:00–18:59 IST inclusive.

    This is our voluntary policy modelled on RBI recovery-agent norms,
    not a law binding on merchants.
    """
    if action.action not in CONTACT_ACTIONS:
        return True, "not a contact action"
    minute = _minutes(ctx.now)
    if not (8 * 60 <= minute <= 18 * 60 + 59):
        return False, "outside contact hours 08:00–18:59 IST"
    return True, "inside contact hours"


def r07_halt(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block every action if the customer asked to stop, promised to pay, or disputed."""
    if ctx.stop_requested:
        return False, "stop requested"
    if ctx.promise_to_pay:
        return False, "promise to pay on file"
    if ctx.dispute_open:
        return False, "dispute open"
    return True, "no halt flag"


RuleFn = Callable[[ProposedAction, GateContext], tuple[bool, str]]

_RULES_IN_ORDER: tuple[tuple[str, RuleFn], ...] = (
    (R01_RAIL_CAP, r01_rail_cap),
    (R02_PREDEBIT_NOTICE, r02_predebit_notice),
    (R03_DLT_TEMPLATE, r03_dlt_template),
    (R04_WHATSAPP_POLICY, r04_whatsapp_policy),
    (R05_CONSENT, r05_consent),
    (R06_QUIET_HOURS, r06_quiet_hours),
    (R07_HALT, r07_halt),
)


def evaluate(action: ProposedAction, ctx: GateContext) -> GateDecision:
    if action.action is ActionType.SUPPRESS:
        return GateDecision(
            event_id=action.event_id,
            approved=True,
            rule_ids_passed=list(RULES),
            blocked_by=None,
            reason="suppress is never blocked",
        )

    passed: list[str] = []
    for rule_id, fn in _RULES_IN_ORDER:
        ok, why = fn(action, ctx)
        if not ok:
            return GateDecision(
                event_id=action.event_id,
                approved=False,
                rule_ids_passed=passed,
                blocked_by=rule_id,
                reason=why,
            )
        passed.append(rule_id)
    return GateDecision(
        event_id=action.event_id,
        approved=True,
        rule_ids_passed=passed,
        blocked_by=None,
        reason="all applicable rules passed",
    )
