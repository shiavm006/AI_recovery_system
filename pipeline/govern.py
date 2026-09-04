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

# NPCI permits one execution plus three retries per mandate cycle. Exported so
# the allocator can screen out mandates with no headroom rather than proposing
# attempts this gate would refuse.
NPCI_RETRY_CAP = 3

# The two clock-dependent rules, as half-open [start, end) minute-of-day
# windows in IST. Exported for the same reason as the cap: allocate schedules
# around these rather than into them, and a second copy of the boundaries
# would eventually disagree with the rule that enforces them.
PEAK_RAIL_WINDOWS: tuple[tuple[int, int], ...] = (
    (10 * 60, 13 * 60),  # 10:00–12:59, rails frozen
    (17 * 60, 21 * 60 + 31),  # 17:00–21:30 inclusive, rails frozen
)
CONTACT_WINDOW: tuple[int, int] = (8 * 60, 19 * 60)  # 08:00–18:59, contact allowed

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

BINDING = "binding"  # a regulator can penalise you for breaking it
CONTRACT = "contract"  # a platform can cut you off for breaking it
VOLUNTARY = "voluntary"  # nobody makes us do this; we do it anyway

# Kept beside RULES so the claim about a rule and the rule itself move
# together. Overstating standing is the failure mode worth guarding against:
# calling a self-imposed courtesy a legal requirement is the kind of thing
# that survives right up until someone in the audience actually knows the
# regulation.
RULE_STANDING: dict[str, tuple[str, str]] = {
    R01_RAIL_CAP: (BINDING, "NPCI mandate-cycle retry limits"),
    R02_PREDEBIT_NOTICE: (BINDING, "RBI/NPCI 24-hour pre-debit notification"),
    R03_DLT_TEMPLATE: (BINDING, "TRAI TCCCPR 2018, DLT-registered templates"),
    R04_WHATSAPP_POLICY: (CONTRACT, "Meta WhatsApp Business Messaging Policy"),
    R05_CONSENT: (BINDING, "DPDP Act 2023 consent for commercial contact"),
    # RBI's recovery-agent hours bind lenders and their agents. A merchant
    # chasing a failed subscription debit is not a recovery agent, so this is
    # borrowed, not owed.
    R06_QUIET_HOURS: (VOLUNTARY, "modelled on RBI recovery-agent hours"),
    # Honouring a stop request is binding under DPDP consent withdrawal.
    # Standing down for a promise-to-pay or an open dispute is not required of
    # anyone; it is restraint, and it is the part worth pointing at.
    R07_HALT: (VOLUNTARY, "stop request is binding; promise-to-pay and dispute are restraint"),
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


def to_ist(stamp: datetime) -> datetime:
    if stamp.tzinfo is None:
        return stamp.replace(tzinfo=IST)
    return stamp.astimezone(IST)


def minute_of_day(stamp: datetime) -> int:
    local = to_ist(stamp)
    return local.hour * 60 + local.minute


def in_peak_rail_window(stamp: datetime) -> bool:
    minute = minute_of_day(stamp)
    return any(start <= minute < end for start, end in PEAK_RAIL_WINDOWS)


def in_contact_hours(stamp: datetime) -> bool:
    start, end = CONTACT_WINDOW
    return start <= minute_of_day(stamp) < end


def r01_rail_cap(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block RETRY/MANDATE_REPRESENT at NPCI retry cap or in peak IST windows."""
    if action.action not in RAIL_ACTIONS:
        return True, "not a rail action"
    if ctx.retries_used >= NPCI_RETRY_CAP:
        return False, (
            f"retries_used={ctx.retries_used} (>= {NPCI_RETRY_CAP}; "
            "NPCI: one execution plus three retries)"
        )
    if in_peak_rail_window(ctx.now):
        return False, "peak window 10:00–13:00 or 17:00–21:30 IST"
    return True, "rail cap ok"


def r02_predebit_notice(action: ProposedAction, ctx: GateContext) -> tuple[bool, str]:
    """Block MANDATE_REPRESENT unless a notice was sent at least 24 hours ago."""
    if action.action is not ActionType.MANDATE_REPRESENT:
        return True, "not mandate represent"
    if ctx.last_notice_sent_at is None:
        return False, "no pre-debit notice on file"
    if to_ist(ctx.now) - to_ist(ctx.last_notice_sent_at) < timedelta(hours=24):
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
    if not in_contact_hours(ctx.now):
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
