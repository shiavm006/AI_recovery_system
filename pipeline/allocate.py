"""Scarce attempt/contact allocation. No LLM, no network.

SINGLE-EVENT CALLS ARE DEGENERATE: one candidate cannot be ranked. Admission
only means something if remaining budget persists across calls — webhook.py
passes that via ``LiveBudget``.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta

from models import ActionType, Diagnosis, FailureEvent, ProposedAction, RootCause
from pipeline.govern import (
    CONTACT_ACTIONS,
    CONTACT_WINDOW,
    NPCI_RETRY_CAP,
    PEAK_RAIL_WINDOWS,
    RAIL_ACTIONS,
    GateContext,
    minute_of_day,
    to_ist,
)

# Recovery probability per root cause, before context.
#
# INVENTED. There is no public per-cause recovery table for Indian recurring
# payments, so these are ordering assumptions, not measurements. Treat the
# ordering as the claim and the magnitudes as placeholders to be replaced by
# observed outcomes once the ledger has enough history.
BASE_RECOVERY_PROBABILITY: dict[RootCause, float] = {
    RootCause.NETWORK_TIMEOUT: 0.80,
    RootCause.ISSUER_DOWNTIME: 0.75,
    RootCause.INSUFFICIENT_FUNDS: 0.45,
    RootCause.DEAD_MANDATE: 0.35,
    RootCause.HARD_DECLINE: 0.02,
    RootCause.RISK_BLOCK: 0.01,
    # Not an estimate. UNKNOWN means diagnosis failed, and zero keeps the
    # scoring function total so ranking cannot raise on it; the SUPPRESS
    # mapping below is what actually keeps it away from the budget.
    RootCause.UNKNOWN: 0.0,
}

# Indian salary credits cluster at month end and month start, so a balance
# failure is much more recoverable then. The window follows that payroll
# pattern; the two adjustment magnitudes are INVENTED.
PAYDAY_DAYS: frozenset[int] = frozenset(range(1, 6)) | frozenset(range(25, 32))
PAYDAY_UPLIFT = 0.25
MID_MONTH_PENALTY = -0.20

# Batch-level throttles. These are NOT the NPCI cap: NPCI's one execution plus
# three retries is per mandate cycle, so a 500-event batch spread over 481
# distinct subscriptions permits 1054 attempts. We cap the batch at roughly a
# fifth of that because firing a thousand debits in one window is an
# issuer-relationship and reputation decision, not a compliance one. CHOSEN,
# not derived — the regulator sets the per-mandate ceiling, we set this one.
DEFAULT_RETRY_BUDGET = 120
# Contacts are throttled harder than attempts: a failed retry is invisible to
# the customer, a message is not. CHOSEN.
DEFAULT_CONTACT_BUDGET = 60

# Budget units consumed by one action, used as the denominator of the index.
# A retry or re-presentation burns one of the capped mandate attempts, which is
# the genuinely scarce resource. A contact spends customer patience instead.
# ATTEMPT_COST_UNITS is the unit of the cap; CONTACT_COST_UNITS is INVENTED.
ATTEMPT_COST_UNITS = 1.0
CONTACT_COST_UNITS = 0.25

# The earliest we are willing to act, in hours after the failure. These are
# floors, not the schedule: next_permitted_time pushes the action forward from
# here to the first moment the gate's clock rules allow it.
#
# INVENTED, except DEAD_MANDATE: govern.py r02 requires a pre-debit notice at
# least 24 hours before a re-presentation, so anything sooner would be blocked
# at the gate anyway.
ACTION_DELAY_HOURS: dict[RootCause, int] = {
    RootCause.NETWORK_TIMEOUT: 1,
    RootCause.ISSUER_DOWNTIME: 6,
    RootCause.INSUFFICIENT_FUNDS: 24,
    RootCause.DEAD_MANDATE: 48,  # 24h notice (r02) plus margin
}

# One best action per cause. HARD_DECLINE and RISK_BLOCK are not worth an
# attempt at any amount, so they map straight to SUPPRESS.
INTERVENTION_BY_CAUSE: dict[RootCause, ActionType] = {
    RootCause.NETWORK_TIMEOUT: ActionType.RETRY,
    RootCause.ISSUER_DOWNTIME: ActionType.RETRY,
    RootCause.INSUFFICIENT_FUNDS: ActionType.RETRY,
    RootCause.DEAD_MANDATE: ActionType.MANDATE_REPRESENT,
    RootCause.HARD_DECLINE: ActionType.SUPPRESS,
    RootCause.RISK_BLOCK: ActionType.SUPPRESS,
    # Belt and braces. Every UNKNOWN this pipeline produces comes from
    # diagnose._fallback at confidence 0.0, so MIN_ACTIONABLE_CONFIDENCE below
    # would already exclude it. The mapping is not redundant though: it is the
    # only guard against an UNKNOWN arriving with high confidence, which the
    # floor would wave straight through.
    RootCause.UNKNOWN: ActionType.SUPPRESS,
}

# Act only when confidence ≥ 0.5. UNKNOWN maps to SUPPRESS and scores 0;
# this floor still stops low-confidence real causes from taking scarce attempts.
# INVENTED threshold, but the coin flip is the natural line: act only when the
# diagnosis is more likely right than wrong.
MIN_ACTIONABLE_CONFIDENCE = 0.5

SUPPRESSED_FOR_BUDGET = "excluded for budget"
SUPPRESSED_FOR_LOW_VALUE = "excluded for low expected value"
SUPPRESSED_FOR_LOW_CONFIDENCE = "excluded for low diagnostic confidence"
SUPPRESSED_FOR_NO_HEADROOM = "excluded for no mandate headroom"
SUPPRESSED_FOR_NO_DIAGNOSIS = "no diagnosis available"
SUPPRESSED_FOR_NO_WINDOW = "excluded for no permitted window"

# How far past the earliest acceptable time the scheduler will hunt for a legal
# window before giving up. An NPCI attempt belongs to the current mandate
# cycle, so an action pushed more than three days out is not this cycle's
# recovery any more. CHOSEN. With the current windows nothing comes close to
# needing it — the longest wait is the overnight gap before contact hours — so
# it is a guard against a future rule that closes a whole day, not a live
# constraint.
SCHEDULING_HORIZON_HOURS = 72

RETRY_POOL = "retry"
CONTACT_POOL = "contact"


def _near_payday(day_of_month: int) -> bool:
    return day_of_month in PAYDAY_DAYS


def has_rail_headroom(event: FailureEvent) -> bool:
    return event.retries_used < NPCI_RETRY_CAP


def _probability_terms(event: FailureEvent, diagnosis: Diagnosis) -> dict[str, float]:
    """Additive terms whose sum is the recovery probability.

    Additive rather than multiplicative so a single ranking decision can be
    explained as a list of contributions that add up to the score.
    """
    base = BASE_RECOVERY_PROBABILITY[diagnosis.cause]
    payday = 0.0
    if diagnosis.cause is RootCause.INSUFFICIENT_FUNDS:
        payday = PAYDAY_UPLIFT if _near_payday(event.day_of_month) else MID_MONTH_PENALTY
    clamped = min(1.0, max(0.0, base + payday))
    return {
        "base": base,
        "payday_adjustment": payday,
        "clamp_adjustment": clamped - base - payday,
    }


def recovery_probability(event: FailureEvent, diagnosis: Diagnosis) -> float:
    return sum(_probability_terms(event, diagnosis).values())


def intervention_for(event: FailureEvent, diagnosis: Diagnosis) -> ActionType:
    """The single best action for this cause.

    A balance failure away from payday is the one exception: asking the
    customer costs a contact, where a retry would burn a scarce attempt on a
    balance that probably has not arrived yet.

    A mandate already at the NPCI cap gets no rail action at all. govern r01
    would refuse it, and an action the gate is certain to block still consumes
    a budget unit here — paying for something that cannot happen.
    """
    if diagnosis.cause is RootCause.INSUFFICIENT_FUNDS and not _near_payday(
        event.day_of_month
    ):
        return ActionType.PAYMENT_LINK
    action = INTERVENTION_BY_CAUSE[diagnosis.cause]
    if action in RAIL_ACTIONS and not has_rail_headroom(event):
        return ActionType.SUPPRESS
    return action


def _blocking_window_ends_at(action: ActionType, when: datetime) -> datetime | None:
    """When the window currently blocking ``action`` lifts, or None if it is open.

    IST has no daylight saving, so adding minutes to a local time is the same
    as adding them to the instant. This would need care in a zone that shifts.
    """
    local = to_ist(when)
    minute = minute_of_day(local)
    if action in RAIL_ACTIONS:
        for start, end in PEAK_RAIL_WINDOWS:
            if start <= minute < end:
                return local + timedelta(minutes=end - minute)
        return None
    if action in CONTACT_ACTIONS:
        start, end = CONTACT_WINDOW
        if minute < start:
            return local + timedelta(minutes=start - minute)
        if minute >= end:
            return local + timedelta(minutes=24 * 60 - minute + start)
        return None
    return None


def next_permitted_time(
    action: ActionType, earliest: datetime, ctx: GateContext
) -> datetime | None:
    """The first moment at or after ``earliest`` when the clock permits ``action``.

    The gate is not a wall to walk into. Two of its rules depend only on the
    time we pick — r01 freezes the rails inside ``PEAK_RAIL_WINDOWS``, r06
    confines contact to ``CONTACT_WINDOW`` — so an action refused for either
    reason was not forbidden, merely booked at the wrong hour. Both window
    definitions are imported from govern rather than restated here; a second
    copy of the boundaries is a copy that eventually disagrees with the rule
    enforcing them.

    Returns None when no window opens inside ``SCHEDULING_HORIZON_HOURS``, and
    when ``ctx`` rules the action out for a reason no hour can fix: a mandate
    at the NPCI cap is refused at every moment, so there is no time to find.
    Callers must suppress rather than schedule in that case.
    """
    if action in RAIL_ACTIONS and ctx.retries_used >= NPCI_RETRY_CAP:
        return None

    deadline = earliest + timedelta(hours=SCHEDULING_HORIZON_HOURS)
    when = earliest
    while when <= deadline:
        opens_at = _blocking_window_ends_at(action, when)
        if opens_at is None:
            return when
        when = opens_at
    return None


def _schedule_context(event: FailureEvent, when: datetime) -> GateContext:
    """The slice of GateContext the scheduler reads: the clock and the cap.

    :func:`next_permitted_time` consults ``now`` and ``retries_used`` and
    nothing else. The customer state in the remaining fields is not known at
    allocation time — run.py assembles the real context at the gate — so they
    are set to values that cannot change the answer. If a clock-dependent rule
    ever starts reading one of them, this stops being safe and the scheduler
    has to be handed the real context instead.
    """
    return GateContext(
        now=when,
        retries_used=event.retries_used,
        contacts_this_cycle=0,
        last_notice_sent_at=None,
        stop_requested=False,
        promise_to_pay=False,
        dispute_open=False,
        consent_logged=False,
        channel=None,
    )


def schedule_for(
    event: FailureEvent, diagnosis: Diagnosis, action: ActionType
) -> datetime | None:
    earliest = event.occurred_at + timedelta(
        hours=ACTION_DELAY_HOURS[diagnosis.cause]
    )
    return next_permitted_time(action, earliest, _schedule_context(event, earliest))


def budget_pool(action: ActionType) -> str | None:
    if action in RAIL_ACTIONS:
        return RETRY_POOL
    if action in CONTACT_ACTIONS:
        return CONTACT_POOL
    return None


COST_BY_POOL: dict[str, float] = {
    RETRY_POOL: ATTEMPT_COST_UNITS,
    CONTACT_POOL: CONTACT_COST_UNITS,
}


def _budget_cost(action: ActionType) -> float:
    pool = budget_pool(action)
    return 0.0 if pool is None else COST_BY_POOL[pool]


def index_score(event: FailureEvent, diagnosis: Diagnosis) -> tuple[float, dict]:
    """Expected recovered paise per budget unit: confidence × P(recovery) × amount / cost.

    Confidence scales the ranking; ``MIN_ACTIONABLE_CONFIDENCE`` in allocate is the floor.
    """
    action = intervention_for(event, diagnosis)
    cost = _budget_cost(action)
    terms = _probability_terms(event, diagnosis)
    probability = sum(terms.values())
    per_unit = 0.0 if cost == 0.0 else diagnosis.confidence * event.amount_paise / cost
    contributions = {name: per_unit * value for name, value in terms.items()}
    score = sum(contributions.values())
    breakdown = {
        "action": action.value,
        "cause": diagnosis.cause.value,
        "amount_paise": event.amount_paise,
        "probability": probability,
        "confidence": diagnosis.confidence,
        "expected_recovery_paise": round(
            diagnosis.confidence * probability * event.amount_paise
        ),
        "budget_cost_units": cost,
        "contributions": contributions,
        "score": score,
    }
    return score, breakdown


def _granted(
    event: FailureEvent,
    diagnosis: Diagnosis,
    action: ActionType,
    score: float,
    breakdown: dict,
    scheduled_for: datetime,
) -> ProposedAction:
    return ProposedAction(
        event_id=event.event_id,
        action=action,
        rationale=(
            f"{diagnosis.cause.value} -> {action.value}: "
            f"p={breakdown['probability']:.2f} at confidence "
            f"{diagnosis.confidence:.2f} ({diagnosis.method}) on "
            f"{event.amount_paise} paise, index {score:.0f} per budget unit, "
            f"booked for {to_ist(scheduled_for):%Y-%m-%d %H:%M} IST"
        ),
        expected_recovery_paise=breakdown["expected_recovery_paise"],
        scheduled_for=scheduled_for,
    )


def _suppressed(event: FailureEvent, rationale: str) -> ProposedAction:
    return ProposedAction(
        event_id=event.event_id,
        action=ActionType.SUPPRESS,
        rationale=rationale,
        expected_recovery_paise=0,
        scheduled_for=None,
    )


def allocate(
    events: list[FailureEvent],
    diagnoses: list[Diagnosis],
    retry_budget: int = DEFAULT_RETRY_BUDGET,
    contact_budget: int = DEFAULT_CONTACT_BUDGET,
) -> tuple[list[ProposedAction], dict]:
    """Rank by index score; each pool cuts independently when its budget is dry.

    Below ``MIN_ACTIONABLE_CONFIDENCE`` never reaches either pool. One ProposedAction
    per input event, in ranked order.
    """
    if retry_budget < 0 or contact_budget < 0:
        raise ValueError("budgets must be non-negative")
    by_event = {diagnosis.event_id: diagnosis for diagnosis in diagnoses}
    missing = [event.event_id for event in events if event.event_id not in by_event]
    if missing:
        raise ValueError(f"no diagnosis for {len(missing)} event(s): {missing[:3]}")

    scored = [(index_score(event, by_event[event.event_id]), event) for event in events]
    scored.sort(key=lambda row: (-row[0][0], row[1].event_id))

    limits = {RETRY_POOL: retry_budget, CONTACT_POOL: contact_budget}
    spent = {RETRY_POOL: 0, CONTACT_POOL: 0}
    cut: dict[str, float | None] = {RETRY_POOL: None, CONTACT_POOL: None}

    actions: list[ProposedAction] = []
    for (score, breakdown), event in scored:
        diagnosis = by_event[event.event_id]
        action = ActionType(breakdown["action"])
        if action is ActionType.SUPPRESS:
            if diagnosis.cause is RootCause.UNKNOWN:
                rationale = (
                    f"{SUPPRESSED_FOR_NO_DIAGNOSIS}: "
                    f"{'; '.join(diagnosis.evidence) or 'reason not recorded'}"
                )
            elif not has_rail_headroom(event):
                rationale = (
                    f"{SUPPRESSED_FOR_NO_HEADROOM}: {diagnosis.cause.value} needs a "
                    f"rail attempt but retries_used={event.retries_used} is at the "
                    f"NPCI cap of {NPCI_RETRY_CAP}"
                )
            else:
                rationale = (
                    f"{SUPPRESSED_FOR_LOW_VALUE}: {diagnosis.cause.value} is not "
                    f"recoverable by an attempt (p={breakdown['probability']:.2f})"
                )
            actions.append(_suppressed(event, rationale))
            continue
        if diagnosis.confidence < MIN_ACTIONABLE_CONFIDENCE:
            actions.append(
                _suppressed(
                    event,
                    f"{SUPPRESSED_FOR_LOW_CONFIDENCE}: {diagnosis.cause.value} at "
                    f"confidence {diagnosis.confidence:.2f} ({diagnosis.method}) is "
                    f"below the {MIN_ACTIONABLE_CONFIDENCE} floor",
                )
            )
            continue
        scheduled = schedule_for(event, diagnosis, action)
        if scheduled is None:
            actions.append(
                _suppressed(
                    event,
                    f"{SUPPRESSED_FOR_NO_WINDOW}: no moment in the next "
                    f"{SCHEDULING_HORIZON_HOURS}h lets {action.value} through the "
                    "gate's clock rules",
                )
            )
            continue
        pool = budget_pool(action)
        if pool is not None:
            if spent[pool] >= limits[pool]:
                actions.append(
                    _suppressed(
                        event,
                        f"{SUPPRESSED_FOR_BUDGET}: {pool} budget {limits[pool]} "
                        f"exhausted; index {score:.0f} fell below the {pool} cut",
                    )
                )
                continue
            spent[pool] += 1
            cut[pool] = score
        actions.append(_granted(event, diagnosis, action, score, breakdown, scheduled))

    counts = Counter(action.action.value for action in actions)
    stats = {
        "by_action": dict(counts),
        "total_expected_recovery_paise": sum(
            action.expected_recovery_paise for action in actions
        ),
        "retry_budget_spent": spent[RETRY_POOL],
        "contact_budget_spent": spent[CONTACT_POOL],
        "retry_cut_index_score": cut[RETRY_POOL],
        "contact_cut_index_score": cut[CONTACT_POOL],
        "suppressed_for_budget": sum(
            action.rationale.startswith(SUPPRESSED_FOR_BUDGET) for action in actions
        ),
        "suppressed_for_low_value": sum(
            action.rationale.startswith(SUPPRESSED_FOR_LOW_VALUE) for action in actions
        ),
        "suppressed_for_low_confidence": sum(
            action.rationale.startswith(SUPPRESSED_FOR_LOW_CONFIDENCE)
            for action in actions
        ),
        "suppressed_for_no_headroom": sum(
            action.rationale.startswith(SUPPRESSED_FOR_NO_HEADROOM) for action in actions
        ),
        "suppressed_for_no_window": sum(
            action.rationale.startswith(SUPPRESSED_FOR_NO_WINDOW) for action in actions
        ),
        "suppressed_for_no_diagnosis": sum(
            action.rationale.startswith(SUPPRESSED_FOR_NO_DIAGNOSIS)
            for action in actions
        ),
    }
    return actions, stats
