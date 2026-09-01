"""NPCI permits one execution plus three retries per mandate cycle. Recovery is
therefore a constrained allocation problem, not a schedule — the question is
which failures receive the scarce attempts and which receive none. This is an
index policy in the spirit of a restless multi-armed bandit; we do not claim a
proven Whittle index.

No LLM calls, no network calls.
"""

from __future__ import annotations

from collections import Counter
from datetime import timedelta

from models import ActionType, Diagnosis, FailureEvent, ProposedAction, RootCause
from pipeline.govern import CONTACT_ACTIONS, RAIL_ACTIONS

# Recovery probability per root cause, before context.
#
# INVENTED. There is no public per-cause recovery table for Indian recurring
# payments, so these are ordering assumptions, not measurements. Treat the
# ordering as the claim and the magnitudes as placeholders to be replaced by
# observed outcomes once the ledger has enough history.
BASE_RECOVERY_PROBABILITY: dict[RootCause, float] = {
    RootCause.NETWORK_TIMEOUT: 0.80,  # transient gateway fault; nothing is wrong with the payer
    RootCause.ISSUER_DOWNTIME: 0.75,  # bank-side outage, so the failure was not the customer's
    RootCause.INSUFFICIENT_FUNDS: 0.45,  # depends on the payer's balance; see the payday terms
    RootCause.DEAD_MANDATE: 0.35,  # recoverable, but only by re-presenting the mandate
    RootCause.HARD_DECLINE: 0.02,  # the issuer refused the instrument outright
    RootCause.RISK_BLOCK: 0.01,  # blocked deliberately; retrying is the wrong answer
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

# Hours to wait after the failure before acting. INVENTED, except DEAD_MANDATE:
# govern.py r02 requires a pre-debit notice at least 24 hours before a
# re-presentation, so anything sooner would be blocked at the gate anyway.
ACTION_DELAY_HOURS: dict[RootCause, int] = {
    RootCause.NETWORK_TIMEOUT: 1,  # retry once the blip clears
    RootCause.ISSUER_DOWNTIME: 6,  # let the bank's outage window pass
    RootCause.INSUFFICIENT_FUNDS: 24,  # give the payday credit time to land
    RootCause.DEAD_MANDATE: 48,  # 24h notice plus margin
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
}

# A diagnosis we do not believe is not worth a scarce attempt. diagnose.py
# emits confidence 0.0 for its deterministic fallback, and that fallback
# defaults to INSUFFICIENT_FUNDS — the highest-scoring cause here — so without
# this floor a fabricated diagnosis outranks real work. INVENTED threshold, but
# the coin flip is the natural line: act only when the diagnosis is more likely
# right than wrong.
MIN_ACTIONABLE_CONFIDENCE = 0.5

SUPPRESSED_FOR_BUDGET = "excluded for budget"
SUPPRESSED_FOR_LOW_VALUE = "excluded for low expected value"
SUPPRESSED_FOR_LOW_CONFIDENCE = "excluded for low diagnostic confidence"

# The two scarce pools. Mandate attempts are capped by NPCI; contacts are
# capped by how often we are willing to bother one customer in a cycle.
RETRY_POOL = "retry"
CONTACT_POOL = "contact"


def _near_payday(day_of_month: int) -> bool:
    return day_of_month in PAYDAY_DAYS


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
    """
    if diagnosis.cause is RootCause.INSUFFICIENT_FUNDS and not _near_payday(
        event.day_of_month
    ):
        return ActionType.PAYMENT_LINK
    return INTERVENTION_BY_CAUSE[diagnosis.cause]


def _budget_pool(action: ActionType) -> str | None:
    """Which budget the action draws on, or None when it consumes neither."""
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
    pool = _budget_pool(action)
    return 0.0 if pool is None else COST_BY_POOL[pool]


def index_score(event: FailureEvent, diagnosis: Diagnosis) -> tuple[float, dict]:
    """Expected recovered paise per budget unit consumed, plus a breakdown.

    Recovery is only worth what the diagnosis is worth, so every term is
    weighted by ``diagnosis.confidence``: the expected value is
    ``P(cause is right) * P(recovery | cause) * amount``. Confidence scales the
    ranking; ``MIN_ACTIONABLE_CONFIDENCE`` in :func:`allocate` is the hard floor.

    ``breakdown["contributions"]`` sums to the returned score.
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
) -> ProposedAction:
    return ProposedAction(
        event_id=event.event_id,
        action=action,
        rationale=(
            f"{diagnosis.cause.value} -> {action.value}: "
            f"p={breakdown['probability']:.2f} at confidence "
            f"{diagnosis.confidence:.2f} ({diagnosis.method}) on "
            f"{event.amount_paise} paise, index {score:.0f} per budget unit"
        ),
        expected_recovery_paise=breakdown["expected_recovery_paise"],
        scheduled_for=event.occurred_at
        + timedelta(hours=ACTION_DELAY_HOURS[diagnosis.cause]),
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
    """Rank every event by index score and spend both budgets in one pass.

    Attempts draw on ``retry_budget`` and contacts on ``contact_budget``. An
    action is granted only while its own pool has room; once that pool is dry
    the event falls through to SUPPRESS naming the budget that ran out, so the
    two pools cut at different places in the same ranking.

    A diagnosis below ``MIN_ACTIONABLE_CONFIDENCE`` never reaches either pool:
    scarce budget is not spent on a guess, however large the amount.

    Returns one ProposedAction per input event, in ranked order.
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
            actions.append(
                _suppressed(
                    event,
                    f"{SUPPRESSED_FOR_LOW_VALUE}: {diagnosis.cause.value} is not "
                    f"recoverable by an attempt (p={breakdown['probability']:.2f})",
                )
            )
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
        pool = _budget_pool(action)
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
        actions.append(_granted(event, diagnosis, action, score, breakdown))

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
    }
    return actions, stats
