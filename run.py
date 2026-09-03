from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable
from datetime import datetime, timedelta
from pathlib import Path

from ledger import Ledger
from models import (
    ActionType,
    BatchResult,
    Diagnosis,
    FailureEvent,
    GateDecision,
    ProposedAction,
    RootCause,
)
from pipeline import govern
from pipeline.allocate import (
    CONTACT_POOL,
    DEFAULT_CONTACT_BUDGET,
    DEFAULT_RETRY_BUDGET,
    RETRY_POOL,
    SUPPRESSED_FOR_BUDGET,
    SUPPRESSED_FOR_LOW_CONFIDENCE,
    SUPPRESSED_FOR_LOW_VALUE,
    SUPPRESSED_FOR_NO_DIAGNOSIS,
    SUPPRESSED_FOR_NO_HEADROOM,
    SUPPRESSED_FOR_NO_WINDOW,
    allocate,
    budget_pool,
    index_score,
    recovery_probability,
)
from pipeline.diagnose import diagnose_batch, run_config
from pipeline.govern import CONTACT_ACTIONS, GateContext

AGENT = "agent"
CONTROL = "control"
POLICIES = (AGENT, CONTROL)

ENTRY_INGESTED = "ingested"
ENTRY_DIAGNOSED = "diagnosed"
ENTRY_PROPOSED = "proposed"
ENTRY_GATED = "gated"
TRACE_ENTRY_TYPES = (ENTRY_INGESTED, ENTRY_DIAGNOSED, ENTRY_PROPOSED, ENTRY_GATED)

# What actually happened when an approved action fired. Every event gets the
# four trace entries above; only an approved action can get this fifth one, so
# it is not part of TRACE_ENTRY_TYPES.
#
# NOTHING WRITES THIS YET — the simulator computes outcomes in memory and
# throws them away. The schema exists because BASE_RECOVERY_PROBABILITY's whole
# claim is that its invented constants get replaced by observed rates, and that
# is impossible without a durable record of which attempts actually recovered.
ENTRY_EXECUTED = "executed"
ALL_ENTRY_TYPES = TRACE_ENTRY_TYPES + (ENTRY_EXECUTED,)


def write_execution_outcome(
    ledger: Ledger,
    event_id: str,
    action: ActionType,
    attempted_at: datetime,
    succeeded: bool,
    recovered_paise: int = 0,
    detail: str | None = None,
) -> None:
    """Append the outcome of one approved action.

    Grouping these by the diagnosed cause is what turns
    BASE_RECOVERY_PROBABILITY from a table of guesses into a measurement.
    """
    ledger.append(
        event_id,
        ENTRY_EXECUTED,
        {
            "action": action.value,
            "attempted_at": attempted_at,
            "succeeded": succeeded,
            "recovered_paise": recovered_paise,
            "detail": detail,
        },
    )

# The fixed offset the control arm retries at. Dunning tools ship a schedule
# like T+24h/T+72h; we model the first rung only, since the batch is one cycle.
CONTROL_RETRY_DELAY_HOURS = 24

# WhatsApp is the default contact channel: it is the dominant dunning channel
# in India and, unlike SMS and voice, it needs logged consent (govern r04)
# rather than a registered TRAI DLT template (r03) that we do not have.
CONTACT_CHANNEL = "whatsapp"


def gate_context_for(event: FailureEvent, action: ProposedAction) -> GateContext:
    """Build the gate context from the event, not from wishful defaults.

    ``now`` is the moment the action would actually fire, not wall-clock time.
    Two of the rules are clock-dependent — r01 freezes rails in peak IST
    windows and r06 confines contact to daytime — so evaluating against
    ``datetime.now()`` would make the same batch legal or illegal depending on
    when the script ran. Using ``scheduled_for`` asks the correct question:
    would this be permitted at the time we intend to do it?

    ``retries_used`` comes off the event. The rest of GateContext describes
    customer state that FailureEvent does not carry, so each default below is
    a decision:

    * ``last_notice_sent_at`` — None, FAILING CLOSED. With no proof a pre-debit
      notice went out, r02 blocks every MANDATE_REPRESENT. A gate that assumes
      the notice happened is not a gate.
    * ``consent_logged`` — derived from the mandate. A subscription exists only
      because the customer completed an authenticated UPI Autopay or e-NACH
      registration. That is authorisation to debit, and we treat a message
      about a failed debit on that mandate as a service message it covers. It
      is a proxy, not the real thing: marketing consent is a separate record in
      a CRM, and a mandate-less one-off payment gets no contact at all.
    * ``stop_requested`` / ``promise_to_pay`` / ``dispute_open`` — False. These
      are the one place we cannot fail closed: defaulting them True halts the
      whole batch and the run proves nothing. This is the widest gap between
      this harness and production, and all three live in a CRM feed we do not
      have.
    * ``contacts_this_cycle`` — 0. No rule reads it today; a real deployment
      would feed it from the messaging log.
    """
    return GateContext(
        now=action.scheduled_for or event.occurred_at,
        retries_used=event.retries_used,
        contacts_this_cycle=0,
        last_notice_sent_at=None,
        stop_requested=False,
        promise_to_pay=False,
        dispute_open=False,
        consent_logged=event.subscription_id is not None,
        channel=CONTACT_CHANNEL if action.action in CONTACT_ACTIONS else None,
    )


def _ground_truth(event: FailureEvent, cause: RootCause) -> Diagnosis:
    return Diagnosis(
        event_id=event.event_id,
        cause=cause,
        confidence=1.0,
        evidence=["simulator ground truth, not a pipeline output"],
        method="rule",
    )


def simulate_outcomes(
    actions: list[ProposedAction],
    events: list[FailureEvent],
    diagnoses: list[Diagnosis],
    seed: int,
) -> dict[str, bool]:
    """Flip one deterministic coin per event. ``actions`` are the ones the gate approved.

    The simulator is the world, not the agent, so it resolves against
    ``true_cause`` and only falls back to the diagnosed cause when there is no
    ground truth. This matters: if outcomes were drawn from what the agent
    *believed*, a confidently wrong diagnosis would be rewarded exactly as
    much as a correct one, and the control arm — which diagnoses nothing —
    could not be simulated at all. Reading the truth here is what makes the
    two arms comparable; the pipeline itself still never sees it.

    The draw is keyed by event_id rather than by draw order, so an event that
    recovers under one policy recovers under the other. Both arms face the same
    world and the difference between them is selection, not luck.

    Anything not acted on — suppressed, or blocked at the gate and therefore
    absent from ``actions`` — recovers nothing.
    """
    by_event = {diagnosis.event_id: diagnosis for diagnosis in diagnoses}
    acting = {
        action.event_id: action
        for action in actions
        if action.action is not ActionType.SUPPRESS
    }

    outcomes: dict[str, bool] = {}
    for event in events:
        if event.event_id not in acting:
            outcomes[event.event_id] = False
            continue
        cause = event.true_cause
        if cause is None:
            diagnosis = by_event.get(event.event_id)
            if diagnosis is None:
                raise ValueError(
                    f"cannot simulate {event.event_id}: no true_cause and no diagnosis"
                )
            cause = diagnosis.cause
        # ponytail: the wrong instrument is not penalised — a control RETRY on a
        # DEAD_MANDATE scores the full re-presentation probability even though a
        # plain retry cannot fix a dead mandate. That is generous to the
        # baseline, so the agent's margin is a lower bound. Upgrade path: a
        # (cause, action) recovery matrix instead of a per-cause vector.
        probability = recovery_probability(event, _ground_truth(event, cause))
        draw = random.Random(f"{seed}:{event.event_id}").random()
        outcomes[event.event_id] = draw < probability
    return outcomes


Propose = Callable[[list[FailureEvent], int, int], list[ProposedAction]]


def _propose_agent(diagnoses: list[Diagnosis]) -> Propose:
    def propose(
        pending: list[FailureEvent], retry_left: int, contact_left: int
    ) -> list[ProposedAction]:
        return allocate(pending, diagnoses, retry_left, contact_left)[0]

    return propose


def _propose_control(
    pending: list[FailureEvent], retry_left: int, _contact_left: int
) -> list[ProposedAction]:
    """Retry the first ``retry_left`` failures in arrival order, suppress the rest.

    No diagnosis and no ranking — that absence is the baseline. It never draws
    on the contact budget, because a fixed schedule has no reason to prefer a
    payment link over a retry.
    """
    actions: list[ProposedAction] = []
    for index, event in enumerate(pending):
        if index < retry_left:
            actions.append(
                ProposedAction(
                    event_id=event.event_id,
                    action=ActionType.RETRY,
                    rationale=(
                        f"control: fixed-schedule retry at "
                        f"T+{CONTROL_RETRY_DELAY_HOURS}h, no diagnosis"
                    ),
                    expected_recovery_paise=0,
                    scheduled_for=event.occurred_at
                    + timedelta(hours=CONTROL_RETRY_DELAY_HOURS),
                )
            )
            continue
        actions.append(
            ProposedAction(
                event_id=event.event_id,
                action=ActionType.SUPPRESS,
                rationale=(
                    f"{SUPPRESSED_FOR_BUDGET}: retry budget spent further up the "
                    "arrival order"
                ),
                expected_recovery_paise=0,
                scheduled_for=None,
            )
        )
    return actions


def _plan_and_gate(
    events: list[FailureEvent], propose: Propose, retry_budget: int, contact_budget: int
) -> dict:
    """Propose, gate, reclaim the budget the gate refused, and propose again.

    A blocked action places nothing, so it must not consume the budget it was
    granted — otherwise a run reports spending its whole allowance while the
    gate quietly discarded part of it, and the events just below the cut are
    denied an attempt that was never actually used. Each pass returns the
    refused units to the pool and re-proposes over the events that were held
    back for budget.

    Both policies get this. It is not an intelligence: a fixed-schedule tool
    that finds its retry rejected also moves to the next name on its list.
    Withholding reclamation from the control would let the agent place more
    attempts than the baseline, and the comparison has to differ only in which
    events are chosen.

    Terminates because a pass either finalises at least one event or the
    pending set is unchanged, which breaks the loop.
    """
    by_id = {event.event_id: event for event in events}
    final: dict[str, ProposedAction] = {}
    decisions: dict[str, GateDecision] = {}
    passes: dict[str, int] = {}
    remaining = {RETRY_POOL: retry_budget, CONTACT_POOL: contact_budget}

    pending = [event.event_id for event in events]
    current_pass = 0
    while True:
        current_pass += 1
        held_for_budget: list[str] = []
        for action in propose(
            [by_id[i] for i in pending], remaining[RETRY_POOL], remaining[CONTACT_POOL]
        ):
            event = by_id[action.event_id]
            decision = govern.evaluate(action, gate_context_for(event, action))
            final[action.event_id] = action
            decisions[action.event_id] = decision
            passes[action.event_id] = current_pass

            if action.action is ActionType.SUPPRESS:
                if action.rationale.startswith(SUPPRESSED_FOR_BUDGET):
                    held_for_budget.append(action.event_id)
                continue
            if decision.approved:
                pool = budget_pool(action.action)
                if pool is not None:
                    remaining[pool] -= 1

        if not held_for_budget or len(held_for_budget) == len(pending):
            break
        pending = held_for_budget

    return {
        "actions": [final[event.event_id] for event in events],
        "decisions": decisions,
        "passes": passes,
        "retry_spent": retry_budget - remaining[RETRY_POOL],
        "contact_spent": contact_budget - remaining[CONTACT_POOL],
        "allocation_passes": current_pass,
    }


def _write_trace(
    ledger: Ledger,
    event: FailureEvent,
    diagnosis: Diagnosis | None,
    action: ProposedAction,
    decision: GateDecision,
    score: float | None,
    allocation_pass: int,
) -> None:
    """Four entries per event, always in the same order, whatever the outcome."""
    ledger.append(
        event.event_id,
        ENTRY_INGESTED,
        {
            "payment_id": event.payment_id,
            "subscription_id": event.subscription_id,
            "amount_paise": event.amount_paise,
            "method": event.method,
            "issuer": event.issuer,
            "error_code": event.error_code,
            "error_reason": event.error_reason,
            "retries_used": event.retries_used,
            "occurred_at": event.occurred_at,
        },
    )
    ledger.append(
        event.event_id,
        ENTRY_DIAGNOSED,
        {
            "cause": diagnosis.cause.value if diagnosis else None,
            "method": diagnosis.method if diagnosis else "none",
            "confidence": diagnosis.confidence if diagnosis else None,
            "evidence": diagnosis.evidence if diagnosis else ["control policy does not diagnose"],
        },
    )
    ledger.append(
        event.event_id,
        ENTRY_PROPOSED,
        {
            "action": action.action.value,
            "index_score": score,
            "allocation_pass": allocation_pass,
            "rationale": action.rationale,
            "expected_recovery_paise": action.expected_recovery_paise,
            "scheduled_for": action.scheduled_for,
        },
    )
    ledger.append(
        event.event_id,
        ENTRY_GATED,
        {
            "approved": decision.approved,
            "blocked_by": decision.blocked_by,
            "reason": decision.reason,
            "rule_ids_passed": decision.rule_ids_passed,
        },
    )


def run_batch(
    events: list[FailureEvent],
    policy: str,
    ledger_path: str,
    retry_budget: int | None = None,
    contact_budget: int | None = None,
    seed: int = 42,
) -> BatchResult:
    """Thread one batch through plan, gate, ledger and outcome simulation.

    Both policies write every outcome to the ledger, approved and blocked
    alike: an audit trail that only records the actions you were allowed to
    take is not an audit trail.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    retry_budget = DEFAULT_RETRY_BUDGET if retry_budget is None else retry_budget
    contact_budget = DEFAULT_CONTACT_BUDGET if contact_budget is None else contact_budget

    if policy == AGENT:
        diagnoses, _ = diagnose_batch(events)
        propose = _propose_agent(diagnoses)
    else:
        diagnoses = []
        propose = _propose_control
    plan = _plan_and_gate(events, propose, retry_budget, contact_budget)

    by_diagnosis = {diagnosis.event_id: diagnosis for diagnosis in diagnoses}
    scores = {
        event.event_id: index_score(event, by_diagnosis[event.event_id])[0]
        for event in events
        if event.event_id in by_diagnosis
    }

    ledger = Ledger(ledger_path)
    for event, action in zip(events, plan["actions"]):
        _write_trace(
            ledger,
            event,
            by_diagnosis.get(event.event_id),
            action,
            plan["decisions"][event.event_id],
            scores.get(event.event_id),
            plan["passes"][event.event_id],
        )

    approved = [
        action
        for action in plan["actions"]
        if plan["decisions"][action.event_id].approved
    ]
    outcomes = simulate_outcomes(approved, events, diagnoses, seed)

    def suppressed_for(prefix: str) -> int:
        return sum(action.rationale.startswith(prefix) for action in plan["actions"])

    recovered_by_method: Counter[str] = Counter()
    for event in events:
        if outcomes[event.event_id]:
            recovered_by_method[event.method] += event.amount_paise

    return BatchResult(
        policy=policy,
        events_processed=len(events),
        actions_by_type=dict(
            Counter(action.action.value for action in plan["actions"])
        ),
        blocked_by_rule=dict(
            Counter(
                decision.blocked_by
                for decision in plan["decisions"].values()
                if not decision.approved
            )
        ),
        gross_recovered_paise=sum(
            event.amount_paise for event in events if outcomes[event.event_id]
        ),
        recovered_by_method=dict(recovered_by_method),
        retry_budget_spent=plan["retry_spent"],
        contact_budget_spent=plan["contact_spent"],
        allocation_passes=plan["allocation_passes"],
        suppressed_for_budget=suppressed_for(SUPPRESSED_FOR_BUDGET),
        suppressed_for_low_value=suppressed_for(SUPPRESSED_FOR_LOW_VALUE),
        suppressed_for_low_confidence=suppressed_for(SUPPRESSED_FOR_LOW_CONFIDENCE),
        suppressed_for_no_headroom=suppressed_for(SUPPRESSED_FOR_NO_HEADROOM),
        suppressed_for_no_window=suppressed_for(SUPPRESSED_FOR_NO_WINDOW),
        unknown_diagnoses=sum(d.cause is RootCause.UNKNOWN for d in diagnoses),
        suppressed_for_no_diagnosis=suppressed_for(SUPPRESSED_FOR_NO_DIAGNOSIS),
        ledger_path=ledger_path,
        # Control diagnoses nothing by design, so its block is empty rather
        # than absent — that is the baseline's defining property, not a gap.
        run_config=run_config(diagnoses),
    )


def _print_comparison(agent: BatchResult, control: BatchResult) -> None:
    def row(label: str, left: object, right: object) -> None:
        print(f"  {label:<32}{str(left):>16}{str(right):>16}")

    print(f"\n  {'':<32}{'AGENT':>16}{'CONTROL':>16}")
    print("  " + "-" * 64)
    row("events processed", agent.events_processed, control.events_processed)
    row("UNDIAGNOSED (cause=UNKNOWN)", agent.unknown_diagnoses, control.unknown_diagnoses)
    for action in ActionType:
        if agent.actions_by_type.get(action.value) or control.actions_by_type.get(
            action.value
        ):
            row(
                f"action {action.value}",
                agent.actions_by_type.get(action.value, 0),
                control.actions_by_type.get(action.value, 0),
            )
    row("blocked at gate", sum(agent.blocked_by_rule.values()), sum(control.blocked_by_rule.values()))
    for rule_id in govern.RULES:
        if agent.blocked_by_rule.get(rule_id) or control.blocked_by_rule.get(rule_id):
            row(
                f"  blocked {rule_id}",
                agent.blocked_by_rule.get(rule_id, 0),
                control.blocked_by_rule.get(rule_id, 0),
            )
    row("retry budget placed", agent.retry_budget_spent, control.retry_budget_spent)
    row("contact budget placed", agent.contact_budget_spent, control.contact_budget_spent)
    row("allocation passes", agent.allocation_passes, control.allocation_passes)
    row("suppressed for budget", agent.suppressed_for_budget, control.suppressed_for_budget)
    row("suppressed low value", agent.suppressed_for_low_value, control.suppressed_for_low_value)
    row("suppressed low confidence", agent.suppressed_for_low_confidence, control.suppressed_for_low_confidence)
    row("suppressed no headroom", agent.suppressed_for_no_headroom, control.suppressed_for_no_headroom)
    row("suppressed no window", agent.suppressed_for_no_window, control.suppressed_for_no_window)
    row("suppressed no diagnosis", agent.suppressed_for_no_diagnosis, control.suppressed_for_no_diagnosis)
    print("  " + "-" * 64)
    row("gross recovered (paise)", agent.gross_recovered_paise, control.gross_recovered_paise)
    row(
        "gross recovered (rupees)",
        f"{agent.gross_recovered_paise / 100:,.0f}",
        f"{control.gross_recovered_paise / 100:,.0f}",
    )
    lift = agent.gross_recovered_paise - control.gross_recovered_paise
    ratio = (
        agent.gross_recovered_paise / control.gross_recovered_paise
        if control.gross_recovered_paise
        else float("inf")
    )
    print(f"\n  agent recovers {lift / 100:,.0f} rupees more ({ratio:.2f}x control)")
    if agent.unknown_diagnoses:
        share = agent.unknown_diagnoses / agent.events_processed
        print(
            f"\n  DEGRADED: {agent.unknown_diagnoses} events ({share:.0%}) could not be "
            "diagnosed and were declined, not guessed at."
        )


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    frozen = root / "data" / "batch_seed42.parquet"
    if not frozen.exists():
        from generator.generate import freeze

        freeze(str(root / "data"), seed=42)

    from pipeline.diagnose import _load_frozen_batch

    batch = _load_frozen_batch(frozen)

    results = []
    for policy in POLICIES:
        path = root / "data" / f"ledger_{policy}.db"
        path.unlink(missing_ok=True)
        results.append(run_batch(batch, policy=policy, ledger_path=str(path)))

    _print_comparison(*results)

    print("\n  ledger verification")
    for result in results:
        check = Ledger(result.ledger_path).verify()
        entries = len(Ledger(result.ledger_path).read_all())
        detail = "intact" if check.ok else f"{check.failure.upper()} at seq {check.seq}"
        print(
            f"    {result.policy:<8} {entries:>5} entries "
            f"({entries // len(batch)} per event)  {detail}  {result.ledger_path}"
        )
