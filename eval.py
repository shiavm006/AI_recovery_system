from __future__ import annotations

import contextlib
import os
import statistics
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Callable, Hashable

import chainladder as cl
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from generator.generate import (
    AMOUNT_PAISE,
    VALUATION_MONTH,
    generate_batch,
    generate_history,
)
from models import BatchResult, Diagnosis, FailureEvent, RootCause
from pipeline.diagnose import diagnose_batch, run_config
from run import AGENT, CONTROL, POLICIES, run_batch

# generate_history draws this many payments and disputes ~2.5% of them, but
# only the disputed rows survive into the frame, so the denominator is not
# recoverable from the history itself. In production this is settled card
# volume from the payments table; here it is the generator's own exposure.
HISTORY_PAYMENTS = 50_000
HISTORY_EXPOSURE_PAISE = int(HISTORY_PAYMENTS * sum(AMOUNT_PAISE) / len(AMOUNT_PAISE))

CARD_METHOD = "card"


def observable_signature(event: FailureEvent) -> tuple[str, ...]:
    """The discrete error fields two events must share to be indistinguishable.

    Deliberately excludes the noisy numeric features. They carry real signal,
    so a ceiling computed with them included runs to 0.98 on 387 groups out of
    500 events — almost every event becomes its own group and the bound stops
    bounding anything. The coarse signature is the conservative reading: the
    honest claim is that accuracy above this line is being earned by the
    features, timing and issuer correlation, not that the line is impassable.
    """
    return (
        event.error_code,
        event.error_source,
        event.error_step,
        event.error_reason,
        event.method,
    )


def bayes_ceiling(
    events: list[FailureEvent],
    key: Callable[[FailureEvent], Hashable] = observable_signature,
) -> tuple[float, int]:
    """Best accuracy any classifier could reach on these inputs, and group count.

    Within a group of identical inputs, the most a classifier can do is answer
    with the most common true cause and be wrong about the rest. Summing the
    modal count across groups gives the irreducible ceiling.
    """
    groups: dict[Hashable, Counter] = {}
    for event in events:
        groups.setdefault(key(event), Counter())[event.true_cause] += 1
    if not events:
        return 1.0, 0
    return sum(max(g.values()) for g in groups.values()) / len(events), len(groups)


def diagnosis_report(events: list[FailureEvent], diagnoses: list[Diagnosis]) -> dict:
    """Confusion matrix, per-class precision and recall, accuracy vs the ceiling."""
    predicted_by_id = {d.event_id: d.cause.value for d in diagnoses}
    truth = [e.true_cause.value for e in events]
    predicted = [
        predicted_by_id.get(e.event_id, RootCause.UNKNOWN.value) for e in events
    ]

    labels = [cause.value for cause in RootCause]
    matrix = confusion_matrix(truth, predicted, labels=labels)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, zero_division=0
    )

    correct = sum(t == p for t, p in zip(truth, predicted))
    answered = [
        (t, p) for t, p in zip(truth, predicted) if p != RootCause.UNKNOWN.value
    ]
    answered_correct = sum(t == p for t, p in answered)

    accuracy = correct / len(events) if events else 0.0
    ceiling, groups = bayes_ceiling(events)
    fine_ceiling, fine_groups = bayes_ceiling(
        events,
        key=lambda e: observable_signature(e)
        + (e.days_since_mandate_created // 30, e.day_of_month, e.issuer),
    )

    return {
        "events": len(events),
        "labels": labels,
        "confusion_matrix": matrix.tolist(),
        "per_class": {
            label: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "accuracy_all_events": round(accuracy, 4),
        "accuracy_when_answered": round(
            answered_correct / len(answered) if answered else 0.0, 4
        ),
        "answered": len(answered),
        "undiagnosed": len(events) - len(answered),
        "bayes_ceiling": round(ceiling, 4),
        "bayes_ceiling_groups": groups,
        "share_of_ceiling": round(accuracy / ceiling, 4) if ceiling else 0.0,
        "fine_ceiling": round(fine_ceiling, 4),
        "fine_ceiling_groups": fine_groups,
    }

def _by_origin(triangle_like) -> pd.Series:
    frame = triangle_like.to_frame(origin_as_datetime=False)
    series = frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame
    return series.fillna(0.0)


def reserve_report(
    history: pd.DataFrame,
    batch_recovered_paise: int,
    exposure_paise: int = HISTORY_EXPOSURE_PAISE,
) -> dict:
    """Chain-ladder the dispute triangle and net the batch's recovery against it.

    Recovering money is not the same as keeping it: a card payment recovered
    today can be charged back for months afterwards, and the disputes that
    have not arrived yet are exactly what a reserve model is for.

    ``history`` may contain post-valuation disputes. If it does they are held
    out of the fit and used to grade the projection, which is a reserve model
    checked against known truth — normally impossible, and only available here
    because the future was seeded. Pass an observed-only frame (as production
    would) and the grading section comes back empty.

    ``batch_recovered_paise`` should be card volume only; chargeback rights are
    a card-network mechanism.
    """
    disputes = history.copy()
    disputes["cohort_month"] = pd.to_datetime(disputes["cohort_month"])
    disputes["dispute_month"] = pd.to_datetime(disputes["dispute_month"])

    observed = disputes.loc[disputes["dispute_month"] <= VALUATION_MONTH]
    held_out = disputes.loc[disputes["dispute_month"] > VALUATION_MONTH]

    triangle = cl.Triangle(
        data=observed,
        origin="cohort_month",
        development="dispute_month",
        columns="amount_paise",
        cumulative=False,
    ).incr_to_cum()
    model = cl.Chainladder().fit(triangle)

    projected_ibnr = _by_origin(model.ibnr_)
    ultimate = _by_origin(model.ultimate_)
    reported_paise = float(observed["amount_paise"].sum())
    ultimate_paise = float(ultimate.sum())

    # Ultimate rather than reported: reserving on what has already arrived
    # under-books every cohort that is still developing.
    ultimate_rate = ultimate_paise / exposure_paise if exposure_paise else 0.0
    clawback = round(batch_recovered_paise * ultimate_rate)

    truth_by_cohort = (
        held_out.groupby(held_out["cohort_month"].dt.to_period("M"))["amount_paise"]
        .sum()
        .astype(float)
    )
    graded = []
    for origin, projected in projected_ibnr.items():
        period = pd.Period(str(origin), freq="M")
        actual = float(truth_by_cohort.get(period, 0.0))
        graded.append(
            {
                "cohort": str(period),
                "projected_ibnr_paise": round(float(projected)),
                "true_ibnr_paise": round(actual),
                "error_paise": round(float(projected) - actual),
            }
        )

    total_projected = sum(row["projected_ibnr_paise"] for row in graded)
    total_true = sum(row["true_ibnr_paise"] for row in graded)

    triangle_frame = triangle.to_frame(origin_as_datetime=False)
    triangle_frame.index = [str(origin) for origin in triangle_frame.index]
    # chainladder pads the factor row out to its internal grid; past the last
    # development age the triangle actually has, every factor is a placeholder
    # 1.0 that would read as a fitted result.
    ldf_row = model.ldf_.to_frame(origin_as_datetime=False).iloc[0]

    return {
        # Shipped as split-orient records so the console can rebuild the frame
        # without re-fitting the model on every rerender.
        "triangle": triangle_frame.to_dict("split"),
        "ldf": {
            str(age): round(float(factor), 4)
            for age, factor in ldf_row.items()
        },
        "ldf_fitted_ages": len(triangle_frame.columns) - 1,
        "cohorts": int(observed["cohort_month"].dt.to_period("M").nunique()),
        "observed_disputes": len(observed),
        "held_out_disputes": len(held_out),
        "reported_paise": round(reported_paise),
        "ultimate_paise": round(ultimate_paise),
        "exposure_paise": exposure_paise,
        "reported_dispute_rate": round(reported_paise / exposure_paise, 6)
        if exposure_paise
        else 0.0,
        "ultimate_dispute_rate": round(ultimate_rate, 6),
        "gross_recovered_paise": batch_recovered_paise,
        "projected_clawback_paise": clawback,
        "net_recovered_paise": batch_recovered_paise - clawback,
        "ibnr_by_cohort": graded,
        "total_projected_ibnr_paise": total_projected,
        "total_true_ibnr_paise": total_true,
        "total_ibnr_error_paise": total_projected - total_true,
        "total_ibnr_error_pct": round(
            100.0 * (total_projected - total_true) / total_true, 2
        )
        if total_true
        else None,
    }


def _executed(result: BatchResult) -> int:
    proposed = sum(
        count
        for action, count in result.actions_by_type.items()
        if action != "SUPPRESS"
    )
    return proposed - sum(result.blocked_by_rule.values())


def _policy_view(result: BatchResult, ultimate_dispute_rate: float | None) -> dict:
    card = result.recovered_by_method.get(CARD_METHOD, 0)
    clawback = (
        round(card * ultimate_dispute_rate) if ultimate_dispute_rate is not None else 0
    )
    placed = result.retry_budget_spent + result.contact_budget_spent
    return {
        "gross_recovered_paise": result.gross_recovered_paise,
        "card_recovered_paise": card,
        "projected_clawback_paise": clawback,
        "net_recovered_paise": result.gross_recovered_paise - clawback,
        "actions_executed": _executed(result),
        "actions_suppressed": result.actions_by_type.get("SUPPRESS", 0),
        "actions_blocked": sum(result.blocked_by_rule.values()),
        "blocked_by_rule": dict(result.blocked_by_rule),
        "retry_budget_spent": result.retry_budget_spent,
        "contact_budget_spent": result.contact_budget_spent,
        "recovery_per_retry_paise": round(
            result.gross_recovered_paise / result.retry_budget_spent
        )
        if result.retry_budget_spent
        else None,
        "recovery_per_action_paise": round(result.gross_recovered_paise / placed)
        if placed
        else None,
        "suppression_breakdown": {
            "budget": result.suppressed_for_budget,
            "low_value": result.suppressed_for_low_value,
            "low_confidence": result.suppressed_for_low_confidence,
            "no_headroom": result.suppressed_for_no_headroom,
            "no_window": result.suppressed_for_no_window,
            "no_diagnosis": result.suppressed_for_no_diagnosis,
        },
        "undiagnosed": result.unknown_diagnoses,
    }


def compare(
    agent: BatchResult,
    control: BatchResult,
    ultimate_dispute_rate: float | None = None,
) -> dict:
    """Side-by-side, on gross and on net.

    ``ultimate_dispute_rate`` comes from :func:`reserve_report`. Without it the
    net columns equal the gross ones and say so, rather than quietly implying
    no chargeback exposure exists.
    """
    left = _policy_view(agent, ultimate_dispute_rate)
    right = _policy_view(control, ultimate_dispute_rate)

    def lift(key: str) -> dict:
        a, c = left[key], right[key]
        # A per-retry figure is None when that arm placed no retries. There is
        # no lift over an arm that never played, so the comparison stays empty
        # rather than treating "did not attempt" as "attempted and got zero".
        comparable = a is not None and c is not None
        return {
            "agent": a,
            "control": c,
            "absolute": a - c if comparable else None,
            "ratio": round(a / c, 4) if comparable and c else None,
        }

    return {
        "net_is_modelled": ultimate_dispute_rate is not None,
        "ultimate_dispute_rate": ultimate_dispute_rate,
        "gross": lift("gross_recovered_paise"),
        "net": lift("net_recovered_paise"),
        "recovery_per_retry": lift("recovery_per_retry_paise"),
        "agent_detail": left,
        "control_detail": right,
    }


@contextlib.contextmanager
def _provider_override(provider: str | None):
    """Swap NAKAD_LLM_PROVIDER for the duration, restoring it after."""
    if provider is None:
        yield
        return
    previous = os.environ.get("NAKAD_LLM_PROVIDER")
    os.environ["NAKAD_LLM_PROVIDER"] = provider
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("NAKAD_LLM_PROVIDER", None)
        else:
            os.environ["NAKAD_LLM_PROVIDER"] = previous


def _merge_run_configs(configs: list[dict]) -> dict:
    """Collapse per-seed provenance into one block for the whole sweep."""
    if not configs:
        return {}
    totals: Counter[str] = Counter()
    for config in configs:
        totals.update(config["by_method"])
    return {
        "provider": configs[0]["provider"],
        "model": configs[0]["model"],
        "seeds": len(configs),
        "events": sum(config["events"] for config in configs),
        "by_method": dict(sorted(totals.items())),
        "unknown": sum(config["unknown"] for config in configs),
        "degraded": any(config["degraded"] for config in configs),
    }


def multi_seed(
    seeds: list[int] | tuple[int, ...] = (42, 43, 44, 45, 46),
    provider: str | None = None,
) -> dict:
    """Rerun both policies on a fresh batch per seed and spread the lift.

    One seed cannot distinguish a policy that works from a batch that happened
    to suit it.

    ``provider`` overrides NAKAD_LLM_PROVIDER for these runs only. Five seeds
    is five times the LLM spend of the headline run and will exhaust a free
    daily quota, so the stability check can be run with the provider off while
    the headline run keeps it live. The returned ``run_config`` records which,
    so the two are never read as one number.
    """
    runs = []
    configs: list[dict] = []
    with _provider_override(provider), tempfile.TemporaryDirectory() as workspace:
        for seed in seeds:
            events = generate_batch(seed=seed)
            results = {
                policy: run_batch(
                    events,
                    policy=policy,
                    ledger_path=str(Path(workspace) / f"{policy}_{seed}.db"),
                    seed=seed,
                )
                for policy in POLICIES
            }
            agent, control = results[AGENT], results[CONTROL]
            configs.append(agent.run_config)
            runs.append(
                {
                    "seed": seed,
                    "agent_gross_paise": agent.gross_recovered_paise,
                    "control_gross_paise": control.gross_recovered_paise,
                    "absolute_lift_paise": agent.gross_recovered_paise
                    - control.gross_recovered_paise,
                    "ratio": round(
                        agent.gross_recovered_paise / control.gross_recovered_paise, 4
                    )
                    if control.gross_recovered_paise
                    else None,
                    "undiagnosed": agent.unknown_diagnoses,
                }
            )

    config = _merge_run_configs(configs)
    ratios = [run["ratio"] for run in runs if run["ratio"] is not None]
    absolutes = [run["absolute_lift_paise"] for run in runs]

    def spread(values: list[float]) -> dict:
        return {
            "mean": round(statistics.fmean(values), 4),
            "stdev": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    return {
        "seeds": list(seeds),
        "runs": runs,
        "ratio": spread(ratios),
        "absolute_lift_paise": spread([float(v) for v in absolutes]),
        "run_config": config,
    }


def _rupees(paise: float) -> str:
    return f"{paise / 100:,.0f}"


def _header(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def _print_run_config(config: dict, label: str) -> None:
    """Print the provenance that every number below it depends on."""
    if not config:
        print(f"  {label:<26}(no diagnoses)")
        return
    provider = config["provider"]
    model = config["model"] or "—"
    mix = ", ".join(f"{name} {count}" for name, count in config["by_method"].items())
    state = (
        f"🟠 DEGRADED — {config['unknown']} of {config['events']} undiagnosed"
        if config["degraded"]
        else "✅ all events diagnosed"
    )
    print(f"  {label:<26}provider {provider}  model {model}")
    print(f"  {'':<26}{mix or 'nothing diagnosed'}")
    print(f"  {'':<26}{state}")


def _print_diagnosis(report: dict) -> None:
    _header("1. DIAGNOSIS — against true_cause")
    labels = report["labels"]
    width = max(len(label) for label in labels) + 2
    print("\n  confusion matrix (rows = true, columns = predicted)\n")
    print(" " * width + "".join(f"{label[:6]:>8}" for label in labels))
    for label, row in zip(labels, report["confusion_matrix"]):
        print(f"  {label:<{width - 2}}" + "".join(f"{value:>8}" for value in row))

    print(f"\n  {'class':<22}{'precision':>11}{'recall':>9}{'f1':>8}{'support':>9}")
    for label in labels:
        stats = report["per_class"][label]
        print(
            f"  {label:<22}{stats['precision']:>11.3f}{stats['recall']:>9.3f}"
            f"{stats['f1']:>8.3f}{stats['support']:>9}"
        )

    print(
        f"\n  accuracy over all {report['events']} events      "
        f"{report['accuracy_all_events']:.3f}"
    )
    print(
        f"  accuracy over the {report['answered']} answered      "
        f"{report['accuracy_when_answered']:.3f}  "
        f"({report['undiagnosed']} undiagnosed, excluded)"
    )
    print(
        f"  Bayes ceiling ({report['bayes_ceiling_groups']} signatures)   "
        f"{report['bayes_ceiling']:.3f}   "
        "— ambiguous signatures no model can split"
    )
    print(f"  share of the ceiling reached  {report['share_of_ceiling']:.1%}")


def _print_reserves(report: dict) -> None:
    _header("2. RESERVES — chain ladder on the dispute triangle")
    print(
        f"\n  {report['cohorts']} cohorts, {report['observed_disputes']} disputes "
        f"observed to {VALUATION_MONTH.date()}, "
        f"{report['held_out_disputes']} held out as truth"
    )
    print(
        f"  reported {_rupees(report['reported_paise'])} -> ultimate "
        f"{_rupees(report['ultimate_paise'])} rupees"
    )
    print(
        f"  dispute rate: reported {report['reported_dispute_rate']:.4%}, "
        f"ultimate {report['ultimate_dispute_rate']:.4%}"
    )

    print(f"\n  {'cohort':<12}{'projected':>14}{'true':>14}{'error':>14}")
    for row in report["ibnr_by_cohort"]:
        print(
            f"  {row['cohort']:<12}{_rupees(row['projected_ibnr_paise']):>14}"
            f"{_rupees(row['true_ibnr_paise']):>14}"
            f"{_rupees(row['error_paise']):>14}"
        )
    error_pct = report["total_ibnr_error_pct"]
    print(
        f"  {'TOTAL':<12}{_rupees(report['total_projected_ibnr_paise']):>14}"
        f"{_rupees(report['total_true_ibnr_paise']):>14}"
        f"{_rupees(report['total_ibnr_error_paise']):>14}"
        + (f"   ({error_pct:+.1f}%)" if error_pct is not None else "")
    )

    print(
        f"\n  agent card recovery {_rupees(report['gross_recovered_paise'])} "
        f"- clawback {_rupees(report['projected_clawback_paise'])} "
        f"= net {_rupees(report['net_recovered_paise'])} rupees"
    )


def _print_compare(report: dict, config: dict | None = None) -> None:
    _header("3. AGENT vs CONTROL")
    if config:
        _print_run_config(config, "run config")
    agent, control = report["agent_detail"], report["control_detail"]

    def row(label: str, left: object, right: object) -> None:
        print(f"  {label:<30}{str(left):>16}{str(right):>16}")

    def money_row(label: str, key: str) -> None:
        row(label, _rupees(agent[key] or 0), _rupees(control[key] or 0))

    print(f"\n  {'':<30}{'AGENT':>16}{'CONTROL':>16}")
    print("  " + "-" * 62)
    money_row("gross recovered (rupees)", "gross_recovered_paise")
    if report["net_is_modelled"]:
        money_row("projected clawback", "projected_clawback_paise")
        money_row("net recovered (rupees)", "net_recovered_paise")
    for label, key in (
        ("actions executed", "actions_executed"),
        ("actions suppressed", "actions_suppressed"),
        ("actions blocked at gate", "actions_blocked"),
    ):
        row(label, agent[key], control[key])
    for rule in sorted(set(agent["blocked_by_rule"]) | set(control["blocked_by_rule"])):
        row(
            f"  {rule}",
            agent["blocked_by_rule"].get(rule, 0),
            control["blocked_by_rule"].get(rule, 0),
        )
    row("retries placed", agent["retry_budget_spent"], control["retry_budget_spent"])
    row(
        "contacts placed",
        agent["contact_budget_spent"],
        control["contact_budget_spent"],
    )
    money_row("recovery per retry (rupees)", "recovery_per_retry_paise")
    print("  suppression by reason")
    for reason, count in agent["suppression_breakdown"].items():
        row(f"  {reason}", count, control["suppression_breakdown"][reason])
    print("  " + "-" * 62)

    gross, net = report["gross"], report["net"]
    print(
        f"\n  gross lift {_rupees(gross['absolute'])} rupees "
        f"({gross['ratio']}x control)"
    )
    if report["net_is_modelled"]:
        print(
            f"  net lift   {_rupees(net['absolute'])} rupees "
            f"({net['ratio']}x control)"
        )


def _print_multi_seed(report: dict) -> None:
    _header("4. STABILITY — the same comparison across seeds")
    # Printed again here because the sweep may run under a different provider
    # than the headline: five seeds is five times the LLM spend.
    _print_run_config(report.get("run_config", {}), "run config")
    print(f"\n  {'seed':<8}{'agent':>14}{'control':>14}{'lift':>14}{'ratio':>9}{'undiag':>9}")
    for run in report["runs"]:
        print(
            f"  {run['seed']:<8}{_rupees(run['agent_gross_paise']):>14}"
            f"{_rupees(run['control_gross_paise']):>14}"
            f"{_rupees(run['absolute_lift_paise']):>14}"
            f"{run['ratio']:>9}{run['undiagnosed']:>9}"
        )
    ratio, absolute = report["ratio"], report["absolute_lift_paise"]
    print(
        f"\n  ratio     mean {ratio['mean']:.2f}x  sd {ratio['stdev']:.2f}  "
        f"range {ratio['min']:.2f}–{ratio['max']:.2f}"
    )
    print(
        f"  lift      mean {_rupees(absolute['mean'])}  "
        f"sd {_rupees(absolute['stdev'])} rupees"
    )


if __name__ == "__main__":
    frozen_batch = _ROOT / "data" / "batch_seed42.parquet"
    if not frozen_batch.exists():
        from generator.generate import freeze

        freeze(str(_ROOT / "data"), seed=42)

    from pipeline.diagnose import _load_frozen_batch

    events = _load_frozen_batch(frozen_batch)
    diagnoses, _ = diagnose_batch(events)
    headline_config = run_config(diagnoses)

    _header("0. RUN CONFIG — what produced every number below")
    _print_run_config(headline_config, "headline run")

    _print_diagnosis(diagnosis_report(events, diagnoses))

    with tempfile.TemporaryDirectory() as workspace:
        results = {
            policy: run_batch(
                events, policy=policy, ledger_path=str(Path(workspace) / f"{policy}.db")
            )
            for policy in POLICIES
        }

    # The full history including post-valuation disputes, so the projection can
    # be graded. freeze() writes the observed-only cut, which cannot grade.
    history = generate_history(seed=42, valuation_cutoff=None)
    reserves = reserve_report(
        history, results[AGENT].recovered_by_method.get(CARD_METHOD, 0)
    )
    _print_reserves(reserves)

    _print_compare(
        compare(results[AGENT], results[CONTROL], reserves["ultimate_dispute_rate"]),
        headline_config,
    )
    # The sweep is five more full runs. Point it at a cheaper provider with
    # MULTI_SEED_PROVIDER=none when the daily quota will not stretch.
    _print_multi_seed(multi_seed(provider=os.getenv("MULTI_SEED_PROVIDER") or None))
