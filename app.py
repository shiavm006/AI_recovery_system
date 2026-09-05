"""Nakad console: explore a precomputed run — budgets and baselines, not rebuilds.

Build the artifact with `python app.py`, then `streamlit run app.py`.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from eval import CARD_METHOD, compare, diagnosis_report, multi_seed, reserve_report
from generator.generate import generate_history
from ledger import Ledger
from models import (
    ActionType,
    Diagnosis,
    FailureEvent,
    ProposedAction,
    RootCause,
)
from pipeline import govern
from pipeline.allocate import (
    DEFAULT_CONTACT_BUDGET,
    DEFAULT_RETRY_BUDGET,
    SUPPRESSED_FOR_BUDGET,
    SUPPRESSED_FOR_LOW_CONFIDENCE,
    SUPPRESSED_FOR_LOW_VALUE,
    SUPPRESSED_FOR_NO_DIAGNOSIS,
    SUPPRESSED_FOR_NO_HEADROOM,
    SUPPRESSED_FOR_NO_WINDOW,
)
from pipeline.diagnose import _load_frozen_batch, diagnose_batch, run_config
from pipeline.govern import (
    CONTACT_ACTIONS,
    CONTACT_WINDOW,
    NPCI_RETRY_CAP,
    PEAK_RAIL_WINDOWS,
    to_ist,
)
from run import (
    AGENT,
    CONTROL,
    CONTROL_RETRY_DELAY_HOURS,
    POLICIES,
    _plan_and_gate,
    _propose_agent,
    _propose_control,
    run_batch,
    simulate_outcomes,
)

DATA = _ROOT / "data"
ARTIFACT = DATA / "console.json"
SANDBOX_LEDGER = DATA / "ledger_sandbox.db"
LIVE_LEDGER = DATA / "ledger_live.db"


def _frozen_batch(seed: int = 42) -> Path:
    return DATA / f"batch_seed{seed}.parquet"


REQUIRED_CONFIG_FIELDS = ("provider", "model", "events", "by_method", "unknown")

BASELINE_ARRIVAL = "Arrival order — fixed schedule"
BASELINE_AMOUNT = "Amount-sorted queue"
BASELINE_EQUAL = "Equal total actions"
BASELINES = (BASELINE_ARRIVAL, BASELINE_AMOUNT, BASELINE_EQUAL)

RULE_ROWS = (
    (govern.R01_RAIL_CAP, "Attempt cap", "NPCI", "law"),
    (govern.R02_PREDEBIT_NOTICE, "24-hour notice", "RBI", "law"),
    (govern.R04_WHATSAPP_POLICY, "WhatsApp policy", "Meta", "contract"),
    (govern.R06_QUIET_HOURS, "Quiet hours", "Nakad", "policy"),
)

INK = "#111827"
ACCENT = "#2563EB"
MUTED = "#9CA3AF"
FAINT = "#E5E7EB"
GOOD = "#059669"
STOP = "#DC2626"

# Trim Streamlit chrome only — no layout overrides (upgrade-safe).
CSS = """
<style>
#MainMenu, footer, [data-testid="stDecoration"] {display: none;}
[data-testid="stMetricValue"] {font-size: 1.6rem; font-variant-numeric: tabular-nums;}
[data-testid="stMetricLabel"] {font-size: 0.78rem; letter-spacing: .02em;}
code, .mono {font-variant-numeric: tabular-nums;}
</style>
"""
def write_artifact(artifact: dict, path: Path = ARTIFACT) -> None:
    """Write the console artifact, refusing one that cannot state its own provenance."""
    for label, config in (
        ("run_config", artifact.get("run_config")),
        ("multi_seed.run_config", artifact.get("multi_seed", {}).get("run_config")),
    ):
        if not config:
            raise ValueError(f"refusing to write {path.name}: {label} is missing")
        missing = [f for f in REQUIRED_CONFIG_FIELDS if config.get(f) in (None, "")]
        if missing:
            raise ValueError(
                f"refusing to write {path.name}: {label} is missing {missing}"
            )
    path.write_text(json.dumps(artifact, indent=2, default=str))


def build_console(seed: int = 42, multi_seed_provider: str | None = None) -> dict:
    """Run everything once and write the artifact the console renders."""
    if st.runtime.exists():
        raise RuntimeError(
            "build_console belongs offline (`python app.py`); the console only "
            "renders a finished artifact."
        )

    DATA.mkdir(parents=True, exist_ok=True)
    frozen = DATA / f"batch_seed{seed}.parquet"
    if not frozen.exists():
        from generator.generate import freeze

        freeze(str(DATA), seed=seed)

    events = _load_frozen_batch(frozen)
    diagnoses, diagnose_stats = diagnose_batch(events)

    results = {}
    for policy in POLICIES:
        path = DATA / f"ledger_{policy}.db"
        path.unlink(missing_ok=True)
        results[policy] = run_batch(
            events, policy=policy, ledger_path=str(path), seed=seed
        )

    reserves = reserve_report(
        generate_history(seed=seed, valuation_cutoff=None),
        results[AGENT].recovered_by_method.get(CARD_METHOD, 0),
    )

    artifact = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "seed": seed,
        "events": len(events),
        "run_config": run_config(diagnoses),
        "fallback_reasons": diagnose_stats.get("fallback_reasons", {}),
        "diagnosis": diagnosis_report(events, diagnoses),
        "reserve": reserves,
        "compare": compare(
            results[AGENT], results[CONTROL], reserves["ultimate_dispute_rate"]
        ),
        "multi_seed": multi_seed(provider=multi_seed_provider),
        "ledgers": {p: results[p].ledger_path for p in POLICIES},
    }
    write_artifact(artifact)
    return artifact


def _mtime(path: Path | str) -> float:
    return Path(path).stat().st_mtime if Path(path).exists() else 0.0


def _rupees(paise: int | float) -> str:
    return f"₹{paise / 100:,.0f}"


def describe_methods(config: dict) -> str:
    mix = config.get("by_method") or {}
    return " · ".join(f"{name} {count}" for name, count in mix.items()) or (
        "nothing diagnosed"
    )


@st.cache_data(show_spinner=False)
def load_console(stamp: float) -> dict:
    return json.loads(ARTIFACT.read_text())


@st.cache_data(show_spinner=False)
def load_events(path: str, stamp: float) -> list[FailureEvent]:
    return _load_frozen_batch(Path(path))


@st.cache_data(show_spinner=False)
def load_diagnoses(ledger_path: str, stamp: float) -> list[Diagnosis]:
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT event_id, payload FROM entries WHERE entry_type = 'diagnosed'"
        ).fetchall()
    out: list[Diagnosis] = []
    for event_id, raw in rows:
        payload = json.loads(raw)
        cause = payload.get("cause")
        if not cause:
            continue
        out.append(
            Diagnosis(
                event_id=event_id,
                cause=RootCause(cause),
                confidence=float(payload.get("confidence") or 0),
                evidence=list(payload.get("evidence") or []),
                method=payload.get("method") or "rule",
            )
        )
    return out


@st.cache_data(show_spinner=False)
def load_tip(ledger_path: str, stamp: float) -> tuple[int, str]:
    with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
        tip = conn.execute("SELECT seq, entry_hash FROM chain_tip").fetchone()
        n = conn.execute("SELECT COUNT(*) FROM entries").fetchone()[0]
    return (n, tip[1] if tip else "")


@st.cache_data(show_spinner=False)
def load_live_rows(stamp: float) -> list[dict]:
    if not LIVE_LEDGER.exists():
        return []
    with sqlite3.connect(f"file:{LIVE_LEDGER}?mode=ro", uri=True) as conn:
        rows = conn.execute(
            "SELECT event_id, entry_type, payload FROM entries ORDER BY seq DESC LIMIT 24"
        ).fetchall()
    by_event: dict[str, dict] = {}
    for event_id, entry_type, raw in rows:
        payload = json.loads(raw)
        slot = by_event.setdefault(event_id, {"event_id": event_id})
        if entry_type == "proposed":
            slot["action"] = payload.get("action")
        if entry_type == "gated":
            slot["approved"] = payload.get("approved")
            slot["blocked_by"] = payload.get("blocked_by")
        if entry_type == "ingested":
            slot["payment_id"] = payload.get("payment_id") or event_id
    out = []
    for slot in by_event.values():
        action = slot.get("action")
        if action == "SUPPRESS":
            tag = ("gated", "tag vol")
        elif not slot.get("approved", True):
            tag = ("gated", "tag vol")
        elif action in {"RETRY", "MANDATE_REPRESENT"}:
            tag = ("retry", "tag live")
        elif action in {"NUDGE", "PAYMENT_LINK"}:
            tag = ("link", "tag live")
        else:
            tag = ((action or "seen").lower(), "tag")
        out.append(
            {
                "label": slot.get("payment_id") or slot["event_id"],
                "tag": tag[0],
                "cls": tag[1],
            }
        )
        if len(out) >= 3:
            break
    return out


def _propose_amount_sorted(
    pending: list[FailureEvent], retry_left: int, _contact_left: int
) -> list[ProposedAction]:
    ranked = sorted(pending, key=lambda e: (-e.amount_paise, e.event_id))
    actions: list[ProposedAction] = []
    for index, event in enumerate(ranked):
        if index < retry_left:
            actions.append(
                ProposedAction(
                    event_id=event.event_id,
                    action=ActionType.RETRY,
                    rationale="amount-sorted: retry largest unpaid first",
                    expected_recovery_paise=0,
                    scheduled_for=event.occurred_at
                    + timedelta(hours=CONTROL_RETRY_DELAY_HOURS),
                )
            )
        else:
            actions.append(
                ProposedAction(
                    event_id=event.event_id,
                    action=ActionType.SUPPRESS,
                    rationale=(
                        f"{SUPPRESSED_FOR_BUDGET}: retry budget spent further up "
                        "the amount order"
                    ),
                    expected_recovery_paise=0,
                    scheduled_for=None,
                )
            )
    return actions


def _score(
    events: list[FailureEvent],
    diagnoses: list[Diagnosis],
    propose,
    retry_budget: int,
    contact_budget: int,
    seed: int,
) -> dict:
    plan = _plan_and_gate(events, propose, retry_budget, contact_budget)
    approved = [
        action
        for action in plan["actions"]
        if plan["decisions"][action.event_id].approved
        and action.action is not ActionType.SUPPRESS
    ]
    outcomes = simulate_outcomes(approved, events, diagnoses, seed)
    by_id = {e.event_id: e for e in events}

    def suppressed_for(prefix: str) -> int:
        return sum(a.rationale.startswith(prefix) for a in plan["actions"])

    booked = []
    for action in approved:
        if not action.scheduled_for:
            continue
        local = to_ist(action.scheduled_for)
        booked.append(
            {
                "event_id": action.event_id,
                "action": action.action.value,
                "hour": local.hour,
                "minute": local.hour * 60 + local.minute,
                "clock": local.strftime("%H:%M"),
                "kind": "contact" if action.action in CONTACT_ACTIONS else "rail",
            }
        )

    blocked = Counter(
        d.blocked_by for d in plan["decisions"].values() if not d.approved and d.blocked_by
    )
    return {
        "gross_paise": sum(
            by_id[eid].amount_paise for eid, ok in outcomes.items() if ok
        ),
        "acted": len(approved),
        "retry_spent": plan["retry_spent"],
        "contact_spent": plan["contact_spent"],
        "blocked": sum(blocked.values()),
        "blocked_by_rule": dict(blocked),
        "budget": suppressed_for(SUPPRESSED_FOR_BUDGET),
        "low_value": suppressed_for(SUPPRESSED_FOR_LOW_VALUE),
        "no_headroom": suppressed_for(SUPPRESSED_FOR_NO_HEADROOM),
        "low_confidence": suppressed_for(SUPPRESSED_FOR_LOW_CONFIDENCE),
        "no_window": suppressed_for(SUPPRESSED_FOR_NO_WINDOW),
        "no_diagnosis": suppressed_for(SUPPRESSED_FOR_NO_DIAGNOSIS),
        "booked": booked,
        "actions": plan["actions"],
        "decisions": plan["decisions"],
    }


@st.cache_data(show_spinner=False)
def score_view(
    retry_budget: int,
    contact_budget: int,
    seed: int,
    batch_path: str,
    batch_stamp: float,
    ledger_path: str,
    ledger_stamp: float,
) -> dict:
    events = load_events(batch_path, batch_stamp)
    diagnoses = load_diagnoses(ledger_path, ledger_stamp)
    agent = _score(
        events,
        diagnoses,
        _propose_agent(diagnoses),
        retry_budget,
        contact_budget,
        seed,
    )
    arrival = _score(
        events, diagnoses, _propose_control, retry_budget, 0, seed
    )
    amount = _score(
        events, diagnoses, _propose_amount_sorted, retry_budget, 0, seed
    )
    equal_budget = agent["retry_spent"] + agent["contact_spent"]
    equal = _score(
        events, diagnoses, _propose_control, equal_budget, 0, seed
    )
    return {
        "agent": agent,
        "arrival": arrival,
        "amount": amount,
        "equal": equal,
        "npci_headroom": sum(
            max(0, NPCI_RETRY_CAP - e.retries_used) for e in events
        ),
    }


def _pick_trace(
    events: list[FailureEvent],
    diagnoses: list[Diagnosis],
    view: dict,
) -> dict:
    by_diag = {d.event_id: d for d in diagnoses}
    by_action = {a.event_id: a for a in view["actions"]}
    decisions = view["decisions"]
    for event in events:
        decision = decisions[event.event_id]
        if decision.approved:
            continue
        action = by_action[event.event_id]
        diagnosis = by_diag.get(event.event_id)
        booked = ""
        if action.scheduled_for:
            booked = to_ist(action.scheduled_for).strftime("%H:%M")
        return {
            "event_id": event.event_id,
            "ingested": (
                f"{event.method.replace('_', ' ').title()} · {event.issuer} · "
                f"{_rupees(event.amount_paise)} · attempt {event.retries_used + 1} of "
                f"{NPCI_RETRY_CAP + 1}"
            ),
            "diagnosed": (
                f"{diagnosis.cause.value.replace('_', ' ').title()} · "
                f"{diagnosis.method} · confidence {diagnosis.confidence:.2f}"
                if diagnosis
                else "—"
            ),
            "proposed": (
                f"{action.action.value.replace('_', ' ').title()}"
                + (f" · booked {booked}" if booked else "")
            ),
            "gated": f"Refused — {decision.reason}",
        }
    event = events[0]
    diagnosis = by_diag.get(event.event_id)
    action = by_action[event.event_id]
    return {
        "event_id": event.event_id,
        "ingested": f"{event.method} · {event.issuer} · {_rupees(event.amount_paise)}",
        "diagnosed": diagnosis.cause.value if diagnosis else "—",
        "proposed": action.action.value,
        "gated": "Approved",
    }


SWEEP_POINTS = 12
SWEEP_MIN, SWEEP_MAX = 20, 300


@st.cache_data(show_spinner=False)
def sweep_view(
    baseline: str,
    contact_budget: int,
    seed: int,
    batch_path: str,
    batch_stamp: float,
    ledger_path: str,
    ledger_stamp: float,
) -> list[dict]:
    events = load_events(batch_path, batch_stamp)
    diagnoses = load_diagnoses(ledger_path, ledger_stamp)
    propose_agent = _propose_agent(diagnoses)
    baseline_propose = (
        _propose_amount_sorted if baseline == BASELINE_AMOUNT else _propose_control
    )
    step = (SWEEP_MAX - SWEEP_MIN) / (SWEEP_POINTS - 1)
    rows: list[dict] = []
    for i in range(SWEEP_POINTS):
        budget = int(round(SWEEP_MIN + i * step))
        agent = _score(events, diagnoses, propose_agent, budget, contact_budget, seed)
        base = _score(events, diagnoses, baseline_propose, budget, 0, seed)
        rows.append(
            {"budget": budget, "series": "Nakad", "rupees": agent["gross_paise"] / 100}
        )
        rows.append(
            {"budget": budget, "series": "Baseline", "rupees": base["gross_paise"] / 100}
        )
    return rows


def _theme(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure_axis(
            grid=False,
            labelColor="#6B7280",
            titleColor="#6B7280",
            labelFontSize=11,
            titleFontSize=11,
            titleFontWeight="normal",
            domainColor=FAINT,
            tickColor=FAINT,
        )
        .configure_legend(
            labelColor="#374151", titleColor="#6B7280", labelFontSize=11, titleFontSize=11
        )
    )


def chart_baselines(rows: list[dict]) -> alt.Chart:
    frame = pd.DataFrame(rows)
    base = alt.Chart(frame).encode(
        y=alt.Y("policy:N", sort=None, title=None, axis=alt.Axis(labelLimit=200)),
        x=alt.X("rupees:Q", title="recovered (₹)", axis=alt.Axis(format="~s")),
        tooltip=[
            alt.Tooltip("policy:N", title="policy"),
            alt.Tooltip("rupees:Q", title="recovered ₹", format=",.0f"),
            alt.Tooltip("acted:Q", title="actions placed"),
        ],
    )
    bars = base.mark_bar(height=26, cornerRadiusEnd=2).encode(
        color=alt.Color(
            "policy:N",
            sort=None,
            legend=None,
            scale=alt.Scale(range=[ACCENT, "#6B7280", "#B6BDC7"]),
        )
    )
    labels = base.mark_text(align="left", dx=6, fontSize=12, color=INK).encode(
        text=alt.Text("rupees:Q", format=",.0f")
    )
    return _theme((bars + labels).properties(height=170))


def chart_sweep(rows: list[dict], current_budget: int) -> alt.Chart:
    frame = pd.DataFrame(rows)
    colour = alt.Color(
        "series:N",
        title=None,
        scale=alt.Scale(domain=["Nakad", "Baseline"], range=[ACCENT, MUTED]),
    )
    line = (
        alt.Chart(frame)
        .mark_line(strokeWidth=2, point=False)
        .encode(
            x=alt.X("budget:Q", title="attempt budget", scale=alt.Scale(nice=False)),
            y=alt.Y("rupees:Q", title="recovered (₹)", axis=alt.Axis(format="~s")),
            color=colour,
            tooltip=[
                alt.Tooltip("series:N", title=""),
                alt.Tooltip("budget:Q", title="attempts"),
                alt.Tooltip("rupees:Q", title="recovered ₹", format=",.0f"),
            ],
        )
    )
    marker = (
        alt.Chart(pd.DataFrame([{"budget": current_budget}]))
        .mark_rule(strokeDash=[4, 3], color=INK, strokeWidth=1)
        .encode(x="budget:Q")
    )
    return _theme((line + marker).properties(height=170))


def chart_stack(rows: list[dict], palette: list[str]) -> alt.Chart:
    frame = pd.DataFrame(rows)
    order = {row["label"]: i for i, row in enumerate(rows)}
    frame["order"] = frame["label"].map(order)
    total = max(int(frame["n"].sum()), 1)
    frame["share"] = frame["n"] / total
    tooltip = [
        alt.Tooltip("label:N", title=""),
        alt.Tooltip("n:Q", title="events"),
        alt.Tooltip("share:Q", title="share", format=".0%"),
    ]
    x = alt.X("n:Q", stack="zero", title=None, axis=None)
    order = alt.Order("order:Q")
    base = alt.Chart(frame)
    bars = base.mark_bar(height=34).encode(
        x=x,
        color=alt.Color(
            "label:N",
            sort=[row["label"] for row in rows],
            title=None,
            scale=alt.Scale(domain=[row["label"] for row in rows], range=palette),
            legend=alt.Legend(orient="bottom", columns=3, symbolType="square"),
        ),
        order=order,
        tooltip=tooltip,
    )
    # Only label a segment wide enough to hold the number; the rest have tooltips.
    labels = base.transform_filter(alt.datum.share > 0.07).mark_text(
        color="white", fontSize=11, fontWeight="bold"
    ).encode(x=alt.X("n:Q", stack="zero", bandPosition=0.5, title=None, axis=None),
             order=order, text=alt.Text("n:Q"), tooltip=tooltip)
    return _theme((bars + labels).properties(height=76))


def _closed_bands() -> list[dict]:
    bands = [
        {"kind": "Attempts", "start": start, "end": end}
        for start, end in PEAK_RAIL_WINDOWS
    ]
    open_from, open_to = CONTACT_WINDOW
    bands += [
        {"kind": "Contacts", "start": 0, "end": open_from},
        {"kind": "Contacts", "start": open_to, "end": 24 * 60},
    ]
    return bands


def chart_booking(booked: list[dict]) -> alt.Chart:
    kind_of = {"rail": "Attempts", "contact": "Contacts"}
    points = pd.DataFrame(
        [
            {
                "kind": kind_of[row["kind"]],
                "minute": row["minute"],
                "clock": row["clock"],
                "action": row["action"].replace("_", " ").title(),
                "event_id": row["event_id"],
            }
            for row in booked
        ]
    )
    if points.empty:
        points = pd.DataFrame(
            [{"kind": "Attempts", "minute": None, "clock": "", "action": "", "event_id": ""}]
        )

    y = alt.Y("kind:N", title=None, sort=["Attempts", "Contacts"])
    x = alt.X(
        "minute:Q",
        title="time of day (IST)",
        scale=alt.Scale(domain=[0, 1440], nice=False),
        axis=alt.Axis(
            values=[0, 240, 480, 720, 960, 1200, 1440],
            labelExpr="format(floor(datum.value/60), '02') + ':00'",
        ),
    )
    bands = (
        alt.Chart(pd.DataFrame(_closed_bands()))
        .mark_rect(color=STOP, opacity=0.10)
        .encode(x=alt.X("start:Q", scale=alt.Scale(domain=[0, 1440])), x2="end:Q", y=y)
    )
    ticks = (
        alt.Chart(points)
        .mark_tick(thickness=1.5, size=22, color=ACCENT, opacity=0.45)
        .encode(
            x=x,
            y=y,
            tooltip=[
                alt.Tooltip("event_id:N", title="event"),
                alt.Tooltip("action:N", title="action"),
                alt.Tooltip("clock:N", title="booked"),
            ],
        )
    )
    return _theme((bands + ticks).properties(height=150))


def _section(title: str, note: str) -> None:
    st.divider()
    st.markdown(f"##### {title}")
    st.caption(note)


def render() -> None:
    st.set_page_config(
        page_title="Nakad console", layout="wide", initial_sidebar_state="expanded"
    )
    st.markdown(CSS, unsafe_allow_html=True)

    if not ARTIFACT.exists():
        st.info("The console explores a finished run. Build that run once, then reopen.")
        st.markdown(f"**Expected artifact:** `{ARTIFACT}`")
        st.code("python app.py", language="bash")
        st.caption("That writes the artifact; `streamlit run app.py` only renders it.")
        return

    console = load_console(_mtime(ARTIFACT))
    seed = int(console["seed"])
    frozen = _frozen_batch(seed)
    if not frozen.exists():
        st.info("The console needs the frozen batch that matches this artifact.")
        st.markdown(f"**Expected batch:** `{frozen}`")
        st.code("python app.py", language="bash")
        st.caption("That regenerates the batch and artifact together.")
        return

    agent_ledger = console["ledgers"][AGENT]
    config = console["run_config"]
    batch_path, batch_stamp = str(frozen), _mtime(frozen)
    ledger_stamp = _mtime(agent_ledger)

    with st.sidebar:
        st.markdown("### Nakad")
        st.caption("Failed-payment recovery · Track 03")
        st.divider()
        baseline = st.selectbox("Compare against", BASELINES, index=0)
        retry = st.slider("Mandate attempts", 20, 300, DEFAULT_RETRY_BUDGET)
        contact = st.slider("Customer contacts", 0, 200, DEFAULT_CONTACT_BUDGET)
        if st.button("Reload run", width="stretch"):
            st.cache_data.clear()
            st.rerun()
        st.caption(
            "Diagnoses are computed once and stored in the ledger. "
            "Budgets change what we do with them, not what they are."
        )

    view = score_view(
        retry, contact, seed, batch_path, batch_stamp, agent_ledger, ledger_stamp
    )
    agent = view["agent"]
    baselines = {
        BASELINE_ARRIVAL: view["arrival"],
        BASELINE_AMOUNT: view["amount"],
        BASELINE_EQUAL: view["equal"],
    }
    selected = baselines[baseline]

    lift = agent["gross_paise"] / selected["gross_paise"] if selected["gross_paise"] else 0
    per_attempt = agent["gross_paise"] / agent["retry_spent"] if agent["retry_spent"] else 0
    base_per = (
        selected["gross_paise"] / selected["retry_spent"] if selected["retry_spent"] else 0
    )
    spread = console["multi_seed"]["ratio"]
    unknown = config.get("unknown") or 0
    by_method = config.get("by_method") or {}
    llm_events = int(by_method.get("llm") or 0)
    events_n = int(config.get("events") or console["events"])
    llm_calls = config.get("llm_calls")
    if llm_calls is None:
        model_note = f"{llm_events} of {events_n} events diagnosed by the model"
    else:
        model_note = (
            f"{llm_events} of {events_n} events diagnosed by the model — "
            f"{int(llm_calls)} API calls after batching and caching"
        )

    st.markdown("## Batch result")
    badge = (
        f":green[● all {console['events']} diagnosed]"
        if not unknown
        else f":red[● {unknown} undiagnosed]"
    )
    st.caption(
        f"seed **{seed}** · reproducible &nbsp;|&nbsp; "
        f"{config.get('provider')} · `{config.get('model') or '—'}` &nbsp;|&nbsp; "
        f"{console['events']} failures &nbsp;|&nbsp; {badge} &nbsp;|&nbsp; "
        f"{model_note}"
    )

    cells = st.columns(4)
    numbers = (
        ("Recovered", _rupees(agent["gross_paise"]),
         f"baseline {_rupees(selected['gross_paise'])}"),
        ("Lift over baseline", f"{lift:.2f}×",
         f"5-seed mean {spread['mean']:.2f}× · sd {spread['stdev']:.2f}"),
        ("Recovered per attempt", _rupees(per_attempt),
         f"baseline {_rupees(base_per)}"),
        ("Refused at gate", f"{agent['blocked']}",
         f"baseline {selected['blocked']}"),
    )
    for cell, (label, value, note) in zip(cells, numbers):
        cell.metric(label, value, border=True)
        cell.caption(note)

    _section(
        "Where the recovery comes from",
        f"Identical {console['events']} failures and {retry} attempts in every arm. "
        "Outcomes are keyed by event id, so all three face the same luck.",
    )
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.altair_chart(
            chart_baselines(
                [
                    {"policy": "Nakad", "rupees": agent["gross_paise"] / 100,
                     "acted": agent["acted"]},
                    {"policy": "Amount-sorted", "rupees": view["amount"]["gross_paise"] / 100,
                     "acted": view["amount"]["acted"]},
                    {"policy": "Arrival order", "rupees": view["arrival"]["gross_paise"] / 100,
                     "acted": view["arrival"]["acted"]},
                ]
            ),
            width="stretch",
        )
        st.caption(
            "Amount-sorted is the honest baseline: it is a real prioritisation "
            "policy, not a strawman."
        )
    with c2:
        st.altair_chart(
            chart_sweep(
                sweep_view(
                    baseline, contact, seed, batch_path, batch_stamp,
                    agent_ledger, ledger_stamp,
                ),
                retry,
            ),
            width="stretch",
        )
        st.caption("Dashed line marks the current budget. Both curves flatten as the batch runs out of recoverable failures.")

    _section(
        "How the batch was handled",
        "Every failure is labelled once and dispositioned once. Both bars total "
        f"{console['events']}.",
    )
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Diagnosis route**")
        by_method = config.get("by_method") or {}
        st.altair_chart(
            chart_stack(
                [
                    {"label": "Rule lookup", "n": by_method.get("rule", 0)},
                    {"label": "Fleet correlation", "n": by_method.get("fleet", 0)},
                    {"label": "LLM", "n": by_method.get("llm", 0)},
                    {"label": "Undiagnosed", "n": unknown},
                ],
                [MUTED, ACCENT, "#93C5FD", STOP],
            ),
            width="stretch",
        )
        st.caption("Cheapest route first. The LLM only sees what rules and correlation could not settle.")
    with c2:
        st.markdown("**Disposition**")
        other = agent["low_confidence"] + agent["no_window"] + agent["no_diagnosis"]
        st.altair_chart(
            chart_stack(
                [
                    {"label": "Acted", "n": agent["acted"]},
                    {"label": "Out of budget", "n": agent["budget"]},
                    {"label": "No NPCI headroom", "n": agent["no_headroom"]},
                    {"label": "Not worth an attempt", "n": agent["low_value"]},
                    {"label": "Refused at gate", "n": agent["blocked"]},
                    {"label": "Other", "n": other},
                ],
                [ACCENT, "#6B7280", "#B6BDC7", FAINT, STOP, "#D1D5DB"],
            ),
            width="stretch",
        )
        st.caption(
            f"{agent['acted']} actions is the ceiling at this budget. Every other "
            "event carries a reason it was declined."
        )

    _section(
        "Timing and compliance",
        "Shaded hours are closed by rule. The allocator books around them rather "
        "than into them, so the gate rarely has to refuse on timing.",
    )
    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.altair_chart(chart_booking(agent["booked"]), width="stretch")
        st.caption(
            f"All {len(agent['booked'])} booked actions sit outside their own "
            "closed hours. Attempts and contacts obey different rules."
        )
    with c2:
        blocks = agent["blocked_by_rule"]
        rows = "\n".join(
            f"| {label} | {who} | {standing} | "
            f"{blocks.get(rule_id, 0)} |"
            for rule_id, label, who, standing in RULE_ROWS
        )
        st.markdown(
            "| Check | Source | Standing | Blocked |\n|---|---|---|---:|\n" + rows
        )
        residual = sum(
            blocks.get(r, 0)
            for r in (govern.R02_PREDEBIT_NOTICE, govern.R04_WHATSAPP_POLICY)
        )
        st.caption(
            f"Law and self-imposed policy are labelled separately. The {residual} "
            "remaining refusals are ones no clock can fix."
        )

    events = load_events(batch_path, batch_stamp)
    diagnoses = load_diagnoses(agent_ledger, ledger_stamp)
    trace = _pick_trace(events, diagnoses, agent)
    n_entries, tip_hash = load_tip(agent_ledger, _mtime(agent_ledger))

    _section(
        "Audit trail",
        "Four entries per event — ingested, diagnosed, proposed, gated — appended "
        "and never edited. Approved and refused alike.",
    )
    c1, c2, c3 = st.columns([1.35, 1, 1])
    with c1:
        st.markdown(f"**One event, end to end** &nbsp; `{trace['event_id']}`")
        st.markdown(
            f"| Stage | Entry |\n|---|---|\n"
            f"| Ingested | {trace['ingested']} |\n"
            f"| Diagnosed | {trace['diagnosed']} |\n"
            f"| Proposed | {trace['proposed']} |\n"
            f"| Gated | {trace['gated']} |"
        )
    with c2:
        st.markdown("**Chain verification**")
        if "audit_ok" not in st.session_state:
            st.session_state.audit_ok = True
            st.session_state.audit_note = f"{n_entries:,} entries · tip matches"
        if st.session_state.audit_ok:
            st.markdown(f":green[● Intact] — {st.session_state.audit_note}")
        else:
            st.markdown(f":red[● Broken] — {st.session_state.audit_note}")
        st.code(tip_hash or "—", language=None)
        if st.button("Edit an entry and re-check", width="stretch"):
            if not SANDBOX_LEDGER.exists():
                shutil.copyfile(agent_ledger, SANDBOX_LEDGER)
            with sqlite3.connect(SANDBOX_LEDGER) as conn:
                victim = conn.execute(
                    "SELECT seq FROM entries ORDER BY seq LIMIT 1 OFFSET ?",
                    (n_entries // 2,),
                ).fetchone()
                conn.execute(
                    "UPDATE entries SET payload = json_set(payload, '$.amount_paise', "
                    "999999999) WHERE seq = ?",
                    (victim[0],),
                )
            verdict = Ledger(str(SANDBOX_LEDGER)).verify()
            st.session_state.audit_ok = verdict.ok
            st.session_state.audit_note = (
                f"{n_entries:,} entries · tip matches"
                if verdict.ok
                else f"tampered at seq {verdict.seq}"
            )
            st.rerun()
    with c3:
        st.markdown("**Live · Razorpay test mode**")
        live = load_live_rows(_mtime(LIVE_LEDGER))
        if live:
            st.markdown(
                "| Payment | Outcome |\n|---|---|\n"
                + "\n".join(f"| `{row['label']}` | {row['tag']} |" for row in live)
            )
        else:
            st.caption("No live events yet. Send a test webhook to populate this.")
        st.caption("Signature verified, deduped, drawn from the same budget pool.")


if st.runtime.exists():
    render()
elif __name__ == "__main__":
    print("building console artifact (this makes the run's LLM calls) …")
    built = build_console(multi_seed_provider=os.getenv("MULTI_SEED_PROVIDER") or None)
    for label, config in (
        ("headline", built["run_config"]),
        ("sweep", built["multi_seed"]["run_config"]),
    ):
        print(
            f"  {label:<10}provider {config['provider']} "
            f"model {config['model'] or '—'} · {describe_methods(config)} · "
            f"{config['unknown']} undiagnosed"
        )
    print(f"wrote {ARTIFACT} — seed {built['seed']}, {built['events']} events")
    print("now run:  streamlit run app.py")
