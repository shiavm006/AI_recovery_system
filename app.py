from __future__ import annotations

import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
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
from models import BatchResult
from pipeline import govern
from pipeline.diagnose import _load_frozen_batch, diagnose_batch, run_config
from pipeline.govern import CONTACT_WINDOW, PEAK_RAIL_WINDOWS, to_ist
from run import (
    AGENT,
    ALL_ENTRY_TYPES,
    CONTROL,
    ENTRY_PROPOSED,
    POLICIES,
    TRACE_ENTRY_TYPES,
    run_batch,
)

DATA = _ROOT / "data"
ARTIFACT = DATA / "console.json"
SANDBOX_LEDGER = DATA / "ledger_sandbox.db"

SUPPRESSION_LABELS = {
    "budget": "budget exhausted",
    "low_value": "expected value below cut",
    "no_headroom": "no NPCI headroom",
    "no_diagnosis": "no diagnosis (declined)",
    "low_confidence": "confidence below floor",
    "no_window": "no permitted window",
}

STANDING_BADGE = {
    govern.BINDING: "binding",
    govern.CONTRACT: "contract",
    govern.VOLUNTARY: "voluntary",
}


REQUIRED_CONFIG_FIELDS = ("provider", "model", "events", "by_method", "unknown")


def write_artifact(artifact: dict, path: Path = ARTIFACT) -> None:
    """Write the console artifact, refusing one that cannot state its own provenance.

    A shipped artifact is read as the system's performance long after whoever
    built it has forgotten which provider was configured that afternoon. An
    artifact that cannot say is worse than no artifact, so this raises rather
    than writing a file that will be quoted out of context.
    """
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
    """Run everything once and write the artifact the console renders.

    ``multi_seed_provider`` runs the five-seed stability sweep under a
    different provider than the headline run. The sweep is five more full
    batches and will exhaust a free daily quota; both provider states are
    recorded separately so the two sections are never read as one.
    """
    if st.runtime.exists():
        raise RuntimeError(
            "build_console makes LLM calls and must not run in the render path; "
            "run `python app.py` first."
        )

    DATA.mkdir(parents=True, exist_ok=True)
    frozen = DATA / f"batch_seed{seed}.parquet"
    if not frozen.exists():
        from generator.generate import freeze

        freeze(str(DATA), seed=seed)

    events = _load_frozen_batch(frozen)
    diagnoses, diagnose_stats = diagnose_batch(events)

    results: dict[str, BatchResult] = {}
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
        # Provenance for the headline run. The stability sweep carries its own
        # under multi_seed, because it may have run under a different provider.
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


# --------------------------------------------------------------------------
# load (render path — disk only)
# --------------------------------------------------------------------------


@st.cache_data(show_spinner=False)
def load_console(stamp: float) -> dict:
    return json.loads(ARTIFACT.read_text())


@st.cache_data(show_spinner=False)
def load_ledger(path: str, stamp: float) -> pd.DataFrame:
    """Flatten a ledger to a frame. ``stamp`` is the file mtime, so tampering
    with the database invalidates the cache rather than serving stale rows."""
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as conn:
        frame = pd.read_sql_query(
            "SELECT seq, timestamp, event_id, entry_type, payload, entry_hash "
            "FROM entries ORDER BY seq",
            conn,
        )
    frame["payload"] = frame["payload"].map(json.loads)
    return frame


def _mtime(path: Path | str) -> float:
    return Path(path).stat().st_mtime if Path(path).exists() else 0.0


def _rupees(paise: float | None) -> str:
    return "—" if paise is None else f"₹{paise / 100:,.0f}"


def describe_methods(config: dict) -> str:
    """"rule 313 · fleet 63 · llm 124", or a plain statement of nothing."""
    mix = config.get("by_method") or {}
    return " · ".join(f"{name} {count}" for name, count in mix.items()) or (
        "nothing diagnosed"
    )


def booked_times(ledger_frame: pd.DataFrame) -> pd.DataFrame:
    """Every action the allocator actually booked, as an IST minute of day."""
    rows = []
    for _, row in ledger_frame.loc[
        ledger_frame["entry_type"] == ENTRY_PROPOSED
    ].iterrows():
        stamp = row["payload"].get("scheduled_for")
        action = row["payload"].get("action")
        if not stamp or action == "SUPPRESS":
            continue
        local = to_ist(datetime.fromisoformat(stamp))
        rows.append(
            {
                "event_id": row["event_id"],
                "action": action,
                "minute": local.hour * 60 + local.minute,
                "clock": local.strftime("%H:%M"),
                "kind": "contact" if action in {"NUDGE", "PAYMENT_LINK"} else "rail",
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# tabs
# --------------------------------------------------------------------------


def tab_scoreboard(console: dict) -> None:
    report = console["compare"]
    agent, control = report["agent_detail"], report["control_detail"]

    st.subheader("Agent versus control")
    rows = [
        ("Gross recovered", _rupees(agent["gross_recovered_paise"]), _rupees(control["gross_recovered_paise"])),
        ("Projected clawback", _rupees(agent["projected_clawback_paise"]), _rupees(control["projected_clawback_paise"])),
        ("Net of reserve", _rupees(agent["net_recovered_paise"]), _rupees(control["net_recovered_paise"])),
        ("Recovery per retry placed", _rupees(agent["recovery_per_retry_paise"]), _rupees(control["recovery_per_retry_paise"])),
        ("Retries placed", agent["retry_budget_spent"], control["retry_budget_spent"]),
        ("Contacts placed", agent["contact_budget_spent"], control["contact_budget_spent"]),
    ]
    st.dataframe(
        pd.DataFrame(rows, columns=["metric", "agent", "control"]).set_index("metric"),
        width="stretch",
    )

    # The mean leads and the single seed trails, deliberately. One seed cannot
    # tell a policy that works from a batch that happened to suit it, and 2.40×
    # is the flattering end of a range that runs down to 1.86×.
    spread = console["multi_seed"]["ratio"]
    seeds = len(console["multi_seed"]["seeds"])
    lift = st.columns(3)
    lift[0].metric(f"Lift, mean of {seeds} seeds", f"{spread['mean']:.2f}×", f"σ {spread['stdev']:.2f}", delta_color="off")
    lift[1].metric("Range across seeds", f"{spread['min']:.2f}–{spread['max']:.2f}×")
    lift[2].metric(f"This seed ({console['seed']}) alone", f"{report['gross']['ratio']:.2f}×")
    st.caption(
        f"Seed {console['seed']} is one draw, and it is the favourable end of the "
        f"range. The headline is {spread['mean']:.2f}× ± {spread['stdev']:.2f}."
    )

    st.divider()
    st.subheader("Lift by seed")
    sweep = console["multi_seed"]["run_config"]
    st.caption(
        f"Sweep provider **{sweep['provider']}** (`{sweep['model'] or '—'}`) · "
        f"{describe_methods(sweep)} across {sweep['seeds']} seeds · "
        + (
            f"🟠 {sweep['unknown']} of {sweep['events']} undiagnosed"
            if sweep["degraded"]
            else "✅ all events diagnosed"
        )
    )
    runs = pd.DataFrame(console["multi_seed"]["runs"])
    runs["seed"] = runs["seed"].astype(str)
    base = alt.Chart(runs)
    points = base.mark_circle(size=220, color="#1f77b4").encode(
        x=alt.X("ratio:Q", title="agent ÷ control", scale=alt.Scale(zero=False)),
        y=alt.Y("seed:N", title="seed"),
        tooltip=["seed", "ratio", "agent_gross_paise", "control_gross_paise"],
    )
    mean_rule = (
        alt.Chart(pd.DataFrame({"m": [spread["mean"]]}))
        .mark_rule(color="#d62728", strokeWidth=2)
        .encode(x="m:Q")
    )
    mean_text = (
        alt.Chart(pd.DataFrame({"m": [spread["mean"]], "label": [f"mean {spread['mean']:.2f}×"]}))
        .mark_text(align="left", baseline="top", dx=6, color="#d62728", fontWeight="bold")
        .encode(x="m:Q", y=alt.value(4), text="label:N")
    )
    parity = (
        alt.Chart(pd.DataFrame({"m": [1.0]}))
        .mark_rule(color="#999", strokeDash=[4, 4])
        .encode(x="m:Q")
    )
    st.altair_chart((parity + points + mean_rule + mean_text).properties(height=200))

    st.divider()
    st.subheader("Where the suppressions go")
    st.caption(
        "Suppression is arithmetic, not timidity: the NPCI budget is finite and "
        "every bar below is a named reason for not spending it here."
    )
    stacked = pd.DataFrame(
        [
            {
                "policy": name,
                "reason": SUPPRESSION_LABELS.get(reason, reason),
                "events": count,
            }
            for name, detail in (("agent", agent), ("control", control))
            for reason, count in detail["suppression_breakdown"].items()
            if agent["suppression_breakdown"][reason]
            or control["suppression_breakdown"][reason]
        ]
    )
    st.altair_chart(
        alt.Chart(stacked)
        .mark_bar()
        .encode(
            x=alt.X("events:Q", title="events suppressed", stack="zero"),
            y=alt.Y("policy:N", title=None),
            color=alt.Color("reason:N", title="reason", scale=alt.Scale(scheme="tableau10")),
            tooltip=["policy", "reason", "events"],
        )
        .properties(height=160)
    )


def tab_compliance(console: dict) -> None:
    report = console["compare"]
    agent_blocks = report["agent_detail"]["blocked_by_rule"]
    control_blocks = report["control_detail"]["blocked_by_rule"]

    st.subheader("The seven rules, and what actually binds us")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "rule": rule_id,
                    "standing": STANDING_BADGE[govern.RULE_STANDING[rule_id][0]],
                    "source": govern.RULE_STANDING[rule_id][1],
                    "what it does": description,
                    "agent blocks": agent_blocks.get(rule_id, 0),
                    "control blocks": control_blocks.get(rule_id, 0),
                }
                for rule_id, description in govern.RULES.items()
            ]
        ).set_index("rule"),
        width="stretch",
    )
    st.caption(
        "Two of the seven are not law. r06's contact hours are borrowed from RBI "
        "recovery-agent norms that bind lenders, not merchants; standing down for "
        "a promise-to-pay is restraint nobody requires. They are marked as such "
        "because claiming otherwise is the kind of thing that gets checked."
    )

    st.divider()
    st.subheader("The agent schedules around the gate, not into it")
    booked = booked_times(load_ledger(console["ledgers"][AGENT], _mtime(console["ledgers"][AGENT])))
    if booked.empty:
        st.info("No booked actions in this run.")
        return

    # The two constraints are drawn in separate lanes rather than layered. They
    # genuinely overlap in time — contact hours run straight through the
    # afternoon rail freeze — and stacking translucent bands over one plot turns
    # the overlap into a third colour that means nothing.
    scale = alt.Scale(domain=[0, 24 * 60], nice=False)
    axis = alt.Axis(
        values=list(range(0, 24 * 60 + 1, 120)),
        # utcFormat, not timeFormat: the minute-of-day is already IST, so
        # letting the browser apply its own offset would shift every label.
        labelExpr="utcFormat(datum.value * 60 * 1000, '%H:%M')",
        title="time of day (IST)",
    )

    def lane(kind: str, windows: list[tuple[int, int]], shade: str, bar: str, title: str):
        band = alt.Chart(pd.DataFrame(windows, columns=["start", "end"])).mark_rect(
            color=shade, opacity=0.25
        ).encode(x=alt.X("start:Q", scale=scale, axis=axis), x2="end:Q")
        bars = (
            alt.Chart(booked.loc[booked["kind"] == kind])
            .mark_bar(color=bar, size=7)
            .encode(
                x=alt.X("minute:Q", scale=scale, axis=axis, bin=alt.Bin(step=10)),
                y=alt.Y("count():Q", title="actions booked"),
                tooltip=[alt.Tooltip("clock:N", title="booked"), alt.Tooltip("count():Q")],
            )
        )
        return (band + bars).properties(height=230, title=title, width=920)

    st.altair_chart(
        alt.vconcat(
            lane(
                "rail",
                list(PEAK_RAIL_WINDOWS),
                "#d62728",
                "#1f77b4",
                "Rail actions — red is the r01 peak freeze, and nothing is booked inside it",
            ),
            lane(
                "contact",
                [CONTACT_WINDOW],
                "#2ca02c",
                "#ff7f0e",
                "Contact actions — green is the r06 permitted window, and nothing is booked outside it",
            ),
            spacing=30,
        )
    )

    busiest = (
        booked.groupby(["clock", "action"]).size().sort_values(ascending=False).head(3)
    )
    st.caption(
        "Not luck: the scheduler reads the same window constants the gate "
        "enforces, so it books for the minute a freeze lifts rather than walking "
        "into one. Busiest slots — "
        + ", ".join(
            f"**{clock} {action}** ({count})" for (clock, action), count in busiest.items()
        )
        + ". 21:31 is the first legal minute after the evening rail freeze and "
        "08:00 is the first legal minute of contact hours."
    )
    st.caption(
        "What that leaves is the point: every remaining block is substantive "
        "rather than a timing accident — "
        + ", ".join(f"**{rule}** ({count})" for rule, count in sorted(agent_blocks.items()))
        + ". r02 is a genuine compliance refusal: no pre-debit notice on file, so "
        "the re-presentation does not go out."
    )


def tab_ledger(console: dict) -> None:
    st.subheader("Every event, four entries, whatever the outcome")
    policy = st.radio("ledger", POLICIES, horizontal=True, label_visibility="collapsed")
    path = console["ledgers"][policy]
    frame = load_ledger(path, _mtime(path))

    controls = st.columns([2, 2])
    query = controls[0].text_input("Filter by event id", placeholder="evt_…")
    kinds = controls[1].multiselect(
        "Entry type", ALL_ENTRY_TYPES, default=list(TRACE_ENTRY_TYPES)
    )

    view = frame.loc[frame["entry_type"].isin(kinds)]
    if query:
        view = view.loc[view["event_id"].str.contains(query, case=False, regex=False)]

    st.caption(f"{len(view)} of {len(frame)} entries")
    st.dataframe(
        view.assign(
            payload=view["payload"].map(lambda p: json.dumps(p, default=str)),
            entry_hash=view["entry_hash"].str.slice(0, 12) + "…",
        ).set_index("seq"),
        width="stretch",
        height=380,
    )

    st.divider()
    st.subheader("Tamper-evidence, live")
    st.caption(
        "This runs against a throwaway copy, so the demonstration cannot damage "
        "the real trail. Tamper-evident is not tamper-proof: anyone who can "
        "write the file can rewrite every row and recompute the tip. Real "
        "detection needs the tip published somewhere this process cannot reach."
    )

    if not SANDBOX_LEDGER.exists():
        shutil.copyfile(path, SANDBOX_LEDGER)

    buttons = st.columns(4)
    if buttons[0].button("Verify", width="stretch"):
        st.session_state["verdict"] = Ledger(str(SANDBOX_LEDGER)).verify()
    if buttons[1].button("Edit one payload", width="stretch"):
        with sqlite3.connect(SANDBOX_LEDGER) as conn:
            victim = conn.execute(
                "SELECT seq FROM entries ORDER BY seq LIMIT 1 OFFSET ?",
                (len(frame) // 2,),
            ).fetchone()
            conn.execute(
                "UPDATE entries SET payload = json_set(payload, '$.amount_paise', 999999999) "
                "WHERE seq = ?",
                (victim[0],),
            )
        st.session_state["verdict"] = Ledger(str(SANDBOX_LEDGER)).verify()
        st.session_state["note"] = f"edited the payload of seq {victim[0]}"
    if buttons[2].button("Delete the tail", width="stretch"):
        with sqlite3.connect(SANDBOX_LEDGER) as conn:
            conn.execute(
                "DELETE FROM entries WHERE seq > (SELECT MAX(seq) - 5 FROM entries)"
            )
        st.session_state["verdict"] = Ledger(str(SANDBOX_LEDGER)).verify()
        st.session_state["note"] = "deleted the last 5 entries"
    if buttons[3].button("Reset copy", width="stretch"):
        SANDBOX_LEDGER.unlink(missing_ok=True)
        shutil.copyfile(path, SANDBOX_LEDGER)
        st.session_state["verdict"] = Ledger(str(SANDBOX_LEDGER)).verify()
        st.session_state["note"] = "restored from the real ledger"

    verdict = st.session_state.get("verdict")
    if verdict is not None:
        if note := st.session_state.get("note"):
            st.write(f"Last action: {note}")
        if verdict.ok:
            st.success("verify() → intact. Every link recomputes and the tip agrees.")
        else:
            st.error(
                f"verify() → **{verdict.failure.upper()}** at seq {verdict.seq}. "
                + (
                    "Rows are missing: the chain is shorter than the tip says."
                    if verdict.failure != "tampered"
                    else "A row's contents no longer match its hash."
                )
            )
        tip = sqlite3.connect(SANDBOX_LEDGER).execute(
            "SELECT seq, entry_hash FROM chain_tip"
        ).fetchone()
        if tip:
            st.code(f"chain_tip  seq={tip[0]}  entry_hash={tip[1]}", language=None)


def tab_diagnosis(console: dict) -> None:
    report = console["diagnosis"]
    labels = report["labels"]

    top = st.columns(4)
    top[0].metric("Accuracy, all events", f"{report['accuracy_all_events']:.1%}")
    top[1].metric("Accuracy when answered", f"{report['accuracy_when_answered']:.1%}")
    top[2].metric("Bayes ceiling", f"{report['bayes_ceiling']:.1%}")
    top[3].metric("Share of ceiling", f"{report['share_of_ceiling']:.1%}")
    st.caption(
        f"{report['undiagnosed']} events carry **UNKNOWN** — the pipeline declined "
        f"rather than guessed, so they count against the all-events figure and are "
        f"excluded from the answered one. The ceiling is the best score any model "
        f"could reach on {report['bayes_ceiling_groups']} observable signatures, "
        "some of which are shared by two causes and cannot be split by anything."
    )

    st.subheader("Confusion matrix")
    cells = pd.DataFrame(
        [
            {"true": labels[r], "predicted": labels[c], "n": value}
            for r, row in enumerate(report["confusion_matrix"])
            for c, value in enumerate(row)
        ]
    )
    heat = (
        alt.Chart(cells)
        .mark_rect()
        .encode(
            x=alt.X("predicted:N", sort=labels, title="predicted"),
            y=alt.Y("true:N", sort=labels, title="true cause"),
            color=alt.Color("n:Q", scale=alt.Scale(scheme="blues"), title="events"),
            tooltip=["true", "predicted", "n"],
        )
    )
    text = heat.mark_text(fontSize=12).encode(
        text=alt.Text("n:Q"),
        color=alt.condition(alt.datum.n > cells["n"].max() / 2, alt.value("white"), alt.value("#333")),
    )
    st.altair_chart((heat + text).properties(height=340))

    st.subheader("Per class")
    st.dataframe(
        pd.DataFrame(report["per_class"]).T.rename_axis("cause"),
        width="stretch",
    )
    st.caption(
        "Precision at 1.000 on every real cause means the pipeline is never wrong "
        "when it answers — but note this run's answers come from the rule and "
        "fleet layers, which match known signatures. The recall gap is refusals, "
        "not errors."
    )


def tab_reserve(console: dict) -> None:
    report = console["reserve"]

    top = st.columns(4)
    top[0].metric("Reported disputes", _rupees(report["reported_paise"]))
    top[1].metric("Ultimate", _rupees(report["ultimate_paise"]))
    top[2].metric("Reported rate", f"{report['reported_dispute_rate']:.3%}")
    top[3].metric("Ultimate rate", f"{report['ultimate_dispute_rate']:.3%}")

    st.subheader("Development triangle (cumulative disputed paise)")
    triangle = pd.DataFrame(**report["triangle"]).rename_axis("cohort")
    st.dataframe(
        triangle.style.background_gradient(cmap="Blues", axis=None).format("{:,.0f}", na_rep=""),
        width="stretch",
    )

    fitted = report["ldf_fitted_ages"]
    ldf = {age: f for age, f in list(report["ldf"].items())[:fitted]}
    st.caption("Selected development factors (age to age)")
    st.dataframe(
        pd.DataFrame([ldf], index=["LDF"]),
        width="stretch",
    )
    st.caption(
        "The 1-2 factor does the work: a cohort's disputes roughly triple after "
        "their first month, which is why reserving on reported volume alone "
        "under-books every cohort still developing."
    )

    st.divider()
    st.subheader("Projected IBNR against seeded truth")
    st.caption(
        "A reserve model graded against known truth, which is normally "
        "impossible — the future disputes were seeded and held out of the fit."
    )
    ibnr = pd.DataFrame(report["ibnr_by_cohort"])
    ibnr = ibnr.loc[(ibnr["projected_ibnr_paise"] > 0) | (ibnr["true_ibnr_paise"] > 0)]
    melted = ibnr.melt(
        id_vars="cohort",
        value_vars=["projected_ibnr_paise", "true_ibnr_paise"],
        var_name="series",
        value_name="paise",
    ).replace({"projected_ibnr_paise": "projected", "true_ibnr_paise": "true"})
    st.altair_chart(
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("cohort:N", title="cohort"),
            y=alt.Y("paise:Q", title="IBNR (paise)"),
            xOffset="series:N",
            color=alt.Color(
                "series:N",
                title=None,
                scale=alt.Scale(domain=["projected", "true"], range=["#1f77b4", "#7f7f7f"]),
            ),
            tooltip=["cohort", "series", "paise"],
        )
        .properties(height=300)
    )
    error_pct = report["total_ibnr_error_pct"]
    st.caption(
        f"Projected {_rupees(report['total_projected_ibnr_paise'])} against a true "
        f"{_rupees(report['total_true_ibnr_paise'])}"
        + (f" — over-reserved by {error_pct:+.1f}%" if error_pct is not None else "")
        + ", concentrated in the youngest cohort where the triangle is thinnest. "
        "That is the textbook chain-ladder failure mode and it errs conservative; "
        "a model that under-reserved would be the actual problem."
    )

    st.divider()
    st.subheader("Gross to net, this batch")
    st.dataframe(
        pd.DataFrame(
            [
                ("Card volume recovered", _rupees(report["gross_recovered_paise"])),
                ("Projected clawback at the ultimate rate", "− " + _rupees(report["projected_clawback_paise"])),
                ("Net kept", _rupees(report["net_recovered_paise"])),
            ],
            columns=["", "paise"],
        ).set_index(""),
        width="stretch",
    )
    st.caption(
        "Card volume only: chargeback rights are a card-network mechanism, so "
        "applying this rate to UPI Autopay or NACH recovery would invent an "
        "exposure that does not exist."
    )


# --------------------------------------------------------------------------


def render() -> None:
    st.set_page_config(page_title="nakad console", layout="wide")

    if not ARTIFACT.exists():
        st.error("No precomputed run found.")
        st.code("python app.py", language="bash")
        st.caption(
            "The console renders a precomputed run and never builds one, so that "
            "no interaction can trigger an LLM call."
        )
        return

    console = load_console(_mtime(ARTIFACT))
    st.title("nakad — recovery under a hard budget")

    config = console["run_config"]
    degraded = config["unknown"]
    identity = (
        f"seed **{console['seed']}** · **{console['events']}** events · "
        f"provider **{config['provider']}** (`{config['model'] or '—'}`) · "
        f"{describe_methods(config)} · built {console['built_at']}"
    )
    if degraded:
        share = degraded / console["events"]
        st.warning(
            f"{identity} · 🟠 **DEGRADED** — {degraded} events ({share:.0%}) "
            "could not be diagnosed and were declined, not guessed at."
        )
    else:
        st.info(f"{identity} · ✅ all events diagnosed")

    sweep = console["multi_seed"]["run_config"]
    if sweep["provider"] != config["provider"]:
        st.caption(
            f"The stability sweep on the Scoreboard tab ran under provider "
            f"**{sweep['provider']}**, not **{config['provider']}** — five seeds "
            "is five times the LLM spend. Its spread is not comparable to the "
            "headline figures above."
        )

    tabs = st.tabs(["Scoreboard", "Compliance", "Ledger", "Diagnosis", "Reserve"])
    with tabs[0]:
        tab_scoreboard(console)
    with tabs[1]:
        tab_compliance(console)
    with tabs[2]:
        tab_ledger(console)
    with tabs[3]:
        tab_diagnosis(console)
    with tabs[4]:
        tab_reserve(console)


if st.runtime.exists():
    render()
elif __name__ == "__main__":
    print("building console artifact (this makes the run's LLM calls) …")
    # The sweep is five more full batches. Point it at a cheaper provider with
    # MULTI_SEED_PROVIDER=none when the daily quota will not stretch.
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
