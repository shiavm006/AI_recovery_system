"""Console charts recompute from disk — no LLM."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import app as console
from models import Diagnosis, RootCause
from run import _propose_agent, _propose_control

DATA = Path(__file__).resolve().parents[1] / "data"
LEDGER = DATA / "ledger_agent.db"
BATCH = DATA / "batch_seed42.parquet"


def _diagnoses() -> list[Diagnosis]:
    rows = sqlite3.connect(LEDGER).execute(
        "SELECT event_id, payload FROM entries WHERE entry_type = 'diagnosed'"
    )
    out = []
    for event_id, raw in rows:
        payload = json.loads(raw)
        out.append(
            Diagnosis(
                event_id=event_id,
                cause=RootCause(payload["cause"]),
                confidence=float(payload["confidence"]),
                evidence=list(payload.get("evidence") or []),
                method=payload.get("method") or "rule",
            )
        )
    return out


def test_default_budgets_match_mockup_scale():
    events = console._load_frozen_batch(BATCH)
    diagnoses = _diagnoses()
    agent = console._score(events, diagnoses, _propose_agent(diagnoses), 120, 60, 42)
    arrival = console._score(events, diagnoses, _propose_control, 120, 0, 42)
    amount = console._score(
        events, diagnoses, console._propose_amount_sorted, 120, 0, 42
    )
    assert agent["gross_paise"] == 8_794_200
    assert arrival["gross_paise"] == 3_503_600
    assert 1.50 <= agent["gross_paise"] / amount["gross_paise"] <= 1.53
    tight = console._score(events, diagnoses, _propose_agent(diagnoses), 40, 10, 42)
    assert tight["gross_paise"] < agent["gross_paise"]


def test_every_chart_builds_from_a_real_scored_view():
    """Altair validates its own spec, so building is the schema check."""
    events = console._load_frozen_batch(BATCH)
    diagnoses = _diagnoses()
    agent = console._score(events, diagnoses, _propose_agent(diagnoses), 120, 60, 42)

    console.chart_baselines(
        [
            {"policy": "Nakad", "rupees": 87_942, "acted": agent["acted"]},
            {"policy": "Amount-sorted", "rupees": 57_928, "acted": 120},
            {"policy": "Arrival order", "rupees": 35_036, "acted": 120},
        ]
    ).to_dict()
    console.chart_stack(
        [{"label": "Rule lookup", "n": 315}, {"label": "LLM", "n": 131}],
        [console.MUTED, console.ACCENT],
    ).to_dict()
    console.chart_sweep(
        [
            {"budget": 20, "series": "Nakad", "rupees": 1.0},
            {"budget": 20, "series": "Baseline", "rupees": 0.5},
        ],
        120,
    ).to_dict()
    console.chart_booking(agent["booked"]).to_dict()


def test_booking_chart_never_plots_an_action_inside_its_own_closed_window():
    """The timing panel claims every booking clears its rule. Hold it to that."""
    events = console._load_frozen_batch(BATCH)
    diagnoses = _diagnoses()
    agent = console._score(events, diagnoses, _propose_agent(diagnoses), 120, 60, 42)

    closed: dict[str, list[tuple[int, int]]] = {}
    for band in console._closed_bands():
        closed.setdefault(band["kind"], []).append((band["start"], band["end"]))
    lane = {"rail": "Attempts", "contact": "Contacts"}

    assert agent["booked"], "no bookings to check"
    for row in agent["booked"]:
        for start, end in closed[lane[row["kind"]]]:
            assert not start <= row["minute"] < end, (
                f"{row['action']} booked {row['clock']} inside a closed window"
            )


def test_disposition_segments_account_for_every_event():
    """The stacked bar is captioned as covering the batch, so it must total it.

    Gate refusals sit in no suppression bucket — leaving them out silently
    shrinks the bar, which is exactly the kind of quiet lie a console must not
    tell.
    """
    events = console._load_frozen_batch(BATCH)
    diagnoses = _diagnoses()
    agent = console._score(events, diagnoses, _propose_agent(diagnoses), 120, 60, 42)

    segments = (
        agent["acted"]
        + agent["budget"]
        + agent["no_headroom"]
        + agent["low_value"]
        + agent["blocked"]
        + agent["low_confidence"]
        + agent["no_window"]
        + agent["no_diagnosis"]
    )
    assert segments == len(events)
