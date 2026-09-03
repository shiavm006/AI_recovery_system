from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from generator.generate import generate_batch
from models import RootCause
from pipeline.diagnose import _cache_key, _sig, diagnose_by_rule, detect_outages


def test_ambiguous_features_correlate_without_separating():
    batch = generate_batch(n=500, seed=42)
    leftover = [
        e for e in batch if diagnose_by_rule(e, detect_outages(batch)) is None
    ]
    assert len({_sig(e) for e in leftover}) > 50
    assert len({_cache_key(e) for e in leftover}) < len({_sig(e) for e in leftover})

    def mean(cause: RootCause, attr: str) -> tuple[float, float, float, float]:
        rows = [e for e in batch if e.true_cause is cause]
        others = [e for e in batch if e.true_cause is not cause]
        return (
            sum(getattr(e, attr) for e in rows) / len(rows),
            sum(getattr(e, attr) for e in others) / len(others),
            min(getattr(e, attr) for e in rows),
            max(getattr(e, attr) for e in others),
        )

    dead_m, rest_m, dead_min, rest_max = mean(RootCause.DEAD_MANDATE, "days_since_mandate_created")
    assert dead_m > rest_m
    assert dead_min < rest_max

    nsf = [e for e in batch if e.true_cause is RootCause.INSUFFICIENT_FUNDS]
    other = [e for e in batch if e.true_cause is not RootCause.INSUFFICIENT_FUNDS]
    nsf_late = sum(e.day_of_month >= 24 for e in nsf) / len(nsf)
    other_late = sum(e.day_of_month >= 24 for e in other) / len(other)
    assert nsf_late > other_late
    assert any(e.day_of_month < 24 for e in nsf)

    down_m, rest_r, down_min, rest_rmax = mean(
        RootCause.ISSUER_DOWNTIME, "issuer_recent_failure_rate"
    )
    assert down_m > rest_r
    assert down_min < rest_rmax

    risk_m, rest_a, risk_min, rest_amax = mean(RootCause.RISK_BLOCK, "amount_vs_customer_avg")
    assert risk_m > rest_a
    assert risk_min < rest_amax
