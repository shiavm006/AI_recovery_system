"""Synthetic failure batch and chargeback history. Types come from models.py."""

from __future__ import annotations

import string
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from models import FailureEvent, RootCause

# Vocabulary from https://razorpay.com/docs/errors/payments/ and
# https://razorpay.com/docs/errors/payments/payment-methods-error-parameters/
AMOUNT_PAISE: tuple[int, ...] = (14900, 29900, 49900, 99900)
METHODS: tuple[str, ...] = ("upi_autopay", "card", "nach")
METHOD_WEIGHTS: tuple[float, ...] = (0.55, 0.30, 0.15)
ANCHOR: datetime = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
VALUATION_MONTH: pd.Timestamp = pd.Timestamp("2026-09-01")
LAG_MU: float = float(np.log(42.0))
LAG_SIGMA: float = 0.45
_ID_CHARS = list(string.ascii_letters + string.digits)

# Shared generic signature — INSUFFICIENT_FUNDS and DEAD_MANDATE both land here.
AMBIGUOUS_SIGNAL: tuple[str, str, str, str] = (
    "BAD_REQUEST_ERROR",
    "issuer_bank",
    "payment_authorization",
    "payment_failed",
)

# (error_code, error_source, error_step, error_reason) from Razorpay error tables.
CAUSE_SIGNALS: dict[RootCause, tuple[tuple[str, str, str, str], ...]] = {
    RootCause.INSUFFICIENT_FUNDS: (
        ("BAD_REQUEST_ERROR", "customer", "payment_authorization", "insufficient_funds"),
        ("BAD_REQUEST_ERROR", "customer", "payment_authorization", "insufficient_funds"),
        ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization", "debit_declined"),
    ),
    RootCause.DEAD_MANDATE: (
        ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization", "debit_instrument_inactive"),
        ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization", "debit_instrument_blocked"),
        ("BAD_REQUEST_ERROR", "customer", "payment_authorization", "bank_account_invalid"),
        ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "mandate_creation_failed"),
    ),
    RootCause.HARD_DECLINE: (
        ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization", "card_declined"),
        ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "payment_declined"),
        ("BAD_REQUEST_ERROR", "customer", "payment_authorization", "transaction_limit_exceeded"),
        ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization", "debit_instrument_blocked"),
    ),
    RootCause.ISSUER_DOWNTIME: (
        ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "bank_technical_error"),
        ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "bank_not_available"),
        ("GATEWAY_ERROR", "gateway", "payment_authorization", "gateway_technical_error"),
        ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "issuer_technical_error"),
    ),
    RootCause.NETWORK_TIMEOUT: (
        ("GATEWAY_ERROR", "gateway", "payment_authorization", "payment_timed_out"),
        ("GATEWAY_ERROR", "gateway", "payment_authorization", "request_timed_out"),
        ("BAD_REQUEST_ERROR", "gateway", "payment_authorization", "payment_timed_out"),
    ),
    RootCause.RISK_BLOCK: (
        ("BAD_REQUEST_ERROR", "issuer_bank", "payment_authorization", "payment_risk_check_failed"),
        ("BAD_REQUEST_ERROR", "gateway", "payment_authorization", "payment_risk_check_failed"),
        ("BAD_REQUEST_ERROR", "internal", "payment_initiation", "payment_risk_check_failed"),
    ),
}

OUTAGE_SIGNALS: tuple[tuple[str, str, str, str], ...] = (
    ("GATEWAY_ERROR", "gateway", "payment_authorization", "payment_timed_out"),
    ("GATEWAY_ERROR", "gateway", "payment_authorization", "request_timed_out"),
    ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "bank_technical_error"),
    ("GATEWAY_ERROR", "issuer_bank", "payment_authorization", "bank_not_available"),
    ("GATEWAY_ERROR", "gateway", "payment_authorization", "gateway_technical_error"),
)

BUSINESS_CAUSES: tuple[RootCause, ...] = (
    RootCause.INSUFFICIENT_FUNDS,
    RootCause.DEAD_MANDATE,
    RootCause.HARD_DECLINE,
)
BUSINESS_WEIGHTS: tuple[float, ...] = (0.70, 0.18, 0.12)
TECH_CAUSES: tuple[RootCause, ...] = (
    RootCause.ISSUER_DOWNTIME,
    RootCause.NETWORK_TIMEOUT,
)
TECH_WEIGHTS: tuple[float, ...] = (0.55, 0.45)


def _load_issuers() -> list[dict]:
    rows = yaml.safe_load(Path(__file__).with_name("issuers.yaml").read_text())
    total = sum(float(row["volume_share"]) for row in rows)
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"issuer volume_share must sum to 1.0, got {total}")
    return rows


def _rzp_id(rng: np.random.Generator, prefix: str) -> str:
    return prefix + "".join(rng.choice(_ID_CHARS, size=14))


def _signal(
    rng: np.random.Generator, cause: RootCause, ambiguous: bool
) -> tuple[str, str, str, str]:
    if ambiguous:
        return AMBIGUOUS_SIGNAL
    options = CAUSE_SIGNALS[cause]
    return options[int(rng.integers(len(options)))]


def _true_cause(
    rng: np.random.Generator, p_tech: float
) -> RootCause:
    if rng.random() < 0.03:
        return RootCause.RISK_BLOCK
    if rng.random() < p_tech:
        return TECH_CAUSES[int(rng.choice(len(TECH_CAUSES), p=TECH_WEIGHTS))]
    return BUSINESS_CAUSES[int(rng.choice(len(BUSINESS_CAUSES), p=BUSINESS_WEIGHTS))]


def generate_batch(n: int = 500, seed: int = 42) -> list[FailureEvent]:
    rng = np.random.default_rng(seed)
    issuers = _load_issuers()
    names = [row["name"] for row in issuers]
    shares = np.array([row["volume_share"] for row in issuers], dtype=float)
    td_rates = np.array([row["td_rate"] for row in issuers], dtype=float)
    td_bar = float(shares @ td_rates)
    p_tech = np.clip(0.18 * td_rates / td_bar, 0.0, 1.0)

    issuer_idx = rng.choice(len(names), size=n, p=shares)
    amounts = rng.choice(AMOUNT_PAISE, size=n).astype(int)
    methods = rng.choice(METHODS, size=n, p=METHOD_WEIGHTS)
    retries = rng.choice(4, size=n, p=(0.50, 0.25, 0.15, 0.10))
    offsets = rng.random(n) * 24 * 3600
    ambiguous_mask = rng.random(n) < 0.25

    # Mix target is ~18% technical. The two outage windows add extra
    # ISSUER_DOWNTIME on top, so seed 42 lands at ~28% technical
    # (139/500: 99 downtime + 40 timeout). That lift is intentional.
    span_start = ANCHOR - timedelta(hours=24)
    windows: list[tuple[str, datetime, datetime]] = []
    outage_of = np.full(n, -1, dtype=int)
    used: set[int] = set()
    outage_issuers = rng.choice(len(names), size=2, replace=False)
    for w, bank_i in enumerate(outage_issuers):
        dur_min = int(rng.integers(8, 16))
        latest_start = ANCHOR - timedelta(minutes=dur_min)
        start = span_start + timedelta(
            seconds=float(rng.random() * (latest_start - span_start).total_seconds())
        )
        end = start + timedelta(minutes=dur_min)
        k = int(rng.integers(25, 41))
        pool = [j for j in range(n) if j not in used]
        chosen = rng.choice(pool, size=min(k, len(pool)), replace=False)
        windows.append((names[int(bank_i)], start, end))
        for j in chosen:
            used.add(int(j))
            outage_of[int(j)] = w

    events: list[FailureEvent] = []
    for i in range(n):
        w = int(outage_of[i])
        method = str(methods[i])
        sub: str | None
        if method == "card" and rng.random() < 0.2:
            sub = None
        else:
            sub = _rzp_id(rng, "sub_")
        if w >= 0:
            issuer, start, end = windows[w]
            cause = RootCause.ISSUER_DOWNTIME
            code, source, step, reason = OUTAGE_SIGNALS[
                int(rng.integers(len(OUTAGE_SIGNALS)))
            ]
            occurred_at = start + timedelta(
                seconds=float(rng.random() * (end - start).total_seconds())
            )
        else:
            bank_i = int(issuer_idx[i])
            issuer = names[bank_i]
            cause = _true_cause(rng, float(p_tech[bank_i]))
            code, source, step, reason = _signal(rng, cause, bool(ambiguous_mask[i]))
            occurred_at = ANCHOR - timedelta(seconds=float(offsets[i]))
        events.append(
            FailureEvent(
                event_id=_rzp_id(rng, "evt_"),
                payment_id=_rzp_id(rng, "pay_"),
                subscription_id=sub,
                amount_paise=int(amounts[i]),
                method=method,
                issuer=issuer,
                error_code=code,
                error_source=source,
                error_step=step,
                error_reason=reason,
                occurred_at=occurred_at,
                retries_used=int(retries[i]),
                true_cause=cause,
            )
        )

    generate_batch.outage_windows = windows  # type: ignore[attr-defined]
    return events


generate_batch.outage_windows = []  # type: ignore[attr-defined]


def generate_history(
    n: int = 50_000,
    cohorts: int = 18,
    seed: int = 42,
    valuation_cutoff: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    origin_months = pd.date_range(end=VALUATION_MONTH, periods=cohorts, freq="MS")
    cohort_idx = np.arange(n) % cohorts
    rng.shuffle(cohort_idx)
    disputed = rng.random(n) < 0.025
    lag_days = np.clip(rng.lognormal(LAG_MU, LAG_SIGMA, size=n), 1.0, 120.0)
    frame = pd.DataFrame(
        {
            "cohort_month": origin_months[cohort_idx],
            "amount_paise": rng.choice(AMOUNT_PAISE, size=n).astype(int),
            "disputed": disputed,
            "lag_days": lag_days,
        }
    )
    frame = frame.loc[frame["disputed"]].copy()
    frame["dispute_month"] = (
        (frame["cohort_month"] + pd.to_timedelta(frame["lag_days"], unit="D"))
        .dt.to_period("M")
        .dt.to_timestamp()
    )
    out = frame[["cohort_month", "dispute_month", "amount_paise"]].reset_index(drop=True)
    if valuation_cutoff is not None:
        cutoff = pd.Timestamp(valuation_cutoff)
        out = out.loc[out["dispute_month"] <= cutoff].reset_index(drop=True)
    return out


def freeze(out_dir: str = "data", seed: int = 42) -> None:
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    batch = generate_batch(seed=seed)
    history = generate_history(seed=seed, valuation_cutoff=VALUATION_MONTH)
    rows = []
    for event in batch:
        payload = event.model_dump()
        cause = payload["true_cause"]
        payload["true_cause"] = cause.value if isinstance(cause, RootCause) else cause
        rows.append(payload)
    pd.DataFrame(rows).to_parquet(dest / f"batch_seed{seed}.parquet", index=False)
    history.to_parquet(dest / f"history_seed{seed}.parquet", index=False)


def _in_outage(event: FailureEvent) -> bool:
    for issuer, start, end in generate_batch.outage_windows:  # type: ignore[attr-defined]
        if event.issuer == issuer and start <= event.occurred_at <= end:
            return True
    return False


if __name__ == "__main__":
    batch = generate_batch(n=500, seed=42)
    history = generate_history(n=50_000, cohorts=18, seed=42, valuation_cutoff=VALUATION_MONTH)
    print("true_cause distribution:")
    for cause, count in Counter(event.true_cause for event in batch).most_common():
        print(f"  {cause}: {count}")
    print(f"outage-window events: {sum(_in_outage(event) for event in batch)}")
    print(
        "ambiguous (payment_failed): "
        f"{sum(event.error_reason == 'payment_failed' for event in batch)}"
    )
    print(f"history disputed rows: {len(history)}")
