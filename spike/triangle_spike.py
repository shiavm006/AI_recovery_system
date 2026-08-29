"""Throwaway spike: validate chainladder Triangle API on RAA and synthetic chargebacks."""

from __future__ import annotations

import numpy as np
import pandas as pd
import chainladder as cl

# Pin the valuation month so the run is reproducible regardless of clock.
VALUATION_MONTH: pd.Timestamp = pd.Timestamp("2026-09-01")
AMOUNT_PAISE: tuple[int, ...] = (14900, 29900, 49900, 99900)
AMOUNT_WEIGHTS: tuple[float, ...] = (0.40, 0.35, 0.18, 0.07)
# ponytail: lognormal lag is a shape heuristic (median ~42d, mass in 20–70, clip at 120).
LAG_MU: float = float(np.log(42.0))
LAG_SIGMA: float = 0.45


def generate_disputes(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """Synthetic recovered-card rows; returns only the disputed subset."""
    rng = np.random.default_rng(seed)
    origin_months = pd.date_range(end=VALUATION_MONTH, periods=6, freq="MS")
    cohort_month = rng.choice(origin_months.to_numpy(), size=n)
    amount_paise = rng.choice(AMOUNT_PAISE, size=n, p=AMOUNT_WEIGHTS).astype(int)
    disputed = rng.random(n) < 0.02
    lag_days = np.clip(rng.lognormal(LAG_MU, LAG_SIGMA, size=n), 1.0, 120.0)

    frame = pd.DataFrame(
        {
            "cohort_month": pd.to_datetime(cohort_month),
            "amount_paise": amount_paise,
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
    return frame[["cohort_month", "dispute_month", "amount_paise"]].reset_index(drop=True)


def selected_ldfs(model: cl.Chainladder) -> np.ndarray:
    values = np.asarray(model.ldf_.values, dtype=float).ravel()
    return values[np.isfinite(values)]


def ibnr_by_origin(model: cl.Chainladder) -> pd.Series:
    frame = model.ibnr_.to_frame(origin_as_datetime=False)
    series = frame.iloc[:, 0] if isinstance(frame, pd.DataFrame) else frame
    series.name = "ibnr"
    return series


def print_fit(triangle: cl.Triangle, model: cl.Chainladder) -> None:
    print("\n-- triangle --")
    print(triangle)
    print("\n-- link ratios (age-to-age) --")
    print(triangle.link_ratio)
    print("\n-- development factors (selected LDF) --")
    print(model.ldf_)
    print("\n-- IBNR by origin --")
    print(model.ibnr_)


def verdict(model: cl.Chainladder) -> None:
    print("\n" + "=" * 72)
    print("PART C — VERDICT (synthetic chargeback triangle)")
    print("=" * 72)

    ldfs = selected_ldfs(model)
    ibnr = ibnr_by_origin(model)
    ibnr_filled = ibnr.fillna(0.0)
    total_ibnr = float(ibnr_filled.sum())
    most_recent = ibnr_filled.index[-1]
    largest_origin = ibnr_filled.idxmax()

    checks: list[tuple[str, bool, str]] = [
        (
            "all development factors >= 1.0",
            bool(len(ldfs) > 0 and np.all(ldfs >= 1.0)),
            f"ldfs={np.round(ldfs, 4).tolist()}",
        ),
        (
            "development factors monotonically decreasing toward 1.0",
            bool(
                len(ldfs) >= 2
                and np.all(np.diff(ldfs) <= 1e-12)
                and ldfs[-1] <= ldfs[0]
            ),
            f"ldfs={np.round(ldfs, 4).tolist()}",
        ),
        (
            "total IBNR positive",
            total_ibnr > 0,
            f"total_ibnr={total_ibnr:.2f}",
        ),
        (
            "IBNR largest for the most recent cohort",
            largest_origin == most_recent,
            f"largest={largest_origin} value={float(ibnr_filled.max()):.2f}; most_recent={most_recent}",
        ),
    ]

    for label, passed, detail in checks:
        mark = "PASS" if passed else "FAIL"
        print(f"[{mark}] {label}  ({detail})")


def main() -> None:
    np.random.seed(42)

    print("=" * 72, flush=True)
    print("PART A — RAA sample (CAS 10x10 general-liability triangle)", flush=True)
    print("=" * 72, flush=True)
    raa = cl.load_sample("raa")
    raa_model = cl.Chainladder().fit(raa)
    print_fit(raa, raa_model)

    print("\n" + "=" * 72, flush=True)
    print("PART B — synthetic chargeback triangle", flush=True)
    print("=" * 72, flush=True)
    disputes = generate_disputes(n=2000, seed=42)
    observed = disputes.loc[disputes["dispute_month"] <= VALUATION_MONTH].copy()
    held_out = disputes.loc[disputes["dispute_month"] > VALUATION_MONTH]
    true_ibnr = held_out.groupby(held_out["cohort_month"].dt.to_period("M"))[
        "amount_paise"
    ].sum()
    print(
        f"disputed rows={len(disputes)}  "
        f"observed through {VALUATION_MONTH.date()}={len(observed)}  "
        f"(future disputes held out as IBNR)"
    )
    print("true held-out IBNR by cohort (paise):")
    print(true_ibnr if not true_ibnr.empty else "(none)")

    triangle = cl.Triangle(
        data=observed,
        origin="cohort_month",
        development="dispute_month",
        columns="amount_paise",
        cumulative=False,
    ).incr_to_cum()
    model = cl.Chainladder().fit(triangle)
    print_fit(triangle, model)

    verdict(model)


if __name__ == "__main__":
    main()
