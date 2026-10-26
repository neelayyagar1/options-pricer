r"""Variance risk premium: implied variance minus the variance that followed.

At each date t the VIX is the market's 30 calendar day implied variance (it is
computed from a strip of options and is variance-swap-like, so it contains the whole
skew, not just an ATM Black-Scholes vol). The matching realized quantity is the
annualized sum of squared daily log returns over the next ``horizon`` trading days:

.. math:: RV_t^2 = \frac{252}{h} \sum_{i=1}^{h} r_{t+i}^2, \qquad
          VRP_t = (VIX_t/100)^2 - RV_t^2

Both the variance premium (VIX^2 - RV^2) and the volatility premium (VIX - RV) are
reported, since both appear in the literature.

Overlapping windows: consecutive observations share h-1 days of returns, so the
series is strongly autocorrelated and ordinary standard errors are far too small.
The headline t-statistic uses Newey-West errors with ``horizon`` lags, and a second
estimate uses non-overlapping windows only.

This is a measured risk premium, not a trading edge. It exists because option sellers
are paid to carry crash risk, and the VIX is not directly tradable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm

__all__ = ["PremiumSummary", "premium_frame", "summarize_premium"]

TRADING_DAYS = 252


@dataclass(frozen=True, slots=True)
class PremiumSummary:
    """Mean premium with overlapping (Newey-West) and non-overlapping inference."""

    mean: float
    newey_west_se: float
    newey_west_t: float
    n_overlapping: int
    nonoverlapping_mean: float
    nonoverlapping_se: float
    nonoverlapping_t: float
    n_nonoverlapping: int
    fraction_positive: float


def premium_frame(history: pd.DataFrame, horizon: int = 21) -> pd.DataFrame:
    """Build implied and forward-realized variance and vol per date.

    ``history`` needs ``vix`` (in index points) and ``spx`` columns. The last
    ``horizon`` rows have no forward window and are dropped.
    """
    log_ret = np.log(history["spx"]).diff()
    squared = log_ret**2
    # Sum of r^2 over t+1..t+h: rolling sum over the window ending at t+h, shifted back.
    forward_sum = squared.rolling(horizon).sum().shift(-horizon)
    out = pd.DataFrame(index=history.index)
    out["iv"] = history["vix"] / 100.0
    out["rv"] = np.sqrt(forward_sum * TRADING_DAYS / horizon)
    out["iv_var"] = out["iv"] ** 2
    out["rv_var"] = out["rv"] ** 2
    out["variance_premium"] = out["iv_var"] - out["rv_var"]
    out["vol_premium"] = out["iv"] - out["rv"]
    return out.dropna()


def summarize_premium(series: pd.Series, horizon: int = 21) -> PremiumSummary:
    """Mean of ``series`` with Newey-West and non-overlapping standard errors."""
    values = series.to_numpy(dtype=float)
    fit = sm.OLS(values, np.ones_like(values)).fit(cov_type="HAC", cov_kwds={"maxlags": horizon})
    sampled = values[::horizon]
    mean = float(sampled.mean())
    se = float(sampled.std(ddof=1) / math.sqrt(sampled.size))
    return PremiumSummary(
        mean=float(values.mean()),
        newey_west_se=float(fit.bse[0]),
        newey_west_t=float(fit.tvalues[0]),
        n_overlapping=int(values.size),
        nonoverlapping_mean=mean,
        nonoverlapping_se=se,
        nonoverlapping_t=mean / se,
        n_nonoverlapping=int(sampled.size),
        fraction_positive=float(np.mean(values > 0.0)),
    )
