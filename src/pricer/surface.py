r"""SVI fit of the implied volatility smile, with arbitrage diagnostics.

Raw SVI models total implied variance w = sigma^2 T as a function of log-moneyness
k = ln(K / F):

.. math:: w(k) = a + b\left[\rho (k - m) + \sqrt{(k - m)^2 + s^2}\right]

Fits are done one expiry at a time. The least squares problem is not convex, so a
single starting point can land in a bad local minimum. This uses several starts and
keeps the best. (Zeliade's quasi-explicit method, which reduces the search to two
parameters, is the more robust production approach.)

Two static no-arbitrage conditions are checked rather than assumed:

- butterfly, Gatheral's g(k) >= 0, which is what keeps the implied density positive
- calendar, total variance non-decreasing in maturity at fixed k

A fit can match the quotes well and still fail these, and the diagnostics report it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares

__all__ = [
    "SVIParams",
    "SVISlice",
    "butterfly_g",
    "calendar_violations",
    "fit_svi",
    "svi_total_variance",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class SVIParams:
    """Raw SVI parameters."""

    a: float
    b: float
    rho: float
    m: float
    s: float


@dataclass(frozen=True, slots=True)
class SVISlice:
    """A fitted slice and how well it fits."""

    maturity: float
    params: SVIParams
    rmse_vol: float
    """Root-mean-square error in implied vol (decimal, so 0.002 is 0.2 vol points)."""
    n_points: int


def svi_total_variance(k: FloatArray, p: SVIParams) -> FloatArray:
    """Evaluate w(k)."""
    x = k - p.m
    return np.asarray(p.a + p.b * (p.rho * x + np.sqrt(x * x + p.s * p.s)), dtype=np.float64)


def _derivatives(k: FloatArray, p: SVIParams) -> tuple[FloatArray, FloatArray]:
    x = k - p.m
    root = np.sqrt(x * x + p.s * p.s)
    first = p.b * (p.rho + x / root)
    second = p.b * p.s * p.s / root**3
    return np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)


def butterfly_g(k: FloatArray, p: SVIParams) -> FloatArray:
    """Gatheral's g(k). The implied density is non-negative where g >= 0."""
    w = svi_total_variance(k, p)
    wp, wpp = _derivatives(k, p)
    g = (1.0 - k * wp / (2.0 * w)) ** 2 - wp**2 / 4.0 * (1.0 / w + 0.25) + wpp / 2.0
    return np.asarray(g, dtype=np.float64)


def calendar_violations(
    slices: Sequence[SVISlice], k_grid: FloatArray
) -> list[tuple[float, float, float]]:
    """List (T_short, T_long, worst_gap) for adjacent slices where w decreases.

    ``worst_gap`` is the most negative value of w_long(k) - w_short(k) on the grid.
    """
    ordered = sorted(slices, key=lambda s: s.maturity)
    found = []
    for early, late in pairwise(ordered):
        gap = svi_total_variance(k_grid, late.params) - svi_total_variance(k_grid, early.params)
        if gap.min() < 0.0:
            found.append((early.maturity, late.maturity, float(gap.min())))
    return found


def fit_svi(k: FloatArray, vols: FloatArray, maturity: float) -> SVISlice:
    """Fit raw SVI to one expiry's (log-moneyness, implied vol) points."""
    if k.size < 6:
        raise ValueError(f"need at least 6 points to fit 5 parameters, got {k.size}")
    w = vols**2 * maturity
    w_scale = float(w.max())
    k_lo, k_hi = float(k.min()), float(k.max())

    lower = [-w_scale, 1e-4, -0.999, k_lo - 0.25, 1e-3]
    upper = [w_scale, 2.0, 0.999, k_hi + 0.25, 1.0]

    def residuals(x: FloatArray) -> FloatArray:
        p = SVIParams(*map(float, x))
        fit = svi_total_variance(k, p)
        # Penalise a negative minimum total variance, a + b s sqrt(1 - rho^2) < 0.
        floor = p.a + p.b * p.s * np.sqrt(1.0 - p.rho**2)
        penalty = 10.0 * max(0.0, -floor)
        return np.append(fit - w, penalty)

    best = None
    for rho0 in (-0.7, -0.3, 0.0, 0.3):
        for m0 in (0.0, 0.5 * k_lo, 0.5 * k_hi):
            for s0 in (0.05, 0.2):
                start = np.array([0.5 * float(w.min()), 0.1 * w_scale + 1e-3, rho0, m0, s0])
                start = np.clip(start, np.array(lower) + 1e-9, np.array(upper) - 1e-9)
                try:
                    sol = least_squares(residuals, start, bounds=(lower, upper), xtol=1e-12)
                except ValueError:
                    continue
                if best is None or sol.cost < best.cost:
                    best = sol
    if best is None:
        raise RuntimeError("SVI fit failed from every starting point")

    params = SVIParams(*map(float, best.x))
    fitted_vol = np.sqrt(np.maximum(svi_total_variance(k, params), 0.0) / maturity)
    rmse = float(np.sqrt(np.mean((fitted_vol - vols) ** 2)))
    return SVISlice(maturity=maturity, params=params, rmse_vol=rmse, n_points=int(k.size))
