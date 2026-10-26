r"""Heston stochastic volatility: pricing, simulation, calibration, hedging deltas.

.. math::

    dS = (r - q) S\,dt + \sqrt{v}\,S\,dW_1, \qquad
    dv = \kappa(\theta - v)\,dt + \xi\sqrt{v}\,dW_2, \qquad
    d\langle W_1, W_2\rangle = \rho\,dt

Variance is random and mean reverting, and a negative rho (stock down, vol up) makes
the smile skew the way equity index options do.

Everything is written in terms of the forward F and x = ln(S_T / F), whose
characteristic function does not depend on r or q, so one calibration works for any
expiry once its forward is known. With k = ln(F/K), Lewis' formula gives the call
price divided by the discounted forward, D F:

.. math::

    n(k, v_0) = 1 - \frac{e^{-k/2}}{\pi}\int_0^\infty
    \frac{\mathrm{Re}\left[e^{iuk}\,\varphi(u - i/2)\right]}{u^2 + 1/4}\,du

The characteristic function uses the "little trap" form (Albrecher et al.), which
avoids the branch cut problem of the original formulation.

Simulation uses full truncation Euler. It has a discretization bias that shrinks with
the step, and it is the simplest scheme that stays stable when the Feller condition
2 kappa theta >= xi^2 fails (it usually does for fitted equity parameters).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.optimize import least_squares
from scipy.special import ndtr

from pricer.paths import SeedLike

__all__ = [
    "HestonParams",
    "calibrate_heston",
    "heston_call_normalized",
    "heston_implied_vols",
    "simulate_heston",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HestonParams:
    """Heston parameters: initial variance, speed, long-run variance, vol of vol, correlation."""

    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        """Validate ranges."""
        if min(self.v0, self.kappa, self.theta, self.xi) <= 0.0:
            raise ValueError("v0, kappa, theta and xi must be positive")
        if not -1.0 < self.rho < 1.0:
            raise ValueError("rho must lie in (-1, 1)")

    @property
    def feller_ratio(self) -> float:
        """2 kappa theta / xi^2. Above one, variance cannot reach zero."""
        return 2.0 * self.kappa * self.theta / self.xi**2


def _log_cf(w: np.ndarray, v0: FloatArray, maturity: float, p: HestonParams) -> np.ndarray:
    """E[exp(w X)], X = ln(S_T / F), for each (w, v0) pair as a (len(v0), len(w)) array."""
    a = p.kappa - p.rho * p.xi * w
    d = np.sqrt(a * a + p.xi**2 * (w - w * w))
    g = (a - d) / (a + d)
    e = np.exp(-d * maturity)
    big_d = (a - d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    big_c = (p.kappa * p.theta / p.xi**2) * (
        (a - d) * maturity - 2.0 * np.log((1.0 - g * e) / (1.0 - g))
    )
    return np.asarray(np.exp(big_c[None, :] + v0[:, None] * big_d[None, :]), dtype=np.complex128)


def _u_grid(maturity: float, v_min: float, p: HestonParams) -> FloatArray:
    """Integration grid long enough that the truncated tail is negligible.

    The integrand decays like exp(-beta u) with beta = v0 sqrt(1 - rho^2) / xi, and
    the tail is weighted only by 1/u^2, so stopping early leaves an error of order
    1/u_max in the price. Run out to about 15 / beta.
    """
    beta = max(v_min, 1e-3) * math.sqrt(1.0 - p.rho**2) / p.xi
    u_max = min(max(80.0, 15.0 / beta, 12.0 / math.sqrt(max(maturity, 1e-6) * 0.02)), 10_000.0)
    n = min(40_000, max(4_000, int(8 * u_max)))
    return np.linspace(0.0, u_max, n)


def heston_call_normalized(
    k: FloatArray, v0: FloatArray, maturity: float, p: HestonParams
) -> tuple[FloatArray, FloatArray]:
    """Return (n, dn/dk), each of shape (len(k), len(v0)).

    ``n`` is the call price divided by D*F, with k = ln(F/K). The initial variance in
    ``p`` is ignored here: ``v0`` is the grid of variances to evaluate at.
    """
    k = np.atleast_1d(np.asarray(k, dtype=np.float64))
    v0 = np.atleast_1d(np.asarray(v0, dtype=np.float64))
    u = _u_grid(maturity, float(v0.min()), p)
    w = 1j * (u - 0.5j)  # i * (u - i/2) = i u + 1/2
    phi = _log_cf(w, v0, maturity, p)  # (nv, nu)
    denom = u * u + 0.25
    weights = np.full(u.size, u[1] - u[0])
    weights[[0, -1]] *= 0.5

    base = (phi * (weights / denom)[None, :]).T  # (nu, nv)
    exp_k = np.exp((1j * u[None, :] - 0.5) * k[:, None])  # e^{(iu - 1/2) k}, (nk, nu)
    price_int = (exp_k @ base).real
    slope_int = ((1j * u - 0.5)[None, :] * exp_k @ base).real
    return 1.0 - price_int / math.pi, -slope_int / math.pi


def _normalized_black(
    k: FloatArray, vol: FloatArray, maturity: float
) -> tuple[FloatArray, FloatArray]:
    """Black-76 call / (D F) and its vega, for log forward-moneyness k."""
    total = vol * math.sqrt(maturity)
    d1 = (k + 0.5 * total**2) / total
    d2 = d1 - total
    price = ndtr(d1) - np.exp(-k) * ndtr(d2)
    vega = np.exp(-0.5 * d1**2) / math.sqrt(2.0 * math.pi) * math.sqrt(maturity)
    return np.asarray(price), np.asarray(vega)


def _invert_normalized(price: FloatArray, k: FloatArray, maturity: float) -> FloatArray:
    """Vectorized bisection for the vol whose normalized Black price equals ``price``.

    Prices are floored just above intrinsic first: rounding can leave a price a hair
    below it (or negative far out of the money), which has no implied vol.
    """
    price = np.maximum(price, np.maximum(1.0 - np.exp(-k), 0.0) + 1e-12)
    lo = np.full(price.shape, 1e-4)
    hi = np.full(price.shape, 3.0)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        too_low = _normalized_black(k, mid, maturity)[0] < price
        lo = np.where(too_low, mid, lo)
        hi = np.where(too_low, hi, mid)
    return np.asarray(0.5 * (lo + hi), dtype=np.float64)


def heston_implied_vols(log_moneyness: FloatArray, maturity: float, p: HestonParams) -> FloatArray:
    """Black implied vols of Heston call prices at ln(K/F), the convention of ``surface``."""
    k = -np.asarray(log_moneyness, dtype=np.float64)  # the pricer uses ln(F/K)
    price, _ = heston_call_normalized(k, np.array([p.v0]), maturity, p)
    return _invert_normalized(price[:, 0], k, maturity)


def calibrate_heston(
    slices: list[tuple[float, FloatArray, FloatArray]],
) -> tuple[HestonParams, list[float]]:
    """Fit Heston to target smiles, each given as (maturity, ln(K/F), implied vol).

    Residuals are model implied vol minus target implied vol, equally weighted. A
    price-space objective is cheaper, but far out of the money a tiny price error is a
    large vol error, and fitting prices hides that. Returns the parameters and the
    per-slice RMSE in implied vol.
    """
    targets = [(m, np.asarray(lm, dtype=np.float64), np.asarray(v)) for m, lm, v in slices]

    def residuals(x: FloatArray) -> FloatArray:
        p = HestonParams(*map(float, x))
        return np.concatenate([heston_implied_vols(lm, m, p) - v for m, lm, v in targets])

    lower = [1e-4, 0.05, 1e-4, 0.05, -0.99]
    upper = [0.5, 20.0, 0.5, 4.0, -0.01]
    atm = float(np.mean([np.mean(v) for _, _, v in slices])) ** 2
    best = None
    for kappa0, xi0, rho0 in ((2.0, 0.8, -0.7), (6.0, 1.5, -0.8), (0.8, 0.5, -0.5)):
        start = np.array([atm, kappa0, atm, xi0, rho0])
        sol = least_squares(
            residuals, start, bounds=(lower, upper), xtol=1e-10, ftol=1e-10, x_scale="jac"
        )
        if best is None or sol.cost < best.cost:
            best = sol
    assert best is not None
    params = HestonParams(*map(float, best.x))
    rmses = [
        float(np.sqrt(np.mean((heston_implied_vols(lm, m, params) - v) ** 2)))
        for m, lm, v in targets
    ]
    return params, rmses


def simulate_heston(
    p: HestonParams,
    spot: float,
    rate: float,
    maturity: float,
    n_steps: int,
    n_paths: int,
    seed: SeedLike,
    *,
    substeps: int = 4,
) -> tuple[FloatArray, FloatArray]:
    """Simulate (S, v) at ``n_steps + 1`` dates, with ``substeps`` Euler steps between.

    Full truncation Euler: the variance fed to drift and diffusion is max(v, 0).
    """
    rng = np.random.default_rng(seed)
    dt = maturity / (n_steps * substeps)
    sqrt_dt = math.sqrt(dt)
    log_s = np.full(n_paths, math.log(spot))
    v = np.full(n_paths, p.v0)
    s_out = np.empty((n_paths, n_steps + 1))
    v_out = np.empty((n_paths, n_steps + 1))
    s_out[:, 0], v_out[:, 0] = spot, p.v0
    corr = math.sqrt(1.0 - p.rho**2)

    for step in range(1, n_steps + 1):
        for _ in range(substeps):
            z1 = rng.standard_normal(n_paths)
            z2 = p.rho * z1 + corr * rng.standard_normal(n_paths)
            vp = np.maximum(v, 0.0)
            sv = np.sqrt(vp)
            log_s += (rate - 0.5 * vp) * dt + sv * sqrt_dt * z1
            v = v + p.kappa * (p.theta - vp) * dt + p.xi * sv * sqrt_dt * z2
        s_out[:, step] = np.exp(log_s)
        v_out[:, step] = np.maximum(v, 0.0)
    return s_out, v_out
