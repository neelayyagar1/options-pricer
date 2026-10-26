r"""Discrete delta hedging of a short option.

Sell a European option at the Black-Scholes price for implied vol sigma_i, then hedge
with the underlying, rebalancing at n equally spaced dates while the stock actually
moves with realized vol sigma_r. Two results are reproduced:

**Hedging at implied vol.** In the continuous limit the P&L at expiry (short option)
is

.. math::

    \mathrm{P\&L} = \int_0^T e^{r(T-t)}\,\tfrac{1}{2} S_t^2\,\Gamma_t^{(\sigma_i)}
    \left(\sigma_i^2 - \sigma_r^2\right) dt

which is random (path dependent) but positive in expectation when sigma_i > sigma_r.

**Hedging at realized vol.** The P&L is deterministic, equal to
(BS(sigma_i) - BS(sigma_r)) e^{rT}, but the mark-to-market path along the way is not
smooth.

With a *finite* number of rebalances there is also hedging error, whose standard
deviation falls like n^(-1/2) (Boyle and Emanuel). That is the cost of the
continuous-trading assumption.

The real-world drift mu never appears in either formula. For the realized-vol hedge
that means the P&L does not depend on it at all. For the implied-vol hedge only the
formula is drift-free: its expectation still depends on mu, because drift changes how
long the stock spends near the strike where gamma is large.

No dividends: the hedge holds shares and the simulation does not model a yield.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.interpolate import RegularGridInterpolator

from pricer import black_scholes as bs
from pricer.heston import (
    HestonParams,
    _invert_normalized,
    heston_call_normalized,
    simulate_heston,
)
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from pricer.paths import SeedLike, draw_normals, gbm_paths

__all__ = [
    "HedgeResult",
    "hedge_on_paths",
    "hedge_under_heston",
    "pnl_from_deltas",
    "simulate_delta_hedge",
]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class HedgeResult:
    """Terminal P&L (in time-T money) of the short option plus hedge, one per path."""

    pnl: FloatArray
    premium: float
    n_rebalances: int

    @property
    def mean(self) -> float:
        """Average P&L."""
        return float(self.pnl.mean())

    @property
    def std(self) -> float:
        """Standard deviation of P&L."""
        return float(self.pnl.std(ddof=1))


def pnl_from_deltas(
    paths: FloatArray,
    deltas: FloatArray,
    option: EuropeanOption,
    *,
    rate: float,
    premium: float,
) -> HedgeResult:
    """Terminal P&L of a short option hedged with a given share count per date.

    ``deltas`` has shape (n_paths, n): the shares held after rebalancing at dates
    0..n-1. Everything is rolled forward at ``rate`` and settled at expiry.
    """
    n = paths.shape[1] - 1
    growth = math.exp(rate * option.maturity / n)
    kind = option.option_type

    shares = deltas[:, 0]
    cash = premium - shares * paths[:, 0]
    for j in range(1, n):
        new_shares = deltas[:, j]
        cash = cash * growth - (new_shares - shares) * paths[:, j]
        shares = new_shares

    terminal = paths[:, -1]
    payoff = np.maximum(kind.sign * (terminal - option.strike), 0.0)
    pnl = cash * growth + shares * terminal - payoff
    return HedgeResult(pnl=pnl, premium=premium, n_rebalances=n)


def hedge_on_paths(
    paths: FloatArray,
    option: EuropeanOption,
    *,
    rate: float,
    implied_vol: float,
    hedge_vol: float,
) -> HedgeResult:
    """Run the Black-Scholes delta hedge on given paths (columns are dates 0..n)."""
    n = paths.shape[1] - 1
    dt = option.maturity / n
    strike, kind = option.strike, option.option_type

    premium = float(bs.price(paths[0, 0], strike, option.maturity, rate, implied_vol, kind))
    deltas = np.empty((paths.shape[0], n))
    for j in range(n):
        remaining = option.maturity - j * dt
        deltas[:, j] = bs.delta(paths[:, j], strike, remaining, rate, hedge_vol, kind)
    return pnl_from_deltas(paths, deltas, option, rate=rate, premium=premium)


def simulate_delta_hedge(
    spot: float,
    option: EuropeanOption,
    *,
    rate: float,
    implied_vol: float,
    realized_vol: float,
    n_rebalances: int,
    n_paths: int,
    seed: SeedLike,
    hedge_vol: float | None = None,
    drift: float | None = None,
) -> HedgeResult:
    """Simulate paths with realized vol and hedge with ``hedge_vol`` (default implied)."""
    if option.option_type is not OptionType.CALL and option.option_type is not OptionType.PUT:
        raise ValueError("unsupported option type")
    world = MarketParams(
        spot=spot,
        volatility=realized_vol,
        rate=rate if drift is None else drift,
        dividend_yield=0.0,
    )
    paths = gbm_paths(world, option.maturity, draw_normals(n_paths, n_rebalances, seed))
    return hedge_on_paths(
        paths,
        option,
        rate=rate,
        implied_vol=implied_vol,
        hedge_vol=implied_vol if hedge_vol is None else hedge_vol,
    )


def _heston_delta_tables(
    p: HestonParams, remaining: float, k_grid: FloatArray, v_grid: FloatArray
) -> tuple[RegularGridInterpolator, RegularGridInterpolator]:
    """Heston call delta and minimum-variance delta on a (k, variance) grid.

    The call is S * n(k, v) with k = ln(S e^{r tau} / K), so the model delta
    (derivative at fixed variance) is n + dn/dk. Variance moves with the stock, with
    correlation rho, so the share count that minimizes the variance of the hedged
    position is Cov(dC, dS) / Var(dS) = delta + rho * xi * dn/dv.
    """
    n, dn = heston_call_normalized(k_grid, v_grid, remaining, p)
    dn_dv = np.gradient(n, v_grid, axis=1)
    model = RegularGridInterpolator((k_grid, v_grid), n + dn)
    min_var = RegularGridInterpolator((k_grid, v_grid), n + dn + p.rho * p.xi * dn_dv)
    return model, min_var


def hedge_under_heston(
    p: HestonParams,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    *,
    n_rebalances: int,
    n_paths: int,
    seed: SeedLike,
    substeps: int = 4,
) -> dict[str, HedgeResult]:
    """Sell a call at its Heston price and hedge it three ways in a Heston world.

    The stock follows the Heston model, drift ``rate``. The call is sold for its model
    price, and the same paths are hedged with:

    ``black_scholes_flat``
        BS delta at the call's implied vol at inception, never updated.
    ``black_scholes_instantaneous``
        BS delta at the current instantaneous vol sqrt(v).
    ``heston``
        The model delta (derivative at fixed variance), using the current state.
    ``heston_min_variance``
        The model delta plus rho * xi * dC/dv / S, which accounts for variance
        moving with the stock and is the variance-minimizing share count.

    The last two use the simulated variance, which a real hedger cannot observe, so
    they show what is available with perfect knowledge of the state, not what is
    achievable.
    """
    option = EuropeanOption(strike, maturity, OptionType.CALL)
    s, v = simulate_heston(p, spot, rate, maturity, n_rebalances, n_paths, seed, substeps=substeps)
    forward0 = spot * math.exp(rate * maturity)
    k0 = np.array([math.log(forward0 / strike)])
    n0, _ = heston_call_normalized(k0, np.array([p.v0]), maturity, p)
    premium = float(spot * n0[0, 0])
    flat_vol = float(_invert_normalized(n0[:, 0], k0, maturity)[0])

    dt = maturity / n_rebalances
    flat = np.empty((n_paths, n_rebalances))
    inst = np.empty_like(flat)
    model = np.empty_like(flat)
    min_var = np.empty_like(flat)
    k_grid = np.linspace(-0.8, 0.8, 81)
    v_grid = np.linspace(5e-4, max(0.5, 1.05 * float(v.max())), 30)
    for j in range(n_rebalances):
        remaining = maturity - j * dt
        flat[:, j] = bs.delta(s[:, j], strike, remaining, rate, flat_vol, OptionType.CALL)
        inst_vol = np.maximum(np.sqrt(v[:, j]), 1e-4)
        inst[:, j] = bs.delta(s[:, j], strike, remaining, rate, inst_vol, OptionType.CALL)
        table, table_mv = _heston_delta_tables(p, remaining, k_grid, v_grid)
        k_path = np.clip(np.log(s[:, j] * math.exp(rate * remaining) / strike), -0.8, 0.8)
        v_path = np.clip(v[:, j], v_grid[0], v_grid[-1])
        point = np.column_stack([k_path, v_path])
        model[:, j] = table(point)
        min_var[:, j] = table_mv(point)

    return {
        name: pnl_from_deltas(s, deltas, option, rate=rate, premium=premium)
        for name, deltas in (
            ("black_scholes_flat", flat),
            ("black_scholes_instantaneous", inst),
            ("heston", model),
            ("heston_min_variance", min_var),
        )
    }
