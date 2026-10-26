r"""Monte Carlo Greeks, two ways, to compare with the analytic ones.

Finite difference bumps an input and reprices. It only works with common random
numbers: both bumped runs must use the same normal draws, or the noise in each price
swamps the small difference being measured. The bump size trades truncation bias
(large h) against noise (small h).

Pathwise differentiates the payoff inside the expectation, for a call
(phi = +1, put = -1, ITM = 1 when phi * (S_T - K) > 0):

.. math::

    \Delta = e^{-rT} E\left[\phi\,\mathbf{1}_{ITM}\, S_T / S_0\right], \qquad
    \mathcal{V} = e^{-rT} E\left[\phi\,\mathbf{1}_{ITM}\, S_T
    \frac{\ln(S_T/S_0) - (r - q + \sigma^2/2)T}{\sigma}\right]

It uses one simulation and has lower variance, but it needs a payoff that is
continuous in the parameter. For a digital the derivative is zero almost everywhere
and the estimator is simply wrong (it returns 0). Gamma fails too, since the vanilla
payoff has a kink and no second derivative. Likelihood ratio is the usual fix there.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from pricer.instruments import EuropeanOption
from pricer.market import MarketParams
from pricer.monte_carlo import simulate_european
from pricer.paths import SeedLike, draw_normals

__all__ = ["FiniteDifferenceGreeks", "PathwiseGreeks", "fd_greeks", "pathwise_greeks"]


@dataclass(frozen=True, slots=True)
class FiniteDifferenceGreeks:
    """Raw-unit Greeks from bump-and-revalue."""

    delta: float
    gamma: float
    vega: float
    theta: float
    rho: float


@dataclass(frozen=True, slots=True)
class PathwiseGreeks:
    """Pathwise estimates, each with its standard error."""

    delta: float
    delta_se: float
    vega: float
    vega_se: float
    rho: float
    rho_se: float


def fd_greeks(
    market: MarketParams,
    option: EuropeanOption,
    *,
    n_paths: int,
    seed: int,
    rel_bump: float = 0.01,
    common_random_numbers: bool = True,
) -> FiniteDifferenceGreeks:
    """Central finite-difference Greeks.

    ``rel_bump`` scales the spot and vol bumps (relative); rate and maturity use
    fixed small absolute bumps. With ``common_random_numbers=False`` every bumped
    run gets its own seed, which is only here to show why that fails.
    """
    counter = iter(range(10_000))

    def price(mkt: MarketParams, opt: EuropeanOption) -> float:
        run_seed = seed if common_random_numbers else seed + 1 + next(counter)
        return simulate_european(mkt, opt, n_paths=n_paths, seed=run_seed).price

    s0, vol = market.spot, market.volatility
    hs, hv, hr = rel_bump * s0, rel_bump * vol, 1e-4
    ht = min(1e-2, 0.5 * option.maturity)

    base = price(market, option)
    up_s, dn_s = (
        price(market.with_(spot=s0 + hs), option),
        price(market.with_(spot=s0 - hs), option),
    )
    up_v = price(market.with_(volatility=vol + hv), option)
    dn_v = price(market.with_(volatility=vol - hv), option)
    up_r = price(market.with_(rate=market.rate + hr), option)
    dn_r = price(market.with_(rate=market.rate - hr), option)
    up_t = price(market, replace(option, maturity=option.maturity + ht))
    dn_t = price(market, replace(option, maturity=option.maturity - ht))

    return FiniteDifferenceGreeks(
        delta=(up_s - dn_s) / (2 * hs),
        gamma=(up_s - 2 * base + dn_s) / hs**2,
        vega=(up_v - dn_v) / (2 * hv),
        theta=-(up_t - dn_t) / (2 * ht),
        rho=(up_r - dn_r) / (2 * hr),
    )


def pathwise_greeks(
    market: MarketParams,
    option: EuropeanOption,
    *,
    n_paths: int,
    seed: SeedLike,
) -> PathwiseGreeks:
    """Pathwise delta, vega and rho for a European vanilla (one exact step)."""
    t, k = option.maturity, option.strike
    phi = option.option_type.sign
    sigma, r, q = market.volatility, market.rate, market.dividend_yield

    z = draw_normals(n_paths, 1, seed)[:, 0]
    log_ratio = (r - q - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z
    s_t = market.spot * np.exp(log_ratio)
    itm = phi * (s_t - k) > 0.0
    df = math.exp(-r * t)

    delta = df * phi * itm * s_t / market.spot
    vega = df * phi * itm * s_t * (log_ratio - (r - q + 0.5 * sigma**2) * t) / sigma
    payoff = np.maximum(phi * (s_t - k), 0.0)
    rho = df * (phi * itm * s_t * t - t * payoff)

    def stat(x: np.ndarray) -> tuple[float, float]:
        return float(x.mean()), float(x.std(ddof=1) / math.sqrt(n_paths))

    (d, d_se), (v, v_se), (rh, rh_se) = stat(delta), stat(vega), stat(rho)
    return PathwiseGreeks(d, d_se, v, v_se, rh, rh_se)
