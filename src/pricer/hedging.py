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

from pricer import black_scholes as bs
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from pricer.paths import SeedLike, draw_normals, gbm_paths

__all__ = ["HedgeResult", "hedge_on_paths", "simulate_delta_hedge"]

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


def hedge_on_paths(
    paths: FloatArray,
    option: EuropeanOption,
    *,
    rate: float,
    implied_vol: float,
    hedge_vol: float,
) -> HedgeResult:
    """Run the hedge on given paths, columns being the rebalance dates 0..n."""
    n = paths.shape[1] - 1
    dt = option.maturity / n
    growth = math.exp(rate * dt)
    strike, kind = option.strike, option.option_type

    premium = float(bs.price(paths[0, 0], strike, option.maturity, rate, implied_vol, kind))

    def delta_at(j: int) -> FloatArray:
        remaining = option.maturity - j * dt
        return np.asarray(
            bs.delta(paths[:, j], strike, remaining, rate, hedge_vol, kind), dtype=np.float64
        )

    # The option is sold, so its price is received and the hedge is bought with it.
    shares = delta_at(0)
    cash = premium - shares * paths[:, 0]
    for j in range(1, n):
        new_shares = delta_at(j)
        cash = cash * growth - (new_shares - shares) * paths[:, j]
        shares = new_shares

    terminal = paths[:, -1]
    payoff = np.maximum(kind.sign * (terminal - strike), 0.0)
    pnl = cash * growth + shares * terminal - payoff
    return HedgeResult(pnl=pnl, premium=premium, n_rebalances=n)


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
