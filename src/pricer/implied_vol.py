"""Implied volatility: invert Black-Scholes for sigma.

Newton-Raphson using the analytic vega, with Brent's method as the fallback when
Newton misbehaves (it does for deep out-of-the-money options where vega is close to
zero). Prices outside the no-arbitrage bounds are rejected instead of returning a
meaningless root.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from pricer import black_scholes as bs
from pricer.data.loader import ExpiryChain
from pricer.forward import ImpliedForward
from pricer.instruments import OptionType

__all__ = ["implied_vol", "otm_smile"]

SIGMA_LOW = 1e-6
SIGMA_HIGH = 5.0


def implied_vol(
    price: float,
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    *,
    tol: float = 1e-10,
    max_iter: int = 50,
) -> float:
    """Return sigma such that the Black-Scholes price equals ``price``."""
    discounted_spot = spot * math.exp(-dividend_yield * maturity)
    discounted_strike = strike * math.exp(-rate * maturity)
    if option_type is OptionType.CALL:
        lower, upper = max(discounted_spot - discounted_strike, 0.0), discounted_spot
    else:
        lower, upper = max(discounted_strike - discounted_spot, 0.0), discounted_strike
    if not lower < price < upper:
        raise ValueError(f"price {price} outside no-arbitrage bounds ({lower}, {upper})")

    def model(sigma: float) -> float:
        return float(bs.price(spot, strike, maturity, rate, sigma, option_type, dividend_yield))

    # Brenner-Subrahmanyam starting guess, clipped to the search interval.
    sigma = min(max(math.sqrt(2.0 * math.pi / maturity) * price / spot, 0.05), 2.0)
    for _ in range(max_iter):
        diff = model(sigma) - price
        if abs(diff) < tol:
            return sigma
        vega = float(bs.vega(spot, strike, maturity, rate, sigma, option_type, dividend_yield))
        if vega < 1e-12:
            break
        step = diff / vega
        sigma_new = sigma - step
        if not SIGMA_LOW < sigma_new < SIGMA_HIGH:
            break
        sigma = sigma_new

    return float(brentq(lambda s: model(s) - price, SIGMA_LOW, SIGMA_HIGH, xtol=1e-14, rtol=1e-12))


def otm_smile(
    chain: ExpiryChain,
    fwd: ImpliedForward,
    *,
    max_relative_spread: float = 0.25,
    min_mid: float = 0.05,
    log_moneyness_range: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Implied vols from out-of-the-money mids, as (log-moneyness k, sigma).

    Calls are used above the forward and puts below it. OTM options are the liquid
    ones, and for American-style quotes they carry little early-exercise premium.
    Options are priced with the implied forward and discount factor (Black-76 form:
    spot = F, q = r), so the surface is consistent with parity.
    """
    r, f = fwd.rate, fwd.forward
    ks, vols = [], []
    for quotes, option_type in ((chain.calls, OptionType.CALL), (chain.puts, OptionType.PUT)):
        mid, spread = quotes.mid, quotes.relative_spread
        for strike, m, s in zip(quotes.strike, mid, spread, strict=True):
            k = math.log(strike / f)
            if (
                np.isnan(m)
                or m < min_mid
                or s > max_relative_spread
                or abs(k) > log_moneyness_range
            ):
                continue
            if (option_type is OptionType.CALL) != (strike >= f):
                continue
            try:
                vol = implied_vol(m, f, strike, chain.maturity, r, option_type, r)
            except ValueError:
                continue
            ks.append(k)
            vols.append(vol)
    order = np.argsort(ks)
    return np.asarray(ks)[order], np.asarray(vols)[order]
