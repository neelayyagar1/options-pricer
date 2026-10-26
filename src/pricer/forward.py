r"""Implied forward and discount factor from put-call parity.

For European options on the same expiry, C(K) - P(K) = D (F - K), so regressing
C - P on K gives the discount factor D from the slope and the forward F from the
intercept (b = D F). That avoids sourcing a risk-free rate and a dividend yield
separately, and it is consistent with the quotes the volatilities are inverted from.

In practice the slope is not well determined: over a narrow strike range it is noisy
(implied rates of -50% show up on short expiries), and SPY options are American so
parity only holds approximately. So the default is to fix D from a supplied rate and
estimate F as the average of K + (C - P) / D, which is stable. The free regression
(``rate=None``) is kept as a diagnostic and is exact on clean European quotes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pricer.data.loader import ExpiryChain

__all__ = ["ImpliedForward", "implied_forward"]


@dataclass(frozen=True, slots=True)
class ImpliedForward:
    """Result of the parity regression for one expiry."""

    maturity: float
    discount_factor: float
    forward: float
    r_squared: float
    n_strikes: int

    @property
    def rate(self) -> float:
        """Continuously compounded rate, -ln(D) / T."""
        return -math.log(self.discount_factor) / self.maturity

    def dividend_yield(self, spot: float) -> float:
        """Dividend yield implied by F = S exp((r - q) T)."""
        return self.rate - math.log(self.forward / spot) / self.maturity


def implied_forward(
    chain: ExpiryChain,
    spot: float,
    *,
    band: float = 0.05,
    max_relative_spread: float = 0.25,
    min_strikes: int = 5,
    rate: float | None = None,
) -> ImpliedForward:
    """Fit C - P = D (F - K) over liquid strikes within ``band`` of spot.

    With ``rate`` given, D = exp(-rate * T) is fixed and only F is estimated.
    """
    calls, puts = chain.calls, chain.puts
    call_mid = dict(zip(calls.strike, calls.mid, strict=True))
    call_spread = dict(zip(calls.strike, calls.relative_spread, strict=True))
    put_mid = dict(zip(puts.strike, puts.mid, strict=True))
    put_spread = dict(zip(puts.strike, puts.relative_spread, strict=True))

    strikes, diffs = [], []
    for k in sorted(set(call_mid) & set(put_mid)):
        if abs(k / spot - 1.0) > band:
            continue
        c, p = call_mid[k], put_mid[k]
        if np.isnan(c) or np.isnan(p):
            continue
        if call_spread[k] > max_relative_spread or put_spread[k] > max_relative_spread:
            continue
        strikes.append(k)
        diffs.append(c - p)

    if len(strikes) < min_strikes:
        raise ValueError(f"only {len(strikes)} usable strikes, need {min_strikes}")

    x, y = np.asarray(strikes), np.asarray(diffs)
    if rate is None:
        slope, intercept = np.polyfit(x, y, deg=1)
        if slope >= 0.0:
            raise ValueError("parity regression slope is not negative, quotes are unusable")
        discount = -float(slope)
    else:
        discount = math.exp(-rate * chain.maturity)
        intercept = float(np.mean(y + discount * x))
        slope = -discount
    fitted = slope * x + intercept
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return ImpliedForward(
        maturity=chain.maturity,
        discount_factor=discount,
        forward=float(intercept) / discount,
        r_squared=1.0 - ss_res / ss_tot,
        n_strikes=len(strikes),
    )
