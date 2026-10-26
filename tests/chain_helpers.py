"""Build synthetic chains from Black-Scholes so parity and IV code have a known answer."""

from __future__ import annotations

from collections.abc import Callable
from datetime import date

import numpy as np

from pricer import black_scholes as bs
from pricer.data.loader import ExpiryChain, QuoteSet
from pricer.instruments import OptionType


def synthetic_expiry(
    spot: float,
    maturity: float,
    rate: float,
    dividend_yield: float,
    vol: float = 0.2,
    *,
    vol_of_k: Callable[[np.ndarray], np.ndarray] | None = None,
    half_spread: float = 0.0,
    noise: float = 0.0,
    seed: int = 0,
) -> ExpiryChain:
    """European chain priced from a flat vol or a callable vol_of_k(log K/F)."""
    rng = np.random.default_rng(seed)
    strikes = np.arange(round(spot * 0.8), round(spot * 1.2) + 1, 1.0)
    forward = spot * np.exp((rate - dividend_yield) * maturity)

    def quotes(option_type: OptionType) -> QuoteSet:
        sigma: float | np.ndarray = vol if vol_of_k is None else vol_of_k(np.log(strikes / forward))
        mid = np.asarray(
            bs.price(spot, strikes, maturity, rate, sigma, option_type, dividend_yield)
        ) + noise * rng.standard_normal(strikes.size)
        mid = np.maximum(mid, 0.02)
        return QuoteSet(
            strike=strikes,
            bid=mid - half_spread,
            ask=mid + half_spread,
            last=mid,
            volume=np.full(strikes.size, 100.0),
            open_interest=np.full(strikes.size, 100.0),
        )

    return ExpiryChain(date(2030, 1, 1), maturity, quotes(OptionType.CALL), quotes(OptionType.PUT))
