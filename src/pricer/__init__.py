"""Options pricing engine: closed-form, Monte Carlo, and empirical volatility analysis.

The package is organised so that each module answers one question:

``black_scholes``
    What is the exact value under the Black-Scholes-Merton assumptions?
``paths`` / ``payoffs`` / ``estimators`` / ``monte_carlo``
    What is the value when no closed form exists, and how large is the error?
``binomial`` / ``american``
    What is it worth if the holder may exercise early?
``greeks``
    How does the value move, computed three independent ways?
``forward`` / ``implied_vol`` / ``surface``
    What do live market prices imply, and where does the model break?
``hedging``
    What does that breakage cost when the position is actually hedged?
"""

from __future__ import annotations

from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams

__all__ = ["EuropeanOption", "MarketParams", "OptionType"]

__version__ = "0.1.0"
