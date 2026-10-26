"""Cox-Ross-Rubinstein binomial tree.

This is the independent check on the American Monte Carlo: a lattice and a regression
based simulation agreeing means more than two simulations agreeing.
"""

from __future__ import annotations

import math

import numpy as np

from pricer.instruments import OptionType
from pricer.market import MarketParams

__all__ = ["binomial_price"]


def _tree_price(
    market: MarketParams,
    strike: float,
    maturity: float,
    option_type: OptionType,
    n_steps: int,
    american: bool,
) -> float:
    dt = maturity / n_steps
    up = math.exp(market.volatility * math.sqrt(dt))
    down = 1.0 / up
    growth = math.exp(market.cost_of_carry * dt)
    p_up = (growth - down) / (up - down)
    if not 0.0 < p_up < 1.0:
        raise ValueError("tree is not arbitrage free, increase n_steps")
    df = math.exp(-market.rate * dt)
    phi = option_type.sign

    j = np.arange(n_steps + 1)
    spots = market.spot * up ** (n_steps - j) * down**j
    values = np.maximum(phi * (spots - strike), 0.0)
    for step in range(n_steps - 1, -1, -1):
        values = df * (p_up * values[:-1] + (1.0 - p_up) * values[1:])
        if american:
            j = np.arange(step + 1)
            spots = market.spot * up ** (step - j) * down**j
            values = np.maximum(values, phi * (spots - strike))
    return float(values[0])


def binomial_price(
    market: MarketParams,
    strike: float,
    maturity: float,
    option_type: OptionType,
    *,
    american: bool = False,
    n_steps: int = 2000,
) -> float:
    """Price a vanilla on a CRR tree.

    Tree prices oscillate with the step count, so this averages the n and n+1 step
    trees to damp that.
    """
    if n_steps < 2:
        raise ValueError("n_steps must be at least 2")
    a = _tree_price(market, strike, maturity, option_type, n_steps, american)
    b = _tree_price(market, strike, maturity, option_type, n_steps + 1, american)
    return 0.5 * (a + b)
