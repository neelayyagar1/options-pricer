r"""Closed form for a discretely monitored geometric Asian option.

The Kemna-Vorst formula assumes continuous averaging. The simulation averages over n
dates, so the control variate has to use the discrete version or its known mean would
not match what is simulated, and the price would be biased.

With monitoring dates t_i = i*dt, i = 1..n, T = n*dt, ln G = (1/n) sum ln S_{t_i} is
normal with

.. math::

    \mu_G = \ln S_0 + (r - q - \tfrac{1}{2}\sigma^2)\,\frac{(n+1)\,dt}{2}, \qquad
    \sigma_G^2 = \frac{\sigma^2\, dt\,(n+1)(2n+1)}{6n}

The variance comes from sum_{i,j} min(i, j) = n(n+1)(2n+1)/6. The price is then a
Black-Scholes style expression in G.
"""

from __future__ import annotations

import math

from scipy.special import ndtr

from pricer.instruments import OptionType

__all__ = ["geometric_asian_moments", "geometric_asian_price"]


def geometric_asian_moments(
    spot: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_dates: int,
    dividend_yield: float = 0.0,
) -> tuple[float, float]:
    """Return (mu_G, sigma_G^2), the mean and variance of ln G."""
    if n_dates <= 0:
        raise ValueError(f"n_dates must be positive, got {n_dates}")
    dt = maturity / n_dates
    drift = rate - dividend_yield - 0.5 * volatility**2
    mu = math.log(spot) + drift * dt * (n_dates + 1) / 2.0
    var = volatility**2 * dt * (n_dates + 1) * (2 * n_dates + 1) / (6.0 * n_dates)
    return mu, var


def geometric_asian_price(
    spot: float,
    strike: float,
    maturity: float,
    rate: float,
    volatility: float,
    n_dates: int,
    option_type: OptionType,
    dividend_yield: float = 0.0,
) -> float:
    """Price a discretely monitored geometric-average Asian option."""
    mu, var = geometric_asian_moments(spot, maturity, rate, volatility, n_dates, dividend_yield)
    discount = math.exp(-rate * maturity)
    if var <= 0.0:
        forward_g = math.exp(mu)
        return discount * max(option_type.sign * (forward_g - strike), 0.0)

    sd = math.sqrt(var)
    mean_g = math.exp(mu + 0.5 * var)
    d1 = (mu - math.log(strike) + var) / sd
    d2 = d1 - sd
    phi = option_type.sign
    return discount * phi * (mean_g * float(ndtr(phi * d1)) - strike * float(ndtr(phi * d2)))
