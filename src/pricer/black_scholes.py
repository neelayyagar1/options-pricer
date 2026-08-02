r"""Black-Scholes-Merton closed-form prices and analytic Greeks.

Everything here is exact: no discretisation, no simulation. That is the point —
this module is the ground truth against which the Monte Carlo engine in
:mod:`pricer.monte_carlo` is validated, so any error here would silently
invalidate the correctness proof that the rest of the project rests on.

The model
---------
Under the risk-neutral measure the underlying follows

.. math:: dS_t = (r - q) S_t\,dt + \sigma S_t\,dW_t

and the value of a European option with :math:`\phi = +1` for a call and
:math:`\phi = -1` for a put is

.. math::

    V = \phi\left[S e^{-qT} N(\phi d_1) - K e^{-rT} N(\phi d_2)\right],
    \qquad
    d_{1,2} = \frac{\ln(S/K) + (r - q \pm \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}

Notice what is *absent*: the real-world drift :math:`\mu`. The hedging argument
removes risk preferences from the problem entirely, which is why the one parameter
that would be hardest to estimate never has to be estimated.

Conventions
-----------
All Greeks are returned in raw mathematical units — per unit of the underlying,
per unit of volatility, per *year* of time. Trading desks quote vega per volatility
*point* and theta per calendar *day*; :class:`Greeks` exposes both, because
silently mixing the two conventions is a common and very visible error.

Degenerate inputs
-----------------
When :math:`\sigma\sqrt{T} = 0` the option has no optionality left and the
formulas above are ``0/0``. Rather than returning ``nan``, this module returns the
correct limits (see :func:`price` and :func:`greeks`), which keeps the boundaries
of a Greeks surface usable and makes the ``T -> 0`` and ``sigma -> 0`` unit tests
meaningful rather than skipped.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, cast

import numpy as np
import numpy.typing as npt
from scipy.special import ndtr

from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams

__all__ = [
    "DAYS_PER_YEAR",
    "VOL_POINT",
    "Greeks",
    "delta",
    "gamma",
    "greeks",
    "greeks_contract",
    "price",
    "price_contract",
    "rho",
    "theta",
    "vega",
]

FloatArray = npt.NDArray[np.float64]
ArrayLike = float | FloatArray
Number = np.float64 | FloatArray

#: :math:`1/\sqrt{2\pi}`, the standard normal density's normalising constant.
_INV_SQRT_2PI = 0.398_942_280_401_432_7

#: Calendar days per year, for quoting theta per day.
DAYS_PER_YEAR = 365.0

#: One volatility point (1%), for quoting vega per point.
VOL_POINT = 0.01


@dataclass(frozen=True, slots=True)
class Greeks:
    r"""First- and second-order sensitivities in raw units.

    Attributes
    ----------
    price
        Option value.
    delta
        :math:`\partial V/\partial S`, per unit of underlying.
    gamma
        :math:`\partial^2 V/\partial S^2`.
    vega
        :math:`\partial V/\partial \sigma`, per unit of volatility (i.e. per
        100 volatility points). See :attr:`vega_per_point`.
    theta
        :math:`\partial V/\partial t`, per year. Negative for most long options.
        See :attr:`theta_per_day`.
    rho
        :math:`\partial V/\partial r`, per unit of rate.
    """

    price: Number
    delta: Number
    gamma: Number
    vega: Number
    theta: Number
    rho: Number

    @property
    def vega_per_point(self) -> Number:
        """Return vega per one volatility point (1%), the desk convention."""
        return self.vega * VOL_POINT

    @property
    def theta_per_day(self) -> Number:
        """Return theta per calendar day, the desk convention."""
        return self.theta / DAYS_PER_YEAR


class _Core(NamedTuple):
    """Shared intermediates, computed once per call."""

    spot: FloatArray
    strike: FloatArray
    maturity: FloatArray
    d1: FloatArray
    d2: FloatArray
    volatility: FloatArray
    total_vol: FloatArray
    r""":math:`\sigma\sqrt{T}`, with zeros replaced by one to guard division."""
    sqrt_maturity: FloatArray
    dividend_discount: FloatArray
    """:math:`e^{-qT}`."""
    rate_discount: FloatArray
    """:math:`e^{-rT}`."""
    discounted_spot: FloatArray
    """:math:`S e^{-qT}`."""
    discounted_strike: FloatArray
    """:math:`K e^{-rT}`."""
    rate: FloatArray
    dividend_yield: FloatArray
    degenerate: npt.NDArray[np.bool_]
    r"""True where :math:`\sigma\sqrt{T} = 0` and the closed form is ``0/0``."""


def _norm_pdf(x: FloatArray) -> FloatArray:
    """Return the standard normal density :math:`n(x)`."""
    return _INV_SQRT_2PI * np.exp(-0.5 * x * x)


def _scalarize(values: FloatArray) -> Number:
    """Collapse a 0-d array to a NumPy scalar, leaving real arrays untouched.

    Keeps scalar inputs producing scalar-looking outputs without giving up
    vectorisation for array inputs.
    """
    return cast(Number, values[()])


def _core(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    dividend_yield: ArrayLike,
) -> _Core:
    """Validate, broadcast, and compute the shared intermediates.

    Denominators are made safe *before* division rather than cleaned up after, so
    no spurious ``RuntimeWarning`` is ever raised for inputs that are perfectly
    legitimate (an expired option, a zero-volatility limit).
    """
    s, k, t, r, sigma, q = (
        np.asarray(value, dtype=np.float64)
        for value in (spot, strike, maturity, rate, volatility, dividend_yield)
    )

    if np.any(s <= 0.0):
        raise ValueError("spot must be positive")
    if np.any(k <= 0.0):
        raise ValueError("strike must be positive")
    if np.any(t < 0.0):
        raise ValueError("maturity must be non-negative")
    if np.any(sigma < 0.0):
        raise ValueError("volatility must be non-negative")

    s, k, t, r, sigma, q = np.broadcast_arrays(s, k, t, r, sigma, q)

    sqrt_t = np.sqrt(t)
    total_vol = sigma * sqrt_t
    degenerate = total_vol <= 0.0
    safe_total_vol = np.where(degenerate, 1.0, total_vol)

    d1 = (np.log(s / k) + (r - q) * t) / safe_total_vol + 0.5 * safe_total_vol
    d2 = d1 - safe_total_vol

    dividend_discount = np.exp(-q * t)
    rate_discount = np.exp(-r * t)

    return _Core(
        spot=s,
        strike=k,
        maturity=t,
        d1=d1,
        d2=d2,
        volatility=sigma,
        total_vol=safe_total_vol,
        sqrt_maturity=sqrt_t,
        dividend_discount=dividend_discount,
        rate_discount=rate_discount,
        discounted_spot=s * dividend_discount,
        discounted_strike=k * rate_discount,
        rate=r,
        dividend_yield=q,
        degenerate=degenerate,
    )


def _degenerate_indicator(core: _Core, phi: float) -> FloatArray:
    r"""Return the :math:`\sigma\sqrt{T} \to 0` limit of :math:`N(\phi d_{1,2})`.

    With no volatility the terminal spot is deterministic, so both normal CDFs
    collapse to the same indicator of finishing in the money against the *forward*.
    The at-the-money-forward case is assigned ``0.5``, the symmetric limit.
    """
    moneyness = phi * (core.discounted_spot - core.discounted_strike)
    return cast(
        FloatArray,
        np.where(moneyness > 0.0, 1.0, np.where(moneyness < 0.0, 0.0, 0.5)),
    )


def price(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Number:
    r"""Return the Black-Scholes-Merton value of a European option.

    All numeric arguments broadcast against one another, so a whole strike ladder
    or volatility surface can be priced in a single call.

    Parameters
    ----------
    spot
        Underlying price :math:`S`.
    strike
        Strike :math:`K`.
    maturity
        Time to expiry :math:`T` in years.
    rate
        Continuously compounded risk-free rate :math:`r`.
    volatility
        Annualised volatility :math:`\sigma`.
    option_type
        Call or put.
    dividend_yield
        Continuous dividend yield :math:`q`.

    Returns
    -------
    numpy.float64 or numpy.ndarray
        Option value. Where :math:`\sigma\sqrt{T} = 0` the discounted intrinsic
        value :math:`\max(\phi(Se^{-qT} - Ke^{-rT}), 0)` is returned, which is the
        exact limit and covers both an expired option and a zero-volatility one.
    """
    core = _core(spot, strike, maturity, rate, volatility, dividend_yield)
    phi = option_type.sign

    regular = phi * (
        core.discounted_spot * ndtr(phi * core.d1) - core.discounted_strike * ndtr(phi * core.d2)
    )
    limit = np.maximum(phi * (core.discounted_spot - core.discounted_strike), 0.0)

    return _scalarize(np.where(core.degenerate, limit, regular))


def delta(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Number:
    r"""Return :math:`\partial V/\partial S = \phi e^{-qT} N(\phi d_1)`."""
    return greeks(spot, strike, maturity, rate, volatility, option_type, dividend_yield).delta


def gamma(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Number:
    r"""Return :math:`\partial^2 V/\partial S^2`, identical for calls and puts."""
    return greeks(spot, strike, maturity, rate, volatility, option_type, dividend_yield).gamma


def vega(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Number:
    r"""Return :math:`\partial V/\partial\sigma`, identical for calls and puts.

    This must be exact: the Newton-Raphson implied-volatility solver in
    :mod:`pricer.implied_vol` uses it as its derivative.
    """
    return greeks(spot, strike, maturity, rate, volatility, option_type, dividend_yield).vega


def theta(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Number:
    r"""Return :math:`\partial V/\partial t` per year (note: *not* per day)."""
    return greeks(spot, strike, maturity, rate, volatility, option_type, dividend_yield).theta


def rho(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Number:
    r"""Return :math:`\partial V/\partial r`, holding the dividend yield fixed."""
    return greeks(spot, strike, maturity, rate, volatility, option_type, dividend_yield).rho


def greeks(
    spot: ArrayLike,
    strike: ArrayLike,
    maturity: ArrayLike,
    rate: ArrayLike,
    volatility: ArrayLike,
    option_type: OptionType,
    dividend_yield: ArrayLike = 0.0,
) -> Greeks:
    r"""Return price and all analytic Greeks, sharing intermediates.

    Every quantity is differentiated in closed form rather than bumped, so these
    values carry no truncation error and serve as the reference against which the
    Monte Carlo estimators in :mod:`pricer.greeks` are judged.

    Notes
    -----
    In the degenerate limit :math:`\sigma\sqrt{T} \to 0` the normal CDFs collapse
    to an indicator (see :func:`_degenerate_indicator`) and gamma and vega vanish.
    Gamma's true limit is a Dirac delta concentrated at the forward-at-the-money
    point; ``0`` is returned as the almost-everywhere limit.
    """
    core = _core(spot, strike, maturity, rate, volatility, dividend_yield)
    phi = option_type.sign

    n_d1 = _norm_pdf(core.d1)
    cdf_d1 = ndtr(phi * core.d1)
    cdf_d2 = ndtr(phi * core.d2)
    indicator = _degenerate_indicator(core, phi)

    # Denominators are guarded so the masked-out degenerate entries never divide
    # by zero; their values are discarded by the np.where below.
    safe_two_sqrt_t = np.where(core.degenerate, 1.0, 2.0 * core.sqrt_maturity)

    value = np.where(
        core.degenerate,
        np.maximum(phi * (core.discounted_spot - core.discounted_strike), 0.0),
        phi * (core.discounted_spot * cdf_d1 - core.discounted_strike * cdf_d2),
    )

    # delta = phi e^{-qT} N(phi d1)
    delta_value = np.where(
        core.degenerate,
        phi * core.dividend_discount * indicator,
        phi * core.dividend_discount * cdf_d1,
    )

    # gamma = e^{-qT} n(d1) / (S sigma sqrt(T)).  Note the discount factor, not the
    # discounted *spot*: gamma differentiates delta, which has already shed one
    # factor of S. Its true limit as sigma sqrt(T) -> 0 is a Dirac delta at the
    # forward-at-the-money point, so 0 is returned as the almost-everywhere limit.
    gamma_value = np.where(
        core.degenerate,
        0.0,
        core.dividend_discount * n_d1 / (core.spot * core.total_vol),
    )

    vega_value = np.where(
        core.degenerate,
        0.0,
        core.discounted_spot * n_d1 * core.sqrt_maturity,
    )

    # theta = -S e^{-qT} n(d1) sigma / (2 sqrt(T))  +  phi (q S e^{-qT} N(phi d1)
    #                                                      - r K e^{-rT} N(phi d2))
    # The first term is the decay of optionality and vanishes as sigma -> 0; the
    # second is the carry on the two legs of the replicating portfolio, which
    # survives the limit weighted by the in-the-money indicator.
    time_decay = -core.discounted_spot * n_d1 * core.volatility / safe_two_sqrt_t
    carry = phi * (
        core.dividend_yield * core.discounted_spot * cdf_d1
        - core.rate * core.discounted_strike * cdf_d2
    )
    degenerate_carry = (
        phi
        * indicator
        * (core.dividend_yield * core.discounted_spot - core.rate * core.discounted_strike)
    )
    theta_value = np.where(core.degenerate, degenerate_carry, time_decay + carry)

    # rho = phi K T e^{-rT} N(phi d2), i.e. the discounted strike leg scaled by
    # its own maturity. The CDF collapses to the indicator in the degenerate limit.
    rho_scale = phi * core.discounted_strike * core.maturity
    rho_value = np.where(core.degenerate, rho_scale * indicator, rho_scale * cdf_d2)

    return Greeks(
        price=_scalarize(value),
        delta=_scalarize(delta_value),
        gamma=_scalarize(gamma_value),
        vega=_scalarize(vega_value),
        theta=_scalarize(theta_value),
        rho=_scalarize(rho_value),
    )


def price_contract(option: EuropeanOption, market: MarketParams) -> Number:
    """Price a :class:`~pricer.instruments.EuropeanOption` in a given market."""
    return price(
        market.spot,
        option.strike,
        option.maturity,
        market.rate,
        market.volatility,
        option.option_type,
        market.dividend_yield,
    )


def greeks_contract(option: EuropeanOption, market: MarketParams) -> Greeks:
    """Return price and Greeks for a contract in a given market."""
    return greeks(
        market.spot,
        option.strike,
        option.maturity,
        market.rate,
        market.volatility,
        option.option_type,
        market.dividend_yield,
    )
