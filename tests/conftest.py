"""Shared fixtures and numerical-differentiation helpers for the test suite.

The differentiation helpers here exist so that analytic Greeks can be checked
against the derivative of the *price function itself*, independently of any Monte
Carlo. Richardson extrapolation is used to push the truncation error far below the
tolerances asserted in the tests, so a failure indicates a genuine error in the
calculus rather than a badly chosen step size.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from pricer.instruments import OptionType

# A deliberately awkward parameter battery: deep in and out of the money, very
# short and very long dated, low and high volatility, with and without a yield.
PARAMETER_BATTERY: list[tuple[float, float, float, float, float, float]] = [
    # spot, strike, maturity, rate, volatility, dividend_yield
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
    (100.0, 100.0, 1.00, 0.05, 0.20, 0.03),
    (100.0, 80.0, 0.50, 0.03, 0.35, 0.01),
    (100.0, 130.0, 2.00, 0.04, 0.15, 0.02),
    (100.0, 100.0, 0.02, 0.05, 0.60, 0.00),
    (100.0, 100.0, 5.00, 0.01, 0.10, 0.05),
    (42.0, 40.0, 0.50, 0.10, 0.20, 0.00),
    (5.0, 100.0, 0.25, 0.05, 0.40, 0.00),
    (500.0, 100.0, 0.25, 0.05, 0.40, 0.00),
    (100.0, 100.0, 1.00, -0.005, 0.25, 0.00),
]


@pytest.fixture(params=list(OptionType))
def option_type(request: pytest.FixtureRequest) -> Iterator[OptionType]:
    """Parametrise a test over both calls and puts."""
    yield request.param


def central_difference(
    func: Callable[[float], float],
    x: float,
    step: float,
) -> float:
    """Return a Richardson-extrapolated first derivative of ``func`` at ``x``.

    Combines central differences at ``step`` and ``step / 2`` to cancel the
    leading :math:`O(h^2)` truncation term, leaving :math:`O(h^4)`.
    """
    coarse = (func(x + step) - func(x - step)) / (2.0 * step)
    fine = (func(x + step / 2.0) - func(x - step / 2.0)) / step
    return (4.0 * fine - coarse) / 3.0


def second_difference(
    func: Callable[[float], float],
    x: float,
    step: float,
) -> float:
    """Return a Richardson-extrapolated second derivative of ``func`` at ``x``.

    Second differences lose roughly twice as many digits to cancellation as first
    differences, so ``step`` must be chosen larger here than for
    :func:`central_difference`.
    """
    centre = func(x)
    coarse = (func(x + step) - 2.0 * centre + func(x - step)) / step**2
    half = step / 2.0
    fine = (func(x + half) - 2.0 * centre + func(x - half)) / half**2
    return (4.0 * fine - coarse) / 3.0
