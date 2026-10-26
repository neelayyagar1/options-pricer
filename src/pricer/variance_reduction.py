r"""Control variate estimator.

Price Y with a correlated payoff X whose expectation is known exactly:

.. math::

    \hat{Y}_{cv} = \bar{Y} - c\,(\bar{X} - E[X]), \qquad
    c^* = \frac{\mathrm{Cov}(Y, X)}{\mathrm{Var}(X)}

The coefficient is estimated on a separate pilot run. Estimating it on the same paths
that get averaged adds a small O(1/N) bias, and the pilot avoids that for free.
The standard error comes from the adjusted samples Y - c(X - E[X]), not from Y.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from pricer.estimators import MCResult, RunningMoments, reduce_antithetic
from pricer.market import MarketParams
from pricer.monte_carlo import DEFAULT_MAX_ELEMENTS, _as_seed_sequence, _chunk_size
from pricer.paths import SeedLike, antithetic_normals, draw_normals, gbm_paths
from pricer.payoffs import Payoff

__all__ = ["ControlVariateResult", "simulate_control_variate"]


@dataclass(frozen=True, slots=True)
class ControlVariateResult:
    """Price plus the coefficient and pilot correlation behind it."""

    result: MCResult
    coefficient: float
    correlation: float


def simulate_control_variate(
    market: MarketParams,
    maturity: float,
    payoff: Payoff,
    control: Payoff,
    control_price: float,
    *,
    n_paths: int,
    seed: SeedLike,
    n_steps: int,
    n_pilot: int = 20_000,
    antithetic: bool = False,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> ControlVariateResult:
    """Price ``payoff`` using ``control`` as a control variate.

    ``control_price`` is the exact *discounted* expectation of the control payoff.
    """
    if n_paths <= 0 or n_pilot < 2:
        raise ValueError("n_paths must be positive and n_pilot at least 2")
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"antithetic sampling requires an even n_paths, got {n_paths}")

    discount = market.discount_factor(maturity)
    pilot_seed, main_seed = _as_seed_sequence(seed).spawn(2)

    pilot_paths = gbm_paths(market, maturity, draw_normals(n_pilot, n_steps, pilot_seed))
    y_pilot = discount * payoff(pilot_paths)
    x_pilot = discount * control(pilot_paths)
    var_x = float(np.var(x_pilot, ddof=1))
    if var_x <= 0.0:
        raise ValueError("control payoff has zero variance in the pilot run")
    cov = float(np.cov(y_pilot, x_pilot, ddof=1)[0, 1])
    coefficient = cov / var_x
    correlation = cov / math.sqrt(var_x * float(np.var(y_pilot, ddof=1)))

    n_samples = n_paths // 2 if antithetic else n_paths
    chunk = _chunk_size(n_steps, max_elements)
    children = main_seed.spawn(math.ceil(n_samples / chunk))

    started = time.perf_counter()
    moments = RunningMoments()
    remaining = n_samples
    for child in children:
        size = min(chunk, remaining)
        remaining -= size
        normals = (
            antithetic_normals(size, n_steps, child)
            if antithetic
            else draw_normals(size, n_steps, child)
        )
        paths = gbm_paths(market, maturity, normals)
        adjusted = discount * payoff(paths) - coefficient * (
            discount * control(paths) - control_price
        )
        if antithetic:
            adjusted = reduce_antithetic(adjusted)
        moments.update(adjusted)

    result = MCResult(
        price=moments.mean,
        standard_error=moments.standard_error,
        n_samples=moments.count,
        n_paths=n_paths,
        elapsed_seconds=time.perf_counter() - started,
    )
    return ControlVariateResult(result=result, coefficient=coefficient, correlation=correlation)
