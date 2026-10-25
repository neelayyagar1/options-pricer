r"""American options by Longstaff-Schwartz least squares Monte Carlo.

Go backward from expiry. At each exercise date, regress the discounted cashflow the
holder actually goes on to receive on the current spot, using only in-the-money
paths. Exercise where intrinsic value beats the fitted continuation value.

Fitting the policy and valuing it on the same paths lets the regression see the
outcomes it is predicting, which biases the price up. So this is done in two passes:
the policy is fitted on one set of paths and valued on an independent set. Any fixed
policy is at best optimal, so the result is a low-biased estimate (a lower bound).

The regression is a scikit-learn estimator, polynomial features into a linear model
by default, and any estimator with ``fit``/``predict`` can be passed in.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from pricer.estimators import MCResult
from pricer.instruments import OptionType
from pricer.market import MarketParams
from pricer.monte_carlo import _as_seed_sequence
from pricer.paths import SeedLike, draw_normals, gbm_paths

__all__ = ["default_regressor", "price_american_lsm"]

FloatArray = npt.NDArray[np.float64]
RegressorFactory = Callable[[], Any]


def default_regressor(degree: int = 3) -> Pipeline:
    """Polynomial basis in S/K feeding ordinary least squares."""
    return make_pipeline(PolynomialFeatures(degree, include_bias=False), LinearRegression())


def price_american_lsm(
    market: MarketParams,
    strike: float,
    maturity: float,
    option_type: OptionType,
    *,
    n_paths: int,
    n_steps: int,
    seed: SeedLike,
    regressor_factory: RegressorFactory = default_regressor,
    min_itm_paths: int = 20,
) -> MCResult:
    """Price an American vanilla, exercisable on ``n_steps`` equally spaced dates."""
    if n_paths < 2 or n_steps < 2:
        raise ValueError("need at least 2 paths and 2 steps")

    train_seed, value_seed = _as_seed_sequence(seed).spawn(2)
    dt = maturity / n_steps
    df = math.exp(-market.rate * dt)
    phi = option_type.sign
    started = time.perf_counter()

    # Pass 1: fit one regression per exercise date, working backward.
    train = gbm_paths(market, maturity, draw_normals(n_paths, n_steps, train_seed))
    cash = np.maximum(phi * (train[:, -1] - strike), 0.0)
    models: dict[int, Any] = {}
    for t in range(n_steps - 1, 0, -1):
        cash = cash * df
        spot = train[:, t]
        intrinsic = np.maximum(phi * (spot - strike), 0.0)
        itm = intrinsic > 0.0
        if itm.sum() < min_itm_paths:
            continue
        model = regressor_factory()
        model.fit((spot[itm] / strike).reshape(-1, 1), cash[itm])
        models[t] = model
        continuation = model.predict((spot[itm] / strike).reshape(-1, 1))
        exercise = intrinsic[itm] > continuation
        idx = np.flatnonzero(itm)[exercise]
        cash[idx] = intrinsic[idx]

    # Pass 2: value that policy on fresh paths, going forward.
    fresh = gbm_paths(market, maturity, draw_normals(n_paths, n_steps, value_seed))
    value = np.zeros(n_paths)
    alive = np.ones(n_paths, dtype=bool)
    for t in range(1, n_steps):
        model = models.get(t)
        if model is None:
            continue
        spot = fresh[:, t]
        intrinsic = np.maximum(phi * (spot - strike), 0.0)
        cand = alive & (intrinsic > 0.0)
        if not cand.any():
            continue
        continuation = model.predict((spot[cand] / strike).reshape(-1, 1))
        idx = np.flatnonzero(cand)[intrinsic[cand] > continuation]
        value[idx] = intrinsic[idx] * math.exp(-market.rate * t * dt)
        alive[idx] = False
    final = np.maximum(phi * (fresh[:, -1] - strike), 0.0)
    value[alive] = final[alive] * math.exp(-market.rate * maturity)

    return MCResult(
        price=float(value.mean()),
        standard_error=float(value.std(ddof=1) / math.sqrt(n_paths)),
        n_samples=n_paths,
        n_paths=n_paths,
        elapsed_seconds=time.perf_counter() - started,
    )
