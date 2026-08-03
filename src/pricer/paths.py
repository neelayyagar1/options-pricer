r"""Simulation of the underlying under the risk-neutral measure.

Two ideas structure this module, and both matter more than they first appear.

**Drawing normals is separated from turning them into paths.** Every technique the
project relies on later — common random numbers for bumped Greeks, antithetic
variates, quasi-Monte Carlo — is a statement about *which normals to draw*, not
about the price process. Keeping the two apart means each composes with the others
instead of requiring its own bespoke simulator.

**Stepping is exact, not an Euler discretisation.** Under geometric Brownian motion
the log-price increment is exactly normal, so the transition can be sampled in
closed form:

.. math::

    S_{t+\Delta t} = S_t \exp\!\left[\left(r - q - \tfrac{1}{2}\sigma^2\right)
    \Delta t + \sigma\sqrt{\Delta t}\,Z\right]

There is therefore **no discretisation bias in the marginal distribution** of the
simulated price, at any step count. An Euler scheme would carry one. This matters
for interpreting the convergence study: every residual error is Monte Carlo noise
or a path-dependent *feature* being sampled discretely (where a barrier is
monitored, where an average is taken) — never the price process itself.

Reproducibility
---------------
Functions take a *seed*, not a live :class:`numpy.random.Generator`. Rebuilding the
generator from a seed is what makes common random numbers trivial: price twice with
the same seed and the two runs consume identical draws, so a finite-difference bump
measures the bump rather than the noise.
"""

from __future__ import annotations

import math
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri
from scipy.stats import qmc

from pricer.market import MarketParams

__all__ = [
    "RandomSource",
    "SeedLike",
    "antithetic_normals",
    "draw_normals",
    "gbm_paths",
    "sobol_normals",
]

FloatArray = npt.NDArray[np.float64]
SeedLike = int | np.random.SeedSequence

# The smallest and largest doubles strictly inside (0, 1). Sobol points may land on
# exactly 0, whose normal quantile is -inf; nudging to the neighbouring double keeps
# the draw finite (about -38 sigma) without distorting the tail, which clipping to
# an arbitrary epsilon such as 1e-12 would.
_SMALLEST_ABOVE_ZERO = float(np.nextafter(0.0, 1.0))
_LARGEST_BELOW_ONE = float(np.nextafter(1.0, 0.0))


class RandomSource(StrEnum):
    """Which sequence drives the simulation.

    Attributes
    ----------
    PSEUDO_RANDOM
        PCG64 pseudo-random draws. Independent, so the sample standard error is a
        valid estimate of the estimator's error. This is the default and the only
        source used for reported prices.
    SOBOL
        A scrambled Sobol low-discrepancy sequence. Fills the unit cube far more
        evenly than random draws, buying error decay closer to :math:`O(N^{-1})`
        instead of :math:`O(N^{-1/2})` — but the points are *not* independent, so
        the usual standard-error formula does not apply. Used only as a benchmark
        arm in the one-dimensional convergence study.
    """

    PSEUDO_RANDOM = "pseudo_random"
    SOBOL = "sobol"


def draw_normals(n_paths: int, n_steps: int, seed: SeedLike) -> FloatArray:
    """Draw i.i.d. standard normals of shape ``(n_paths, n_steps)``.

    Parameters
    ----------
    n_paths
        Number of independent paths.
    n_steps
        Number of time steps per path.
    seed
        Seed or :class:`numpy.random.SeedSequence`. The same seed always yields the
        same draws, which is what makes common random numbers work.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    generator = np.random.default_rng(seed)
    return generator.standard_normal((n_paths, n_steps))


def antithetic_normals(n_pairs: int, n_steps: int, seed: SeedLike) -> FloatArray:
    r"""Draw ``n_pairs`` normals and stack them above their negations.

    Returns an array of shape ``(2 * n_pairs, n_steps)`` in which row ``i`` and row
    ``i + n_pairs`` are mirror images. Downstream code relies on that exact layout
    to pair payoffs before computing a standard error.

    Notes
    -----
    Antithetic sampling reduces variance only when
    :math:`\operatorname{Cov}(f(Z), f(-Z)) < 0`, which holds when the payoff is
    monotone in :math:`Z` — true for vanilla calls and puts. For payoffs that are
    symmetric about the forward, such as a straddle, the covariance can be positive
    and the technique *increases* variance. It is not free, and this package
    measures rather than assumes its benefit.
    """
    primal = draw_normals(n_pairs, n_steps, seed)
    return np.concatenate((primal, -primal), axis=0)


def sobol_normals(n_paths: int, n_steps: int, seed: SeedLike) -> FloatArray:
    r"""Draw ``n_paths`` scrambled Sobol points mapped to standard normals.

    Parameters
    ----------
    n_paths
        Number of points. Must be a power of two — Sobol's equidistribution
        properties hold on balanced blocks of :math:`2^m` points, and taking an
        arbitrary prefix degrades the uniformity that motivates using it at all.
    n_steps
        Dimension of the sequence, one coordinate per time step.
    seed
        Seed for the Owen scrambling.

    Raises
    ------
    ValueError
        If ``n_paths`` is not a power of two.

    Notes
    -----
    The convergence advantage of QMC degrades with **effective dimension**. For a
    one-dimensional problem — a European payoff, which needs only the terminal
    price — the improvement is large and clearly visible. Applied naively to a
    252-step path it is largely lost, because the later coordinates of a Sobol
    sequence are much less well distributed than the first few. Recovering the
    benefit there requires a Brownian-bridge or PCA path construction that
    concentrates variance in the leading dimensions; that is deliberately out of
    scope, so this package restricts QMC to the low-dimensional case where the
    claimed speed-up is real.
    """
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")

    exponent = round(math.log2(n_paths))
    if 2**exponent != n_paths:
        raise ValueError(
            f"n_paths must be a power of two for a balanced Sobol block, got {n_paths}"
        )

    # scipy's QMC engines take a Generator or an integer, not a SeedSequence, so
    # the seed is realised here rather than passed through.
    engine = qmc.Sobol(d=n_steps, scramble=True, seed=np.random.default_rng(seed))
    uniforms = engine.random_base2(exponent)
    bounded = np.clip(uniforms, _SMALLEST_ABOVE_ZERO, _LARGEST_BELOW_ONE)
    return np.asarray(ndtri(bounded), dtype=np.float64)


def gbm_paths(
    market: MarketParams,
    maturity: float,
    normals: FloatArray,
) -> FloatArray:
    r"""Convert standard normals into geometric Brownian motion paths.

    Parameters
    ----------
    market
        Spot, volatility, rate and dividend yield.
    maturity
        Total simulated horizon :math:`T` in years, divided evenly across the
        columns of ``normals``.
    normals
        Standard normal draws of shape ``(n_paths, n_steps)``.

    Returns
    -------
    numpy.ndarray
        Paths of shape ``(n_paths, n_steps + 1)``. Column 0 is the spot, so column
        ``j`` is the price at time ``j * T / n_steps`` and the final column is
        :math:`S_T`.

    Notes
    -----
    The log-price is accumulated with :func:`numpy.cumsum` and exponentiated once at
    the end. Working in log space keeps the increments additive and exactly normal,
    and avoids the compounding round-off of repeated multiplication.
    """
    if normals.ndim != 2:
        raise ValueError(f"normals must be two-dimensional, got shape {normals.shape}")
    if maturity < 0.0:
        raise ValueError(f"maturity must be non-negative, got {maturity}")

    n_paths, n_steps = normals.shape
    step = maturity / n_steps

    drift = (market.cost_of_carry - 0.5 * market.volatility**2) * step
    diffusion = market.volatility * math.sqrt(step)

    log_increments = drift + diffusion * normals
    log_relative = np.concatenate(
        (np.zeros((n_paths, 1), dtype=np.float64), np.cumsum(log_increments, axis=1)),
        axis=1,
    )
    # asarray rather than a bare product: multiplying a Python float by an ndarray
    # is typed as Any under some NumPy stub versions, which would silently widen
    # this function's return type. It is a no-op at runtime for a float64 array.
    return np.asarray(market.spot * np.exp(log_relative), dtype=np.float64)
