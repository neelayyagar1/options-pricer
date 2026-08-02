r"""The Monte Carlo pricing engine.

Risk-neutral valuation says the price of a claim is the discounted expectation of
its payoff under a particular measure. Monte Carlo takes that literally: simulate
the risk-neutral world many times, average the payoff, discount. The law of large
numbers does the rest.

What it costs is speed of convergence. The error of an average of :math:`N`
independent samples falls as :math:`O(N^{-1/2})`, so buying one more decimal place
of accuracy costs a hundred times the compute. That single fact is why the variance
reduction in :mod:`pricer.estimators` is not an optimisation but a necessity, and
why every result carries an error bar.

Memory
------
A naive one-million-path, 252-step simulation allocates roughly two gigabytes.
Simulation is therefore chunked: paths are generated, converted to payoffs, and
folded into running moments a block at a time, so peak memory is bounded by the
chunk rather than by the run. The chunk size is derived deterministically from the
step count so that results stay reproducible.

Reproducibility contract
------------------------
A run is reproducible given the same ``seed``, ``n_paths``, ``n_steps`` and
``max_elements``. Chunk boundaries influence which draws land on which path, so
changing the memory budget changes the sample — not its distribution, and not the
price beyond Monte Carlo error, but the exact digits. The defaults are fixed
constants precisely so the default path is bit-for-bit repeatable.
"""

from __future__ import annotations

import math
import time

import numpy as np

from pricer.estimators import MCResult, RunningMoments, reduce_antithetic
from pricer.instruments import EuropeanOption
from pricer.market import MarketParams
from pricer.paths import (
    RandomSource,
    SeedLike,
    antithetic_normals,
    draw_normals,
    gbm_paths,
    sobol_normals,
)
from pricer.payoffs import Payoff, VanillaPayoff

__all__ = ["DEFAULT_MAX_ELEMENTS", "simulate_european", "simulate_price"]

#: Maximum number of floating-point elements held in memory at once, i.e. about
#: 64 MB of float64. Fixed rather than tuned to the host so that runs reproduce
#: across machines.
DEFAULT_MAX_ELEMENTS = 8_000_000


def _as_seed_sequence(seed: SeedLike) -> np.random.SeedSequence:
    """Normalise a seed into a :class:`numpy.random.SeedSequence`."""
    if isinstance(seed, np.random.SeedSequence):
        return seed
    return np.random.SeedSequence(seed)


def _chunk_size(n_steps: int, max_elements: int) -> int:
    """Return how many paths may be simulated at once within the memory budget."""
    return max(1, max_elements // (n_steps + 1))


def simulate_price(
    market: MarketParams,
    maturity: float,
    payoff: Payoff,
    *,
    n_paths: int,
    seed: SeedLike,
    n_steps: int | None = None,
    antithetic: bool = False,
    source: RandomSource = RandomSource.PSEUDO_RANDOM,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> MCResult:
    r"""Price a claim by simulating the risk-neutral measure.

    Parameters
    ----------
    market
        Spot, volatility, rate and dividend yield.
    maturity
        Time to expiry :math:`T` in years.
    payoff
        Any :class:`~pricer.payoffs.Payoff`.
    n_paths
        Number of paths to simulate. Under ``antithetic`` this must be even, and
        only ``n_paths / 2`` of them are independent.
    seed
        Seed controlling every draw in the run.
    n_steps
        Time steps per path. Defaults to ``1`` for a payoff that is not path
        dependent, which is *exact* rather than merely cheap: GBM's transition
        density is known in closed form, so one step reaches :math:`S_T` with no
        discretisation error. Required for a path-dependent payoff, where the step
        count is part of the contract's definition.
    antithetic
        Pair each draw :math:`Z` with :math:`-Z`.
    source
        Pseudo-random (the default) or Sobol. See :class:`~pricer.paths.RandomSource`.
    max_elements
        Memory budget, in float64 elements, for a single simulation chunk.

    Returns
    -------
    MCResult
        Price, standard error, sample counts and elapsed time.

    Raises
    ------
    ValueError
        If the arguments are inconsistent — an odd path count under antithetic
        sampling, a path-dependent payoff with no step count, or a combination
        Sobol cannot support.
    """
    if n_paths <= 0:
        raise ValueError(f"n_paths must be positive, got {n_paths}")
    if maturity < 0.0:
        raise ValueError(f"maturity must be non-negative, got {maturity}")

    if n_steps is None:
        if payoff.path_dependent:
            raise ValueError(
                "n_steps is required for a path-dependent payoff: the monitoring "
                "schedule is part of the contract, not a numerical detail"
            )
        n_steps = 1
    elif n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")

    discount = market.discount_factor(maturity)
    started = time.perf_counter()

    result: tuple[float, float | None, int]
    if source is RandomSource.SOBOL:
        result = _simulate_sobol(
            market, maturity, payoff, n_paths, n_steps, seed, discount, antithetic
        )
    else:
        result = _simulate_pseudo_random(
            market,
            maturity,
            payoff,
            n_paths,
            n_steps,
            seed,
            discount,
            antithetic,
            max_elements,
        )

    price, standard_error, n_samples = result
    return MCResult(
        price=price,
        standard_error=standard_error,
        n_samples=n_samples,
        n_paths=n_paths,
        elapsed_seconds=time.perf_counter() - started,
    )


def _simulate_sobol(
    market: MarketParams,
    maturity: float,
    payoff: Payoff,
    n_paths: int,
    n_steps: int,
    seed: SeedLike,
    discount: float,
    antithetic: bool,
) -> tuple[float, None, int]:
    """Run the quasi-Monte Carlo arm, which reports no standard error.

    Sobol points form a single designed block. They cannot be split into chunks
    without breaking the equidistribution that motivates them, and mirroring them
    antithetically would destroy the same structure, so both are refused rather
    than silently degraded.
    """
    if antithetic:
        raise ValueError(
            "antithetic sampling is incompatible with a Sobol sequence: mirroring "
            "the points destroys the equidistribution that makes QMC converge faster"
        )

    normals = sobol_normals(n_paths, n_steps, seed)
    paths = gbm_paths(market, maturity, normals)
    samples = discount * payoff(paths)
    return float(np.mean(samples)), None, n_paths


def _simulate_pseudo_random(
    market: MarketParams,
    maturity: float,
    payoff: Payoff,
    n_paths: int,
    n_steps: int,
    seed: SeedLike,
    discount: float,
    antithetic: bool,
    max_elements: int,
) -> tuple[float, float, int]:
    """Run the pseudo-random arm with chunked simulation and running moments."""
    if antithetic and n_paths % 2 != 0:
        raise ValueError(f"antithetic sampling requires an even n_paths, got {n_paths}")

    # Under antithetic sampling a "sample" is a mirrored *pair*, not a path.
    n_samples = n_paths // 2 if antithetic else n_paths
    if n_samples < 2:
        raise ValueError(
            f"at least two independent samples are needed to estimate an error, got {n_samples}"
        )

    chunk = _chunk_size(n_steps, max_elements)
    n_chunks = math.ceil(n_samples / chunk)
    children = _as_seed_sequence(seed).spawn(n_chunks)

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
        values = discount * payoff(paths)
        if antithetic:
            values = reduce_antithetic(values)
        moments.update(values)

    return moments.mean, moments.standard_error, moments.count


def simulate_european(
    market: MarketParams,
    option: EuropeanOption,
    *,
    n_paths: int,
    seed: SeedLike,
    antithetic: bool = False,
    source: RandomSource = RandomSource.PSEUDO_RANDOM,
    max_elements: int = DEFAULT_MAX_ELEMENTS,
) -> MCResult:
    """Price a European vanilla by simulation.

    A convenience wrapper over :func:`simulate_price`. Because the payoff depends
    only on :math:`S_T`, the simulation uses a single exact step and therefore
    differs from the closed form of :mod:`pricer.black_scholes` by Monte Carlo error
    alone — which is what makes the convergence study a genuine correctness proof
    rather than a plausibility check.
    """
    return simulate_price(
        market,
        option.maturity,
        VanillaPayoff(strike=option.strike, option_type=option.option_type),
        n_paths=n_paths,
        seed=seed,
        n_steps=1,
        antithetic=antithetic,
        source=source,
        max_elements=max_elements,
    )
