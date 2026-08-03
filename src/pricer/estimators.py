r"""Monte Carlo estimators, and the error bars that make them trustworthy.

This module exists because of one specific, common, credibility-destroying bug.

A Monte Carlo price is worthless without an honest estimate of its own error, and
the moment a variance-reduction technique is introduced the naive formula stops
being correct. Under **antithetic sampling** the :math:`2N` simulated payoffs are
not independent — each is the mirror of another. The estimator is the mean of
:math:`N` **pair averages**

.. math:: Y_i = \tfrac{1}{2}\left[f(Z_i) + f(-Z_i)\right]

and those pair averages, not the raw payoffs, are the independent quantities whose
spread measures the error.

Why the naive formula is worse than merely wrong
------------------------------------------------
Writing :math:`\rho = \operatorname{Corr}(f(Z), f(-Z))` and
:math:`\sigma_f^2 = \operatorname{Var} f(Z)`, the pair average has variance
:math:`\tfrac{1}{2}\sigma_f^2(1 + \rho)`, so the estimator's true standard error is

.. math:: \text{SE} = \sigma_f\sqrt{\frac{1+\rho}{2N}},
   \qquad\text{while}\qquad
   \text{SE}_{\text{naive}} = \frac{\sigma_f}{\sqrt{2N}}

The naive figure is precisely the plain-Monte-Carlo error at :math:`2N` paths, and
it does not depend on :math:`\rho` **at all**. It is therefore blind in both
directions at once:

- When antithetic sampling *works* (:math:`\rho < 0`) it overstates the error and
  silently erases the very reduction the technique was introduced to achieve.
- When antithetic sampling *backfires* (:math:`\rho > 0`, as it can for a payoff
  symmetric about the forward such as a straddle) it understates the error and
  gives no warning at all.

A number that cannot distinguish a technique working from that same technique
doing harm is not a conservative approximation; it is an absence of information
dressed up as a measurement. Every price this package reports therefore travels
with a standard error computed from genuinely independent samples, together with
the count of those samples.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.special import ndtri

__all__ = ["MCResult", "RunningMoments", "reduce_antithetic"]

FloatArray = npt.NDArray[np.float64]


def reduce_antithetic(values: FloatArray) -> FloatArray:
    r"""Collapse mirrored payoffs into independent pair averages.

    Expects the layout produced by :func:`pricer.paths.antithetic_normals`: the
    first half of ``values`` are the primal draws and the second half their mirror
    images, so element ``i`` pairs with element ``i + n/2``.

    Parameters
    ----------
    values
        One-dimensional array of even length.

    Returns
    -------
    numpy.ndarray
        Array of half the length, containing :math:`\tfrac{1}{2}(f(Z_i) + f(-Z_i))`.
    """
    if values.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {values.shape}")
    if values.size % 2 != 0:
        raise ValueError(f"antithetic sampling requires an even count, got {values.size}")

    half = values.size // 2
    return 0.5 * (values[:half] + values[half:])


class RunningMoments:
    r"""Accumulate a mean and variance across chunks without storing the samples.

    Chunked simulation exists so that a ten-million-path run does not need a
    two-gigabyte array; that only works if the summary statistics can be built
    incrementally.

    Notes
    -----
    Sums are accumulated **around a shift** taken from the first batch rather than
    around zero. The textbook one-pass formula
    :math:`\sum x^2 - (\sum x)^2 / n` subtracts two large and nearly equal numbers,
    and loses relative precision in proportion to :math:`(\mu/\sigma)^2` — which is
    exactly the deep-in-the-money regime where the payoff mean dwarfs its spread.
    Shifting by an approximate mean makes the subtraction well conditioned at no
    meaningful cost.
    """

    __slots__ = ("_count", "_shift", "_sum", "_sum_of_squares")

    def __init__(self) -> None:
        self._shift: float | None = None
        self._count = 0
        self._sum = 0.0
        self._sum_of_squares = 0.0

    def update(self, values: FloatArray) -> None:
        """Fold a batch of independent samples into the running totals."""
        if values.size == 0:
            return
        if self._shift is None:
            self._shift = float(np.mean(values))

        centred = values - self._shift
        self._count += values.size
        self._sum += float(np.sum(centred))
        self._sum_of_squares += float(np.sum(centred * centred))

    @property
    def count(self) -> int:
        """Return the number of independent samples accumulated."""
        return self._count

    @property
    def mean(self) -> float:
        """Return the sample mean."""
        if self._count == 0:
            raise ValueError("no samples accumulated")
        assert self._shift is not None
        return self._shift + self._sum / self._count

    @property
    def variance(self) -> float:
        """Return the unbiased (``ddof=1``) sample variance."""
        if self._count < 2:
            raise ValueError("at least two samples are required to estimate a variance")
        centred_mean = self._sum / self._count
        total = self._sum_of_squares - self._count * centred_mean * centred_mean
        # Round-off can drive an exactly-constant sample slightly negative.
        return max(total, 0.0) / (self._count - 1)

    @property
    def standard_error(self) -> float:
        r"""Return :math:`s/\sqrt{n}`, the standard error of the mean."""
        return math.sqrt(self.variance / self._count)


@dataclass(frozen=True, slots=True)
class MCResult:
    r"""A Monte Carlo price together with its uncertainty.

    Attributes
    ----------
    price
        The discounted estimate.
    standard_error
        Standard error of ``price``, or ``None`` when the estimator has no valid
        one. Quasi-Monte Carlo is the case that matters: Sobol points are a
        designed, dependent set, so the sample standard deviation does not estimate
        the error of their average. Returning ``None`` rather than a number that
        looks usable is deliberate.
    n_samples
        Number of **independent** samples behind the estimate.
    n_paths
        Number of paths actually simulated. Under antithetic sampling this is twice
        ``n_samples``; the gap between the two is the whole reason this class
        distinguishes them.
    elapsed_seconds
        Wall-clock simulation time, needed to convert a variance reduction into an
        honest speed-up per unit of compute.
    """

    price: float
    standard_error: float | None
    n_samples: int
    n_paths: int
    elapsed_seconds: float

    def confidence_interval(self, level: float = 0.95) -> tuple[float, float]:
        """Return a central-limit-theorem confidence interval for the price.

        Parameters
        ----------
        level
            Coverage probability, e.g. ``0.95``.

        Raises
        ------
        ValueError
            If the estimator has no valid standard error (see
            :attr:`standard_error`).
        """
        if self.standard_error is None:
            raise ValueError(
                "this estimator has no valid standard error, so no confidence "
                "interval can be formed"
            )
        if not 0.0 < level < 1.0:
            raise ValueError(f"level must lie in (0, 1), got {level}")

        z = float(ndtri(0.5 * (1.0 + level)))
        margin = z * self.standard_error
        return self.price - margin, self.price + margin

    def variance_reduction_versus(self, baseline: MCResult) -> float:
        r"""Return the variance reduction factor at **equal path count**.

        Normalising by ``n_paths`` rather than ``n_samples`` is deliberate and is
        the difference between an honest number and an inflated one. An antithetic
        sample is a *pair*, and consumes two path evaluations; comparing per-sample
        variances would therefore credit the technique with a factor of two it did
        not earn. Measured this way the antithetic factor is
        :math:`1/(1+\rho)`, the standard result, and equals one exactly when the
        mirrored payoffs are uncorrelated.
        """
        if self.standard_error is None or baseline.standard_error is None:
            raise ValueError("both estimators must have a valid standard error")

        own = self.standard_error**2 * self.n_paths
        other = baseline.standard_error**2 * baseline.n_paths
        if own <= 0.0:
            raise ValueError("estimator variance is zero; the ratio is undefined")
        return other / own

    def efficiency_versus(self, baseline: MCResult) -> float:
        r"""Return the cost-adjusted speed-up relative to ``baseline``.

        A technique that cuts variance tenfold while doubling runtime delivers a
        fivefold improvement, not a tenfold one. The quantity that actually matters
        is error per unit of compute, :math:`1/(\sigma^2 t)`, and this is its ratio.
        """
        if self.standard_error is None or baseline.standard_error is None:
            raise ValueError("both estimators must have a valid standard error")
        if self.elapsed_seconds <= 0.0 or baseline.elapsed_seconds <= 0.0:
            raise ValueError("elapsed time must be positive to compare efficiency")

        own = self.standard_error**2 * self.elapsed_seconds
        other = baseline.standard_error**2 * baseline.elapsed_seconds
        return other / own
