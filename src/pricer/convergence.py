r"""Measure the MC convergence rate.

RMSE is averaged over independent replications at each N, then log RMSE is
regressed on log N. A single seed gives a slope that means nothing.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["ConvergenceStudy", "measure_convergence"]

FloatArray = npt.NDArray[np.float64]

#: A pricing routine to study: given a path count and a seed, return a price.
PriceFunction = Callable[[int, np.random.SeedSequence], float]


@dataclass(frozen=True, slots=True)
class ConvergenceStudy:
    r"""Result of a replication-based convergence measurement.

    Attributes
    ----------
    path_counts
        The path counts studied.
    rmse
        Root-mean-square error against the reference, over the replications.
    slope
        Fitted exponent :math:`b` in :math:`\text{RMSE} \propto N^{b}`. Standard
        Monte Carlo predicts :math:`-1/2`; a low-dimensional quasi-Monte Carlo arm
        should come out measurably steeper.
    intercept
        Fitted :math:`\log C`.
    replications
        Independent repetitions used at each path count.
    """

    path_counts: FloatArray
    rmse: FloatArray
    slope: float
    intercept: float
    replications: int

    def predicted_rmse(self, path_counts: FloatArray) -> FloatArray:
        """Evaluate the fitted power law at arbitrary path counts."""
        return np.exp(self.intercept + self.slope * np.log(path_counts))


def measure_convergence(
    price_function: PriceFunction,
    reference: float,
    path_counts: Sequence[int],
    *,
    replications: int = 30,
    seed: int = 20_260_802,
) -> ConvergenceStudy:
    r"""Estimate the convergence exponent of a Monte Carlo pricer.

    Parameters
    ----------
    price_function
        Called as ``price_function(n_paths, seed_sequence)``. Each call must be
        independent of the others given its seed.
    reference
        The exact value to measure error against, typically a closed-form price.
    path_counts
        Path counts to study. Spanning at least two orders of magnitude gives the
        regression enough leverage to distinguish a slope of :math:`-1/2` from one
        of :math:`-1`.
    replications
        Independent repetitions per path count. Averaging over these is what turns
        a noisy single-seed error into an estimate of RMSE.
    seed
        Root seed. Every replication derives a distinct child seed from it, so the
        whole study is reproducible.

    Returns
    -------
    ConvergenceStudy
        The measured errors and the fitted power law.
    """
    if replications < 2:
        raise ValueError(f"replications must be at least 2, got {replications}")
    if len(path_counts) < 2:
        raise ValueError("at least two path counts are needed to fit a slope")

    counts = np.asarray(path_counts, dtype=np.float64)
    if np.any(counts <= 0):
        raise ValueError("path counts must be positive")

    errors = np.empty(len(path_counts), dtype=np.float64)
    for i, n_paths in enumerate(path_counts):
        deviations = np.empty(replications, dtype=np.float64)
        for j in range(replications):
            # spawn_key makes the child seed a deterministic function of (i, j),
            # so a replication's draws do not depend on how many others were run.
            child = np.random.SeedSequence(entropy=seed, spawn_key=(i, j))
            deviations[j] = price_function(int(n_paths), child) - reference
        errors[i] = float(np.sqrt(np.mean(deviations**2)))

    if np.any(errors <= 0.0):
        raise ValueError("an exactly zero error cannot be fitted on a log scale")

    slope, intercept = np.polyfit(np.log(counts), np.log(errors), deg=1)

    return ConvergenceStudy(
        path_counts=counts,
        rmse=errors,
        slope=float(slope),
        intercept=float(intercept),
        replications=replications,
    )
