r"""Payoff functions: pure maps from simulated paths to terminal cashflows.

A payoff receives an array of shape ``(n_paths, n_steps + 1)`` and returns one
undiscounted cashflow per path. Nothing here knows how the paths were produced, and
nothing in :mod:`pricer.paths` knows what will be done with them. That separation is
the reason an Asian or barrier option costs a few lines rather than a new pricer.

Path dependence
---------------
Each payoff declares whether it actually looks at the interior of the path. A
European vanilla does not — it needs only :math:`S_T` — so the engine may simulate
it with a single exact step, which is both faster and free of any discretisation
question. A payoff that *is* path dependent forces the engine to respect the caller's
step count, because for those contracts the step count is not a numerical
convenience but part of the contract's definition (where the average is sampled,
where the barrier is monitored).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pricer.instruments import OptionType

__all__ = ["Payoff", "VanillaPayoff"]

FloatArray = npt.NDArray[np.float64]


@runtime_checkable
class Payoff(Protocol):
    """Structural type for a terminal cashflow function.

    Declared as a read-only property rather than a mutable attribute so that
    frozen dataclasses — which is what every payoff in this package is — satisfy
    the protocol.
    """

    @property
    def path_dependent(self) -> bool:
        """Whether the payoff inspects the path between inception and expiry.

        The engine uses this to decide whether a single exact step suffices.
        """
        ...

    def __call__(self, paths: FloatArray) -> FloatArray:
        """Return the undiscounted cashflow for each row of ``paths``."""
        ...


@dataclass(frozen=True, slots=True)
class VanillaPayoff:
    r"""European call or put: :math:`\max(\phi(S_T - K), 0)`.

    Parameters
    ----------
    strike
        Strike :math:`K`.
    option_type
        Call or put, supplying the sign :math:`\phi`.
    """

    strike: float
    option_type: OptionType
    path_dependent: bool = False

    def __post_init__(self) -> None:
        """Validate the strike."""
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.path_dependent:
            raise ValueError("a vanilla payoff depends only on the terminal price")

    def __call__(self, paths: FloatArray) -> FloatArray:
        """Return the terminal intrinsic value of each path."""
        terminal = paths[:, -1]
        return np.maximum(self.option_type.sign * (terminal - self.strike), 0.0)
