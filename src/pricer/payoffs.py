r"""Payoffs: functions from simulated paths to undiscounted cashflows.

Paths come in as (n_paths, n_steps + 1). A payoff knows nothing about how the paths
were made, which is why adding a new contract is only a few lines.
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
    frozen dataclasses, which is what every payoff in this package is, satisfy
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
