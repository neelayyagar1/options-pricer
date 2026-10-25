r"""Payoffs: functions from simulated paths to undiscounted cashflows.

Paths come in as (n_paths, n_steps + 1). A payoff knows nothing about how the paths
were made, which is why adding a new contract is only a few lines.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

from pricer.instruments import OptionType

__all__ = ["AsianKind", "AsianPayoff", "BarrierPayoff", "BarrierStyle", "Payoff", "VanillaPayoff"]

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


class AsianKind(StrEnum):
    """Which average the Asian option pays on."""

    ARITHMETIC = "arithmetic"
    GEOMETRIC = "geometric"


@dataclass(frozen=True, slots=True)
class AsianPayoff:
    """Fixed-strike Asian option on the average of the monitoring dates.

    The average uses columns 1..n (the spot at inception is not a monitoring date),
    which matches the closed form in :mod:`pricer.analytic`.
    """

    strike: float
    option_type: OptionType
    kind: AsianKind = AsianKind.ARITHMETIC
    path_dependent: bool = True

    def __post_init__(self) -> None:
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")

    def __call__(self, paths: FloatArray) -> FloatArray:
        """Return the average-price payoff per path."""
        monitored = paths[:, 1:]
        if self.kind is AsianKind.ARITHMETIC:
            average = np.mean(monitored, axis=1)
        else:
            average = np.exp(np.mean(np.log(monitored), axis=1))
        return np.maximum(self.option_type.sign * (average - self.strike), 0.0)


class BarrierStyle(StrEnum):
    """Barrier direction and knock behaviour."""

    UP_AND_OUT = "up_and_out"
    UP_AND_IN = "up_and_in"
    DOWN_AND_OUT = "down_and_out"
    DOWN_AND_IN = "down_and_in"

    @property
    def is_up(self) -> bool:
        """True for an up barrier."""
        return self in (BarrierStyle.UP_AND_OUT, BarrierStyle.UP_AND_IN)

    @property
    def is_knock_out(self) -> bool:
        """True for a knock-out, False for a knock-in."""
        return self in (BarrierStyle.UP_AND_OUT, BarrierStyle.DOWN_AND_OUT)


@dataclass(frozen=True, slots=True)
class BarrierPayoff:
    r"""Single-barrier option, no rebate.

    Checking the barrier only on the simulation grid misses crossings between steps,
    so knock-outs come out too expensive. With ``bridge=True`` each step gets the
    probability that a Brownian bridge between its endpoints touched the barrier B:

    .. math:: p_i = \exp\left[-\frac{2\ln(B/S_i)\ln(B/S_{i+1})}{\sigma^2 \Delta t}\right]

    and the payoff is weighted by the survival probability product of (1 - p_i)
    (or one minus that for a knock-in). A step with an endpoint already past the
    barrier counts as a certain hit. With ``bridge=False`` the barrier is monitored
    on the grid only, which is the right model for a contract monitored at those
    dates.

    ``volatility`` and ``maturity`` are needed for the bridge probability.
    """

    strike: float
    barrier: float
    option_type: OptionType
    style: BarrierStyle
    volatility: float
    maturity: float
    bridge: bool = True
    path_dependent: bool = True

    def __post_init__(self) -> None:
        if self.strike <= 0.0 or self.barrier <= 0.0:
            raise ValueError("strike and barrier must be positive")
        if self.volatility <= 0.0 or self.maturity <= 0.0:
            raise ValueError("volatility and maturity must be positive")

    def __call__(self, paths: FloatArray) -> FloatArray:
        """Return the barrier-weighted vanilla payoff per path."""
        n_steps = paths.shape[1] - 1
        dt = self.maturity / n_steps
        vanilla = np.maximum(self.option_type.sign * (paths[:, -1] - self.strike), 0.0)

        sign = 1.0 if self.style.is_up else -1.0
        past = sign * (paths - self.barrier) >= 0.0
        hit_on_grid = past.any(axis=1)

        if not self.bridge:
            survive = (~hit_on_grid).astype(np.float64)
        else:
            left, right = paths[:, :-1], paths[:, 1:]
            safe_left = np.where(past[:, :-1], self.barrier * (1 - sign * 1e-9), left)
            safe_right = np.where(past[:, 1:], self.barrier * (1 - sign * 1e-9), right)
            exponent = (
                -2.0
                * np.log(self.barrier / safe_left)
                * np.log(self.barrier / safe_right)
                / (self.volatility**2 * dt)
            )
            p_cross = np.exp(exponent)
            p_cross = np.where(past[:, :-1] | past[:, 1:], 1.0, p_cross)
            survive = np.prod(1.0 - p_cross, axis=1)

        weight = survive if self.style.is_knock_out else 1.0 - survive
        return np.asarray(vanilla * weight, dtype=np.float64)
