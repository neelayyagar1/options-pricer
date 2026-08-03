"""Contract specifications.

Instruments are immutable value objects: they describe *what* is being priced and
carry no pricing logic. Every model in this package accepts an instrument plus a
:class:`~pricer.market.MarketParams`, so a new payoff never requires touching a
pricing engine.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

__all__ = ["EuropeanOption", "OptionType"]


class OptionType(StrEnum):
    r"""Right conveyed by the contract.

    The :attr:`sign` property exposes the standard :math:`\phi = \pm 1` convention
    that lets a single formula express both calls and puts:

    .. math:: V = \phi\left[S e^{-qT} N(\phi d_1) - K e^{-rT} N(\phi d_2)\right]

    Beyond brevity this matters for vectorisation — branching on option type inside
    an array expression would force a Python-level loop.
    """

    CALL = "call"
    PUT = "put"

    @property
    def sign(self) -> float:
        """Return ``+1.0`` for a call and ``-1.0`` for a put."""
        return 1.0 if self is OptionType.CALL else -1.0

    @property
    def opposite(self) -> OptionType:
        """Return the other option type, used by put-call parity checks."""
        return OptionType.PUT if self is OptionType.CALL else OptionType.CALL


@dataclass(frozen=True, slots=True)
class EuropeanOption:
    """A European-exercise vanilla option.

    Parameters
    ----------
    strike
        Strike price :math:`K`, in the same currency units as the underlying.
    maturity
        Time to expiry :math:`T` in years. Zero is permitted and yields the
        intrinsic value.
    option_type
        Call or put.
    """

    strike: float
    maturity: float
    option_type: OptionType

    def __post_init__(self) -> None:
        """Validate that the contract is economically meaningful."""
        if self.strike <= 0.0:
            raise ValueError(f"strike must be positive, got {self.strike}")
        if self.maturity < 0.0:
            raise ValueError(f"maturity must be non-negative, got {self.maturity}")
