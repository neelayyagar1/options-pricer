"""Market state: the observable (or implied) inputs shared by every model.

Separating market state from contract terms keeps the pricing functions honest —
a model may only depend on what is in these two objects, which makes it obvious
what would have to change to relax an assumption. ``volatility`` living here, as a
single number rather than a function of strike and maturity, *is* the constant-
volatility assumption of Black-Scholes made structurally visible.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

__all__ = ["MarketParams"]


@dataclass(frozen=True, slots=True)
class MarketParams:
    r"""Risk-neutral market inputs.

    Parameters
    ----------
    spot
        Current price :math:`S_0` of the underlying.
    volatility
        Annualised volatility :math:`\sigma`, as a decimal (``0.20`` is 20%).
    rate
        Continuously compounded risk-free rate :math:`r`, as a decimal.
    dividend_yield
        Continuous dividend yield :math:`q`, as a decimal. For an equity index this
        is the aggregate yield; for a non-dividend-paying stock it is zero.

    Notes
    -----
    ``rate`` and ``dividend_yield`` are *continuously compounded*. Quoted Treasury
    yields are not, and neither is a discrete dividend stream — converting is the
    caller's responsibility. In practice this package prefers to imply both from
    put-call parity on a live option chain rather than source them separately;
    see :mod:`pricer.forward`.
    """

    spot: float
    volatility: float
    rate: float
    dividend_yield: float = 0.0

    def __post_init__(self) -> None:
        """Validate that the market state is economically meaningful."""
        if self.spot <= 0.0:
            raise ValueError(f"spot must be positive, got {self.spot}")
        if self.volatility < 0.0:
            raise ValueError(f"volatility must be non-negative, got {self.volatility}")
        for name in ("spot", "volatility", "rate", "dividend_yield"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value}")

    @property
    def cost_of_carry(self) -> float:
        """Return the risk-neutral drift :math:`r - q` of the spot price."""
        return self.rate - self.dividend_yield

    def forward(self, maturity: float) -> float:
        """Return the forward price :math:`F = S_0 e^{(r-q)T}`.

        This is the risk-neutral expectation of the terminal spot, and therefore the
        exact mean of the control variate used for vanilla payoffs in
        :mod:`pricer.estimators`.
        """
        return self.spot * math.exp(self.cost_of_carry * maturity)

    def discount_factor(self, maturity: float) -> float:
        """Return :math:`e^{-rT}`."""
        return math.exp(-self.rate * maturity)

    def with_(self, **changes: float) -> MarketParams:
        """Return a copy with selected fields replaced.

        Used by the bump-and-revalue Greeks in :mod:`pricer.greeks`; the frozen
        dataclass guarantees the original state cannot be mutated by a bump.
        """
        return replace(self, **changes)
