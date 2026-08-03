"""Validation of the Black-Scholes-Merton closed form and its analytic Greeks.

These tests are the foundation of the project's correctness argument: the Monte
Carlo engine is later validated *against* this module, so an undetected error here
would invalidate everything downstream. They are therefore written as proofs of
mathematical properties (parity, exact limits, derivative identities) rather than
as spot checks against remembered numbers.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricer import black_scholes as bs
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from tests.conftest import PARAMETER_BATTERY, central_difference, second_difference


class TestTextbookValues:
    """Cross-checks against published reference values."""

    def test_hull_european_call_and_put(self) -> None:
        """Reproduce Hull, *Options, Futures and Other Derivatives*, ch. 15.

        S=42, K=40, r=10%, sigma=20%, T=0.5. Hull publishes 4.76 and 0.81, so the
        assertion is made at the precision the textbook actually states; the extra
        digits are pinned separately as a regression guard.
        """
        args = (42.0, 40.0, 0.5, 0.10, 0.20)
        call = float(bs.price(*args, OptionType.CALL))
        put = float(bs.price(*args, OptionType.PUT))

        assert call == pytest.approx(4.76, abs=5e-3)
        assert put == pytest.approx(0.81, abs=5e-3)

        assert call == pytest.approx(4.759_422_392_871_5, rel=1e-12)
        assert put == pytest.approx(0.808_599_372_900_1, rel=1e-12)

    def test_at_the_money_forward_call_matches_bachelier_style_approximation(self) -> None:
        """An ATM-forward option is worth about ``0.4 * S * sigma * sqrt(T)``.

        The exact value is ``S(2N(sigma sqrt(T)/2) - 1)``; for small total
        volatility this is close to ``S sigma sqrt(T) / sqrt(2 pi)``. This checks
        the formula against an independent asymptotic expansion rather than a
        stored constant.
        """
        spot, maturity, volatility, rate = 100.0, 1.0, 0.02, 0.0
        value = bs.price(spot, spot, maturity, rate, volatility, OptionType.CALL)
        approximation = spot * volatility * math.sqrt(maturity) / math.sqrt(2.0 * math.pi)

        assert float(value) == pytest.approx(approximation, rel=1e-4)


class TestPutCallParity:
    """C - P = S e^{-qT} - K e^{-rT} must hold to machine precision."""

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_parity_holds(self, params: tuple[float, float, float, float, float, float]) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params

        call = bs.price(spot, strike, maturity, rate, volatility, OptionType.CALL, dividend_yield)
        put = bs.price(spot, strike, maturity, rate, volatility, OptionType.PUT, dividend_yield)

        expected = spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity)
        assert float(call) - float(put) == pytest.approx(expected, abs=1e-12, rel=1e-12)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_greeks_obey_parity(
        self, params: tuple[float, float, float, float, float, float]
    ) -> None:
        """Differentiating parity constrains every Greek, not just the price."""
        spot, strike, maturity, rate, volatility, dividend_yield = params
        shared = (spot, strike, maturity, rate, volatility)

        call = bs.greeks(*shared, OptionType.CALL, dividend_yield)
        put = bs.greeks(*shared, OptionType.PUT, dividend_yield)

        discounted_spot = spot * math.exp(-dividend_yield * maturity)
        discounted_strike = strike * math.exp(-rate * maturity)

        # d/dS: delta_call - delta_put = e^{-qT}
        assert float(call.delta) - float(put.delta) == pytest.approx(
            math.exp(-dividend_yield * maturity), abs=1e-12
        )
        # Gamma and vega are identical for calls and puts.
        assert float(call.gamma) == pytest.approx(float(put.gamma), abs=1e-14)
        assert float(call.vega) == pytest.approx(float(put.vega), abs=1e-12)
        # d/dr: rho_call - rho_put = T K e^{-rT}
        assert float(call.rho) - float(put.rho) == pytest.approx(
            maturity * discounted_strike, abs=1e-10
        )
        # Theta is dV/dt while T runs to expiry, so d/dt = -d/dT and
        # theta_call - theta_put = -d/dT[S e^{-qT} - K e^{-rT}]
        #                        = q S e^{-qT} - r K e^{-rT}.
        assert float(call.theta) - float(put.theta) == pytest.approx(
            dividend_yield * discounted_spot - rate * discounted_strike, abs=1e-10
        )


class TestAnalyticGreeks:
    """Each analytic Greek must equal the numerical derivative of the price."""

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_delta_matches_numerical_derivative(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        analytic = float(
            bs.delta(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        )
        numeric = central_difference(
            lambda s: float(
                bs.price(s, strike, maturity, rate, volatility, option_type, dividend_yield)
            ),
            spot,
            step=spot * 1e-4,
        )
        assert analytic == pytest.approx(numeric, abs=1e-8, rel=1e-7)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_gamma_matches_second_derivative(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        analytic = float(
            bs.gamma(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        )
        numeric = second_difference(
            lambda s: float(
                bs.price(s, strike, maturity, rate, volatility, option_type, dividend_yield)
            ),
            spot,
            step=spot * 3e-3,
        )
        assert analytic == pytest.approx(numeric, abs=1e-8, rel=1e-4)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_vega_matches_numerical_derivative(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        analytic = float(
            bs.vega(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        )
        numeric = central_difference(
            lambda v: float(bs.price(spot, strike, maturity, rate, v, option_type, dividend_yield)),
            volatility,
            step=volatility * 1e-4,
        )
        assert analytic == pytest.approx(numeric, abs=1e-8, rel=1e-7)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_rho_matches_numerical_derivative(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        analytic = float(
            bs.rho(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        )
        numeric = central_difference(
            lambda r: float(
                bs.price(spot, strike, maturity, r, volatility, option_type, dividend_yield)
            ),
            rate,
            step=1e-5,
        )
        assert analytic == pytest.approx(numeric, abs=1e-7, rel=1e-6)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_theta_matches_negative_maturity_derivative(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        """Theta is dV/dt; with T measured to expiry that is -dV/dT."""
        spot, strike, maturity, rate, volatility, dividend_yield = params
        analytic = float(
            bs.theta(spot, strike, maturity, rate, volatility, option_type, dividend_yield)
        )
        numeric = -central_difference(
            lambda t: float(
                bs.price(spot, strike, t, rate, volatility, option_type, dividend_yield)
            ),
            maturity,
            step=min(maturity * 1e-4, 1e-5),
        )
        assert analytic == pytest.approx(numeric, abs=1e-6, rel=1e-6)


class TestDesignatedLimits:
    """Degenerate inputs must return exact limits, never nan."""

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_zero_maturity_gives_intrinsic_value(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        spot, strike, _, rate, volatility, dividend_yield = params
        value = float(bs.price(spot, strike, 0.0, rate, volatility, option_type, dividend_yield))
        intrinsic = max(option_type.sign * (spot - strike), 0.0)
        assert value == pytest.approx(intrinsic, abs=1e-14)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_zero_volatility_gives_discounted_forward_intrinsic(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        """With no volatility the terminal spot is known, so the value is certain."""
        spot, strike, maturity, rate, _, dividend_yield = params
        value = float(bs.price(spot, strike, maturity, rate, 0.0, option_type, dividend_yield))
        expected = max(
            option_type.sign
            * (spot * math.exp(-dividend_yield * maturity) - strike * math.exp(-rate * maturity)),
            0.0,
        )
        assert value == pytest.approx(expected, abs=1e-14)

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_limits_converge_from_the_interior(
        self,
        params: tuple[float, float, float, float, float, float],
        option_type: OptionType,
    ) -> None:
        """The returned limit must agree with the formula evaluated just short of it.

        This is what makes the limits *correct* rather than merely convenient: a
        wrong constant would show up as a discontinuity here.
        """
        spot, strike, maturity, rate, _, dividend_yield = params
        limit = float(bs.price(spot, strike, maturity, rate, 0.0, option_type, dividend_yield))
        nearly = float(bs.price(spot, strike, maturity, rate, 1e-7, option_type, dividend_yield))
        assert nearly == pytest.approx(limit, abs=1e-8)

    def test_no_nan_or_warning_at_degenerate_inputs(self, option_type: OptionType) -> None:
        """Expired and zero-volatility inputs must produce finite Greeks.

        ``filterwarnings = error::RuntimeWarning`` in ``pyproject.toml`` means an
        unguarded division here would fail the suite rather than pass silently.
        """
        for maturity, volatility in ((0.0, 0.2), (1.0, 0.0), (0.0, 0.0)):
            result = bs.greeks(100.0, 100.0, maturity, 0.05, volatility, option_type, 0.02)
            for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
                value = float(getattr(result, name))
                assert math.isfinite(value), f"{name} not finite at T={maturity}, s={volatility}"


class TestArbitrageBounds:
    """Prices must respect the model-free no-arbitrage bounds."""

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_call_within_bounds(
        self, params: tuple[float, float, float, float, float, float]
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        value = float(
            bs.price(spot, strike, maturity, rate, volatility, OptionType.CALL, dividend_yield)
        )
        discounted_spot = spot * math.exp(-dividend_yield * maturity)
        lower = max(discounted_spot - strike * math.exp(-rate * maturity), 0.0)

        assert value >= lower - 1e-12
        assert value <= discounted_spot + 1e-12

    @pytest.mark.parametrize("params", PARAMETER_BATTERY)
    def test_put_within_bounds(
        self, params: tuple[float, float, float, float, float, float]
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        value = float(
            bs.price(spot, strike, maturity, rate, volatility, OptionType.PUT, dividend_yield)
        )
        discounted_strike = strike * math.exp(-rate * maturity)
        lower = max(discounted_strike - spot * math.exp(-dividend_yield * maturity), 0.0)

        assert value >= lower - 1e-12
        assert value <= discounted_strike + 1e-12


class TestMonotonicity:
    """Structural properties that any correct pricer must satisfy."""

    def test_price_increases_with_volatility(self, option_type: OptionType) -> None:
        """Vega is positive, so value is strictly increasing in volatility."""
        volatilities = np.linspace(0.01, 1.5, 60)
        values = np.asarray(bs.price(100.0, 105.0, 1.0, 0.05, volatilities, option_type, 0.01))
        assert np.all(np.diff(values) > 0.0)

    def test_call_increases_and_put_decreases_with_spot(self) -> None:
        spots = np.linspace(20.0, 250.0, 120)
        calls = np.asarray(bs.price(spots, 100.0, 1.0, 0.05, 0.25, OptionType.CALL))
        puts = np.asarray(bs.price(spots, 100.0, 1.0, 0.05, 0.25, OptionType.PUT))

        assert np.all(np.diff(calls) > 0.0)
        assert np.all(np.diff(puts) < 0.0)

    def test_delta_stays_within_theoretical_range(self, option_type: OptionType) -> None:
        spots = np.linspace(20.0, 250.0, 120)
        deltas = np.asarray(bs.delta(spots, 100.0, 1.0, 0.05, 0.25, option_type, 0.02))
        bound = math.exp(-0.02 * 1.0)

        if option_type is OptionType.CALL:
            assert np.all((deltas >= 0.0) & (deltas <= bound))
        else:
            assert np.all((deltas <= 0.0) & (deltas >= -bound))

    def test_gamma_and_vega_are_positive(self, option_type: OptionType) -> None:
        spots = np.linspace(20.0, 250.0, 120)
        result = bs.greeks(spots, 100.0, 1.0, 0.05, 0.25, option_type)

        assert np.all(np.asarray(result.gamma) > 0.0)
        assert np.all(np.asarray(result.vega) > 0.0)


class TestDeepMoneyness:
    """Extreme strikes must stay numerically stable rather than overflow."""

    @pytest.mark.parametrize("strike", [1e-3, 1e-1, 1e3, 1e6])
    def test_extreme_strikes_are_finite(self, strike: float, option_type: OptionType) -> None:
        result = bs.greeks(100.0, strike, 1.0, 0.05, 0.25, option_type, 0.01)
        for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
            assert math.isfinite(float(getattr(result, name)))

    def test_deep_in_the_money_call_approaches_forward_minus_discounted_strike(self) -> None:
        value = float(bs.price(1e4, 100.0, 1.0, 0.05, 0.2, OptionType.CALL))
        expected = 1e4 - 100.0 * math.exp(-0.05)
        assert value == pytest.approx(expected, rel=1e-12)

    def test_deep_out_of_the_money_call_is_effectively_worthless(self) -> None:
        value = float(bs.price(1.0, 1e4, 1.0, 0.05, 0.2, OptionType.CALL))
        assert 0.0 <= value < 1e-30


class TestVectorisation:
    """Broadcasting must agree elementwise with scalar evaluation."""

    def test_broadcast_matches_scalar_loop(self, option_type: OptionType) -> None:
        spots = np.array([80.0, 100.0, 120.0])
        strikes = np.array([[90.0], [110.0]])

        grid = np.asarray(bs.price(spots, strikes, 1.0, 0.05, 0.25, option_type, 0.01))
        assert grid.shape == (2, 3)

        for i, strike in enumerate(strikes.ravel()):
            for j, spot in enumerate(spots):
                scalar = float(bs.price(spot, float(strike), 1.0, 0.05, 0.25, option_type, 0.01))
                assert grid[i, j] == pytest.approx(scalar, abs=1e-14)

    def test_scalar_input_returns_scalar(self, option_type: OptionType) -> None:
        value = bs.price(100.0, 100.0, 1.0, 0.05, 0.2, option_type)
        assert np.ndim(value) == 0


class TestContractInterface:
    """The dataclass wrappers must agree with the primitive interface."""

    def test_price_contract_matches_price(self, option_type: OptionType) -> None:
        option = EuropeanOption(strike=95.0, maturity=0.75, option_type=option_type)
        market = MarketParams(spot=100.0, volatility=0.22, rate=0.04, dividend_yield=0.015)

        assert float(bs.price_contract(option, market)) == pytest.approx(
            float(bs.price(100.0, 95.0, 0.75, 0.04, 0.22, option_type, 0.015)),
            abs=1e-14,
        )

    def test_greeks_contract_matches_greeks(self, option_type: OptionType) -> None:
        option = EuropeanOption(strike=95.0, maturity=0.75, option_type=option_type)
        market = MarketParams(spot=100.0, volatility=0.22, rate=0.04, dividend_yield=0.015)

        from_contract = bs.greeks_contract(option, market)
        direct = bs.greeks(100.0, 95.0, 0.75, 0.04, 0.22, option_type, 0.015)

        for name in ("price", "delta", "gamma", "vega", "theta", "rho"):
            assert float(getattr(from_contract, name)) == pytest.approx(
                float(getattr(direct, name)), abs=1e-14
            )


class TestDeskConventions:
    """Quoted Greeks must be scaled exactly as a desk would expect."""

    def test_vega_per_point_and_theta_per_day(self) -> None:
        result = bs.greeks(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)

        assert float(result.vega_per_point) == pytest.approx(float(result.vega) / 100.0)
        assert float(result.theta_per_day) == pytest.approx(float(result.theta) / 365.0)


class TestInputValidation:
    """Invalid inputs must fail loudly rather than return nonsense."""

    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"spot": 0.0}, "spot must be positive"),
            ({"spot": -1.0}, "spot must be positive"),
            ({"strike": 0.0}, "strike must be positive"),
            ({"maturity": -0.5}, "maturity must be non-negative"),
            ({"volatility": -0.1}, "volatility must be non-negative"),
        ],
    )
    def test_rejects_invalid_inputs(self, kwargs: dict[str, float], message: str) -> None:
        base = {
            "spot": 100.0,
            "strike": 100.0,
            "maturity": 1.0,
            "rate": 0.05,
            "volatility": 0.2,
        }
        base.update(kwargs)
        with pytest.raises(ValueError, match=message):
            bs.price(**base, option_type=OptionType.CALL)

    def test_market_params_rejects_negative_volatility(self) -> None:
        with pytest.raises(ValueError, match="volatility must be non-negative"):
            MarketParams(spot=100.0, volatility=-0.1, rate=0.05)

    def test_european_option_rejects_negative_maturity(self) -> None:
        with pytest.raises(ValueError, match="maturity must be non-negative"):
            EuropeanOption(strike=100.0, maturity=-1.0, option_type=OptionType.CALL)
