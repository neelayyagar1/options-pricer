"""Asian and barrier payoffs, the geometric closed form, and the control variate."""

from __future__ import annotations

import math

import pytest

from pricer import black_scholes as bs
from pricer.analytic import geometric_asian_moments, geometric_asian_price
from pricer.instruments import OptionType
from pricer.market import MarketParams
from pricer.monte_carlo import simulate_price
from pricer.payoffs import AsianKind, AsianPayoff, BarrierPayoff, BarrierStyle, VanillaPayoff
from pricer.variance_reduction import simulate_control_variate

MARKET = MarketParams(spot=100.0, volatility=0.25, rate=0.04, dividend_yield=0.01)
T = 1.0
N_DATES = 50


class TestGeometricAsianClosedForm:
    @pytest.mark.parametrize("option_type", list(OptionType))
    def test_matches_simulation(self, option_type: OptionType) -> None:
        exact = geometric_asian_price(
            100.0, 100.0, T, 0.04, 0.25, N_DATES, option_type, dividend_yield=0.01
        )
        payoff = AsianPayoff(100.0, option_type, AsianKind.GEOMETRIC)
        mc = simulate_price(MARKET, T, payoff, n_paths=400_000, seed=11, n_steps=N_DATES)
        assert mc.standard_error is not None
        assert abs(mc.price - exact) < 3.5 * mc.standard_error

    def test_put_call_parity(self) -> None:
        mu, var = geometric_asian_moments(100.0, T, 0.04, 0.25, N_DATES, 0.01)
        call = geometric_asian_price(100.0, 95.0, T, 0.04, 0.25, N_DATES, OptionType.CALL, 0.01)
        put = geometric_asian_price(100.0, 95.0, T, 0.04, 0.25, N_DATES, OptionType.PUT, 0.01)
        expected = math.exp(-0.04 * T) * (math.exp(mu + 0.5 * var) - 95.0)
        assert call - put == pytest.approx(expected, abs=1e-10)

    def test_one_date_is_a_vanilla(self) -> None:
        asian = geometric_asian_price(100.0, 105.0, T, 0.04, 0.25, 1, OptionType.CALL, 0.01)
        vanilla = float(bs.price(100.0, 105.0, T, 0.04, 0.25, OptionType.CALL, 0.01))
        assert asian == pytest.approx(vanilla, rel=1e-12)

    def test_converges_to_continuous_kemna_vorst(self) -> None:
        mu, var = geometric_asian_moments(100.0, T, 0.04, 0.25, 1_000_000, 0.01)
        sigma, drift = 0.25, 0.04 - 0.01 - 0.5 * 0.25**2
        assert var == pytest.approx(sigma**2 * T / 3.0, rel=1e-5)
        assert mu == pytest.approx(math.log(100.0) + drift * T / 2.0, rel=1e-5)


class TestControlVariate:
    def test_agrees_with_plain_mc_and_cuts_variance(self) -> None:
        arithmetic = AsianPayoff(100.0, OptionType.CALL, AsianKind.ARITHMETIC)
        geometric = AsianPayoff(100.0, OptionType.CALL, AsianKind.GEOMETRIC)
        control_price = geometric_asian_price(
            100.0, 100.0, T, 0.04, 0.25, N_DATES, OptionType.CALL, 0.01
        )

        plain = simulate_price(MARKET, T, arithmetic, n_paths=200_000, seed=5, n_steps=N_DATES)
        cv = simulate_control_variate(
            MARKET,
            T,
            arithmetic,
            geometric,
            control_price,
            n_paths=200_000,
            seed=5,
            n_steps=N_DATES,
        )
        assert plain.standard_error is not None and cv.result.standard_error is not None
        assert abs(cv.result.price - plain.price) < 4.0 * plain.standard_error
        assert cv.correlation > 0.99
        assert cv.result.variance_reduction_versus(plain) > 10.0

    def test_estimate_is_unbiased_for_the_control_itself(self) -> None:
        # Using the control as its own control variate must return its exact price.
        geometric = AsianPayoff(100.0, OptionType.PUT, AsianKind.GEOMETRIC)
        exact = geometric_asian_price(100.0, 100.0, T, 0.04, 0.25, N_DATES, OptionType.PUT, 0.01)
        cv = simulate_control_variate(
            MARKET, T, geometric, geometric, exact, n_paths=20_000, seed=2, n_steps=N_DATES
        )
        assert cv.result.price == pytest.approx(exact, abs=1e-10)


class TestBarrier:
    def _price(self, style: BarrierStyle, steps: int, bridge: bool, strike: float = 100.0) -> float:
        payoff = BarrierPayoff(
            strike, 130.0, OptionType.CALL, style, MARKET.volatility, T, bridge=bridge
        )
        return simulate_price(MARKET, T, payoff, n_paths=200_000, seed=3, n_steps=steps).price

    def test_in_out_parity(self) -> None:
        out = self._price(BarrierStyle.UP_AND_OUT, 50, True)
        knock_in = self._price(BarrierStyle.UP_AND_IN, 50, True)
        vanilla = float(bs.price(100.0, 100.0, T, 0.04, 0.25, OptionType.CALL, 0.01))
        assert out + knock_in == pytest.approx(vanilla, abs=0.05)

    def test_bridge_is_stable_in_step_count_but_grid_check_is_not(self) -> None:
        coarse_bridge = self._price(BarrierStyle.UP_AND_OUT, 10, True)
        fine_bridge = self._price(BarrierStyle.UP_AND_OUT, 200, True)
        coarse_grid = self._price(BarrierStyle.UP_AND_OUT, 10, False)
        fine_grid = self._price(BarrierStyle.UP_AND_OUT, 200, False)
        assert abs(coarse_bridge - fine_bridge) < 0.15
        assert coarse_grid - fine_grid > 0.5  # grid check overprices knock-outs

    def test_far_barrier_recovers_the_vanilla(self) -> None:
        payoff = BarrierPayoff(
            100.0, 1e6, OptionType.CALL, BarrierStyle.UP_AND_OUT, 0.25, T, bridge=True
        )
        vanilla = VanillaPayoff(100.0, OptionType.CALL)
        barrier = simulate_price(MARKET, T, payoff, n_paths=50_000, seed=8, n_steps=20)
        plain = simulate_price(MARKET, T, vanilla, n_paths=50_000, seed=8, n_steps=20)
        assert barrier.price == pytest.approx(plain.price, rel=1e-6)
