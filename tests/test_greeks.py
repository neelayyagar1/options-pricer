"""Monte Carlo Greeks against the analytic ones."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest

from pricer import black_scholes as bs
from pricer.greeks import fd_greeks, pathwise_greeks
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from pricer.monte_carlo import simulate_price
from pricer.paths import draw_normals

MARKET = MarketParams(spot=100.0, volatility=0.25, rate=0.04, dividend_yield=0.01)
OPTION = EuropeanOption(strike=105.0, maturity=1.0, option_type=OptionType.CALL)
N = 1_000_000


@pytest.fixture(scope="module")
def analytic() -> bs.Greeks:
    return bs.greeks_contract(OPTION, MARKET)


class TestFiniteDifference:
    def test_crn_matches_analytic(self, analytic: bs.Greeks) -> None:
        fd = fd_greeks(MARKET, OPTION, n_paths=N, seed=1)
        assert fd.delta == pytest.approx(float(analytic.delta), abs=3e-3)
        assert fd.gamma == pytest.approx(float(analytic.gamma), rel=0.1)
        assert fd.vega == pytest.approx(float(analytic.vega), rel=0.01)
        assert fd.theta == pytest.approx(float(analytic.theta), rel=0.02)
        assert fd.rho == pytest.approx(float(analytic.rho), rel=0.01)

    def test_crn_is_far_less_noisy_than_independent_runs(self, analytic: bs.Greeks) -> None:
        def rmse(crn: bool) -> float:
            errors = [
                fd_greeks(MARKET, OPTION, n_paths=50_000, seed=s, common_random_numbers=crn).rho
                - float(analytic.rho)
                for s in range(8)
            ]
            return math.sqrt(float(np.mean(np.square(errors))))

        assert rmse(False) > 20.0 * rmse(True)


class TestPathwise:
    def test_matches_analytic_within_error(self, analytic: bs.Greeks) -> None:
        pw = pathwise_greeks(MARKET, OPTION, n_paths=N, seed=2)
        assert abs(pw.delta - float(analytic.delta)) < 3.5 * pw.delta_se
        assert abs(pw.vega - float(analytic.vega)) < 3.5 * pw.vega_se
        assert abs(pw.rho - float(analytic.rho)) < 3.5 * pw.rho_se

    def test_put_matches_analytic(self) -> None:
        put = EuropeanOption(strike=95.0, maturity=0.5, option_type=OptionType.PUT)
        exact = bs.greeks_contract(put, MARKET)
        pw = pathwise_greeks(MARKET, put, n_paths=N, seed=3)
        assert abs(pw.delta - float(exact.delta)) < 3.5 * pw.delta_se
        assert abs(pw.vega - float(exact.vega)) < 3.5 * pw.vega_se
        assert abs(pw.rho - float(exact.rho)) < 3.5 * pw.rho_se

    def test_lower_variance_than_a_finite_difference_sample(self) -> None:
        # Delta SE from pathwise should be tiny relative to the delta itself.
        pw = pathwise_greeks(MARKET, OPTION, n_paths=N, seed=4)
        assert pw.delta_se / pw.delta < 2e-3


@dataclass(frozen=True, slots=True)
class _Digital:
    strike: float
    path_dependent: bool = False

    def __call__(self, paths: np.ndarray) -> np.ndarray:
        return (paths[:, -1] > self.strike).astype(np.float64)


class TestPathwiseFailsOnDiscontinuousPayoffs:
    def test_digital_delta(self) -> None:
        """Bumping with CRN recovers the digital delta, the pathwise formula gives 0."""
        k, t = 100.0, 1.0
        sigma, r, q, s0 = 0.25, 0.04, 0.01, 100.0
        d2 = (math.log(s0 / k) + (r - q - 0.5 * sigma**2) * t) / (sigma * math.sqrt(t))
        true_delta = math.exp(-r * t) * math.exp(-0.5 * d2**2) / math.sqrt(2 * math.pi)
        true_delta /= s0 * sigma * math.sqrt(t)
        h = 1.0
        up = simulate_price(MARKET.with_(spot=s0 + h), t, _Digital(k), n_paths=2_000_000, seed=5)
        dn = simulate_price(MARKET.with_(spot=s0 - h), t, _Digital(k), n_paths=2_000_000, seed=5)
        fd_delta = (up.price - dn.price) / (2 * h)
        assert fd_delta == pytest.approx(true_delta, rel=0.1)

        # Pathwise: differentiate the payoff along each path. A digital is flat
        # except at a jump the sampled paths essentially never sit on.
        z = draw_normals(2_000_000, 1, 5)[:, 0]
        s_t = s0 * np.exp((r - q - 0.5 * sigma**2) * t + sigma * math.sqrt(t) * z)
        eps = 1e-12 * s_t
        slope = (_Digital(k)(np.c_[s_t + eps]) - _Digital(k)(np.c_[s_t - eps])) / (2 * eps)
        pathwise_delta = float(np.mean(math.exp(-r * t) * slope * s_t / s0))
        assert pathwise_delta == 0.0
        assert true_delta > 0.01
