"""Heston pricing, simulation, calibration and hedging."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricer import black_scholes as bs
from pricer.hedging import hedge_under_heston
from pricer.heston import (
    HestonParams,
    calibrate_heston,
    heston_call_normalized,
    heston_implied_vols,
    simulate_heston,
)
from pricer.instruments import OptionType

P = HestonParams(v0=0.04, kappa=2.0, theta=0.05, xi=0.6, rho=-0.7)


class TestParams:
    def test_feller_ratio(self) -> None:
        assert P.feller_ratio == pytest.approx(2 * 2.0 * 0.05 / 0.36)

    @pytest.mark.parametrize(
        "kwargs", [{"v0": -0.1}, {"kappa": 0.0}, {"xi": -1.0}, {"rho": 1.0}, {"rho": -1.5}]
    )
    def test_rejects_invalid(self, kwargs: dict[str, float]) -> None:
        base = {"v0": 0.04, "kappa": 2.0, "theta": 0.05, "xi": 0.6, "rho": -0.7}
        with pytest.raises(ValueError):
            HestonParams(**{**base, **kwargs})


class TestPricing:
    def test_reduces_to_black_scholes_when_volatility_of_variance_vanishes(self) -> None:
        sigma, maturity = 0.2, 0.5
        p = HestonParams(sigma**2, 5.0, sigma**2, 0.02, -0.5)
        k = np.array([-0.2, -0.05, 0.0, 0.1])  # ln(F/K)
        n, _ = heston_call_normalized(k, np.array([sigma**2]), maturity, p)
        forward = 100.0
        expected = [
            float(bs.price(forward, forward * math.exp(-x), maturity, 0.0, sigma, OptionType.CALL))
            / forward
            for x in k
        ]
        np.testing.assert_allclose(n[:, 0], expected, atol=5e-4)

    def test_matches_monte_carlo(self) -> None:
        spot, rate, maturity = 100.0, 0.03, 0.5
        s, _ = simulate_heston(P, spot, rate, maturity, 25, 150_000, 1, substeps=8)
        forward = spot * math.exp(rate * maturity)
        for strike in (90.0, 100.0, 110.0):
            n, _ = heston_call_normalized(
                np.array([math.log(forward / strike)]), np.array([P.v0]), maturity, P
            )
            exact = math.exp(-rate * maturity) * forward * n[0, 0]
            payoff = np.maximum(s[:, -1] - strike, 0.0) * math.exp(-rate * maturity)
            se = payoff.std() / math.sqrt(payoff.size)
            # Euler discretization adds a small positive bias on top of sampling error.
            assert abs(payoff.mean() - exact) < 3.5 * se + 0.02

    def test_prices_respect_arbitrage_bounds_and_are_monotone_in_strike(self) -> None:
        k = np.linspace(-0.4, 0.4, 41)  # ln(F/K), increasing means a lower strike
        n, _ = heston_call_normalized(k, np.array([P.v0]), 0.3, P)
        assert np.all(n[:, 0] >= np.maximum(1.0 - np.exp(-k), 0.0) - 1e-8)
        assert np.all(n[:, 0] <= 1.0 + 1e-8)
        assert np.all(np.diff(n[:, 0]) > 0.0)

    def test_derivative_in_moneyness_matches_finite_difference(self) -> None:
        k, v, h = np.array([-0.2, 0.0, 0.07]), np.array([0.02, 0.05]), 1e-5
        _, dn = heston_call_normalized(k, v, 0.2, P)
        up = heston_call_normalized(k + h, v, 0.2, P)[0]
        down = heston_call_normalized(k - h, v, 0.2, P)[0]
        np.testing.assert_allclose(dn, (up - down) / (2 * h), atol=1e-6)

    def test_negative_rho_gives_a_downward_sloping_skew(self) -> None:
        vols = heston_implied_vols(np.array([-0.1, 0.0, 0.1]), 0.25, P)  # ln(K/F)
        assert vols[0] > vols[1] > vols[2]


class TestSimulation:
    def test_discounted_spot_is_a_martingale_and_variance_mean_reverts(self) -> None:
        s, v = simulate_heston(P, 100.0, 0.03, 1.0, 50, 100_000, 2, substeps=4)
        forward = 100.0 * math.exp(0.03)
        se = s[:, -1].std() / math.sqrt(s.shape[0])
        assert abs(s[:, -1].mean() - forward) < 4 * se
        expected_v = P.theta + (P.v0 - P.theta) * math.exp(-P.kappa)
        assert v[:, -1].mean() == pytest.approx(expected_v, rel=0.03)

    def test_same_seed_reproduces(self) -> None:
        a, _ = simulate_heston(P, 100.0, 0.03, 0.5, 10, 100, 7)
        b, _ = simulate_heston(P, 100.0, 0.03, 0.5, 10, 100, 7)
        np.testing.assert_array_equal(a, b)

    def test_variance_stays_non_negative(self) -> None:
        _, v = simulate_heston(
            HestonParams(0.01, 1.0, 0.02, 2.0, -0.9), 100.0, 0.0, 1.0, 50, 2_000, 3
        )
        assert v.min() >= 0.0


class TestCalibration:
    def test_recovers_a_known_surface(self) -> None:
        truth = HestonParams(0.03, 3.0, 0.045, 1.0, -0.7)
        slices = []
        for maturity in (0.1, 0.25, 0.5, 1.0):
            width = 3.0 * 0.2 * math.sqrt(maturity)
            grid = np.linspace(-width, 0.6 * width, 12)
            slices.append((maturity, grid, heston_implied_vols(grid, maturity, truth)))
        fitted, rmses = calibrate_heston(slices)
        assert max(rmses) < 5e-4  # under 0.05 vol points
        assert fitted.rho == pytest.approx(truth.rho, abs=0.1)


class TestHedging:
    def test_variance_minimizing_delta_beats_the_alternatives(self) -> None:
        p = HestonParams(0.028, 7.3, 0.044, 1.7, -0.72)
        out = hedge_under_heston(
            p, 100.0, 100.0, 0.25, 0.04, n_rebalances=40, n_paths=2_500, seed=1
        )
        premium = out["heston"].premium
        assert premium == pytest.approx(3.7, abs=0.3)
        for result in out.values():
            assert abs(result.mean) < 0.15  # fairly priced and hedged, so no drift
        stds = {name: r.std for name, r in out.items()}
        assert stds["heston_min_variance"] < 0.9 * stds["black_scholes_flat"]
        assert stds["heston"] > stds["black_scholes_flat"]
