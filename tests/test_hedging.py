"""Delta hedging: the P&L identity, the realized-vol result, and error scaling."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pricer import black_scholes as bs
from pricer.hedging import hedge_on_paths, simulate_delta_hedge
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from pricer.paths import draw_normals, gbm_paths

S0, K, T, R = 100.0, 100.0, 0.5, 0.04
CALL = EuropeanOption(K, T, OptionType.CALL)


def _gamma_pnl(paths: np.ndarray, implied: float, realized: float) -> np.ndarray:
    """Integral of e^{r(T-t)} * 0.5 S^2 Gamma (sigma_i^2 - sigma_r^2) dt, trapezoid rule."""
    n = paths.shape[1] - 1
    dt = T / n
    t = np.arange(n + 1) * dt
    remaining = np.maximum(T - t, 1e-12)
    gamma = np.asarray(bs.gamma(paths, K, remaining, R, implied, OptionType.CALL))
    integrand = np.exp(R * (T - t)) * 0.5 * paths**2 * gamma * (implied**2 - realized**2)
    return np.asarray(np.trapezoid(integrand, dx=dt, axis=1))


class TestHedgeAtImpliedVol:
    def test_pnl_matches_the_gamma_integral_path_by_path(self) -> None:
        implied, realized = 0.25, 0.15
        world = MarketParams(S0, realized, R)
        paths = gbm_paths(world, T, draw_normals(4_000, 1_000, 3))
        result = hedge_on_paths(paths, CALL, rate=R, implied_vol=implied, hedge_vol=implied)
        formula = _gamma_pnl(paths, implied, realized)

        assert np.corrcoef(result.pnl, formula)[0, 1] > 0.99
        assert result.mean == pytest.approx(float(formula.mean()), abs=0.05)
        assert result.mean > 0.0

    def test_pnl_is_not_deterministic(self) -> None:
        r = simulate_delta_hedge(
            S0,
            CALL,
            rate=R,
            implied_vol=0.25,
            realized_vol=0.15,
            n_rebalances=250,
            n_paths=5_000,
            seed=1,
        )
        assert r.std > 0.2


class TestHedgeAtRealizedVol:
    def test_pnl_does_not_depend_on_the_real_world_drift(self) -> None:
        def run(drift: float) -> float:
            return simulate_delta_hedge(
                S0,
                CALL,
                rate=R,
                implied_vol=0.25,
                realized_vol=0.15,
                hedge_vol=0.15,
                n_rebalances=500,
                n_paths=5_000,
                seed=2,
                drift=drift,
            ).mean

        assert run(-0.10) == pytest.approx(run(0.30), abs=0.05)

    def test_pnl_is_nearly_deterministic_and_equals_the_vol_gap(self) -> None:
        implied, realized = 0.25, 0.15
        r = simulate_delta_hedge(
            S0,
            CALL,
            rate=R,
            implied_vol=implied,
            realized_vol=realized,
            hedge_vol=realized,
            n_rebalances=1_000,
            n_paths=5_000,
            seed=4,
        )
        gap = (
            float(bs.price(S0, K, T, R, implied, OptionType.CALL))
            - float(bs.price(S0, K, T, R, realized, OptionType.CALL))
        ) * math.exp(R * T)
        assert r.mean == pytest.approx(gap, abs=0.03)
        assert r.std < 0.15 * gap

    def test_spread_is_far_smaller_than_hedging_at_implied(self) -> None:
        kwargs = {
            "rate": R,
            "implied_vol": 0.25,
            "realized_vol": 0.15,
            "n_rebalances": 500,
            "n_paths": 5_000,
            "seed": 5,
        }
        at_real = simulate_delta_hedge(S0, CALL, hedge_vol=0.15, **kwargs)  # type: ignore[arg-type]
        at_impl = simulate_delta_hedge(S0, CALL, hedge_vol=0.25, **kwargs)  # type: ignore[arg-type]
        assert at_real.std < 0.3 * at_impl.std


class TestHedgingErrorScaling:
    def test_error_shrinks_like_one_over_root_n(self) -> None:
        counts = [8, 16, 32, 64, 128, 256]
        stds = [
            simulate_delta_hedge(
                S0,
                CALL,
                rate=R,
                implied_vol=0.2,
                realized_vol=0.2,
                n_rebalances=n,
                n_paths=40_000,
                seed=6,
            ).std
            for n in counts
        ]
        slope = np.polyfit(np.log(counts), np.log(stds), 1)[0]
        assert slope == pytest.approx(-0.5, abs=0.07)

    def test_correct_vol_hedge_has_mean_near_zero(self) -> None:
        r = simulate_delta_hedge(
            S0,
            CALL,
            rate=R,
            implied_vol=0.2,
            realized_vol=0.2,
            n_rebalances=252,
            n_paths=40_000,
            seed=7,
        )
        assert abs(r.mean) < 4.0 * r.std / math.sqrt(40_000) + 1e-3
