"""SVI fitting and its no-arbitrage diagnostics."""

from __future__ import annotations

import numpy as np
import pytest

from pricer.surface import (
    SVIParams,
    SVISlice,
    butterfly_g,
    calendar_violations,
    fit_svi,
    svi_total_variance,
)

K = np.linspace(-0.3, 0.3, 61)
TRUE = SVIParams(a=0.01, b=0.12, rho=-0.5, m=0.02, s=0.15)


def _vols(params: SVIParams, maturity: float, k: np.ndarray = K) -> np.ndarray:
    return np.sqrt(svi_total_variance(k, params) / maturity)


class TestFit:
    def test_recovers_a_noiseless_slice(self) -> None:
        fit = fit_svi(K, _vols(TRUE, 0.5), 0.5)
        assert fit.rmse_vol < 1e-6
        refit = svi_total_variance(K, fit.params)
        np.testing.assert_allclose(refit, svi_total_variance(K, TRUE), atol=1e-7)

    def test_noisy_slice_fits_to_noise_level(self) -> None:
        rng = np.random.default_rng(0)
        vols = _vols(TRUE, 0.5) + 0.002 * rng.standard_normal(K.size)
        fit = fit_svi(K, vols, 0.5)
        assert fit.rmse_vol < 0.0035

    def test_too_few_points_raises(self) -> None:
        with pytest.raises(ValueError, match="at least 6"):
            fit_svi(K[:4], _vols(TRUE, 0.5, K[:4]), 0.5)

    def test_fitted_parameters_respect_bounds(self) -> None:
        p = fit_svi(K, _vols(TRUE, 0.25), 0.25).params
        assert p.b > 0 and abs(p.rho) < 1 and p.s > 0
        assert p.a + p.b * p.s * np.sqrt(1 - p.rho**2) >= -1e-9


class TestButterfly:
    def test_well_behaved_slice_has_nonnegative_g(self) -> None:
        assert butterfly_g(K, TRUE).min() > 0.0

    def test_detects_a_butterfly_arbitrage(self) -> None:
        # Very small s with a large b makes a sharp kink and a negative density.
        bad = SVIParams(a=0.0, b=1.0, rho=0.0, m=0.0, s=0.005)
        assert butterfly_g(np.linspace(-0.05, 0.05, 201), bad).min() < 0.0

    def test_g_matches_finite_difference_density_sign(self) -> None:
        # g is built from w' and w''. Check those against numerical derivatives.
        from pricer.surface import _derivatives

        first, second = _derivatives(K, TRUE)
        h = 1e-5

        def w(x: np.ndarray) -> np.ndarray:
            return svi_total_variance(x, TRUE)

        np.testing.assert_allclose(first, (w(K + h) - w(K - h)) / (2 * h), atol=1e-7)
        np.testing.assert_allclose(second, (w(K + h) - 2 * w(K) + w(K - h)) / h**2, atol=1e-4)


class TestCalendar:
    def _slice(self, maturity: float, scale: float) -> SVISlice:
        p = SVIParams(TRUE.a * scale, TRUE.b * scale, TRUE.rho, TRUE.m, TRUE.s)
        return SVISlice(maturity, p, 0.0, 61)

    def test_increasing_variance_has_no_violations(self) -> None:
        slices = [self._slice(0.25, 1.0), self._slice(0.5, 1.5), self._slice(1.0, 2.0)]
        assert calendar_violations(slices, K) == []

    def test_flags_decreasing_variance(self) -> None:
        slices = [self._slice(0.25, 2.0), self._slice(0.5, 1.0)]
        found = calendar_violations(slices, K)
        assert len(found) == 1 and found[0][2] < 0.0
