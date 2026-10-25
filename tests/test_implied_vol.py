"""Implied volatility solver, implied forward, and the smile extractor."""

from __future__ import annotations

import math

import pytest

from pricer import black_scholes as bs
from pricer.forward import implied_forward
from pricer.implied_vol import implied_vol, otm_smile
from pricer.instruments import OptionType
from tests.chain_helpers import synthetic_expiry


class TestImpliedVol:
    @pytest.mark.parametrize("option_type", list(OptionType))
    @pytest.mark.parametrize("strike", [60.0, 90.0, 100.0, 115.0, 160.0])
    @pytest.mark.parametrize("vol", [0.05, 0.2, 0.6, 1.5])
    def test_round_trip(self, option_type: OptionType, strike: float, vol: float) -> None:
        price = float(bs.price(100.0, strike, 0.75, 0.03, vol, option_type, 0.01))
        intrinsic = max(
            option_type.sign * (100.0 * math.exp(-0.0075) - strike * math.exp(-0.0225)), 0.0
        )
        if price - intrinsic < 1e-10:
            pytest.skip("price indistinguishable from intrinsic, vol not identifiable")
        got = implied_vol(price, 100.0, strike, 0.75, 0.03, option_type, 0.01)
        assert got == pytest.approx(vol, rel=1e-6)

    def test_deep_otm_uses_the_fallback_and_still_converges(self) -> None:
        # vega is ~1e-9 here, so Newton gives up and Brent has to finish.
        price = float(bs.price(100.0, 220.0, 0.25, 0.03, 0.25, OptionType.CALL))
        assert price > 0.0
        assert implied_vol(price, 100.0, 220.0, 0.25, 0.03, OptionType.CALL) == pytest.approx(
            0.25, rel=1e-4
        )

    @pytest.mark.parametrize("price", [-1.0, 0.0, 5.0, 105.0])
    def test_rejects_prices_outside_arbitrage_bounds(self, price: float) -> None:
        # ITM call at K=90 has lower bound ~10.5, upper bound 100. 5.0 is below it.
        with pytest.raises(ValueError, match="no-arbitrage"):
            implied_vol(price, 100.0, 90.0, 1.0, 0.03, OptionType.CALL)


class TestImpliedForward:
    def test_free_regression_recovers_rate_and_dividend_on_clean_european_quotes(self) -> None:
        chain = synthetic_expiry(100.0, 0.5, 0.04, 0.012)
        fwd = implied_forward(chain, 100.0)
        assert fwd.rate == pytest.approx(0.04, abs=1e-8)
        assert fwd.dividend_yield(100.0) == pytest.approx(0.012, abs=1e-8)
        assert fwd.forward == pytest.approx(100.0 * math.exp(0.028 * 0.5), rel=1e-10)
        assert fwd.r_squared > 1.0 - 1e-12

    def test_forward_is_stable_but_free_regression_rate_is_not(self) -> None:
        """The zero crossing (forward) survives quote noise, the slope (rate) does not."""
        truth = 100.0 * math.exp((0.04 - 0.012) * 0.1)
        forward_errors, free_rate_errors = [], []
        for seed in range(10):
            chain = synthetic_expiry(100.0, 0.1, 0.04, 0.012, noise=0.02, seed=seed)
            forward_errors.append(abs(implied_forward(chain, 100.0, rate=0.04).forward - truth))
            free_rate_errors.append(abs(implied_forward(chain, 100.0).rate - 0.04))
        assert max(forward_errors) < 0.03
        assert max(free_rate_errors) > 0.05

    def test_too_few_strikes_raises(self) -> None:
        chain = synthetic_expiry(100.0, 0.5, 0.04, 0.0)
        with pytest.raises(ValueError, match="usable strikes"):
            implied_forward(chain, 100.0, band=0.01)


class TestSmileExtraction:
    def test_flat_vol_is_recovered_on_both_wings(self) -> None:
        chain = synthetic_expiry(100.0, 0.5, 0.04, 0.01, vol=0.22)
        fwd = implied_forward(chain, 100.0, rate=0.04)
        k, vols = otm_smile(chain, fwd)
        assert k.size > 20
        assert k.min() < -0.1 and k.max() > 0.1
        assert vols == pytest.approx(0.22, abs=2e-5)

    def test_recovers_a_skewed_smile(self) -> None:
        chain = synthetic_expiry(
            100.0, 0.5, 0.04, 0.01, vol_of_k=lambda k: 0.2 - 0.3 * k + 0.5 * k**2
        )
        fwd = implied_forward(chain, 100.0, rate=0.04)
        k, vols = otm_smile(chain, fwd)
        assert vols == pytest.approx(0.2 - 0.3 * k + 0.5 * k**2, abs=1e-4)
