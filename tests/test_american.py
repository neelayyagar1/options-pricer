"""Binomial tree and Longstaff-Schwartz checks."""

from __future__ import annotations

import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from pricer import black_scholes as bs
from pricer.american import price_american_lsm
from pricer.binomial import binomial_price
from pricer.instruments import OptionType
from pricer.market import MarketParams

MARKET = MarketParams(spot=100.0, volatility=0.30, rate=0.05, dividend_yield=0.0)


class TestBinomial:
    @pytest.mark.parametrize("option_type", list(OptionType))
    @pytest.mark.parametrize("strike", [80.0, 100.0, 120.0])
    def test_european_matches_black_scholes(self, option_type: OptionType, strike: float) -> None:
        tree = binomial_price(MARKET, strike, 1.0, option_type)
        exact = float(bs.price(100.0, strike, 1.0, 0.05, 0.30, option_type))
        assert tree == pytest.approx(exact, abs=2e-3)

    def test_american_put_is_worth_more_than_european(self) -> None:
        american = binomial_price(MARKET, 110.0, 1.0, OptionType.PUT, american=True)
        european = binomial_price(MARKET, 110.0, 1.0, OptionType.PUT)
        assert american > european + 0.5

    def test_american_call_without_dividends_equals_european(self) -> None:
        # Early exercise of a call is never optimal when q = 0 (Merton).
        american = binomial_price(MARKET, 100.0, 1.0, OptionType.CALL, american=True)
        european = binomial_price(MARKET, 100.0, 1.0, OptionType.CALL)
        assert american == pytest.approx(european, abs=1e-9)

    def test_rejects_too_few_steps(self) -> None:
        with pytest.raises(ValueError):
            binomial_price(MARKET, 100.0, 1.0, OptionType.PUT, n_steps=1)


class TestLongstaffSchwartz:
    @pytest.mark.parametrize("strike", [90.0, 110.0])
    def test_put_agrees_with_binomial(self, strike: float) -> None:
        lsm = price_american_lsm(
            MARKET, strike, 1.0, OptionType.PUT, n_paths=100_000, n_steps=50, seed=4
        )
        tree = binomial_price(MARKET, strike, 1.0, OptionType.PUT, american=True)
        assert lsm.standard_error is not None
        assert abs(lsm.price - tree) < 3.0 * lsm.standard_error

    def test_call_without_dividends_matches_european(self) -> None:
        lsm = price_american_lsm(
            MARKET, 100.0, 1.0, OptionType.CALL, n_paths=100_000, n_steps=50, seed=9
        )
        exact = float(bs.price(100.0, 100.0, 1.0, 0.05, 0.30, OptionType.CALL))
        assert lsm.standard_error is not None
        assert abs(lsm.price - exact) < 3.0 * lsm.standard_error

    def test_price_is_not_above_the_true_value(self) -> None:
        # Two-pass estimate is low-biased, so it should not exceed the tree by much.
        lsm = price_american_lsm(
            MARKET, 110.0, 1.0, OptionType.PUT, n_paths=100_000, n_steps=50, seed=21
        )
        tree = binomial_price(MARKET, 110.0, 1.0, OptionType.PUT, american=True)
        assert lsm.standard_error is not None
        assert lsm.price < tree + 3.0 * lsm.standard_error

    def test_worth_at_least_the_european_put(self) -> None:
        lsm = price_american_lsm(
            MARKET, 110.0, 1.0, OptionType.PUT, n_paths=100_000, n_steps=50, seed=2
        )
        european = float(bs.price(100.0, 110.0, 1.0, 0.05, 0.30, OptionType.PUT))
        assert lsm.price > european

    def test_accepts_a_custom_sklearn_regressor(self) -> None:
        def ridge() -> object:
            return make_pipeline(PolynomialFeatures(3, include_bias=False), Ridge(alpha=1e-6))

        lsm = price_american_lsm(
            MARKET,
            110.0,
            1.0,
            OptionType.PUT,
            n_paths=50_000,
            n_steps=25,
            seed=6,
            regressor_factory=ridge,
        )
        tree = binomial_price(MARKET, 110.0, 1.0, OptionType.PUT, american=True)
        assert abs(lsm.price - tree) < 0.15

    def test_same_seed_is_reproducible(self) -> None:
        a = price_american_lsm(
            MARKET, 100.0, 1.0, OptionType.PUT, n_paths=5_000, n_steps=10, seed=1
        )
        b = price_american_lsm(
            MARKET, 100.0, 1.0, OptionType.PUT, n_paths=5_000, n_steps=10, seed=1
        )
        assert a.price == b.price
