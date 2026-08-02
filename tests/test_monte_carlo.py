"""Validation of the Monte Carlo engine against the closed form.

This is the correctness proof the rest of the project stands on. Three claims are
established here, in increasing order of strength:

1. Simulated prices agree with Black-Scholes to within their own stated error.
2. The error decays at the rate the theory predicts, measured rather than asserted.
3. The stated error is itself correct — including under antithetic sampling, where
   the naive formula is wrong.

Claim 3 is the one most often skipped. Without it, claim 1 is unfalsifiable: an
estimator that overstates its uncertainty passes any "within 3 standard errors"
check trivially.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import numpy as np
import pytest

from pricer import black_scholes as bs
from pricer.convergence import measure_convergence
from pricer.estimators import MCResult, RunningMoments, reduce_antithetic
from pricer.instruments import EuropeanOption, OptionType
from pricer.market import MarketParams
from pricer.monte_carlo import simulate_european, simulate_price
from pricer.paths import (
    RandomSource,
    antithetic_normals,
    draw_normals,
    gbm_paths,
    sobol_normals,
)
from pricer.payoffs import VanillaPayoff

SEED = 20_260_802

MARKET = MarketParams(spot=100.0, volatility=0.25, rate=0.04, dividend_yield=0.015)


class TestPathGeneration:
    """The simulated process must have the right distribution, exactly."""

    def test_paths_start_at_spot(self) -> None:
        normals = draw_normals(1_000, 12, SEED)
        paths = gbm_paths(MARKET, 1.0, normals)

        assert paths.shape == (1_000, 13)
        assert np.all(paths[:, 0] == MARKET.spot)
        assert np.all(paths > 0.0)

    def test_terminal_price_is_a_martingale_under_the_forward(self) -> None:
        """E[S_T] must equal the forward S0 e^{(r-q)T}.

        This is the defining property of the risk-neutral measure. If the drift
        adjustment ``-sigma^2/2`` were wrong or missing, this test fails while
        prices would still look superficially plausible.
        """
        maturity = 2.0
        normals = draw_normals(2_000_000, 1, SEED)
        terminal = gbm_paths(MARKET, maturity, normals)[:, -1]

        expected = MARKET.forward(maturity)
        standard_error = float(np.std(terminal, ddof=1)) / math.sqrt(terminal.size)

        assert abs(float(np.mean(terminal)) - expected) < 4.0 * standard_error

    def test_stepping_is_exact_so_step_count_does_not_change_the_law(self) -> None:
        """One step and many steps must give the same terminal distribution.

        Exact GBM stepping has no discretisation bias, so refining the grid changes
        nothing about the marginal law of S_T. An Euler scheme would fail this.
        """
        maturity = 1.5
        coarse = gbm_paths(MARKET, maturity, draw_normals(400_000, 1, SEED))[:, -1]
        fine = gbm_paths(MARKET, maturity, draw_normals(400_000, 64, SEED + 1))[:, -1]

        for moment in (1, 2):
            a, b = float(np.mean(coarse**moment)), float(np.mean(fine**moment))
            assert a == pytest.approx(b, rel=0.01)

    def test_antithetic_layout_is_mirrored(self) -> None:
        """reduce_antithetic relies on this exact layout, so it is pinned."""
        normals = antithetic_normals(50, 4, SEED)

        assert normals.shape == (100, 4)
        np.testing.assert_allclose(normals[:50], -normals[50:], rtol=0, atol=0)

    def test_same_seed_reproduces_draws(self) -> None:
        """Common random numbers depend on this and nothing else."""
        np.testing.assert_array_equal(draw_normals(500, 3, SEED), draw_normals(500, 3, SEED))
        assert not np.array_equal(draw_normals(500, 3, SEED), draw_normals(500, 3, SEED + 1))

    def test_sobol_requires_a_power_of_two(self) -> None:
        with pytest.raises(ValueError, match="power of two"):
            sobol_normals(1_000, 1, SEED)

    def test_sobol_draws_are_finite_and_standardised(self) -> None:
        normals = sobol_normals(2**14, 1, SEED)

        assert normals.shape == (2**14, 1)
        assert np.all(np.isfinite(normals))
        assert float(np.mean(normals)) == pytest.approx(0.0, abs=0.02)
        assert float(np.std(normals)) == pytest.approx(1.0, abs=0.02)


class TestRunningMoments:
    """Chunked accumulation must equal a single-pass computation."""

    def test_matches_numpy_across_chunk_boundaries(self) -> None:
        rng = np.random.default_rng(SEED)
        values = rng.normal(loc=250.0, scale=0.5, size=10_000)

        moments = RunningMoments()
        for chunk in np.array_split(values, 7):
            moments.update(chunk)

        assert moments.count == values.size
        assert moments.mean == pytest.approx(float(np.mean(values)), rel=1e-12)
        assert moments.variance == pytest.approx(float(np.var(values, ddof=1)), rel=1e-10)

    def test_stays_accurate_when_the_mean_dwarfs_the_spread(self) -> None:
        """The shifted accumulator exists for exactly this case.

        With a mean of 1e6 and a spread of 1, the textbook sum-of-squares formula
        cancels away most of its significant digits.
        """
        rng = np.random.default_rng(SEED)
        values = 1e6 + rng.normal(size=200_000)

        moments = RunningMoments()
        for chunk in np.array_split(values, 13):
            moments.update(chunk)

        assert moments.variance == pytest.approx(float(np.var(values, ddof=1)), rel=1e-9)

    def test_requires_two_samples_for_a_variance(self) -> None:
        moments = RunningMoments()
        moments.update(np.array([1.0]))
        with pytest.raises(ValueError, match="at least two samples"):
            _ = moments.variance


class TestAntitheticPairing:
    """The mirrored payoffs must be collapsed before anything statistical happens."""

    def test_reduce_antithetic_averages_matched_rows(self) -> None:
        values = np.array([1.0, 2.0, 3.0, 10.0, 20.0, 30.0])
        np.testing.assert_allclose(reduce_antithetic(values), [5.5, 11.0, 16.5])

    def test_rejects_odd_length(self) -> None:
        with pytest.raises(ValueError, match="even count"):
            reduce_antithetic(np.ones(5))


class TestStandardErrorIsCorrect:
    """The marquee test: the reported error bar must be the true one.

    Everything else in the suite compares a price to a tolerance expressed in
    standard errors. If the standard error itself were wrong, all of it would be
    vacuous. This establishes the error empirically, by running the estimator many
    times and measuring how much its answer actually moves.
    """

    @staticmethod
    def _price(n_paths: int, seed: int, *, antithetic: bool) -> MCResult:
        option = EuropeanOption(strike=105.0, maturity=1.0, option_type=OptionType.CALL)
        return simulate_european(MARKET, option, n_paths=n_paths, seed=seed, antithetic=antithetic)

    @pytest.mark.parametrize("antithetic", [False, True])
    def test_reported_error_matches_the_empirical_spread(self, antithetic: bool) -> None:
        replications, n_paths = 400, 4_000

        estimates = np.empty(replications)
        reported = np.empty(replications)
        for i in range(replications):
            result = self._price(n_paths, SEED + i, antithetic=antithetic)
            estimates[i] = result.price
            reported[i] = result.standard_error or math.nan

        empirical = float(np.std(estimates, ddof=1))
        assert float(np.mean(reported)) == pytest.approx(empirical, rel=0.10)

    @staticmethod
    def _naive_and_correct_error(
        payoff: Callable[[np.ndarray], np.ndarray],
        maturity: float,
        n_pairs: int,
        seed: int,
    ) -> tuple[float, float]:
        """Return the naive and the correct standard error for one antithetic run."""
        discount = MARKET.discount_factor(maturity)
        normals = antithetic_normals(n_pairs, 1, seed)
        values = discount * payoff(gbm_paths(MARKET, maturity, normals))

        naive = float(np.std(values, ddof=1)) / math.sqrt(values.size)
        pairs = reduce_antithetic(values)
        correct = float(np.std(pairs, ddof=1)) / math.sqrt(pairs.size)
        return naive, correct

    def test_naive_error_hides_the_benefit_when_antithetic_works(self) -> None:
        """For a monotone payoff the naive figure is blind to the reduction achieved.

        SE_naive = sigma_f / sqrt(2N) is exactly the plain-Monte-Carlo error at 2N
        paths and does not involve rho at all, so it reports no improvement no
        matter how well the mirroring worked. Here rho < 0, the true error is
        genuinely smaller, and the naive number fails to show it.
        """
        maturity, n_pairs = 1.0, 4_000
        payoff = VanillaPayoff(strike=105.0, option_type=OptionType.CALL)

        naive, correct = self._naive_and_correct_error(payoff, maturity, n_pairs, SEED)
        empirical = float(
            np.std(
                [self._price(2 * n_pairs, SEED + i, antithetic=True).price for i in range(300)],
                ddof=1,
            )
        )

        assert correct == pytest.approx(empirical, rel=0.12)
        assert naive > 1.10 * empirical

    def test_naive_error_gives_no_warning_when_antithetic_backfires(self) -> None:
        """For a payoff symmetric about the forward, antithetic sampling hurts.

        A straddle pays |S_T - K|, so a large move in either direction pays well and
        the mirrored payoffs become positively correlated. The true error is then
        *larger* than plain Monte Carlo at the same path count, and the naive
        formula — still reporting sigma_f / sqrt(2N) — understates it and raises no
        flag. This is the dangerous direction, and it is why the package never uses
        the naive formula even though it looks conservative in the common case.
        """
        maturity, strike, n_pairs = 1.0, 100.0, 8_000

        def straddle(paths: np.ndarray) -> np.ndarray:
            return np.abs(paths[:, -1] - strike)

        naive, correct = self._naive_and_correct_error(straddle, maturity, n_pairs, SEED)

        assert correct > naive

    def test_confidence_interval_covers_at_the_stated_rate(self) -> None:
        option = EuropeanOption(strike=100.0, maturity=0.75, option_type=OptionType.PUT)
        exact = float(bs.price_contract(option, MARKET))

        replications = 400
        covered = 0
        for i in range(replications):
            result = simulate_european(MARKET, option, n_paths=3_000, seed=SEED + 7_000 + i)
            low, high = result.confidence_interval(0.95)
            covered += low <= exact <= high

        # Binomial standard error of the coverage estimate is about 1.1 points.
        assert 0.92 <= covered / replications <= 0.98


class TestAgreementWithClosedForm:
    """Simulated prices must match Black-Scholes to within their stated error."""

    @pytest.mark.parametrize(
        "params",
        [
            (100.0, 100.0, 1.00, 0.05, 0.20, 0.00),
            (100.0, 80.0, 0.50, 0.03, 0.35, 0.01),
            (100.0, 130.0, 2.00, 0.04, 0.15, 0.02),
            (100.0, 100.0, 0.02, 0.05, 0.60, 0.00),
            (42.0, 40.0, 0.50, 0.10, 0.20, 0.00),
        ],
    )
    @pytest.mark.parametrize("antithetic", [False, True])
    def test_within_three_standard_errors(
        self,
        params: tuple[float, float, float, float, float, float],
        antithetic: bool,
        option_type: OptionType,
    ) -> None:
        spot, strike, maturity, rate, volatility, dividend_yield = params
        market = MarketParams(
            spot=spot, volatility=volatility, rate=rate, dividend_yield=dividend_yield
        )
        option = EuropeanOption(strike=strike, maturity=maturity, option_type=option_type)

        exact = float(bs.price_contract(option, market))
        result = simulate_european(
            market, option, n_paths=200_000, seed=SEED, antithetic=antithetic
        )

        assert result.standard_error is not None
        assert abs(result.price - exact) < 3.0 * result.standard_error

    def test_price_is_stable_across_memory_budgets(self) -> None:
        """Chunking changes which draws land where, but not the answer.

        Reproducibility is contracted on a fixed memory budget, so the two runs are
        different samples; they must still agree to within their combined error.
        """
        option = EuropeanOption(strike=95.0, maturity=1.0, option_type=OptionType.CALL)

        big = simulate_price(
            MARKET,
            option.maturity,
            VanillaPayoff(strike=option.strike, option_type=option.option_type),
            n_paths=200_000,
            seed=SEED,
            n_steps=8,
            max_elements=8_000_000,
        )
        small = simulate_price(
            MARKET,
            option.maturity,
            VanillaPayoff(strike=option.strike, option_type=option.option_type),
            n_paths=200_000,
            seed=SEED,
            n_steps=8,
            max_elements=90_000,
        )

        assert big.standard_error is not None
        assert small.standard_error is not None
        combined = math.hypot(big.standard_error, small.standard_error)
        assert abs(big.price - small.price) < 4.0 * combined

    def test_bitwise_reproducible_with_identical_settings(self) -> None:
        option = EuropeanOption(strike=95.0, maturity=1.0, option_type=OptionType.CALL)
        first = simulate_european(MARKET, option, n_paths=50_000, seed=SEED)
        second = simulate_european(MARKET, option, n_paths=50_000, seed=SEED)

        assert first.price == second.price
        assert first.standard_error == second.standard_error


class TestConvergenceRate:
    """The measured decay exponent must be the one the theory predicts."""

    OPTION = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)

    def test_pseudo_random_error_decays_as_one_over_root_n(self) -> None:
        exact = float(bs.price_contract(self.OPTION, MARKET))

        def price(n_paths: int, seed: np.random.SeedSequence) -> float:
            return simulate_european(MARKET, self.OPTION, n_paths=n_paths, seed=seed).price

        study = measure_convergence(
            price,
            exact,
            path_counts=[2**8, 2**10, 2**12, 2**14, 2**16],
            replications=40,
            seed=SEED,
        )

        assert study.slope == pytest.approx(-0.5, abs=0.06)

    @pytest.mark.slow
    def test_sobol_converges_faster_in_one_dimension(self) -> None:
        """QMC should beat pseudo-random on a genuinely one-dimensional integral.

        A European payoff needs only S_T, so the pricing problem is a 1-D integral —
        the regime where Sobol's advantage is real. The claim being tested is
        deliberately the weak, defensible one: a measurably steeper slope, not a
        specific rate.
        """
        exact = float(bs.price_contract(self.OPTION, MARKET))

        def price(n_paths: int, seed: np.random.SeedSequence) -> float:
            return simulate_european(
                MARKET, self.OPTION, n_paths=n_paths, seed=seed, source=RandomSource.SOBOL
            ).price

        study = measure_convergence(
            price,
            exact,
            path_counts=[2**8, 2**10, 2**12, 2**14],
            replications=30,
            seed=SEED,
        )

        assert study.slope < -0.75


class TestVarianceReductionAccounting:
    """Variance and efficiency comparisons must be like-for-like."""

    def test_antithetic_reduces_variance_for_a_monotone_payoff(self) -> None:
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)

        plain = simulate_european(MARKET, option, n_paths=200_000, seed=SEED)
        anti = simulate_european(MARKET, option, n_paths=200_000, seed=SEED, antithetic=True)

        assert anti.n_samples == plain.n_samples // 2
        assert anti.variance_reduction_versus(plain) > 1.5

    def test_measured_reduction_matches_the_theoretical_factor(self) -> None:
        """The equal-path factor must equal 1/(1+rho), the textbook result.

        This is the test that proves the accounting is right rather than merely
        flattering: rho is measured directly from the mirrored payoffs, and the
        factor reported by MCResult must reproduce it. Normalising by samples
        instead of paths would inflate the left-hand side by exactly two.
        """
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
        payoff = VanillaPayoff(strike=option.strike, option_type=option.option_type)
        n_pairs = 200_000

        normals = antithetic_normals(n_pairs, 1, SEED)
        values = payoff(gbm_paths(MARKET, option.maturity, normals))
        rho = float(np.corrcoef(values[:n_pairs], values[n_pairs:])[0, 1])

        plain = simulate_european(MARKET, option, n_paths=2 * n_pairs, seed=SEED)
        anti = simulate_european(MARKET, option, n_paths=2 * n_pairs, seed=SEED, antithetic=True)

        assert rho < 0.0
        assert anti.variance_reduction_versus(plain) == pytest.approx(1.0 / (1.0 + rho), rel=0.05)

    def test_antithetic_backfires_on_a_symmetric_payoff(self) -> None:
        """A straddle is symmetric about the forward, so mirroring correlates it.

        The reported factor must fall below one. An implementation that assumed
        antithetic sampling is always beneficial would never surface this.
        """
        strike, maturity, n_pairs = 100.0, 1.0, 200_000

        class Straddle:
            path_dependent = False

            def __call__(self, paths: np.ndarray) -> np.ndarray:
                return np.abs(paths[:, -1] - strike)

        plain = simulate_price(MARKET, maturity, Straddle(), n_paths=2 * n_pairs, seed=SEED)
        anti = simulate_price(
            MARKET, maturity, Straddle(), n_paths=2 * n_pairs, seed=SEED, antithetic=True
        )

        assert anti.variance_reduction_versus(plain) < 1.0

    def test_efficiency_accounts_for_runtime(self) -> None:
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
        plain = simulate_european(MARKET, option, n_paths=100_000, seed=SEED)
        anti = simulate_european(MARKET, option, n_paths=100_000, seed=SEED, antithetic=True)

        ratio = anti.variance_reduction_versus(plain) / anti.efficiency_versus(plain)
        expected = anti.elapsed_seconds / plain.elapsed_seconds
        assert ratio == pytest.approx(expected, rel=1e-9)


class TestQuasiMonteCarloHonesty:
    """Sobol must refuse to report statistics it cannot justify."""

    OPTION = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)

    def test_sobol_reports_no_standard_error(self) -> None:
        result = simulate_european(
            MARKET, self.OPTION, n_paths=2**14, seed=SEED, source=RandomSource.SOBOL
        )

        assert result.standard_error is None
        with pytest.raises(ValueError, match="no valid standard error"):
            result.confidence_interval()

    def test_sobol_price_is_still_accurate(self) -> None:
        exact = float(bs.price_contract(self.OPTION, MARKET))
        result = simulate_european(
            MARKET, self.OPTION, n_paths=2**16, seed=SEED, source=RandomSource.SOBOL
        )

        assert result.price == pytest.approx(exact, rel=2e-3)

    def test_sobol_refuses_antithetic(self) -> None:
        with pytest.raises(ValueError, match="incompatible with a Sobol"):
            simulate_european(
                MARKET,
                self.OPTION,
                n_paths=2**12,
                seed=SEED,
                antithetic=True,
                source=RandomSource.SOBOL,
            )


class TestEngineValidation:
    """Misuse must fail loudly."""

    def test_path_dependent_payoff_requires_explicit_steps(self) -> None:
        class Averaging:
            path_dependent = True

            def __call__(self, paths: np.ndarray) -> np.ndarray:
                return np.asarray(paths.mean(axis=1))

        with pytest.raises(ValueError, match="n_steps is required"):
            simulate_price(MARKET, 1.0, Averaging(), n_paths=1_000, seed=SEED)

    def test_antithetic_requires_even_paths(self) -> None:
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
        with pytest.raises(ValueError, match="even n_paths"):
            simulate_european(MARKET, option, n_paths=999, seed=SEED, antithetic=True)

    def test_requires_enough_samples_for_an_error_bar(self) -> None:
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
        with pytest.raises(ValueError, match="at least two independent samples"):
            simulate_european(MARKET, option, n_paths=1, seed=SEED)

    def test_rejects_non_positive_paths(self) -> None:
        option = EuropeanOption(strike=100.0, maturity=1.0, option_type=OptionType.CALL)
        with pytest.raises(ValueError, match="n_paths must be positive"):
            simulate_european(MARKET, option, n_paths=0, seed=SEED)
