"""Variance risk premium construction, inference, and walk-forward forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pricer.data.history import load_history
from pricer.forecast import build_features, qlike, walk_forward_forecast
from pricer.vrp import premium_frame, summarize_premium


def _history(n: int = 300, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    spx = 1000.0 * np.exp(np.cumsum(0.01 * rng.standard_normal(n)))
    vix = 15.0 + 3.0 * rng.standard_normal(n)
    return pd.DataFrame(
        {"vix": np.abs(vix) + 5.0, "spx": spx}, index=pd.bdate_range("2020-01-01", periods=n)
    )


class TestPremiumFrame:
    def test_forward_realized_vol_matches_brute_force(self) -> None:
        hist = _history()
        frame = premium_frame(hist, horizon=21)
        log_ret = np.log(hist["spx"]).diff().to_numpy()
        for date in frame.index[[0, 50, 120]]:
            i = int(hist.index.searchsorted(date))
            window = log_ret[i + 1 : i + 22]
            assert frame.loc[date, "rv"] == pytest.approx(np.sqrt(252 / 21 * np.sum(window**2)))
            assert frame.loc[date, "iv"] == pytest.approx(hist["vix"].iloc[i] / 100.0)

    def test_drops_dates_without_a_full_forward_window(self) -> None:
        hist = _history()
        frame = premium_frame(hist, horizon=21)
        assert frame.index[-1] == hist.index[-22]

    def test_premiums_are_differences(self) -> None:
        frame = premium_frame(_history())
        np.testing.assert_allclose(frame["vol_premium"], frame["iv"] - frame["rv"])
        np.testing.assert_allclose(frame["variance_premium"], frame["iv"] ** 2 - frame["rv"] ** 2)


class TestInference:
    def test_newey_west_matches_naive_for_independent_data(self) -> None:
        x = pd.Series(np.random.default_rng(1).standard_normal(4_000))
        s = summarize_premium(x)
        naive = x.std(ddof=1) / np.sqrt(len(x))
        assert s.newey_west_se == pytest.approx(naive, rel=0.2)

    def test_newey_west_is_much_larger_for_overlapping_windows(self) -> None:
        noise = pd.Series(np.random.default_rng(2).standard_normal(4_000 + 20))
        overlapping = noise.rolling(21).mean().dropna().reset_index(drop=True)
        s = summarize_premium(overlapping)
        naive = overlapping.std(ddof=1) / np.sqrt(len(overlapping))
        assert s.newey_west_se > 3.0 * naive

    def test_non_overlapping_uses_every_horizon_th_point(self) -> None:
        s = summarize_premium(pd.Series(np.arange(210.0)), horizon=21)
        assert s.n_nonoverlapping == 10 and s.n_overlapping == 210


class TestWalkForward:
    def test_training_data_never_overlaps_the_test_window(self) -> None:
        n, horizon = 2_000, 21
        idx = np.arange(n, dtype=float)
        frame = pd.DataFrame({"x": idx, "target": idx}, index=pd.RangeIndex(n))
        seen: list[tuple[float, float]] = []

        class Recorder:
            def __init__(self) -> None:
                self.max_train = 0.0

            def fit(self, x: np.ndarray, y: np.ndarray) -> Recorder:
                self.max_train = float(x.max())
                return self

            def predict(self, x: np.ndarray) -> np.ndarray:
                seen.append((self.max_train, float(x.min())))
                return np.zeros(len(x))

        walk_forward_forecast(frame, ["x"], model_factory=Recorder, horizon=horizon, min_train=500)
        assert len(seen) >= 5
        for last_train, first_test in seen:
            assert first_test - last_train > horizon

    def test_first_rows_are_training_only(self) -> None:
        n = 2_000
        frame = pd.DataFrame({"x": np.arange(n, dtype=float), "target": np.zeros(n)})
        out = walk_forward_forecast(frame, ["x"], min_train=750)
        assert out.iloc[:750].isna().all()
        assert out.iloc[-1:].notna().all()

    def test_learns_a_persistent_signal(self) -> None:
        rng = np.random.default_rng(3)
        n = 3_000
        x = np.zeros(n)
        for t in range(1, n):
            x[t] = 0.98 * x[t - 1] + 0.2 * rng.standard_normal()
        frame = pd.DataFrame({"x": x, "target": x + 0.1 * rng.standard_normal(n)})
        pred = walk_forward_forecast(frame, ["x"])
        ok = pred.notna()
        model_err = np.mean((pred[ok] - frame["target"][ok]) ** 2)
        naive_err = np.mean((frame["target"][ok] - frame["target"][ok].mean()) ** 2)
        assert model_err < 0.1 * naive_err


class TestQlike:
    def test_zero_for_a_perfect_forecast(self) -> None:
        v = np.array([0.1, 0.2, 0.3])
        assert qlike(v, v) == pytest.approx(0.0)

    def test_penalises_under_forecasts_more_than_over_forecasts(self) -> None:
        truth = np.array([0.2])
        assert qlike(np.array([0.1]), truth) > qlike(np.array([0.4]), truth)


class TestCommittedHistory:
    def test_snapshot_gives_a_positive_volatility_premium(self) -> None:
        history = load_history()
        assert len(history) > 4_000 and not history.isna().any().any()
        frame = premium_frame(history)
        summary = summarize_premium(frame["vol_premium"])
        assert summary.mean > 0.0 and summary.newey_west_t > 3.0

    def test_feature_frame_has_no_missing_values(self) -> None:
        history = load_history()
        features = build_features(history, premium_frame(history)["rv"])
        assert not features.isna().any().any()
        assert {"lrv1", "lrv5", "lrv22", "lvix", "target"} <= set(features.columns)
