r"""Walk-forward forecasts of forward realized volatility, with scikit-learn.

Target: log of the realized vol over the next ``horizon`` trading days (the same
quantity the VIX is implicitly forecasting). Features are HAR style, meaning backward
realized vol over 1, 5 and 22 days, optionally with the VIX added.

Validation is walk-forward with :class:`sklearn.model_selection.TimeSeriesSplit` and a
``gap`` of one horizon between train and test. Without the gap the last training
labels overlap the first test window (the target looks ``horizon`` days ahead) and the
score leaks. Every forecast is made by a model that never saw its own target window.

Three candidates are compared: the VIX itself as the forecast, HAR, and HAR plus the
VIX. The VIX carries a premium (it sits above realized on average), which a fitted
model can learn to remove.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

__all__ = ["build_features", "qlike", "ridge_pipeline", "walk_forward_forecast"]

TRADING_DAYS = 252


def build_features(history: pd.DataFrame, target_rv: pd.Series) -> pd.DataFrame:
    """HAR features plus the VIX, aligned with the forward realized vol target.

    ``target_rv`` is the forward realized vol (decimal) from
    :func:`pricer.vrp.premium_frame`. Rows with any missing value are dropped.
    """
    log_ret = np.log(history["spx"]).diff()
    sq = log_ret**2
    frame = pd.DataFrame(index=history.index)
    frame["lrv1"] = np.log(np.maximum(np.sqrt(TRADING_DAYS * sq), 0.01))
    frame["lrv5"] = np.log(np.sqrt(TRADING_DAYS * sq.rolling(5).mean()))
    frame["lrv22"] = np.log(np.sqrt(TRADING_DAYS * sq.rolling(22).mean()))
    frame["lvix"] = np.log(history["vix"] / 100.0)
    frame["target"] = np.log(target_rv)
    return frame.dropna()


def ridge_pipeline(alpha: float = 1.0) -> Pipeline:
    """Standardize then ridge regression."""
    return make_pipeline(StandardScaler(), Ridge(alpha=alpha))


def walk_forward_forecast(
    features: pd.DataFrame,
    columns: list[str],
    *,
    model_factory: Callable[[], Any] = ridge_pipeline,
    n_splits: int = 8,
    horizon: int = 21,
    min_train: int = 750,
) -> pd.Series:
    """Out-of-sample forecasts of the ``target`` column, indexed like ``features``.

    The first ``min_train`` rows are only ever used for training. Dates covered by no
    test fold are NaN in the returned series.
    """
    x = features[columns].to_numpy()
    y = features["target"].to_numpy()
    usable = len(features) - min_train
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon, test_size=usable // n_splits)
    forecast = pd.Series(np.nan, index=features.index)
    offset = len(features) - splitter.test_size * n_splits  # start of the first test fold
    for train, test in splitter.split(x[offset - min_train :]):
        train, test = train + offset - min_train, test + offset - min_train
        model = model_factory().fit(x[train], y[train])
        forecast.iloc[test] = model.predict(x[test])
    return forecast


def qlike(forecast_vol: np.ndarray, realized_vol: np.ndarray) -> float:
    """Mean QLIKE loss between forecast and realized variance (lower is better)."""
    ratio = (realized_vol / forecast_vol) ** 2
    return float(np.mean(ratio - np.log(ratio) - 1.0))
