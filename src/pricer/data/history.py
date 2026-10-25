"""Daily VIX and S&P 500 history, with a committed CSV snapshot.

This is the data behind the variance risk premium study. Unlike option chains, daily
index history is available from yfinance back many years, so the study can use a long
sample.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pricer.data.loader import SNAPSHOT_DIR

__all__ = ["HISTORY_PATH", "fetch_history", "load_history", "save_history"]

HISTORY_PATH = SNAPSHOT_DIR / "vix_spx_daily.csv"


def fetch_history(start: str = "2005-01-01") -> pd.DataFrame:
    """Download daily closes of ^VIX and ^GSPC, aligned on common dates."""
    import yfinance as yf

    raw = yf.download(["^VIX", "^GSPC"], start=start, progress=False, auto_adjust=False)
    closes = raw["Close"].rename(columns={"^VIX": "vix", "^GSPC": "spx"})
    closes.index = pd.DatetimeIndex(closes.index).tz_localize(None)
    aligned: pd.DataFrame = closes[["vix", "spx"]].dropna()
    return aligned


def save_history(frame: pd.DataFrame, path: Path = HISTORY_PATH) -> Path:
    """Write the history to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="date", float_format="%.6f")
    return path


def load_history(path: Path = HISTORY_PATH) -> pd.DataFrame:
    """Read a history CSV written by :func:`save_history`."""
    return pd.read_csv(path, index_col="date", parse_dates=True)
