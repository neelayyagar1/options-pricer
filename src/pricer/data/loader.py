"""Option chain loading with a committed snapshot fallback.

Live data comes from yfinance, which only serves *current* chains (no history). Every
live fetch can be written to a dated JSON file, and the examples read that file by
default, so the figures reproduce with no network. Quotes are whatever yfinance
returned at the time, so a weekend snapshot holds Friday's close.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import numpy.typing as npt

__all__ = [
    "SNAPSHOT_DIR",
    "ExpiryChain",
    "OptionChain",
    "QuoteSet",
    "fetch_chain",
    "load_chain",
    "save_chain",
]

FloatArray = npt.NDArray[np.float64]
SNAPSHOT_DIR = Path(__file__).resolve().parent / "snapshots"
DAYS_PER_YEAR = 365.0


@dataclass(frozen=True, slots=True)
class QuoteSet:
    """Parallel arrays of quotes for one side (calls or puts) of one expiry."""

    strike: FloatArray
    bid: FloatArray
    ask: FloatArray
    last: FloatArray
    volume: FloatArray
    open_interest: FloatArray

    @property
    def mid(self) -> FloatArray:
        """Mid price, NaN where the quote is missing or crossed."""
        valid = (self.bid > 0.0) & (self.ask >= self.bid)
        return np.where(valid, 0.5 * (self.bid + self.ask), np.nan)

    @property
    def relative_spread(self) -> FloatArray:
        """(ask - bid) / mid, NaN where the mid is NaN."""
        return (self.ask - self.bid) / self.mid


@dataclass(frozen=True, slots=True)
class ExpiryChain:
    """Calls and puts for one expiry."""

    expiry: date
    maturity: float
    calls: QuoteSet
    puts: QuoteSet


@dataclass(frozen=True, slots=True)
class OptionChain:
    """A full snapshot: spot plus every stored expiry."""

    ticker: str
    asof: date
    spot: float
    rate: float
    """Continuously compounded risk-free rate (13-week T-bill, approximate)."""
    expiries: tuple[ExpiryChain, ...]


def _quote_set(records: list[dict[str, float | None]]) -> QuoteSet:
    def col(name: str) -> FloatArray:
        return np.array([np.nan if r[name] is None else r[name] for r in records], dtype=float)

    order = np.argsort(col("strike"))
    return QuoteSet(
        strike=col("strike")[order],
        bid=col("bid")[order],
        ask=col("ask")[order],
        last=col("last")[order],
        volume=col("volume")[order],
        open_interest=col("open_interest")[order],
    )


def _records(quotes: QuoteSet) -> list[dict[str, float | None]]:
    def clean(x: float) -> float | None:
        return None if np.isnan(x) else float(x)

    return [
        {
            "strike": float(quotes.strike[i]),
            "bid": clean(quotes.bid[i]),
            "ask": clean(quotes.ask[i]),
            "last": clean(quotes.last[i]),
            "volume": clean(quotes.volume[i]),
            "open_interest": clean(quotes.open_interest[i]),
        }
        for i in range(quotes.strike.size)
    ]


def save_chain(chain: OptionChain, path: Path | None = None) -> Path:
    """Write a chain to JSON. Defaults to ``snapshots/<ticker>_<asof>.json``."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    path = path or SNAPSHOT_DIR / f"{chain.ticker.strip('^').lower()}_{chain.asof:%Y%m%d}.json"
    payload = {
        "ticker": chain.ticker,
        "asof": chain.asof.isoformat(),
        "spot": chain.spot,
        "rate": chain.rate,
        "expiries": [
            {
                "expiry": e.expiry.isoformat(),
                "calls": _records(e.calls),
                "puts": _records(e.puts),
            }
            for e in chain.expiries
        ],
    }
    path.write_text(json.dumps(payload, indent=1))
    return path


def load_chain(path: Path) -> OptionChain:
    """Read a snapshot written by :func:`save_chain`."""
    raw = json.loads(Path(path).read_text())
    asof = date.fromisoformat(raw["asof"])
    expiries = []
    for item in raw["expiries"]:
        expiry = date.fromisoformat(item["expiry"])
        expiries.append(
            ExpiryChain(
                expiry=expiry,
                maturity=(expiry - asof).days / DAYS_PER_YEAR,
                calls=_quote_set(item["calls"]),
                puts=_quote_set(item["puts"]),
            )
        )
    return OptionChain(raw["ticker"], asof, float(raw["spot"]), float(raw["rate"]), tuple(expiries))


def fetch_chain(
    ticker: str = "SPY",
    *,
    min_days: int = 7,
    max_days: int = 400,
    moneyness_band: float = 0.30,
) -> OptionChain:
    """Download the current chain from yfinance (needs the ``data`` extra).

    Keeps expiries between ``min_days`` and ``max_days`` out, and strikes within
    ``moneyness_band`` of spot. ``asof`` is the date of the last close, which is
    what the quotes reflect when markets are shut.
    """
    import yfinance as yf

    handle = yf.Ticker(ticker)
    history = handle.history(period="5d")
    spot = float(history["Close"].iloc[-1])
    asof = history.index[-1].date()
    # ^IRX quotes the 13-week bill yield in percent. Treated as a simple annual yield
    # and converted to continuous compounding, which is close enough for this use.
    irx = yf.Ticker("^IRX").history(period="5d")["Close"].iloc[-1]
    rate = math.log(1.0 + float(irx) / 100.0)

    def to_quotes(frame: object) -> QuoteSet:
        import pandas as pd

        df = frame if isinstance(frame, pd.DataFrame) else pd.DataFrame()
        low, high = spot * (1 - moneyness_band), spot * (1 + moneyness_band)
        df = df[(df.strike >= low) & (df.strike <= high)].sort_values("strike")

        def col(name: str) -> FloatArray:
            return np.asarray(df[name].to_numpy(dtype=float), dtype=np.float64)

        return QuoteSet(
            strike=col("strike"),
            bid=col("bid"),
            ask=col("ask"),
            last=col("lastPrice"),
            volume=col("volume"),
            open_interest=col("openInterest"),
        )

    expiries = []
    for text in handle.options:
        expiry = date.fromisoformat(text)
        days = (expiry - asof).days
        if not min_days <= days <= max_days:
            continue
        chain = handle.option_chain(text)
        expiries.append(
            ExpiryChain(expiry, days / DAYS_PER_YEAR, to_quotes(chain.calls), to_quotes(chain.puts))
        )
    return OptionChain(ticker, asof, spot, rate, tuple(expiries))
