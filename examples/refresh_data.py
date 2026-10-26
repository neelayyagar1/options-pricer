"""Refresh the committed market data snapshots from yfinance (needs network).

Run from the repo root:  python examples/refresh_data.py [TICKER]

Writes a dated option chain snapshot and the VIX/S&P 500 daily history. Commit the
new files to update the figures. Note that yfinance only serves current option
chains, so each run captures one day.
"""

from __future__ import annotations

import sys

from pricer.data.history import fetch_history, save_history
from pricer.data.loader import fetch_chain, save_chain

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print("chain:", save_chain(fetch_chain(ticker)))
    print("history:", save_history(fetch_history()))
