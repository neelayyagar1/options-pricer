"""Snapshot round trip and sanity checks on the committed SPY snapshot."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pricer.data.loader import SNAPSHOT_DIR, load_chain, save_chain
from pricer.forward import implied_forward


@pytest.fixture(scope="module")
def snapshot_path() -> Path:
    paths = sorted(SNAPSHOT_DIR.glob("spy_*.json"))
    assert paths, "no committed snapshot found"
    return paths[-1]


def test_round_trip(snapshot_path: Path, tmp_path: Path) -> None:
    chain = load_chain(snapshot_path)
    again = load_chain(save_chain(chain, tmp_path / "copy.json"))
    assert again.spot == chain.spot and again.asof == chain.asof
    assert len(again.expiries) == len(chain.expiries)
    first, other = chain.expiries[3], again.expiries[3]
    np.testing.assert_array_equal(first.calls.strike, other.calls.strike)
    np.testing.assert_array_equal(first.puts.bid, other.puts.bid)


def test_snapshot_is_well_formed(snapshot_path: Path) -> None:
    chain = load_chain(snapshot_path)
    assert chain.spot > 0 and 0.0 < chain.rate < 0.15
    maturities = [e.maturity for e in chain.expiries]
    assert maturities == sorted(maturities) and maturities[0] > 0.0
    for e in chain.expiries:
        assert np.all(np.diff(e.calls.strike) > 0)
        assert e.calls.strike.size > 0 and e.puts.strike.size > 0


def test_implied_forward_is_consistent_with_spot_and_rate(snapshot_path: Path) -> None:
    chain = load_chain(snapshot_path)
    checked = 0
    for e in chain.expiries:
        try:
            fwd = implied_forward(e, chain.spot, rate=chain.rate)
        except ValueError:
            continue
        carry = math.log(fwd.forward / (chain.spot * math.exp(chain.rate * e.maturity)))
        # Dividends and timing noise move it, but not by more than a few percent.
        assert abs(carry) < 0.03
        checked += 1
    assert checked >= 10
