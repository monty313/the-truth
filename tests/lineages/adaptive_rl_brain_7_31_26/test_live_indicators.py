"""Phase 2 Slice 1: live indicator → confluence flags / votes."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.perception.confluence import (
    confluence_from_confirmation_flags,
)
from lineages.adaptive_rl_brain_7_31_26.perception.live_indicators import (
    confluence_from_confirmation_ohlc,
    dual_confirmation_flags,
    dual_flags_to_confluence_kwargs,
    group_flags_on_tf,
    indicator_frame,
    snapshot_at,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, VelocityStrength


def _ohlc_trend(n: int = 300, *, direction: int = 1) -> pd.DataFrame:
    """Synthetic trend with mild pullbacks so CCI/RSI move vs lag (not flat).

    direction=+1 bull, -1 bear. Thrust legs dominate so dual-confirm can go STRONG.
    """
    close = 100.0 if direction > 0 else 500.0
    closes = []
    for i in range(n):
        # 7 bars with trend, 3 mild counter — keeps RSI off a pure flat ceiling/floor
        thrust = (i % 10) < 7
        if thrust:
            close += direction * 1.5
        else:
            close -= direction * 0.6
        closes.append(close)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": c - direction * 0.2,
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
    })


def _first_full_vote_bar(ohlc: pd.DataFrame, *, want_bull: bool) -> int:
    """Index where all three groups vote the requested side on one TF."""
    f = indicator_frame(ohlc)
    for i in range(len(f)):
        fl = group_flags_on_tf(snapshot_at(f, "t", i))
        if want_bull and all(fl[k][0] for k in fl):
            return i
        if (not want_bull) and all(fl[k][1] for k in fl):
            return i
    raise AssertionError("no full-group vote bar in synthetic series")


def test_indicator_frame_has_required_columns():
    o = _ohlc_trend(200, direction=1)
    f = indicator_frame(o)
    for col in (
        "cci30", "cci30_sma_s4", "cci100", "cci100_sma_s4",
        "rsi5", "rsi5_sma_s4", "rsi14", "rsi14_sma_s4",
        "close", "ch_high_s2", "ch_low_s2",
    ):
        assert col in f.columns, col
    assert len(f) == 200


def test_strong_uptrend_both_confirmations_bull_votes():
    up = _ohlc_trend(300, direction=1)
    i = _first_full_vote_bar(up, want_bull=True)
    c = confluence_from_confirmation_ohlc(
        "official:1", up, up.copy(), bar_a=i, bar_b=i,
    )
    assert c.direction == Direction.BULL
    assert c.velocity == VelocityStrength.STRONG
    fa = indicator_frame(up)
    sa = snapshot_at(fa, "a", i)
    sb = snapshot_at(fa, "b", i)
    dual = dual_confirmation_flags(sa, sb)
    for k in ("cci", "rsi", "channel"):
        assert dual[k][0] is True, k
        assert dual[k][1] is False, k


def test_strong_downtrend_both_confirmations_bear_votes():
    dn = _ohlc_trend(300, direction=-1)
    i = _first_full_vote_bar(dn, want_bull=False)
    c = confluence_from_confirmation_ohlc(
        "official:2", dn, dn.copy(), bar_a=i, bar_b=i,
    )
    assert c.direction == Direction.BEAR
    assert c.velocity == VelocityStrength.STRONG


def test_mixed_confirmations_yield_neutral_or_weaker():
    up = _ohlc_trend(300, direction=1)
    dn = _ohlc_trend(300, direction=-1)
    i_up = _first_full_vote_bar(up, want_bull=True)
    i_dn = _first_full_vote_bar(dn, want_bull=False)
    c = confluence_from_confirmation_ohlc(
        "sub:A", up, dn, bar_a=i_up, bar_b=i_dn,
    )
    dual = dual_confirmation_flags(
        snapshot_at(indicator_frame(up), "a", i_up),
        snapshot_at(indicator_frame(dn), "b", i_dn),
    )
    # Opposite confirmations → no dual-above or dual-below on groups
    assert dual["cci"][0] is False
    assert dual["cci"][1] is False
    assert c.direction == Direction.NEUTRAL
    assert c.velocity == VelocityStrength.NONE


def test_entry_tf_never_required():
    up = _ohlc_trend(180, direction=1)
    i = _first_full_vote_bar(up, want_bull=True)
    c = confluence_from_confirmation_ohlc(
        "official:3", up, up.copy(), bar_a=i, bar_b=i,
    )
    assert c.set_key == "official:3"
    assert isinstance(c.direction, Direction)


def test_entry_tf_data_does_not_affect_votes_even_if_supplied():
    """Defensive: entry_ohlc is accepted but must not change votes."""
    conf = _ohlc_trend(300, direction=1)
    i = _first_full_vote_bar(conf, want_bull=True)
    entry_bear = _ohlc_trend(300, direction=-1)
    # short noisy junk entry
    entry_noise = pd.DataFrame({
        "open": np.linspace(1, 2, 40),
        "high": np.linspace(1.5, 2.5, 40),
        "low": np.linspace(0.5, 1.5, 40),
        "close": np.linspace(1.1, 1.9, 40),
    })

    base = confluence_from_confirmation_ohlc(
        "official:1", conf, conf.copy(), bar_a=i, bar_b=i,
    )
    with_bear_entry = confluence_from_confirmation_ohlc(
        "official:1", conf, conf.copy(), bar_a=i, bar_b=i, entry_ohlc=entry_bear,
    )
    with_noise_entry = confluence_from_confirmation_ohlc(
        "official:1", conf, conf.copy(), bar_a=i, bar_b=i, entry_ohlc=entry_noise,
    )
    assert base.direction == with_bear_entry.direction == with_noise_entry.direction
    assert base.velocity == with_bear_entry.velocity == with_noise_entry.velocity
    assert base.n_bull == with_bear_entry.n_bull == with_noise_entry.n_bull
    assert base.n_bear == with_bear_entry.n_bear == with_noise_entry.n_bear
    assert base.direction == Direction.BULL
    assert base.velocity == VelocityStrength.STRONG


def test_nan_warmup_is_neutral_not_crash():
    # Too short for CCI/RSI/channel refs → NaNs → dual flags false → NEUTRAL
    short = _ohlc_trend(5, direction=1)
    c = confluence_from_confirmation_ohlc("official:4", short, short.copy())
    assert c.direction == Direction.NEUTRAL
    assert c.velocity == VelocityStrength.NONE


def test_flags_feed_existing_confluence_helpers():
    up = _ohlc_trend(300, direction=1)
    i = _first_full_vote_bar(up, want_bull=True)
    sa = snapshot_at(indicator_frame(up), "a", i)
    sb = snapshot_at(indicator_frame(up), "b", i)
    dual = dual_confirmation_flags(sa, sb)
    kw = dual_flags_to_confluence_kwargs(dual)
    c1 = confluence_from_confirmation_flags("k", **kw)
    c2 = confluence_from_confirmation_ohlc("k", up, up.copy(), bar_a=i, bar_b=i)
    assert c1.direction == c2.direction
    assert c1.velocity == c2.velocity
    assert c1.n_bull == c2.n_bull


def test_group_flags_on_tf_shape():
    up = _ohlc_trend(220, direction=1)
    i = _first_full_vote_bar(up, want_bull=True)
    snap = snapshot_at(indicator_frame(up), "t", i)
    flags = group_flags_on_tf(snap)
    assert set(flags) == {"cci", "rsi", "channel"}
    for v in flags.values():
        assert len(v) == 2
        assert all(isinstance(x, bool) for x in v)


if __name__ == "__main__":
    test_indicator_frame_has_required_columns()
    test_strong_uptrend_both_confirmations_bull_votes()
    test_strong_downtrend_both_confirmations_bear_votes()
    test_mixed_confirmations_yield_neutral_or_weaker()
    test_entry_tf_never_required()
    test_entry_tf_data_does_not_affect_votes_even_if_supplied()
    test_nan_warmup_is_neutral_not_crash()
    test_flags_feed_existing_confluence_helpers()
    test_group_flags_on_tf_shape()
    print("test_live_indicators OK")
