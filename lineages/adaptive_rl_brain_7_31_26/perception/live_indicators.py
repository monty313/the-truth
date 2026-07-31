"""Live indicator wiring → Phase 1 confluence flags (Slice 1).

CHANGE LOG:
- 2026-07-31  created — WHY: Phase 2 Slice 1. Real CCI/RSI/channel on
  Confirmation TFs only; feed existing confluence_from_confirmation_flags.
  Reuses features/indicators.py (no reimplementation). Entry TF never votes.

Spec (SPEC_PHASE1 §2):
  Group 1 CCI:  CCI 30 + CCI 100 vs SMA(1) shift +4
  Group 2 RSI:  RSI 5 + RSI 14 vs SMA(1) shift +4
  Group 3 Channel: close vs SMA(4) High/Low shift +2
  Dual-confirm: both Confirmation TFs must agree for above/below.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

from features.indicators import cci, rsi, sma, sma_shifted, shifted
from lineages.adaptive_rl_brain_7_31_26.perception.confluence import (
    confluence_from_confirmation_flags,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import SetConfluence

# Locked periods / shifts from Phase 1 spec.
CCI_FAST, CCI_SLOW = 30, 100
RSI_FAST, RSI_SLOW = 5, 14
REF_SMA_N, REF_SMA_SHIFT = 1, 4
CHANNEL_N, CHANNEL_SHIFT = 4, 2

GROUP_KEYS = ("cci", "rsi", "channel")


@dataclass(frozen=True)
class TfIndicatorSnapshot:
    """One bar on one Confirmation TF (or last ready row for tests)."""
    tf: str
    cci30: float
    cci30_sma: float  # SMA(1) of CCI30, shift +4
    cci100: float
    cci100_sma: float
    rsi5: float
    rsi5_sma: float
    rsi14: float
    rsi14_sma: float
    close: float
    channel_high: float  # SMA(4) high, shift +2
    channel_low: float   # SMA(4) low, shift +2


def indicator_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    """Add CCI/RSI/channel + shifted reference lines. Pure; no set logic.

    Required OHLC columns: open, high, low, close (vol optional).
    Output columns (among others):
      cci30, cci30_sma_s4, cci100, cci100_sma_s4,
      rsi5, rsi5_sma_s4, rsi14, rsi14_sma_s4,
      close, ch_high_s2, ch_low_s2
    """
    o = ohlc.copy()
    for col in ("open", "high", "low", "close"):
        if col not in o.columns:
            raise ValueError(f"indicator_frame: missing OHLC column {col!r}")

    cci30 = cci(o, CCI_FAST)
    cci100 = cci(o, CCI_SLOW)
    r5 = rsi(o["close"], RSI_FAST)
    r14 = rsi(o["close"], RSI_SLOW)

    # SMA(period=1) of indicator is the series itself; then MT5 +shift.
    out = pd.DataFrame(index=o.index)
    out["open"] = o["open"]
    out["high"] = o["high"]
    out["low"] = o["low"]
    out["close"] = o["close"]
    out["cci30"] = cci30
    out["cci30_sma_s4"] = shifted(sma(cci30, REF_SMA_N), REF_SMA_SHIFT)
    out["cci100"] = cci100
    out["cci100_sma_s4"] = shifted(sma(cci100, REF_SMA_N), REF_SMA_SHIFT)
    out["rsi5"] = r5
    out["rsi5_sma_s4"] = shifted(sma(r5, REF_SMA_N), REF_SMA_SHIFT)
    out["rsi14"] = r14
    out["rsi14_sma_s4"] = shifted(sma(r14, REF_SMA_N), REF_SMA_SHIFT)
    out["ch_high_s2"] = sma_shifted(o["high"], CHANNEL_N, CHANNEL_SHIFT)
    out["ch_low_s2"] = sma_shifted(o["low"], CHANNEL_N, CHANNEL_SHIFT)
    return out


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def snapshot_at(frame: pd.DataFrame, tf: str, i: int = -1) -> TfIndicatorSnapshot:
    """Read one bar into a frozen snapshot (NaN allowed → neutral upstream)."""
    if frame is None or len(frame) == 0:
        raise ValueError("snapshot_at: empty frame")
    row = frame.iloc[i]
    return TfIndicatorSnapshot(
        tf=str(tf),
        cci30=_f(row["cci30"]),
        cci30_sma=_f(row["cci30_sma_s4"]),
        cci100=_f(row["cci100"]),
        cci100_sma=_f(row["cci100_sma_s4"]),
        rsi5=_f(row["rsi5"]),
        rsi5_sma=_f(row["rsi5_sma_s4"]),
        rsi14=_f(row["rsi14"]),
        rsi14_sma=_f(row["rsi14_sma_s4"]),
        close=_f(row["close"]),
        channel_high=_f(row["ch_high_s2"]),
        channel_low=_f(row["ch_low_s2"]),
    )


def _ready(*vals: float) -> bool:
    return all(np.isfinite(v) for v in vals)


def _pair_above(a: float, a_ref: float, b: float, b_ref: float) -> bool:
    if not _ready(a, a_ref, b, b_ref):
        return False
    return a > a_ref and b > b_ref


def _pair_below(a: float, a_ref: float, b: float, b_ref: float) -> bool:
    if not _ready(a, a_ref, b, b_ref):
        return False
    return a < a_ref and b < b_ref


def group_flags_on_tf(snap: TfIndicatorSnapshot) -> Dict[str, Tuple[bool, bool]]:
    """Per group on one TF: (both_above, both_below) for that TF only."""
    cci_above = _pair_above(snap.cci30, snap.cci30_sma, snap.cci100, snap.cci100_sma)
    cci_below = _pair_below(snap.cci30, snap.cci30_sma, snap.cci100, snap.cci100_sma)
    rsi_above = _pair_above(snap.rsi5, snap.rsi5_sma, snap.rsi14, snap.rsi14_sma)
    rsi_below = _pair_below(snap.rsi5, snap.rsi5_sma, snap.rsi14, snap.rsi14_sma)

    ch_ready = _ready(snap.close, snap.channel_high, snap.channel_low)
    # Channel: above both lines / below both lines
    ch_above = bool(
        ch_ready
        and snap.close > snap.channel_high
        and snap.close > snap.channel_low
    )
    ch_below = bool(
        ch_ready
        and snap.close < snap.channel_high
        and snap.close < snap.channel_low
    )
    return {
        "cci": (cci_above, cci_below),
        "rsi": (rsi_above, rsi_below),
        "channel": (ch_above, ch_below),
    }


def dual_confirmation_flags(
    conf_a: TfIndicatorSnapshot,
    conf_b: TfIndicatorSnapshot,
) -> Dict[str, Tuple[bool, bool]]:
    """AND across both Confirmation TFs → (both_above_on_both, both_below_on_both)."""
    fa = group_flags_on_tf(conf_a)
    fb = group_flags_on_tf(conf_b)
    out: Dict[str, Tuple[bool, bool]] = {}
    for k in GROUP_KEYS:
        a_up, a_dn = fa[k]
        b_up, b_dn = fb[k]
        out[k] = (bool(a_up and b_up), bool(a_dn and b_dn))
    return out


def dual_flags_to_confluence_kwargs(
    dual: Mapping[str, Tuple[bool, bool]],
) -> Dict[str, bool]:
    """Map dual flag dict → kwargs for confluence_from_confirmation_flags."""
    cci_up, cci_dn = dual["cci"]
    rsi_up, rsi_dn = dual["rsi"]
    ch_up, ch_dn = dual["channel"]
    return {
        "cci_both_above": cci_up,
        "cci_both_below": cci_dn,
        "rsi_both_above": rsi_up,
        "rsi_both_below": rsi_dn,
        "channel_both_above": ch_up,
        "channel_both_below": ch_dn,
    }


def confluence_from_confirmation_ohlc(
    set_key: str,
    ohlc_conf_a: pd.DataFrame,
    ohlc_conf_b: pd.DataFrame,
    *,
    bar_a: int = -1,
    bar_b: int = -1,
    tf_a: str = "conf_a",
    tf_b: str = "conf_b",
    entry_ohlc: Optional[pd.DataFrame] = None,  # accepted but IGNORED (Entry never votes)
) -> SetConfluence:
    """End-to-end: two Confirmation OHLC frames → SetConfluence.

    Entry TF is never used for votes. `entry_ohlc` may be supplied for API
    symmetry / caller convenience but is intentionally ignored.
    """
    # Defensive: touch entry_ohlc only to prove we do not read it for math.
    _ = entry_ohlc  # noqa: F841 — explicit ignore
    fa = indicator_frame(ohlc_conf_a)
    fb = indicator_frame(ohlc_conf_b)
    sa = snapshot_at(fa, tf_a, bar_a)
    sb = snapshot_at(fb, tf_b, bar_b)
    dual = dual_confirmation_flags(sa, sb)
    kw = dual_flags_to_confluence_kwargs(dual)
    return confluence_from_confirmation_flags(set_key, **kw)
