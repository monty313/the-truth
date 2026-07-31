"""Multi-set BB / RSI-BB / SMA envelope strategy features.

HTF only:
  Bollinger on close: period 100, shift 2, deviation 0.5
  SMA(10) high, SMA(10) low

LTF only:
  RSI(5) with Bollinger on RSI: period 10, shift 5, deviation 1
  SMA(10) high, SMA(10) low

Sets (LTF, HTF1, HTF2):
  A: 5m, 30m, 1h
  B: 15m, 1h, 4h
  C: 30m, 4h, 1d

Buy:
  HTF (any one): close > upper BB AND close > SMA(low)
  LTF: RSI crosses up through RSI lower BB AND close > SMA(low)

Sell (true opposite):
  HTF (any one): close < lower BB AND close < SMA(high)
  LTF: RSI crosses down through RSI upper BB AND close < SMA(high)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.indicators import bollinger, rsi, sma


SETS = {
    "A_5m_30m_1h": {"ltf": "5min", "htfs": ["30min", "1h"]},
    "B_15m_1h_4h": {"ltf": "15min", "htfs": ["1h", "4h"]},
    "C_30m_4h_1d": {"ltf": "30min", "htfs": ["4h", "1d"]},
}


def _tf_frame(ohlc: pd.DataFrame, *, kind: str) -> pd.DataFrame:
    """kind = 'htf' | 'ltf' | 'both' (SMAs always; BB/RSI by role)."""
    c = ohlc["close"]
    out = pd.DataFrame(index=ohlc.index)
    out["open"] = ohlc["open"]
    out["high"] = ohlc["high"]
    out["low"] = ohlc["low"]
    out["close"] = c
    out["sma_high"] = sma(ohlc["high"], 10)
    out["sma_low"] = sma(ohlc["low"], 10)

    if kind in ("htf", "both"):
        up, mid, lo = bollinger(c, 100, 0.5, shift=2)
        out["bb_up"] = up
        out["bb_mid"] = mid
        out["bb_lo"] = lo
        out["above_bb"] = c > up
        out["below_bb"] = c < lo
        out["above_sma_low"] = c > out["sma_low"]
        out["below_sma_high"] = c < out["sma_high"]
        out["htf_buy_ok"] = out["above_bb"] & out["above_sma_low"]
        out["htf_sell_ok"] = out["below_bb"] & out["below_sma_high"]

    if kind in ("ltf", "both"):
        r = rsi(c, 5)
        rup, rmid, rlo = bollinger(r, 10, 1.0, shift=5)
        out["rsi"] = r
        out["rsi_bb_up"] = rup
        out["rsi_bb_mid"] = rmid
        out["rsi_bb_lo"] = rlo
        r_prev = r.shift(1)
        rlo_prev = rlo.shift(1)
        rup_prev = rup.shift(1)
        # cross up lower band: was <= lower, now > lower
        out["rsi_cross_up_lo"] = (r_prev <= rlo_prev) & (r > rlo)
        out["rsi_cross_dn_up"] = (r_prev >= rup_prev) & (r < rup)
        out["above_sma_low"] = c > out["sma_low"]
        out["below_sma_high"] = c < out["sma_high"]
        out["ltf_buy"] = out["rsi_cross_up_lo"] & out["above_sma_low"]
        out["ltf_sell"] = out["rsi_cross_dn_up"] & out["below_sma_high"]

    return out


def align_to_ltf(htf: pd.DataFrame, htf_tf: str, ltf_index: pd.DatetimeIndex) -> pd.DataFrame:
    from data_io.loader import TF_DELTA

    f = htf.copy()
    f.index = f.index + TF_DELTA[htf_tf]
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f.reindex(ltf_index, method="ffill")


def build_set_frame(m1: pd.DataFrame, set_name: str) -> pd.DataFrame:
    """Build LTF frame with signals for one set."""
    from data_io.loader import resample

    spec = SETS[set_name]
    ltf = resample(m1, spec["ltf"])
    base = _tf_frame(ltf, kind="ltf")

    htf_buy_any = pd.Series(False, index=base.index)
    htf_sell_any = pd.Series(False, index=base.index)
    for i, tf in enumerate(spec["htfs"]):
        h = resample(m1, tf)
        hf = _tf_frame(h, kind="htf")
        al = align_to_ltf(
            hf[["htf_buy_ok", "htf_sell_ok", "bb_up", "bb_lo", "sma_high", "sma_low", "close"]].rename(
                columns={
                    "htf_buy_ok": f"htf{i}_buy",
                    "htf_sell_ok": f"htf{i}_sell",
                    "bb_up": f"htf{i}_bb_up",
                    "bb_lo": f"htf{i}_bb_lo",
                    "sma_high": f"htf{i}_sma_hi",
                    "sma_low": f"htf{i}_sma_lo",
                    "close": f"htf{i}_close",
                }
            ),
            tf,
            base.index,
        )
        base = pd.concat([base, al], axis=1)
        htf_buy_any = htf_buy_any | al[f"htf{i}_buy"].fillna(False).astype(bool)
        htf_sell_any = htf_sell_any | al[f"htf{i}_sell"].fillna(False).astype(bool)

    base["htf_buy_any"] = htf_buy_any
    base["htf_sell_any"] = htf_sell_any
    base["long_entry"] = base["ltf_buy"].fillna(False) & base["htf_buy_any"]
    base["short_entry"] = base["ltf_sell"].fillna(False) & base["htf_sell_any"]
    return base
