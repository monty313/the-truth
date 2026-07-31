"""SMMA(8) High/Low + RSI(2) — only variants that cleared ~60%+ in tests.
V0: Long RSI cross-down 30; Short cross-down 70
V1: Long RSI turn-up from <30; Short cross-down 70
Slots: 15m+4h and Set C (15m+4h+1d) with HTF SMMA bias.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from features import indicators as ind


def _smma(s: pd.Series, n: int = 8) -> pd.Series:
    return s.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def _m1_ohlc(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    return out


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _raw(o: pd.DataFrame, mode: str = "V1") -> pd.Series:
    r = ind.rsi(o["close"], 2)
    sm_hi = _smma(o["high"], 8)
    sm_lo = _smma(o["low"], 8)
    c = o["close"]
    xdn30 = (r.shift(1) >= 30) & (r < 30)
    xdn70 = (r.shift(1) >= 70) & (r < 70)
    turn_up = (r.shift(1) < 30) & (r > r.shift(1))
    if mode == "V0":
        long_ = (c > sm_lo) & xdn30
        short_ = (c < sm_hi) & xdn70
    else:
        long_ = (c > sm_lo) & turn_up
        short_ = (c < sm_hi) & xdn70
    out = np.where(long_.fillna(False), 1.0, np.where(short_.fillna(False), -1.0, 0.0))
    return pd.Series(out, index=o.index, dtype=np.float32)


def _bias(o: pd.DataFrame):
    sm_hi, sm_lo = _smma(o["high"], 8), _smma(o["low"], 8)
    bull = (o["close"] > sm_lo).fillna(False)
    bear = (o["close"] < sm_hi).fillna(False)
    return bull, bear


def compute(F: pd.DataFrame, ltf: str, htfs: list[str], mode: str = "V1") -> pd.Series:
    m1 = _m1_ohlc(F)
    if m1 is None:
        return _zero(F)
    try:
        from data_io.loader import resample, align_to_m1
        o_ltf = m1 if ltf == "1min" else resample(m1, ltf)
        raw = _raw(o_ltf, mode)
        r = raw.to_numpy()
        long_mask, short_mask = r > 0, r < 0
        for tf in htfs:
            o = resample(m1, tf)
            bull, bear = _bias(o)
            ba = bull.reindex(o_ltf.index, method="ffill").fillna(False).to_numpy()
            sa = bear.reindex(o_ltf.index, method="ffill").fillna(False).to_numpy()
            long_mask = long_mask & ba
            short_mask = short_mask & sa
        filtered = np.where(long_mask, 1.0, np.where(short_mask, -1.0, 0.0)).astype(np.float32)
        ser = pd.Series(filtered, index=o_ltf.index, name="s")
        if ltf == "1min":
            return ser.astype(np.float32).reindex(F.index).fillna(0.0)
        aligned = align_to_m1(ser.to_frame(), ltf, m1.index)["s"]
        return aligned.astype(np.float32).reindex(F.index).fillna(0.0)
    except Exception:
        return _zero(F)


HANDLERS = {
    "smma_rsi_15m_4h_V0": lambda F: compute(F, "15min", ["4h"], "V0"),
    "smma_rsi_15m_4h_V1": lambda F: compute(F, "15min", ["4h"], "V1"),
    "smma_rsi_C_V0": lambda F: compute(F, "15min", ["4h", "1d"], "V0"),
    "smma_rsi_C_V1": lambda F: compute(F, "15min", ["4h", "1d"], "V1"),
}
