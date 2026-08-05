"""RSI(2) extreme turn + EMA8 with HTF EMA8 bias. Slots 67-75."""
from __future__ import annotations
import numpy as np
import pandas as pd
from features import indicators as ind


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _m1_ohlc(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    return out


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _raw_on(o: pd.DataFrame, lo=10.0, hi=90.0) -> pd.Series:
    r = ind.rsi(o["close"], 2)
    e8 = _ema(o["close"], 8)
    c = o["close"]
    long_ = (r.shift(1) < lo) & (r > r.shift(1)) & (c > e8)
    short_ = (r.shift(1) > hi) & (r < r.shift(1)) & (c < e8)
    out = np.where(long_.fillna(False), 1.0, np.where(short_.fillna(False), -1.0, 0.0))
    return pd.Series(out, index=o.index, dtype=np.float32)


def compute(F: pd.DataFrame, ltf: str, htfs: list[str]) -> pd.Series:
    m1 = _m1_ohlc(F)
    if m1 is None:
        return _zero(F)
    try:
        from data_io.loader import resample, align_to_m1
        o_ltf = m1 if ltf == "1min" else resample(m1, ltf)
        raw = _raw_on(o_ltf)
        r = raw.to_numpy()
        long_mask, short_mask = r > 0, r < 0
        for tf in htfs:
            o = resample(m1, tf)
            e8 = _ema(o["close"], 8)
            bull = (o["close"] > e8).fillna(False)
            bear = (o["close"] < e8).fillna(False)
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
    "rsi2_ema_1m_15m": lambda F: compute(F, "1min", ["15min"]),
    "rsi2_ema_1m_30m": lambda F: compute(F, "1min", ["30min"]),
    "rsi2_ema_5m_1h": lambda F: compute(F, "5min", ["1h"]),
    "rsi2_ema_5m_4h": lambda F: compute(F, "5min", ["4h"]),
    "rsi2_ema_15m_4h": lambda F: compute(F, "15min", ["4h"]),
    "rsi2_ema_15m_1d": lambda F: compute(F, "15min", ["1d"]),
    "rsi2_ema_A": lambda F: compute(F, "1min", ["15min", "30min"]),
    "rsi2_ema_B": lambda F: compute(F, "5min", ["1h", "4h"]),
    "rsi2_ema_C": lambda F: compute(F, "15min", ["4h", "1d"]),
}
