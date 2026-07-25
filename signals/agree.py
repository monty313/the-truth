"""High-accuracy agreement signal agents (≥70% at 10 bars in tests).
Independent recomputation of component families, then intersection/vote.
Slots 80+:
  agree_seA_r2A       — stoch_ema_A ∩ rsi2_ema_A (~75% @10)
  agree_seB_r2B_epB   — 2of seB, r2B, ema_pull_B (~70-72%)
  agree_2of_top4      — 2of seA, r2A, seB, smaC (~76/71% @10/20)
  agree_seA_r2A_atr   — seA∩r2A + ATR active (~78-81%)
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from features import indicators as ind


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _m1(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    return out


def _stochastic(o: pd.DataFrame):
    lo = o["low"].rolling(5, min_periods=5).min()
    hi = o["high"].rolling(5, min_periods=5).max()
    denom = (hi - lo).replace(0.0, np.nan)
    k = (100.0 * (o["close"] - lo) / denom).rolling(3, min_periods=3).mean()
    d = k.rolling(3, min_periods=3).mean()
    return k, d


def _stoch_ema_raw(o: pd.DataFrame):
    k, d = _stochastic(o)
    e8 = _ema(o["close"], 8)
    c = o["close"]
    cross_up = (k.shift(1) <= d.shift(1)) & (k > d)
    cross_dn = (k.shift(1) >= d.shift(1)) & (k < d)
    long_ = cross_up & (k < 40) & (c > e8)
    short_ = cross_dn & (k > 60) & (c < e8)
    return long_.fillna(False), short_.fillna(False)


def _stoch_ema_bias(o: pd.DataFrame):
    k, d = _stochastic(o)
    e8 = _ema(o["close"], 8)
    bull = (k > d) & (o["close"] > e8)
    bear = (k < d) & (o["close"] < e8)
    return bull.fillna(False), bear.fillna(False)


def _stoch_ema_htf(m1: pd.DataFrame, ltf: str, htfs: list[str]) -> pd.Series:
    from data_io.loader import resample, align_to_m1
    o = m1 if ltf == "1min" else resample(m1, ltf)
    L, S = _stoch_ema_raw(o)
    for tf in htfs:
        h = resample(m1, tf)
        b, be = _stoch_ema_bias(h)
        L = L & b.reindex(o.index, method="ffill").fillna(False)
        S = S & be.reindex(o.index, method="ffill").fillna(False)
    sig = np.where(L, 1.0, np.where(S, -1.0, 0.0)).astype(np.float32)
    ser = pd.Series(sig, index=o.index, name="s")
    if ltf == "1min":
        return ser.reindex(m1.index).fillna(0.0).astype(np.float32)
    return align_to_m1(ser.to_frame(), ltf, m1.index)["s"].astype(np.float32).reindex(m1.index).fillna(0.0)


def _rsi2_raw(o: pd.DataFrame):
    r = ind.rsi(o["close"], 2)
    e8 = _ema(o["close"], 8)
    long_ = (r.shift(1) < 10) & (r > r.shift(1)) & (o["close"] > e8)
    short_ = (r.shift(1) > 90) & (r < r.shift(1)) & (o["close"] < e8)
    return long_.fillna(False), short_.fillna(False)


def _rsi2_htf(m1: pd.DataFrame, ltf: str, htfs: list[str]) -> pd.Series:
    from data_io.loader import resample, align_to_m1
    o = m1 if ltf == "1min" else resample(m1, ltf)
    L, S = _rsi2_raw(o)
    for tf in htfs:
        h = resample(m1, tf)
        e8 = _ema(h["close"], 8)
        b = (h["close"] > e8).fillna(False)
        be = (h["close"] < e8).fillna(False)
        L = L & b.reindex(o.index, method="ffill").fillna(False)
        S = S & be.reindex(o.index, method="ffill").fillna(False)
    sig = np.where(L, 1.0, np.where(S, -1.0, 0.0)).astype(np.float32)
    ser = pd.Series(sig, index=o.index, name="s")
    if ltf == "1min":
        return ser.reindex(m1.index).fillna(0.0).astype(np.float32)
    return align_to_m1(ser.to_frame(), ltf, m1.index)["s"].astype(np.float32).reindex(m1.index).fillna(0.0)


def _ema_pull_htf(m1: pd.DataFrame, ltf: str, htfs: list[str]) -> pd.Series:
    from data_io.loader import resample, align_to_m1
    o = m1 if ltf == "1min" else resample(m1, ltf)
    e8, e21 = _ema(o["close"], 8), _ema(o["close"], 21)
    c, h, l, op = o["close"], o["high"], o["low"], o["open"]
    long_ = (c > e21) & (e8 > e21) & (l.shift(1) <= e8.shift(1)) & (c.shift(1) <= e8.shift(1)) & (c > e8) & (c > op)
    short_ = (c < e21) & (e8 < e21) & (h.shift(1) >= e8.shift(1)) & (c.shift(1) >= e8.shift(1)) & (c < e8) & (c < op)
    long_, short_ = long_.fillna(False), short_.fillna(False)
    for tf in htfs:
        ht = resample(m1, tf)
        he8, he21 = _ema(ht["close"], 8), _ema(ht["close"], 21)
        bull = ((ht["close"] > he21) & (he8 > he21) & (he21 > he21.shift(2))).reindex(o.index, method="ffill").fillna(False)
        bear = ((ht["close"] < he21) & (he8 < he21) & (he21 < he21.shift(2))).reindex(o.index, method="ffill").fillna(False)
        long_ = long_ & bull
        short_ = short_ & bear
    sig = np.where(long_, 1.0, np.where(short_, -1.0, 0.0)).astype(np.float32)
    ser = pd.Series(sig, index=o.index, name="s")
    if ltf == "1min":
        return ser.reindex(m1.index).fillna(0.0).astype(np.float32)
    return align_to_m1(ser.to_frame(), ltf, m1.index)["s"].astype(np.float32).reindex(m1.index).fillna(0.0)


def _sma_outer_C(m1: pd.DataFrame) -> pd.Series:
    from data_io.loader import resample, align_to_m1

    def band(o, shift):
        hi = o["high"].rolling(4, min_periods=4).mean().shift(shift)
        lo = o["low"].rolling(4, min_periods=4).mean().shift(shift)
        return hi, lo

    o = resample(m1, "15min")
    l_hi, l_lo = band(o, 2)
    htf_up = pd.Series(True, index=o.index)
    htf_dn = pd.Series(True, index=o.index)
    for tf in ("4h", "1d"):
        h = resample(m1, tf)
        hh, hl = band(h, 4)
        hh = hh.reindex(o.index, method="ffill")
        hl = hl.reindex(o.index, method="ffill")
        htf_up = htf_up & (o["close"] > hh) & (o["close"] > hl)
        htf_dn = htf_dn & (o["close"] < hh) & (o["close"] < hl)
    long_ = (o["close"] < l_lo) & htf_up
    short_ = (o["close"] > l_hi) & htf_dn
    sig = np.where(long_.fillna(False), 1.0, np.where(short_.fillna(False), -1.0, 0.0)).astype(np.float32)
    ser = pd.Series(sig, index=o.index, name="s")
    return align_to_m1(ser.to_frame(), "15min", m1.index)["s"].astype(np.float32).reindex(m1.index).fillna(0.0)


def _agree(*arrs, min_votes: int = 2) -> np.ndarray:
    stack = np.stack([np.asarray(a, dtype=np.float32) for a in arrs], axis=0)
    up = (stack > 0).sum(axis=0)
    dn = (stack < 0).sum(axis=0)
    out = np.zeros(stack.shape[1], dtype=np.float32)
    out = np.where(up >= min_votes, 1.0, out)
    out = np.where(dn >= min_votes, -1.0, out)
    conflict = (up >= min_votes) & (dn >= min_votes)
    return np.where(conflict, 0.0, out).astype(np.float32)


def _compute_family(F: pd.DataFrame):
    m1 = _m1(F)
    if m1 is None:
        z = np.zeros(len(F), dtype=np.float32)
        return z, z, z, z, z, z, z
    seA = _stoch_ema_htf(m1, "1min", ["15min", "30min"]).to_numpy()
    r2A = _rsi2_htf(m1, "1min", ["15min", "30min"]).to_numpy()
    seB = _stoch_ema_htf(m1, "5min", ["1h", "4h"]).to_numpy()
    r2B = _rsi2_htf(m1, "5min", ["1h", "4h"]).to_numpy()
    epB = _ema_pull_htf(m1, "5min", ["1h", "4h"]).to_numpy()
    smaC = _sma_outer_C(m1).to_numpy()
    epA = _ema_pull_htf(m1, "1min", ["15min", "30min"]).to_numpy()
    return seA, r2A, seB, r2B, epB, smaC, epA


def agree_seA_r2A(F: pd.DataFrame) -> pd.Series:
    try:
        seA, r2A, *_ = _compute_family(F)
        out = _agree(seA, r2A, min_votes=2)
        return pd.Series(out, index=F.index, dtype=np.float32)
    except Exception:
        return _zero(F)


def agree_seB_r2B_epB(F: pd.DataFrame) -> pd.Series:
    try:
        _, _, seB, r2B, epB, _, _ = _compute_family(F)
        out = _agree(seB, r2B, epB, min_votes=2)
        return pd.Series(out, index=F.index, dtype=np.float32)
    except Exception:
        return _zero(F)


def agree_2of_top4(F: pd.DataFrame) -> pd.Series:
    try:
        seA, r2A, seB, _, _, smaC, _ = _compute_family(F)
        out = _agree(seA, r2A, seB, smaC, min_votes=2)
        return pd.Series(out, index=F.index, dtype=np.float32)
    except Exception:
        return _zero(F)


def agree_seA_r2A_atr(F: pd.DataFrame) -> pd.Series:
    try:
        m1 = _m1(F)
        if m1 is None:
            return _zero(F)
        seA, r2A, *_ = _compute_family(F)
        base = _agree(seA, r2A, min_votes=2)
        atr = (m1["high"] - m1["low"]).rolling(14, min_periods=5).mean()
        active = (atr > atr.rolling(50, min_periods=10).median()).fillna(False).to_numpy()
        out = np.where(active, base, 0.0).astype(np.float32)
        return pd.Series(out, index=F.index, dtype=np.float32)
    except Exception:
        return _zero(F)


HANDLERS = {
    "agree_seA_r2A": agree_seA_r2A,
    "agree_seB_r2B_epB": agree_seB_r2B_epB,
    "agree_2of_top4": agree_2of_top4,
    "agree_seA_r2A_atr": agree_seA_r2A_atr,
}
