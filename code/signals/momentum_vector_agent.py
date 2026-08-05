"""Momentum Vector M — profit-tuned signal agents for RL.

US30 research winners (profit-first, hard SL/TP backtest style):

  Best quality (~+99.7% PF 1.23 Sharpe 0.97):
    30m+4h | entry 35 | HTF|M|≥25 | long only | (BT: SL2.0 TP4.0)

  Strong #3 (~+97.4% PF 1.21):
    30m+4h | entry 35 | HTF|M|≥25 | long only | (BT: SL1.5 TP3.5)
    → same ENTRY signal as best quality (SL/TP only differ in sim)

  Strong #4 (~+81.3% PF 1.12):
    30m+4h | entry 25 | HTF|M|≥12 | long only | (BT: SL2.0 TP4.0)

Signal: +1 long / 0 flat (sticky after cross). RL may ignore.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.momentum_vector import build_mtf_momentum


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _m1(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    return out


def _sticky(long_e: np.ndarray, short_e: np.ndarray, M: np.ndarray) -> np.ndarray:
    n = len(M)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    for i in range(n):
        if long_e[i]:
            pos = 1
        elif short_e[i]:
            pos = -1
        else:
            if pos > 0 and (not np.isfinite(M[i]) or M[i] <= 0):
                pos = 0
            elif pos < 0 and (not np.isfinite(M[i]) or M[i] >= 0):
                pos = 0
        out[i] = float(pos)
    return out


def compute(
    F: pd.DataFrame,
    base_tf: str,
    htf_tf: str,
    *,
    entry_thr: float = 35.0,
    htf_m_min: float = 25.0,
    sides: str = "long",
) -> pd.Series:
    m1 = _m1(F)
    if m1 is None or len(m1) < 500:
        return _zero(F)
    try:
        from data_io.loader import align_to_m1, resample

        base = resample(m1, base_tf)
        htf = resample(m1, htf_tf)
        fr = build_mtf_momentum(base, htf, htf_tf)
        M = fr["M"].to_numpy(float)
        Mp = np.roll(M, 1)
        Mp[0] = np.nan
        hM = fr["htf_M"].to_numpy(float)
        hD = fr["htf_direction"].to_numpy(float)

        long_lv = (M > entry_thr) & (hD > 0) & (hM > htf_m_min)
        short_lv = (M < -entry_thr) & (hD < 0) & (hM < -htf_m_min)
        long_e = long_lv & (Mp <= entry_thr)
        short_e = short_lv & (Mp >= -entry_thr)
        if sides == "long":
            short_e = np.zeros_like(short_e)
        elif sides == "short":
            long_e = np.zeros_like(long_e)
        long_e = np.nan_to_num(long_e.astype(float), nan=0.0).astype(bool)
        short_e = np.nan_to_num(short_e.astype(float), nan=0.0).astype(bool)

        sig = _sticky(long_e, short_e, M)
        ser = pd.Series(sig, index=fr.index, name="s", dtype=np.float32)
        aligned = align_to_m1(ser.to_frame(), base_tf, m1.index)["s"]
        return aligned.astype(np.float32).reindex(F.index).fillna(0.0)
    except Exception:
        return _zero(F)


def mv_best_quality_30m_4h_long(F: pd.DataFrame) -> pd.Series:
    """Best quality: 30m+4h thr35 hm25 long (BT SL2 TP4) ~+99.7% US30."""
    return compute(F, "30min", "4h", entry_thr=35.0, htf_m_min=25.0, sides="long")


def mv_strong3_30m_4h_long(F: pd.DataFrame) -> pd.Series:
    """Strong #3: same entries as best quality (BT SL1.5 TP3.5) ~+97.4% US30."""
    return compute(F, "30min", "4h", entry_thr=35.0, htf_m_min=25.0, sides="long")


def mv_strong4_30m_4h_long(F: pd.DataFrame) -> pd.Series:
    """Strong #4: 30m+4h thr25 hm12 long (BT SL2 TP4) ~+81.3% US30."""
    return compute(F, "30min", "4h", entry_thr=25.0, htf_m_min=12.0, sides="long")


# Aliases (older names kept so nothing breaks)
def mv_profit_30m_4h_long(F: pd.DataFrame) -> pd.Series:
    return mv_best_quality_30m_4h_long(F)


def mv_profit_30m_4h_both(F: pd.DataFrame) -> pd.Series:
    """Optional both-sides thr35 hm12 (high ret research)."""
    return compute(F, "30min", "4h", entry_thr=35.0, htf_m_min=12.0, sides="both")


def mv_eur_1h_1d_hard(F: pd.DataFrame) -> pd.Series:
    return compute(F, "1h", "1d", entry_thr=25.0, htf_m_min=12.0, sides="both")


HANDLERS = {
    "mv_best_quality_30m_4h_long": mv_best_quality_30m_4h_long,
    "mv_strong3_30m_4h_long": mv_strong3_30m_4h_long,
    "mv_strong4_30m_4h_long": mv_strong4_30m_4h_long,
    # aliases
    "mv_profit_30m_4h_long": mv_profit_30m_4h_long,
    "mv_profit_30m_4h_both": mv_profit_30m_4h_both,
    "mv_eur_1h_1d_hard": mv_eur_1h_1d_hard,
}
