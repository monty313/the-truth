"""DVMR multi-TF signal agents for RL observation slots.

Champion (highest backtest return on EURUSD research, 2026-07-29):
  base=1h, HTF=1d, n=5, m=15, htf_thr=0.50, hard-style hold
  IS ~+18.8% | PF~1.92 | Sharpe~0.83 | walk-forward OOS positive

Signal values: +1 long suggestion | -1 short | 0 flat.
RL is free to ignore; this is an observation channel only.

Sticky policy (mirrors the hard SL/TP *intent* without price exits):
  enter on base DVMR zero-cross + HTF confirm + regime filter;
  stay in direction while base DVMR keeps the same sign;
  flip or flatten on opposite entry / zero cross.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.dvmr import (
    DEFAULT_CLIP,
    compute_dvmr_features,
    regime_from_dvmr,
)


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _m1_ohlc(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    return out


def _align_htf_series(htf_s: pd.Series, htf_tf: str, base_index: pd.DatetimeIndex) -> pd.Series:
    """No-look-ahead: HTF bar usable only after it closes."""
    from data_io.loader import TF_DELTA

    s = htf_s.copy()
    s.index = s.index + TF_DELTA[htf_tf]
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return s.reindex(base_index, method="ffill")


def _sticky_from_entries(
    long_entry: np.ndarray,
    short_entry: np.ndarray,
    dvmr: np.ndarray,
) -> np.ndarray:
    """Hold direction after entry while DVMR keeps the same sign."""
    n = len(dvmr)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    for i in range(n):
        if long_entry[i]:
            pos = 1
        elif short_entry[i]:
            pos = -1
        else:
            if pos > 0 and (not np.isfinite(dvmr[i]) or dvmr[i] <= 0):
                pos = 0
            elif pos < 0 and (not np.isfinite(dvmr[i]) or dvmr[i] >= 0):
                pos = 0
        out[i] = float(pos)
    return out


def compute(
    F: pd.DataFrame,
    base_tf: str,
    htf_tf: str,
    *,
    n: int = 5,
    m: int = 15,
    htf_thr: float = 0.50,
    sticky: bool = True,
) -> pd.Series:
    """DVMR signal on base TF, aligned onto F's M1 index."""
    m1 = _m1_ohlc(F)
    if m1 is None or len(m1) < max(m, n) + 50:
        return _zero(F)
    try:
        from data_io.loader import align_to_m1, resample

        base = m1 if base_tf in ("1min", "1m") else resample(m1, base_tf)
        htf = resample(m1, htf_tf)
        if len(base) < max(m, n) + 20 or len(htf) < max(m, n) + 5:
            return _zero(F)

        bf = compute_dvmr_features(base, n=n, m=m, clip=DEFAULT_CLIP)
        hf = compute_dvmr_features(htf, n=n, m=m, clip=DEFAULT_CLIP)
        d = bf["dvmr"]
        d_prev = d.shift(1)
        htf_d = _align_htf_series(hf["dvmr"], htf_tf, base.index)
        reg = regime_from_dvmr(d)

        cross_up = (d_prev <= 0) & (d > 0)
        cross_dn = (d_prev >= 0) & (d < 0)
        long_entry = (cross_up & (htf_d > htf_thr) & reg.isin([1, 2])).fillna(False).to_numpy()
        short_entry = (cross_dn & (htf_d < -htf_thr) & (reg == -1)).fillna(False).to_numpy()

        if sticky:
            sig = _sticky_from_entries(long_entry, short_entry, d.to_numpy(dtype=float))
        else:
            sig = np.where(long_entry, 1.0, np.where(short_entry, -1.0, 0.0)).astype(np.float32)

        ser = pd.Series(sig, index=base.index, name="s", dtype=np.float32)
        if base_tf in ("1min", "1m"):
            return ser.reindex(F.index).fillna(0.0).astype(np.float32)
        aligned = align_to_m1(ser.to_frame(), base_tf, m1.index)["s"]
        return aligned.astype(np.float32).reindex(F.index).fillna(0.0)
    except Exception:
        return _zero(F)


# Champion: highest EURUSD return in DVMR research grid
def dvmr_champ_1h_1d(F: pd.DataFrame) -> pd.Series:
    """1h + 1d | n=5 m=15 | thr=0.50 | sticky hard-style (best return)."""
    return compute(F, "1h", "1d", n=5, m=15, htf_thr=0.50, sticky=True)


def dvmr_30m_4h_v2(F: pd.DataFrame) -> pd.Series:
    """Runner-up: 30m+4h thr=0.50 sticky (also strong on EURUSD)."""
    return compute(F, "30min", "4h", n=5, m=20, htf_thr=0.50, sticky=True)


def dvmr_champ_1h_1d_pulse(F: pd.DataFrame) -> pd.Series:
    """Same champion params but pulse-only (entry bar only)."""
    return compute(F, "1h", "1d", n=5, m=15, htf_thr=0.50, sticky=False)


HANDLERS = {
    "dvmr_champ_1h_1d": dvmr_champ_1h_1d,
    "dvmr_30m_4h_v2": dvmr_30m_4h_v2,
    "dvmr_champ_1h_1d_pulse": dvmr_champ_1h_1d_pulse,
}
