"""Directional Velocity-Momentum Ratio (DVMR) — modular feature generation.

Exact definition (signed velocity / abs momentum):
    Δp_t = p_t - p_{t-n}
    v_t  = Δp_t / n
    M_t  = p_t - p_{t-m}
    DVMR_t  = v_t / (|M_t| + ε)
    DVMR*_t = v_t / (ATR_k + ε)

Defaults: n=5, m=20 (always n < m so the indicator does not collapse to ±1/n).
Extreme DVMR / DVMR* values are clipped to ±clip.

These columns are designed to plug into an RL observation space later.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from features.indicators import atr as atr_indicator

EPS = 1e-6
DEFAULT_N = 5
DEFAULT_M = 20
DEFAULT_ATR_K = 14
DEFAULT_CLIP = 10.0

# Regime thresholds (base TF DVMR)
REGIME_STRONG = 1.2
REGIME_MILD = 0.4


def velocity(close: pd.Series, n: int) -> pd.Series:
    """v_t = (p_t - p_{t-n}) / n  — signed."""
    return (close - close.shift(n)) / float(n)


def momentum(close: pd.Series, m: int) -> pd.Series:
    """M_t = p_t - p_{t-m}."""
    return close - close.shift(m)


def dvmr_raw(
    close: pd.Series,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    eps: float = EPS,
) -> pd.Series:
    """DVMR_t = v_t / (|M_t| + ε). Signed numerator; abs in denominator."""
    if n >= m:
        raise ValueError(f"Require n < m (got n={n}, m={m}); n==m collapses DVMR to ~±1/n.")
    v = velocity(close, n)
    M = momentum(close, m)
    return v / (M.abs() + eps)


def dvmr_atr_norm(
    close: pd.Series,
    ohlc: pd.DataFrame,
    n: int = DEFAULT_N,
    atr_k: int = DEFAULT_ATR_K,
    eps: float = EPS,
) -> pd.Series:
    """DVMR*_t = v_t / (ATR_k + ε)."""
    v = velocity(close, n)
    a = atr_indicator(ohlc, atr_k)
    return v / (a + eps)


def clip_series(s: pd.Series, clip: float = DEFAULT_CLIP) -> pd.Series:
    return s.clip(lower=-clip, upper=clip)


def regime_from_dvmr(dvmr: pd.Series) -> pd.Series:
    """Base-TF regime flag from DVMR.

    2  if DVMR > 1.2
    1  if 0.4 < DVMR ≤ 1.2
    0  if -0.4 ≤ DVMR ≤ 0.4
   -1  if DVMR < -0.4
    """
    x = dvmr.astype(float)
    cond2 = x > REGIME_STRONG
    cond1 = (x > REGIME_MILD) & (x <= REGIME_STRONG)
    cond_m1 = x < -REGIME_MILD
    vals = np.where(cond2, 2, np.where(cond1, 1, np.where(cond_m1, -1, 0)))
    out = pd.Series(vals, index=x.index, dtype=np.int8)
    return out.where(x.notna(), other=0).astype(np.int8)


def compute_dvmr_features(
    ohlc: pd.DataFrame,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    atr_k: int = DEFAULT_ATR_K,
    clip: float = DEFAULT_CLIP,
    prefix: str = "",
) -> pd.DataFrame:
    """Return modular feature frame for one timeframe.

    Columns (with optional prefix, e.g. '' or 'htf_'):
      dvmr, dvmr_star, d_dvmr, atr, regime, velocity, momentum
    """
    if not {"open", "high", "low", "close"}.issubset(ohlc.columns):
        raise ValueError("ohlc needs open/high/low/close")
    close = ohlc["close"]
    raw = dvmr_raw(close, n=n, m=m)
    star = dvmr_atr_norm(close, ohlc, n=n, atr_k=atr_k)
    dvmr = clip_series(raw, clip)
    dvmr_star = clip_series(star, clip)
    a = atr_indicator(ohlc, atr_k)
    v = velocity(close, n)
    M = momentum(close, m)
    reg = regime_from_dvmr(dvmr)
    d_dvmr = dvmr.diff()

    p = prefix
    return pd.DataFrame(
        {
            f"{p}dvmr": dvmr,
            f"{p}dvmr_star": dvmr_star,
            f"{p}d_dvmr": d_dvmr,
            f"{p}atr": a,
            f"{p}regime": reg,
            f"{p}velocity": v,
            f"{p}momentum": M,
        },
        index=ohlc.index,
    )


def align_htf_to_base(
    htf_feat: pd.DataFrame,
    htf_tf: str,
    base_index: pd.DatetimeIndex,
    tf_delta: pd.Timedelta,
) -> pd.DataFrame:
    """Forward-fill HTF features onto base TF with no look-ahead.

    HTF bar labeled at its OPEN becomes usable only after it CLOSES
    (open + tf_delta). Then ffill onto base timestamps.
    """
    f = htf_feat.copy()
    f.index = f.index + tf_delta
    # Drop duplicate index after shift if any, keep last
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f.reindex(base_index, method="ffill")


def build_mtf_feature_frame(
    base_ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame,
    htf_tf: str,
    tf_delta: pd.Timedelta,
    n: int = DEFAULT_N,
    m: int = DEFAULT_M,
    atr_k: int = DEFAULT_ATR_K,
    clip: float = DEFAULT_CLIP,
    htf_cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Base OHLC + base DVMR features + aligned HTF DVMR features.

    Base columns: open, high, low, close, vol (if present), dvmr, dvmr_star, ...
    HTF columns prefixed with htf_ (subset via htf_cols if provided).
    """
    base_feat = compute_dvmr_features(base_ohlc, n=n, m=m, atr_k=atr_k, clip=clip, prefix="")
    htf_feat = compute_dvmr_features(htf_ohlc, n=n, m=m, atr_k=atr_k, clip=clip, prefix="htf_")
    if htf_cols is not None:
        keep = [c for c in htf_feat.columns if c in set(htf_cols) or c.replace("htf_", "") in set(htf_cols)]
        # if user passed bare names, map them
        if not keep:
            keep = [f"htf_{c}" if not c.startswith("htf_") else c for c in htf_cols]
            keep = [c for c in keep if c in htf_feat.columns]
        htf_feat = htf_feat[keep]
    htf_aligned = align_htf_to_base(htf_feat, htf_tf, base_ohlc.index, tf_delta)

    price_cols = [c for c in ("open", "high", "low", "close", "vol") if c in base_ohlc.columns]
    out = pd.concat([base_ohlc[price_cols], base_feat, htf_aligned], axis=1)
    return out
