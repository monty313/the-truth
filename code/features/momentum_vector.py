"""Momentum Vector score M for multi-TF timing.

M = Momentum_Size × Velocity_Factor × Strength_Factor × Direction

  Direction       = sign(CCI)
  Momentum_Size   = |CCI|
  Velocity        = CCI − SMA(CCI, 4)
  Velocity_Factor = tanh(Velocity / 40)
  Strength_Factor = clip(1 − |RSI_BB_pos − 0.5| × 1.4, 0.25, 1.4)
  RSI_BB_pos      = (RSI − BB_lower) / (BB_upper − BB_lower)   # 0=lower, 1=upper

Defaults: CCI(20), SMA_CCI(4), RSI(14), BB on RSI (10, 2σ).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.indicators import bollinger, rsi


def cci_fast(ohlc: pd.DataFrame, n: int = 20) -> pd.Series:
    """Vectorized CCI (MT5-style 0.015 * MAD). Faster than rolling.apply MAD."""
    tp = (ohlc["high"] + ohlc["low"] + ohlc["close"]) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    arr = tp.to_numpy(dtype=float)
    mad = np.full(len(arr), np.nan)
    if len(arr) >= n:
        # sliding windows: shape (N-n+1, n)
        from numpy.lib.stride_tricks import sliding_window_view

        win = sliding_window_view(arr, n)
        # ignore windows with nan
        means = np.nanmean(win, axis=1)
        mad[n - 1 :] = np.nanmean(np.abs(win - means[:, None]), axis=1)
    mad_s = pd.Series(mad, index=ohlc.index)
    denom = (0.015 * mad_s).replace(0.0, np.nan)
    out = (tp - ma) / denom
    out = out.mask((mad_s == 0) & ma.notna(), 0.0)
    return out


def momentum_vector_frame(
    ohlc: pd.DataFrame,
    *,
    cci_len: int = 20,
    cci_sma: int = 4,
    rsi_len: int = 14,
    bb_len: int = 10,
    bb_dev: float = 2.0,
    vel_scale: float = 40.0,
) -> pd.DataFrame:
    """Compute M and components on one timeframe OHLC frame."""
    cci = cci_fast(ohlc, cci_len)
    cci_ma = cci.rolling(cci_sma, min_periods=cci_sma).mean()
    velocity = cci - cci_ma
    velocity_factor = np.tanh(velocity / vel_scale)

    r = rsi(ohlc["close"], rsi_len)
    bb_up, bb_mid, bb_lo = bollinger(r, bb_len, bb_dev)
    band = (bb_up - bb_lo).replace(0.0, np.nan)
    rsi_bb_pos = ((r - bb_lo) / band).clip(0.0, 1.0)

    strength = (1.0 - (rsi_bb_pos - 0.5).abs() * 1.4).clip(0.25, 1.4)
    direction = np.sign(cci).replace(0.0, 0.0)
    momentum_size = cci.abs()

    # M = |CCI| * tanh(v/40) * strength * sign(CCI)  ==  CCI * tanh * strength
    M = momentum_size * velocity_factor * strength * direction

    return pd.DataFrame(
        {
            "cci": cci,
            "cci_sma4": cci_ma,
            "velocity": velocity,
            "velocity_factor": velocity_factor,
            "rsi": r,
            "rsi_bb_pos": rsi_bb_pos,
            "strength_factor": strength,
            "direction": direction,
            "momentum_size": momentum_size,
            "M": M,
        },
        index=ohlc.index,
    )


def align_htf_to_base(
    htf: pd.DataFrame,
    htf_tf: str,
    base_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Forward-fill HTF features onto base index after HTF bar close (no look-ahead)."""
    from data_io.loader import TF_DELTA

    f = htf.copy()
    f.index = f.index + TF_DELTA[htf_tf]
    f = f[~f.index.duplicated(keep="last")].sort_index()
    return f.reindex(base_index, method="ffill")


def build_mtf_momentum(
    base_ohlc: pd.DataFrame,
    htf_ohlc: pd.DataFrame,
    htf_tf: str,
    **kw,
) -> pd.DataFrame:
    """Base price + base M components + aligned HTF M/direction."""
    base_f = momentum_vector_frame(base_ohlc, **kw)
    htf_f = momentum_vector_frame(htf_ohlc, **kw)
    htf_cols = htf_f[["M", "direction", "cci", "strength_factor"]].add_prefix("htf_")
    htf_al = align_htf_to_base(htf_cols, htf_tf, base_ohlc.index)

    price = base_ohlc[["open", "high", "low", "close"]].copy()
    if "vol" in base_ohlc.columns:
        price["vol"] = base_ohlc["vol"]
    out = pd.concat([price, base_f, htf_al], axis=1)
    return out
