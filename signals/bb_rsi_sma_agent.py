"""BB / RSI-BB / SMA multi-set signal agents for RL observation.

Sets (LTF, HTF1, HTF2):
  A: 5m  + 30m, 1h
  B: 15m + 1h, 4h
  C: 30m + 4h, 1d

HTF: Bollinger(close, 100, dev=0.5, shift=2) + SMA10 high/low
LTF: RSI(5) with Bollinger on RSI(10, dev=1, shift=5) + SMA10 high/low

Buy:  any HTF (close > BB_up & close > SMA_low)
      + LTF (RSI cross up lower RSI-BB & close > SMA_low)
Sell: opposite

Values: +1 long | -1 short | 0 flat.
Default agents use **pulse** (entry bar only) so RL sees true fires, not always-on bias.
Sticky / C-long helpers available as extra kinds.
Aligned onto M1 timeline for obs::sig_*.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from features.bb_rsi_sma_sets import SETS, build_set_frame


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _m1(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    return out


def _sticky(long_e: np.ndarray, short_e: np.ndarray) -> np.ndarray:
    n = len(long_e)
    out = np.zeros(n, dtype=np.float32)
    pos = 0
    for i in range(n):
        if long_e[i]:
            pos = 1
        elif short_e[i]:
            pos = -1
        out[i] = float(pos)
    return out


def _pulse(long_e: np.ndarray, short_e: np.ndarray) -> np.ndarray:
    out = np.zeros(len(long_e), dtype=np.float32)
    out[long_e] = 1.0
    out[short_e] = -1.0
    # both same bar -> 0
    both = long_e & short_e
    out[both] = 0.0
    return out


def compute(F: pd.DataFrame, set_name: str, *, mode: str = "sticky") -> pd.Series:
    """Build set signals on LTF, align to F's M1 index."""
    m1 = _m1(F)
    if m1 is None or len(m1) < 500:
        return _zero(F)
    if set_name not in SETS:
        return _zero(F)
    try:
        from data_io.loader import align_to_m1

        fr = build_set_frame(m1, set_name)
        le = fr["long_entry"].fillna(False).to_numpy(bool)
        se = fr["short_entry"].fillna(False).to_numpy(bool)
        if mode == "pulse":
            sig = _pulse(le, se)
        else:
            sig = _sticky(le, se)
        ser = pd.Series(sig, index=fr.index, name="s", dtype=np.float32)
        ltf = SETS[set_name]["ltf"]
        if ltf in ("1min", "1m"):
            return ser.reindex(F.index).fillna(0.0).astype(np.float32)
        aligned = align_to_m1(ser.to_frame(), ltf, m1.index)["s"]
        return aligned.astype(np.float32).reindex(F.index).fillna(0.0)
    except Exception:
        return _zero(F)


def bb_rsi_sma_A(F: pd.DataFrame) -> pd.Series:
    """Set A: 5m + 30m/1h (pulse entry)."""
    return compute(F, "A_5m_30m_1h", mode="pulse")


def bb_rsi_sma_B(F: pd.DataFrame) -> pd.Series:
    """Set B: 15m + 1h/4h (pulse entry)."""
    return compute(F, "B_15m_1h_4h", mode="pulse")


def bb_rsi_sma_C(F: pd.DataFrame) -> pd.Series:
    """Set C: 30m + 4h/1d (pulse entry)."""
    return compute(F, "C_30m_4h_1d", mode="pulse")


def bb_rsi_sma_A_sticky(F: pd.DataFrame) -> pd.Series:
    return compute(F, "A_5m_30m_1h", mode="sticky")


def bb_rsi_sma_B_sticky(F: pd.DataFrame) -> pd.Series:
    return compute(F, "B_15m_1h_4h", mode="sticky")


def bb_rsi_sma_C_sticky(F: pd.DataFrame) -> pd.Series:
    return compute(F, "C_30m_4h_1d", mode="sticky")


def bb_rsi_sma_C_long(F: pd.DataFrame) -> pd.Series:
    """Set C long-only pulse (path research: mild US30 long drift)."""
    s = compute(F, "C_30m_4h_1d", mode="pulse")
    return s.clip(lower=0.0)


HANDLERS = {
    "bb_rsi_sma_A": bb_rsi_sma_A,
    "bb_rsi_sma_B": bb_rsi_sma_B,
    "bb_rsi_sma_C": bb_rsi_sma_C,
    "bb_rsi_sma_A_sticky": bb_rsi_sma_A_sticky,
    "bb_rsi_sma_B_sticky": bb_rsi_sma_B_sticky,
    "bb_rsi_sma_C_sticky": bb_rsi_sma_C_sticky,
    "bb_rsi_sma_C_long": bb_rsi_sma_C_long,
}
