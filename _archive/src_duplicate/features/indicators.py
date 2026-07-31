"""Indicator math — MT5-EXACT semantics (golden-testable).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty.
WHAT:  SMA/EMA/CCI/RSI/ATR/Bollinger + the MT5 'shift' rule: a +N shift
       displaces the drawn line N bars forward, so the value compared at
       bar t is the indicator computed at bar t-N  ==  series.shift(N).
WHEN:  2026-07-19 overnight build.
WHERE: consumed by features/engine.py (4-Set matrix, strategies, masks).
WHY:   The strategies live or die on these formulas matching Monty's
       MT5 charts digit-for-digit (Phase-3 golden tests; screenshots
       from Monty verify on real data later).
INTERCONNECTED WITH: configs/features.yaml, codex/regimes/*,
       tests/test_indicators.py.
NOTE:  ATR uses simple rolling mean of True Range (matches Monty's v2
       foundation script); if golden tests vs MT5 show Wilder smoothing,
       flip ATR_WILDER=True — one switch, documented here.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  RSI all-gains=100, CCI flat=0  — WHY: MT5 edge-case parity (audit R1#6/#7).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

ATR_WILDER = False  # see NOTE above


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def shifted(s: pd.Series, shift: int) -> pd.Series:
    """MT5 '+shift' semantics: the line at bar t shows the value from t-shift."""
    return s.shift(shift)


def sma_shifted(s: pd.Series, n: int, shift: int) -> pd.Series:
    return shifted(sma(s, n), shift)


def cci(o: pd.DataFrame, n: int) -> pd.Series:
    """MT5 CCI: typical price vs its SMA over 0.015 * mean absolute deviation."""
    tp = (o["high"] + o["low"] + o["close"]) / 3.0
    ma = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    out = (tp - ma) / (0.015 * mad.replace(0.0, np.nan))
    return out.mask((mad == 0) & ma.notna(), 0.0)     # MT5: flat window = 0 (R1#7)


def rsi(close: pd.Series, n: int) -> pd.Series:
    """Wilder RSI (MT5 standard)."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    out = 100 - 100 / (1 + up / dn.replace(0.0, np.nan))
    return out.mask((dn == 0) & up.notna(), 100.0)   # MT5: all-gains window = 100 (R1#6)


def atr(o: pd.DataFrame, n: int) -> pd.Series:
    pc = o["close"].shift(1)
    tr = pd.concat([(o["high"] - o["low"]), (o["high"] - pc).abs(),
                    (o["low"] - pc).abs()], axis=1).max(axis=1)
    if ATR_WILDER:
        return tr.ewm(alpha=1 / n, min_periods=n, adjust=False).mean()
    return tr.rolling(n, min_periods=n).mean()


def bollinger(s: pd.Series, n: int, dev: float, shift: int = 0):
    """BB on ANY series (price, CCI, RSI). Returns (upper, mid, lower),
    each displaced by MT5 +shift. Population std (ddof=0) = MT5 StdDev."""
    mid = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + dev * sd, mid - dev * sd
    return shifted(up, shift), shifted(mid, shift), shifted(lo, shift)


def envelope(o: pd.DataFrame, n: int = 4, shift: int = 4):
    """S3/forever-mask envelope: SMA(n) applied to High and to Low, +shift.
    Returns (env_high_line, env_low_line)."""
    return sma_shifted(o["high"], n, shift), sma_shifted(o["low"], n, shift)


