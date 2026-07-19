"""Indicator math tests — MT5 semantics pinned. 5W+I: see test_shell.py header.
CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from features import indicators as ind

def test_shift_semantics():
    s = pd.Series([1., 2., 3., 4., 5., 6.])
    line = ind.sma_shifted(s, 2, 2)          # SMA2 shifted +2
    # at index 5: SMA2 at index 3 = (3+4)/2 = 3.5
    assert line.iloc[5] == 3.5

def test_rsi_all_gains_is_100():
    s = pd.Series(np.arange(1, 40, dtype=float))
    assert abs(ind.rsi(s, 14).iloc[-1] - 100.0) < 1e-9

def test_cci_flat_window_zero():
    o = pd.DataFrame({"high": [5.]*30, "low": [5.]*30, "close": [5.]*30})
    assert ind.cci(o, 14).iloc[-1] == 0.0

def test_bollinger_population_std():
    s = pd.Series([1., 2., 3., 4.])
    up, mid, lo = ind.bollinger(s, 4, 1.0, 0)
    assert abs(mid.iloc[-1] - 2.5) < 1e-9
    assert abs((up.iloc[-1] - mid.iloc[-1]) - s.std(ddof=0)) < 1e-9

def test_envelope_applied_to_high_low():
    o = pd.DataFrame({"high": [10., 11, 12, 13, 14, 15, 16, 17, 18, 19],
                      "low":  [ 8.,  9, 10, 11, 12, 13, 14, 15, 16, 17],
                      "close":[ 9., 10, 11, 12, 13, 14, 15, 16, 17, 18]})
    hi, lo = ind.envelope(o, 4, 4)
    # at i=9: SMA4(high) at i=5 = mean(12,13,14,15)=13.5
    assert hi.iloc[9] == 13.5 and lo.iloc[9] == 11.5
