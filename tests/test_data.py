"""No-look-ahead proofs — the mutation sweep (review R2#5, committed).
CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_io.loader import synthetic_m1, resample, align_to_m1
from features.engine import build_features

def test_mutation_sweep_no_lookahead():
    m1 = synthetic_m1(days=3, seed=5)
    F1 = build_features(m1)
    cut = len(m1) * 2 // 3
    m2 = m1.copy()
    m2.iloc[cut:, :4] = m2.iloc[cut:, :4] + 500.0     # poison the future
    F2 = build_features(m2)
    a, b = F1.iloc[:cut - 1], F2.iloc[:cut - 1]
    diff = (a.fillna(-9e9) != b.fillna(-9e9)).any().any()
    assert not diff, "future mutation leaked into past features"

def test_alignment_visibility():
    m1 = synthetic_m1(days=2, seed=6)
    o30 = resample(m1, "30min")
    al = align_to_m1(o30["close"], "30min", m1.index)
    # bar 00:00-00:30 closes at 00:29 close: usable on the 00:29 row
    assert al.loc[m1.index[29]] == o30["close"].iloc[0]
    assert np.isnan(al.loc[m1.index[28]]) or al.loc[m1.index[28]] != o30["close"].iloc[0]


def test_read_mt5_m1_roundtrip(tmp_path):
    """Audit re-round: read_mt5_m1 must NOT return 0 rows on real MT5 format
    (pandas index-alignment bug wiped every row -> real-data path crashed)."""
    import pandas as pd, numpy as np
    from data_io.loader import read_mt5_m1
    idx = pd.date_range("2026-05-01", periods=300, freq="1min")
    df = pd.DataFrame({
        "<DATE>": idx.strftime("%Y.%m.%d"), "<TIME>": idx.strftime("%H:%M:%S"),
        "<OPEN>": 2400.0, "<HIGH>": 2401.0, "<LOW>": 2399.0, "<CLOSE>": 2400.5,
        "<TICKVOL>": 100, "<VOL>": 0, "<SPREAD>": 12})
    p = tmp_path / "XAUUSD_M1_x.csv"
    df.to_csv(p, sep="\t", index=False)
    out = read_mt5_m1(str(p))
    assert len(out) > 0, "real MT5 CSV must parse to non-empty (index-alignment bug)"
    assert list(out.columns) == ["open", "high", "low", "close", "vol", "spread"]
