"""No-look-ahead proofs — the mutation sweep (review R2#5, committed)."""
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
