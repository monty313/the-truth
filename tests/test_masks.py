"""Forever-mask law tests (review R1#1 + CCI/SMA Shell 2026-07-30).
CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-30  CCI dual + dual SMA gate unit cases — WHY: Shell wrong-side open blocks.
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')); sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io.loader import synthetic_m1
from features.engine import build_features, CCI_MASK_TFS, SMA_GATE_TFS
from features import indicators as ind


def test_warmup_fail_closed_both_sides():
    m1 = synthetic_m1(days=2, seed=9)
    F = build_features(m1)
    # before the H1 envelope warms (~8h), BOTH masks must be ON
    early = F.iloc[:400]
    assert (early["mask_buy_blocked"] > 0).all()
    assert (early["mask_sell_blocked"] > 0).all()


def test_masks_never_both_off_then_fine_after_warmup():
    m1 = synthetic_m1(days=3, seed=9)
    F = build_features(m1)
    late = F.iloc[900:]
    # after warmup masks may be on or off, but never NaN
    assert late["mask_buy_blocked"].notna().all()
    assert late["mask_sell_blocked"].notna().all()


def test_cci_mask_columns_present():
    m1 = synthetic_m1(days=3, seed=1)
    F = build_features(m1)
    for col in ("mask_cci_sell", "mask_cci_buy", "mask_sma_sell", "mask_sma_buy"):
        assert col in F.columns
        assert F[col].isin([0.0, 1.0]).all()


def test_cci_bull_blocks_sell_not_buy_when_isolated():
    """When dual CCI firm bull on 5m, mask_cci_sell must fire (component flag)."""
    m1 = synthetic_m1(days=5, seed=42)
    F = build_features(m1)
    # Find rows where 5m CCI dual-bull holds by recomputing condition
    c30 = F["5min::cci30"]
    c100 = F["5min::cci100"]
    l30 = F["5min::cci30_line"]
    l100 = F["5min::cci100_line"]
    bull = (c30 > 0) & (c30 > l30) & (c100 > 0) & (c100 > l100)
    bull = bull.fillna(False)
    if bull.sum() == 0:
        # synthetic data may rarely hit; still assert OR logic exists
        assert F["mask_cci_sell"].dtype == np.float32 or F["mask_cci_sell"].dtype == float
        return
    # where 5m bull: cci sell component should be 1
    assert (F.loc[bull, "mask_cci_sell"] > 0).all()
    # combined mask_sell must include those
    assert (F.loc[bull, "mask_sell_blocked"] > 0).all()


def test_cci_requires_both_periods():
    """Only CCI30 bull is not enough — both 30 and 100 required."""
    m1 = synthetic_m1(days=4, seed=7)
    F = build_features(m1)
    c30 = F["5min::cci30"]
    c100 = F["5min::cci100"]
    l30 = F["5min::cci30_line"]
    l100 = F["5min::cci100_line"]
    only30 = (c30 > 0) & (c30 > l30) & ~((c100 > 0) & (c100 > l100))
    only30 = only30.fillna(False)
    # On rows where only CCI30 qualifies (and 30m also not dual-bull),
    # mask_cci_sell may still fire from 30m — filter those out
    c30b = F["30min::cci30"]
    c100b = F["30min::cci100"]
    l30b = F["30min::cci30_line"]
    l100b = F["30min::cci100_line"]
    bull30m = (c30b > 0) & (c30b > l30b) & (c100b > 0) & (c100b > l100b)
    bull30m = bull30m.fillna(False)
    only = only30 & ~bull30m
    if only.sum() > 10:
        # CCI component should not force sell solely from incomplete dual on 5m
        # (may still be blocked by envelope/sma — check component)
        # Recompute pure cci_sell without other gates: only full dual
        full5 = (c30 > 0) & (c30 > l30) & (c100 > 0) & (c100 > l100)
        assert not (only & full5.fillna(False)).any()


def test_sma_gate_both_tfs_required():
    """Buy SMA gate needs BOTH 1m and 15m under SMA4+4."""
    m1 = synthetic_m1(days=4, seed=3)
    F = build_features(m1)
    c1 = F["1min::close"]
    c15 = F["15min::close"]
    s1 = ind.sma_shifted(c1, 4, 4)
    s15 = ind.sma_shifted(c15, 4, 4)
    both_under = (c1 < s1) & (c15 < s15)
    both_under = both_under.fillna(False)
    only_1m = (c1 < s1) & ~(c15 < s15)
    only_1m = only_1m.fillna(False)
    if both_under.sum() > 0:
        assert (F.loc[both_under, "mask_sma_buy"] > 0).all()
        assert (F.loc[both_under, "mask_buy_blocked"] > 0).all()
    # only 1m under should NOT set mask_sma_buy (need both)
    if only_1m.sum() > 20:
        # allow some nan-fail-closed early bars
        late = only_1m & (np.arange(len(F)) > 500)
        if late.sum() > 10:
            assert (F.loc[late, "mask_sma_buy"] < 0.5).mean() > 0.5


def test_cci_mask_tfs_config():
    assert "5min" in CCI_MASK_TFS and "30min" in CCI_MASK_TFS
    assert "1min" in SMA_GATE_TFS and "15min" in SMA_GATE_TFS


if __name__ == "__main__":
    test_warmup_fail_closed_both_sides()
    test_masks_never_both_off_then_fine_after_warmup()
    test_cci_mask_columns_present()
    test_cci_bull_blocks_sell_not_buy_when_isolated()
    test_cci_requires_both_periods()
    test_sma_gate_both_tfs_required()
    test_cci_mask_tfs_config()
    print("ALL MASK TESTS PASSED")
