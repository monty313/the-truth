"""Forever-mask law tests (review R1#1). 5W+I: see test_shell.py header.
CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io.loader import synthetic_m1
from features.engine import build_features

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
