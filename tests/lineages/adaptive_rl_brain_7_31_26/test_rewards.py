"""Phase 2 Slice 4: reward dials + credit formula (no training)."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.rewards import (
    DEFAULT_DIALS,
    DIAL_BOUNDS,
    MINDLESS_PENALTY,
    clip_dials,
    credit,
    inactivity_penalty,
    make_dials,
    pnl_unit,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import TradeTag


def test_dial_bounds_locked():
    assert DIAL_BOUNDS["w_with_vector"] == (0.5, 2.0)
    assert DIAL_BOUNDS["w_qualified_macro"] == (0.5, 2.0)
    assert DIAL_BOUNDS["w_qualified_micro"] == (0.15, 0.7)
    assert DIAL_BOUNDS["w_inactivity"] == (0.0, 1.0)


def test_clip_dials_enforces_bounds():
    c = clip_dials({
        "w_with_vector": 99.0,
        "w_qualified_macro": -5.0,
        "w_qualified_micro": 0.01,
        "w_inactivity": 2.0,
    })
    assert c["w_with_vector"] == 2.0
    assert c["w_qualified_macro"] == 0.5
    assert c["w_qualified_micro"] == 0.15
    assert c["w_inactivity"] == 1.0


def test_pnl_unit_clip():
    assert pnl_unit(50.0, 25.0) == 1.0
    assert pnl_unit(-50.0, 25.0) == -1.0
    assert abs(pnl_unit(5.0, 10.0) - 0.5) < 1e-9
    assert pnl_unit(1.0, 0.0) == 0.0


def test_credit_with_vector_formula():
    dials = make_dials(w_with_vector=1.0)
    # +0.5R → credit 0.5
    assert abs(credit(TradeTag.WITH_VECTOR, 5.0, 10.0, dials) - 0.5) < 1e-9
    dials2 = make_dials(w_with_vector=2.0)
    assert abs(credit(TradeTag.WITH_VECTOR, 5.0, 10.0, dials2) - 1.0) < 1e-9


def test_credit_macro_micro_use_own_weights():
    d = make_dials(w_qualified_macro=2.0, w_qualified_micro=0.2)
    assert abs(credit(TradeTag.QUALIFIED_MACRO, 10.0, 10.0, d) - 2.0) < 1e-9
    assert abs(credit(TradeTag.QUALIFIED_MICRO, 10.0, 10.0, d) - 0.2) < 1e-9


def test_mindless_fixed_penalty_ignores_lucky_pnl():
    d = make_dials(w_with_vector=2.0)
    # Even huge profit → still wall penalty
    assert credit(TradeTag.MINDLESS, 1_000_000.0, 1.0, d) == MINDLESS_PENALTY
    assert credit(TradeTag.MINDLESS, -1_000_000.0, 1.0, d) == MINDLESS_PENALTY


def test_inactivity_penalty():
    d = make_dials(w_inactivity=0.3)
    assert abs(inactivity_penalty(d) - (-0.3)) < 1e-9
    d0 = make_dials(w_inactivity=0.0)
    assert inactivity_penalty(d0) == 0.0


def test_defaults_inside_bounds():
    for k, v in DEFAULT_DIALS.items():
        lo, hi = DIAL_BOUNDS[k]
        assert lo <= v <= hi


if __name__ == "__main__":
    test_dial_bounds_locked()
    test_clip_dials_enforces_bounds()
    test_pnl_unit_clip()
    test_credit_with_vector_formula()
    test_credit_macro_micro_use_own_weights()
    test_mindless_fixed_penalty_ignores_lucky_pnl()
    test_inactivity_penalty()
    test_defaults_inside_bounds()
    print("test_rewards OK")
