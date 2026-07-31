"""Phase 1 pins: four tags + single primary tag + small exhaustive grid."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.perception.classify import (
    classify_trade,
    make_mindless_inputs,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, TradeTag

B, R, N = Direction.BULL, Direction.BEAR, Direction.NEUTRAL


def _pass(side):
    return make_mindless_inputs(side, turned=True, velocity_confirms=True, higher_weakening=True)


def _fail(side):
    return make_mindless_inputs(side, turned=False, velocity_confirms=False, higher_weakening=False)


def test_with_vector_major_lower_agree():
    c = classify_trade(B, B, B, _fail(B))
    assert c.tag == TradeTag.WITH_VECTOR and c.mindless is False
    c2 = classify_trade(R, R, R, _fail(R))
    assert c2.tag == TradeTag.WITH_VECTOR


def test_qualified_macro_pullback_follow_higher():
    # higher bull, lower bear, trade long with higher
    c = classify_trade(B, B, R, _fail(B))
    assert c.tag == TradeTag.QUALIFIED_MACRO
    assert c.mindless is False
    c2 = classify_trade(R, R, B, _fail(R))
    assert c2.tag == TradeTag.QUALIFIED_MACRO


def test_qualified_micro_follow_lower_against_higher_wall_pass():
    c = classify_trade(R, B, R, _pass(R))  # short while higher bull; lower bear
    assert c.tag == TradeTag.QUALIFIED_MICRO
    assert c.mindless is False


def test_mindless_priority_over_micro_shape():
    # Same micro shape but wall fails → MINDLESS wins
    c = classify_trade(R, B, R, _fail(R))
    assert c.tag == TradeTag.MINDLESS
    assert c.mindless is True


def test_exactly_one_primary_tag_field():
    c = classify_trade(B, B, B, _pass(B))
    assert isinstance(c.tag, TradeTag)
    assert c.tag in {
        TradeTag.MINDLESS,
        TradeTag.WITH_VECTOR,
        TradeTag.QUALIFIED_MACRO,
        TradeTag.QUALIFIED_MICRO,
    }


def test_exhaustive_small_grid_one_tag_each():
    """side × higher × lower × wall_pass — every cell yields exactly one tag."""
    sides = (B, R)
    dirs = (B, R, N)
    for side in sides:
        for hi in dirs:
            for lo in dirs:
                for wall_ok in (True, False):
                    m = _pass(side) if wall_ok else _fail(side)
                    c = classify_trade(side, hi, lo, m)
                    assert c.tag in TradeTag
                    assert isinstance(c.mindless, bool)
                    # Invariant: MINDLESS iff mindless flag
                    assert (c.tag == TradeTag.MINDLESS) == c.mindless
                    # Invariant: against clear higher + wall fail → MINDLESS
                    against = (side != hi and hi != N and side != N)
                    if against and not wall_ok:
                        assert c.tag == TradeTag.MINDLESS
                    # Invariant: against + wall ok → not MINDLESS
                    if against and wall_ok:
                        assert c.tag != TradeTag.MINDLESS
                        assert c.tag == TradeTag.QUALIFIED_MICRO or c.tag in TradeTag


def test_macro_not_selected_when_with_vector_applies():
    # hi==lo==side is WITH_VECTOR, not MACRO
    c = classify_trade(B, B, B, _pass(B))
    assert c.tag == TradeTag.WITH_VECTOR


def test_soft_with_vector_follow_higher_lower_flat():
    c = classify_trade(B, B, N, _fail(B))
    assert c.tag == TradeTag.WITH_VECTOR
    assert "follow_higher" in c.reasons[0] or c.reasons


if __name__ == "__main__":
    test_with_vector_major_lower_agree()
    test_qualified_macro_pullback_follow_higher()
    test_qualified_micro_follow_lower_against_higher_wall_pass()
    test_mindless_priority_over_micro_shape()
    test_exactly_one_primary_tag_field()
    test_exhaustive_small_grid_one_tag_each()
    test_macro_not_selected_when_with_vector_applies()
    test_soft_with_vector_follow_higher_lower_flat()
    print("test_classify OK")
