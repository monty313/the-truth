"""Phase 1 pins: MINDLESS 3-condition wall only."""
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
    mindless_conditions_hold,
    mindless_wall_blocks,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, TradeTag

B, R, N = Direction.BULL, Direction.BEAR, Direction.NEUTRAL


def _m(side, a=False, b=False, c=False):
    return make_mindless_inputs(side, turned=a, velocity_confirms=b, higher_weakening=c)


def test_conditions_hold_requires_all_three():
    assert mindless_conditions_hold(_m(B, True, True, True)) is True
    assert mindless_conditions_hold(_m(B, False, True, True)) is False
    assert mindless_conditions_hold(_m(B, True, False, True)) is False
    assert mindless_conditions_hold(_m(B, True, True, False)) is False
    assert mindless_conditions_hold(_m(B, False, False, False)) is False


def test_wall_not_armed_when_trade_with_higher():
    # Even with all wall flags false, trading WITH higher is not MINDLESS-by-wall
    blocked, reasons = mindless_wall_blocks(B, B, _m(B, False, False, False))
    assert blocked is False
    assert reasons == ()
    # with agreeing lower → WITH_VECTOR
    c = classify_trade(B, B, B, _m(B, False, False, False))
    assert c.tag == TradeTag.WITH_VECTOR
    assert c.mindless is False


def test_wall_blocks_when_against_and_any_condition_missing():
    # against higher (side B, hi R)
    for a, b, c in [
        (False, True, True),
        (True, False, True),
        (True, True, False),
        (False, False, False),
        (True, False, False),
    ]:
        blocked, _reasons = mindless_wall_blocks(B, R, _m(B, a, b, c))
        assert blocked is True
        cl = classify_trade(B, R, B, _m(B, a, b, c))  # lower bull = trade side
        assert cl.tag == TradeTag.MINDLESS
        assert cl.mindless is True


def test_wall_passes_when_against_and_all_three_true():
    blocked, reasons = mindless_wall_blocks(B, R, _m(B, True, True, True))
    assert blocked is False
    assert "wall_passed" in reasons
    cl = classify_trade(B, R, B, _m(B, True, True, True))
    assert cl.tag == TradeTag.QUALIFIED_MICRO
    assert cl.mindless is False


def test_each_missing_condition_named_in_reasons():
    cl_a = classify_trade(B, R, B, _m(B, False, True, True))
    assert "a_lower_vector_not_turned" in cl_a.reasons
    cl_b = classify_trade(B, R, B, _m(B, True, False, True))
    assert "b_lower_velocity_not_confirm" in cl_b.reasons
    cl_c = classify_trade(B, R, B, _m(B, True, True, False))
    assert "c_higher_not_weakening_or_pullback" in cl_c.reasons


def test_neutral_trade_side_is_mindless():
    cl = classify_trade(N, B, B, _m(N, True, True, True))
    assert cl.tag == TradeTag.MINDLESS
    assert cl.mindless is True


def test_default_mindless_inputs_fail_closed_against_higher():
    # mindless=None → all false → MINDLESS when against
    cl = classify_trade(B, R, B, mindless=None)
    assert cl.tag == TradeTag.MINDLESS


if __name__ == "__main__":
    test_conditions_hold_requires_all_three()
    test_wall_not_armed_when_trade_with_higher()
    test_wall_blocks_when_against_and_any_condition_missing()
    test_wall_passes_when_against_and_all_three_true()
    test_each_missing_condition_named_in_reasons()
    test_neutral_trade_side_is_mindless()
    test_default_mindless_inputs_fail_closed_against_higher()
    print("test_mindless_wall OK")
