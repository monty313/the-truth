"""Phase 1 pins: Pullback + Scale-Conflict."""
from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.perception.structure import (
    is_pullback,
    is_scale_conflict,
    opposite,
    structure_flags,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction

B, R, N = Direction.BULL, Direction.BEAR, Direction.NEUTRAL


# ---- Pullback ----

def test_pullback_true_higher_bull_lower_bear():
    assert is_pullback(B, R) is True


def test_pullback_true_higher_bear_lower_bull():
    assert is_pullback(R, B) is True


def test_pullback_false_full_continuation_bull():
    assert is_pullback(B, B) is False


def test_pullback_false_full_continuation_bear():
    assert is_pullback(R, R) is False


def test_pullback_false_higher_flat():
    assert is_pullback(N, R) is False
    assert is_pullback(N, B) is False
    assert is_pullback(N, N) is False


def test_pullback_false_lower_flat_while_higher_clear():
    # Higher clear trend but lower not opposing (flat) → not pullback
    assert is_pullback(B, N) is False
    assert is_pullback(R, N) is False


def test_pullback_false_when_higher_mixed_encoded_as_neutral():
    # Mixed higher TFs collapse to NEUTRAL at confluence → no pullback
    assert is_pullback(N, B) is False


# ---- Scale Conflict ----

def test_scale_conflict_true_opposite_clear():
    assert is_scale_conflict(B, R) is True
    assert is_scale_conflict(R, B) is True


def test_scale_conflict_false_if_major_flat():
    assert is_scale_conflict(N, B) is False
    assert is_scale_conflict(N, R) is False


def test_scale_conflict_false_if_minor_flat():
    assert is_scale_conflict(B, N) is False
    assert is_scale_conflict(R, N) is False


def test_scale_conflict_false_both_flat():
    assert is_scale_conflict(N, N) is False


def test_scale_conflict_false_when_both_agree():
    assert is_scale_conflict(B, B) is False
    assert is_scale_conflict(R, R) is False


# ---- Combined flags + helpers ----

def test_opposite_helper():
    assert opposite(B) == R
    assert opposite(R) == B
    assert opposite(N) == N


def test_structure_flags_pullback_and_conflict_together():
    # Higher bull, lower bear → pullback True and scale_conflict True (default pair)
    f = structure_flags(higher_direction=B, lower_direction=R)
    assert f.pullback is True
    assert f.scale_conflict is True


def test_structure_flags_continuation_no_conflict():
    f = structure_flags(higher_direction=B, lower_direction=B)
    assert f.pullback is False
    assert f.scale_conflict is False


def test_structure_flags_explicit_major_minor_override():
    # Pullback on entry stack; scale conflict vs a different minor
    f = structure_flags(
        higher_direction=B,
        lower_direction=B,       # continuation → no pullback
        major_direction=B,
        minor_direction=R,       # explicit opposite sub → conflict
    )
    assert f.pullback is False
    assert f.scale_conflict is True


if __name__ == "__main__":
    test_pullback_true_higher_bull_lower_bear()
    test_pullback_true_higher_bear_lower_bull()
    test_pullback_false_full_continuation_bull()
    test_pullback_false_full_continuation_bear()
    test_pullback_false_higher_flat()
    test_pullback_false_lower_flat_while_higher_clear()
    test_pullback_false_when_higher_mixed_encoded_as_neutral()
    test_scale_conflict_true_opposite_clear()
    test_scale_conflict_false_if_major_flat()
    test_scale_conflict_false_if_minor_flat()
    test_scale_conflict_false_both_flat()
    test_scale_conflict_false_when_both_agree()
    test_opposite_helper()
    test_structure_flags_pullback_and_conflict_together()
    test_structure_flags_continuation_no_conflict()
    test_structure_flags_explicit_major_minor_override()
    print("test_structure OK")
