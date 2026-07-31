"""Four trade tags + MINDLESS wall (Phase 1).

CHANGE LOG:
- 2026-07-31  created — WHY: SPEC_PHASE1 §4 classification + 3-condition wall.
  Pure functions; synthetic MindlessInputs / Directions only.

Tag priority:
  1. MINDLESS          — against higher-set Vector and wall fails
  2. WITH_VECTOR       — Major + lower agree; trade follows that side
  3. QUALIFIED_MACRO   — pullback: trade follows higher while lower opposes
  4. QUALIFIED_MICRO   — trade follows lower against higher (or wall-passed counter)

MINDLESS applies only when trade_side is against a clear higher Direction.
Then all three must hold or the wall fires:
  (a) lower Vector M turned trade direction
  (b) lower velocity confirms the turn
  (c) higher sets weakening or pullback (not fresh acceleration)
"""
from __future__ import annotations

from typing import List, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.structure import (
    is_clear,
    is_pullback,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Classification,
    Direction,
    MindlessInputs,
    TradeTag,
)


def mindless_conditions_hold(m: MindlessInputs) -> bool:
    """True iff (a)∧(b)∧(c) all hold."""
    return bool(
        m.lower_vector_turned
        and m.lower_velocity_confirms
        and m.higher_weakening_or_pullback
    )


def is_against_higher(
    trade_side: Direction,
    higher_direction: Direction,
) -> bool:
    """Trade is against higher-set Vector when both clear and opposite."""
    side = Direction(trade_side)
    hi = Direction(higher_direction)
    return is_clear(side) and is_clear(hi) and side != hi


def mindless_wall_blocks(
    trade_side: Direction,
    higher_direction: Direction,
    m: MindlessInputs,
) -> Tuple[bool, Tuple[str, ...]]:
    """Return (blocked, reasons).

    Wall only arms when trading against a clear higher Vector.
    Blocked if any of (a)(b)(c) is missing.
    """
    if Direction(trade_side) == Direction.NEUTRAL:
        return True, ("trade_side_neutral",)
    if not is_against_higher(trade_side, higher_direction):
        return False, ()
    missing: List[str] = []
    if not m.lower_vector_turned:
        missing.append("a_lower_vector_not_turned")
    if not m.lower_velocity_confirms:
        missing.append("b_lower_velocity_not_confirm")
    if not m.higher_weakening_or_pullback:
        missing.append("c_higher_not_weakening_or_pullback")
    if missing:
        return True, tuple(missing)
    return False, ("wall_passed",)


def classify_trade(
    trade_side: Direction,
    higher_direction: Direction,
    lower_direction: Direction,
    mindless: MindlessInputs | None = None,
    *,
    pullback: bool | None = None,
) -> Classification:
    """Assign exactly one primary TradeTag.

    Order: MINDLESS wall → WITH_VECTOR → QUALIFIED_MACRO → QUALIFIED_MICRO.
    """
    side = Direction(trade_side)
    hi = Direction(higher_direction)
    lo = Direction(lower_direction)

    # Default mindless inputs: fail closed if omitted while against higher.
    m = mindless if mindless is not None else MindlessInputs(
        trade_side=side,
        lower_vector_turned=False,
        lower_velocity_confirms=False,
        higher_weakening_or_pullback=False,
    )
    # Keep trade_side on the inputs consistent with the classify call.
    m = MindlessInputs(
        trade_side=side,
        lower_vector_turned=m.lower_vector_turned,
        lower_velocity_confirms=m.lower_velocity_confirms,
        higher_weakening_or_pullback=m.higher_weakening_or_pullback,
    )

    # ----- 1) MINDLESS wall first -----
    blocked, wall_reasons = mindless_wall_blocks(side, hi, m)
    if blocked:
        return Classification(
            tag=TradeTag.MINDLESS,
            mindless=True,
            reasons=wall_reasons,
        )

    # ----- Structure for remaining tags -----
    pb = is_pullback(hi, lo) if pullback is None else bool(pullback)
    against = is_against_higher(side, hi)

    # ----- 2) WITH_VECTOR: major + lower agree; trade follows -----
    if is_clear(hi) and is_clear(lo) and hi == lo and side == hi:
        return Classification(
            tag=TradeTag.WITH_VECTOR,
            mindless=False,
            reasons=("major_lower_agree",),
        )

    # ----- 3) QUALIFIED_MACRO: pullback; trade follows higher -----
    if pb and side == hi:
        return Classification(
            tag=TradeTag.QUALIFIED_MACRO,
            mindless=False,
            reasons=("pullback_follow_higher",),
        )

    # ----- 4) QUALIFIED_MICRO: follow lower against higher, or wall-passed counter -----
    if is_clear(lo) and side == lo and (not is_clear(hi) or lo != hi):
        return Classification(
            tag=TradeTag.QUALIFIED_MICRO,
            mindless=False,
            reasons=("follow_lower",) + (("wall_passed",) if against else ()),
        )

    if against:
        # Against higher but wall passed (otherwise we already returned MINDLESS).
        return Classification(
            tag=TradeTag.QUALIFIED_MICRO,
            mindless=False,
            reasons=("wall_passed_against_higher",),
        )

    # Trade with clear higher while lower is flat / non-opposing (not pullback).
    # Soft WITH_VECTOR residual (approved).
    if is_clear(hi) and side == hi:
        return Classification(
            tag=TradeTag.WITH_VECTOR,
            mindless=False,
            reasons=("follow_higher_lower_not_opposing",),
        )

    # Fail closed: no clear structure → still one primary tag.
    return Classification(
        tag=TradeTag.MINDLESS,
        mindless=True,
        reasons=("unclassified_fail_closed",),
    )


def make_mindless_inputs(
    trade_side: Direction,
    *,
    turned: bool = False,
    velocity_confirms: bool = False,
    higher_weakening: bool = False,
) -> MindlessInputs:
    """Test/helper factory for the three wall booleans."""
    return MindlessInputs(
        trade_side=Direction(trade_side),
        lower_vector_turned=turned,
        lower_velocity_confirms=velocity_confirms,
        higher_weakening_or_pullback=higher_weakening,
    )
