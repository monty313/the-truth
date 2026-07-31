"""Pullback + Scale-Conflict structure rulers.

CHANGE LOG:
- 2026-07-31  created — WHY: Phase 1 structure flags (SPEC_PHASE1 §3).
  Inputs are synthetic clear Directions only (confluence outputs or hand-built).

Pullback (within a Major Set stack):
  - Higher / Confirmation side shows a clear Direction (non-neutral)
  - Lower / Entry (or relevant Sub-Set) shows the opposite clear Direction

Scale Conflict (across scales):
  - A Major Set and a Sub-Set (or smaller set) have opposite clear Directions
  - Both sides must be non-neutral; flat on either side → no conflict
"""
from __future__ import annotations

from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    StructureFlags,
)


def is_clear(direction: Direction) -> bool:
    """True when direction is a non-neutral vote (Bull or Bear)."""
    d = Direction(direction)
    return d != Direction.NEUTRAL


def opposite(direction: Direction) -> Direction:
    """Flip BULL↔BEAR; NEUTRAL stays NEUTRAL."""
    d = Direction(direction)
    if d == Direction.BULL:
        return Direction.BEAR
    if d == Direction.BEAR:
        return Direction.BULL
    return Direction.NEUTRAL


def is_pullback(
    higher_direction: Direction,
    lower_direction: Direction,
) -> bool:
    """Pullback when higher TFs share a clear trend and lower opposes it.

    `higher_direction` = aggregated Confirmation-stack Direction of a Major Set
    (both higher TFs already collapsed to one clear vote via confluence).
    `lower_direction`  = Entry TF Direction, or a relevant Sub-Set Direction.

    True only if:
      - higher is clear (not NEUTRAL)
      - lower is clear (not NEUTRAL)
      - lower == opposite(higher)

    False on full continuation (same side), higher flat/mixed, or lower flat.
    """
    hi = Direction(higher_direction)
    lo = Direction(lower_direction)
    if not is_clear(hi) or not is_clear(lo):
        return False
    return lo == opposite(hi)


def is_scale_conflict(
    major_direction: Direction,
    minor_direction: Direction,
) -> bool:
    """Scale conflict when Major vs Sub/smaller set have opposite clear Directions.

    Both sides must be non-neutral. Flat on either side → False.
    Same clear side → False (aligned, not conflict).
    """
    maj = Direction(major_direction)
    minor = Direction(minor_direction)
    if not is_clear(maj) or not is_clear(minor):
        return False
    return maj != minor  # both clear and unequal ⇒ opposite (only ±1)


def structure_flags(
    *,
    higher_direction: Direction,
    lower_direction: Direction,
    major_direction: Direction | None = None,
    minor_direction: Direction | None = None,
) -> StructureFlags:
    """Build StructureFlags from synthetic directions.

    Pullback uses higher vs lower (Major Confirmation vs Entry/Sub).
    Scale-Conflict uses major vs minor; if omitted, defaults to the same
    higher/lower pair so a single call can score both on one stack.
    """
    maj = higher_direction if major_direction is None else major_direction
    minor = lower_direction if minor_direction is None else minor_direction
    return StructureFlags(
        pullback=is_pullback(higher_direction, lower_direction),
        scale_conflict=is_scale_conflict(maj, minor),
    )
