"""Three confluence groups + simple-majority Direction / Velocity.

CHANGE LOG:
- 2026-07-31  created — WHY: Phase 1 pure aggregation (SPEC_PHASE1 §2).
  Live indicator math deferred; tests + callers pass synthetic votes/flags.

Evaluated conceptually on the TWO Confirmation TFs of a set only.
Groups: CCI, RSI, Price Channel → each Bull / Bear / Neutral.
Direction = majority of non-neutral group votes.
Velocity strength (locked):
  3 agree → STRONG
  2 agree → MEDIUM
  1 agree → WEAK
  0 or conflict → NONE
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    ConfluenceGroup,
    Direction,
    GroupVote,
    SetConfluence,
    VelocityStrength,
)

# Canonical group order for fully evaluated confluence.
GROUP_ORDER: Tuple[ConfluenceGroup, ...] = (
    ConfluenceGroup.CCI,
    ConfluenceGroup.RSI,
    ConfluenceGroup.PRICE_CHANNEL,
)


def vote_from_confirmation_flags(
    group: ConfluenceGroup,
    *,
    both_above_on_both_confirmations: bool,
    both_below_on_both_confirmations: bool,
) -> GroupVote:
    """Map dual-confirmation flags → one group vote (spec Group 1–3 shape).

    Both-above on both Confirmation TFs → BULL.
    Both-below on both Confirmation TFs → BEAR.
    Otherwise (mixed / incomplete / contradictory flags) → NEUTRAL.

    Flags are precomputed by the caller (synthetic in Phase 1 tests;
    live indicators later). Contradictory True/True is treated as Neutral.
    """
    if both_above_on_both_confirmations and not both_below_on_both_confirmations:
        d = Direction.BULL
    elif both_below_on_both_confirmations and not both_above_on_both_confirmations:
        d = Direction.BEAR
    else:
        d = Direction.NEUTRAL
    return GroupVote(group=group, direction=d)


def make_group_votes(
    cci: Direction,
    rsi: Direction,
    price_channel: Direction,
) -> Tuple[GroupVote, GroupVote, GroupVote]:
    """Convenience: build the three GroupVotes from synthetic Directions."""
    return (
        GroupVote(ConfluenceGroup.CCI, Direction(cci)),
        GroupVote(ConfluenceGroup.RSI, Direction(rsi)),
        GroupVote(ConfluenceGroup.PRICE_CHANNEL, Direction(price_channel)),
    )


def _as_directions(votes: Sequence[GroupVote]) -> List[Direction]:
    return [v.direction for v in votes]


def majority_direction(votes: Sequence[GroupVote] | Sequence[Direction]) -> Direction:
    """Simple majority among non-neutral votes; conflict or empty → NEUTRAL.

    - bull > bear → BULL
    - bear > bull → BEAR
    - Exactly one non-neutral → that side (majority of the non-neutral set)
    - Equal bull/bear non-zero, or no non-neutral → NEUTRAL (conflict / flat)
    """
    dirs = _normalize(votes)
    n_bull = sum(1 for d in dirs if d == Direction.BULL)
    n_bear = sum(1 for d in dirs if d == Direction.BEAR)
    if n_bull == 0 and n_bear == 0:
        return Direction.NEUTRAL
    if n_bull > n_bear:
        return Direction.BULL
    if n_bear > n_bull:
        return Direction.BEAR
    return Direction.NEUTRAL  # conflict (e.g. 1–1)


def velocity_strength(
    votes: Sequence[GroupVote] | Sequence[Direction],
    direction: Direction | None = None,
) -> VelocityStrength:
    """Velocity strength from how many groups agree with majority Direction.

    Locked mapping:
      3 agree → STRONG
      2 agree → MEDIUM
      1 agree → WEAK
      0 or conflict (no clear direction) → NONE

    `direction` defaults to majority_direction(votes). Independent of which
    axis produced the votes — callers may pass direction votes or any
    Direction sequence for strength counting.
    """
    dirs = _normalize(votes)
    d = majority_direction(dirs) if direction is None else Direction(direction)
    if d == Direction.NEUTRAL:
        return VelocityStrength.NONE
    n_agree = sum(1 for x in dirs if x == d)
    if n_agree >= 3:
        return VelocityStrength.STRONG
    if n_agree == 2:
        return VelocityStrength.MEDIUM
    if n_agree == 1:
        return VelocityStrength.WEAK
    return VelocityStrength.NONE


def aggregate_confluence(
    set_key: str,
    votes: Sequence[GroupVote],
) -> SetConfluence:
    """Full SetConfluence from three (or any) group votes."""
    vote_t = tuple(votes)
    dirs = _as_directions(vote_t)
    n_bull = sum(1 for d in dirs if d == Direction.BULL)
    n_bear = sum(1 for d in dirs if d == Direction.BEAR)
    n_neutral = sum(1 for d in dirs if d == Direction.NEUTRAL)
    direction = majority_direction(vote_t)
    vel = velocity_strength(vote_t, direction=direction)
    return SetConfluence(
        set_key=set_key,
        direction=direction,
        velocity=vel,
        votes=vote_t,
        n_bull=n_bull,
        n_bear=n_bear,
        n_neutral=n_neutral,
    )


def confluence_from_group_directions(
    set_key: str,
    cci: Direction,
    rsi: Direction,
    price_channel: Direction,
) -> SetConfluence:
    """Synthetic entry point used by unit tests (no indicators)."""
    return aggregate_confluence(set_key, make_group_votes(cci, rsi, price_channel))


def confluence_from_confirmation_flags(
    set_key: str,
    *,
    cci_both_above: bool,
    cci_both_below: bool,
    rsi_both_above: bool,
    rsi_both_below: bool,
    channel_both_above: bool,
    channel_both_below: bool,
) -> SetConfluence:
    """Build confluence from per-group dual-confirmation flags (Confirmation TFs only)."""
    votes = (
        vote_from_confirmation_flags(
            ConfluenceGroup.CCI,
            both_above_on_both_confirmations=cci_both_above,
            both_below_on_both_confirmations=cci_both_below,
        ),
        vote_from_confirmation_flags(
            ConfluenceGroup.RSI,
            both_above_on_both_confirmations=rsi_both_above,
            both_below_on_both_confirmations=rsi_both_below,
        ),
        vote_from_confirmation_flags(
            ConfluenceGroup.PRICE_CHANNEL,
            both_above_on_both_confirmations=channel_both_above,
            both_below_on_both_confirmations=channel_both_below,
        ),
    )
    return aggregate_confluence(set_key, votes)


def _normalize(
    votes: Sequence[GroupVote] | Sequence[Direction],
) -> List[Direction]:
    out: List[Direction] = []
    for v in votes:
        if isinstance(v, GroupVote):
            out.append(v.direction)
        else:
            out.append(Direction(v))
    return out
