"""Live end-to-end: OHLC → indicators → confluence → structure → classify.

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 2 — WHY: wire live Confirmation confluence into
  structure flags + four trade tags / MINDLESS wall. Parallel lineage only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import pandas as pd

from lineages.adaptive_rl_brain_7_31_26.perception.classify import (
    classify_trade,
    make_mindless_inputs,
)
from lineages.adaptive_rl_brain_7_31_26.perception.confluence import (
    majority_direction,
)
from lineages.adaptive_rl_brain_7_31_26.perception.live_indicators import (
    GROUP_KEYS,
    confluence_from_confirmation_ohlc,
    group_flags_on_tf,
    indicator_frame,
    snapshot_at,
)
from lineages.adaptive_rl_brain_7_31_26.perception.structure import (
    structure_flags,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Classification,
    Direction,
    MindlessInputs,
    SetConfluence,
    StructureFlags,
    VelocityStrength,
)


@dataclass(frozen=True)
class LiveAssessment:
    """One bar assessment for a Major (higher) vs lower/entry stack."""
    higher: SetConfluence
    lower_direction: Direction
    structure: StructureFlags
    classification: Classification
    mindless: MindlessInputs
    trade_side: Direction


def direction_from_single_tf_ohlc(
    ohlc: pd.DataFrame,
    *,
    bar: int = -1,
    tf: str = "entry",
) -> Direction:
    """Entry / lower TF direction from group votes on that TF alone.

    Uses the same three groups as confluence; majority of non-neutral group
    directions on this single TF (not dual-confirm).
    """
    frame = indicator_frame(ohlc)
    snap = snapshot_at(frame, tf, bar)
    flags = group_flags_on_tf(snap)
    dirs: list[Direction] = []
    for k in GROUP_KEYS:
        above, below = flags[k]
        if above and not below:
            dirs.append(Direction.BULL)
        elif below and not above:
            dirs.append(Direction.BEAR)
        else:
            dirs.append(Direction.NEUTRAL)
    return majority_direction(dirs)


def velocity_from_single_tf_ohlc(
    ohlc: pd.DataFrame,
    *,
    bar: int = -1,
    tf: str = "entry",
) -> VelocityStrength:
    """Velocity strength of lower TF from how many groups agree with its majority."""
    from lineages.adaptive_rl_brain_7_31_26.perception.confluence import velocity_strength

    frame = indicator_frame(ohlc)
    snap = snapshot_at(frame, tf, bar)
    flags = group_flags_on_tf(snap)
    dirs: list[Direction] = []
    for k in GROUP_KEYS:
        above, below = flags[k]
        if above and not below:
            dirs.append(Direction.BULL)
        elif below and not above:
            dirs.append(Direction.BEAR)
        else:
            dirs.append(Direction.NEUTRAL)
    return velocity_strength(dirs)


def default_mindless_from_structure(
    trade_side: Direction,
    *,
    lower_direction: Direction,
    lower_velocity: VelocityStrength,
    structure: StructureFlags,
    higher_velocity: VelocityStrength,
) -> MindlessInputs:
    """Heuristic Phase-2 mindless booleans from live structure (no Vector M series yet).

    (a) lower_vector_turned: lower clear and equals trade_side
    (b) lower_velocity_confirms: lower velocity is MEDIUM or STRONG
    (c) higher_weakening_or_pullback: structure.pullback OR higher velocity WEAK/NONE
    """
    side = Direction(trade_side)
    lo = Direction(lower_direction)
    turned = lo == side and lo != Direction.NEUTRAL
    vel_ok = lower_velocity in (VelocityStrength.MEDIUM, VelocityStrength.STRONG)
    higher_soft = (
        structure.pullback
        or higher_velocity in (VelocityStrength.WEAK, VelocityStrength.NONE)
    )
    return MindlessInputs(
        trade_side=side,
        lower_vector_turned=turned,
        lower_velocity_confirms=vel_ok,
        higher_weakening_or_pullback=higher_soft,
    )


def assess_trade(
    trade_side: Direction,
    ohlc_conf_a: pd.DataFrame,
    ohlc_conf_b: pd.DataFrame,
    ohlc_entry: pd.DataFrame,
    *,
    set_key: str = "official:live",
    bar_a: int = -1,
    bar_b: int = -1,
    bar_entry: int = -1,
    mindless: Optional[MindlessInputs] = None,
    pullback: Optional[bool] = None,
) -> LiveAssessment:
    """OHLC (Confirmation pair + Entry) → structure → Classification.

    Confirmation TFs drive higher-set confluence. Entry OHLC drives lower
    direction only (never higher votes).
    """
    side = Direction(trade_side)
    higher = confluence_from_confirmation_ohlc(
        set_key,
        ohlc_conf_a,
        ohlc_conf_b,
        bar_a=bar_a,
        bar_b=bar_b,
        entry_ohlc=ohlc_entry,  # ignored for votes
    )
    lower_dir = direction_from_single_tf_ohlc(ohlc_entry, bar=bar_entry)
    lower_vel = velocity_from_single_tf_ohlc(ohlc_entry, bar=bar_entry)
    struct = structure_flags(
        higher_direction=higher.direction,
        lower_direction=lower_dir,
    )
    if mindless is None:
        mindless = default_mindless_from_structure(
            side,
            lower_direction=lower_dir,
            lower_velocity=lower_vel,
            structure=struct,
            higher_velocity=higher.velocity,
        )
    cl = classify_trade(
        side,
        higher.direction,
        lower_dir,
        mindless,
        pullback=pullback if pullback is not None else struct.pullback,
    )
    return LiveAssessment(
        higher=higher,
        lower_direction=lower_dir,
        structure=struct,
        classification=cl,
        mindless=mindless,
        trade_side=side,
    )


def assess_from_directions(
    trade_side: Direction,
    higher_direction: Direction,
    lower_direction: Direction,
    mindless: Optional[MindlessInputs] = None,
    *,
    higher_velocity: VelocityStrength = VelocityStrength.NONE,
    lower_velocity: VelocityStrength = VelocityStrength.NONE,
) -> LiveAssessment:
    """Direction-only path (unit tests without full OHLC)."""
    side = Direction(trade_side)
    hi = Direction(higher_direction)
    lo = Direction(lower_direction)
    struct = structure_flags(higher_direction=hi, lower_direction=lo)
    if mindless is None:
        mindless = default_mindless_from_structure(
            side,
            lower_direction=lo,
            lower_velocity=lower_velocity,
            structure=struct,
            higher_velocity=higher_velocity,
        )
    higher = SetConfluence(
        set_key="synthetic",
        direction=hi,
        velocity=higher_velocity,
        votes=(),
        n_bull=1 if hi == Direction.BULL else 0,
        n_bear=1 if hi == Direction.BEAR else 0,
        n_neutral=1 if hi == Direction.NEUTRAL else 0,
    )
    cl = classify_trade(side, hi, lo, mindless, pullback=struct.pullback)
    return LiveAssessment(
        higher=higher,
        lower_direction=lo,
        structure=struct,
        classification=cl,
        mindless=mindless,
        trade_side=side,
    )
