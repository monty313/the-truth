"""Mark / ENTJ multi-set opportunity scan.

CHANGE LOG:
- 2026-08-04  created — WHY: claim heuristic collapsed eyes to Official Set 2
  only (5m). Mark (ENTJ, MARK HERE clone) scans ALL 4 official sets and takes
  the best aligned opportunity. Fast logical scalping — not single-stack thrash.

MARK SETS LAW (see MARK_SETS_LAW.md — immutable):
  1: 1m  · 15m, 30m
  2: 5m  · 30m, 1h
  3: 15m · 1h, 4h
  4: 30m · 4h, 1d

Mark rule (plain):
  - Scan ALL four sets every decision (never set-2 only).
  - HTF (last two) = trend confirm / permission.
  - LTF (first) = pullbacks, continuations, adds only WITH HTF.
  - Prefer HTF-clear + LTF-aligned (continuation / release).
  - Pullback (HTF clear, LTF opposite) = wait slingshot, do not thrash reverse.
  - Multi-set agreement = higher conviction.
  - Conflict / no LTF trigger = HOLD.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    OfficialSet,
    SetConfluence,
    VelocityStrength,
)

# Velocity ranks for scoring (ENTJ: strength matters)
_VEL_RANK = {
    VelocityStrength.NONE: 0,
    VelocityStrength.WEAK: 1,
    VelocityStrength.MEDIUM: 2,
    VelocityStrength.STRONG: 3,
}

# Macro bias weighs more for direction; micro still counts for speed when aligned
_SET_WEIGHT = {
    1: 1.0,   # micro scalp
    2: 1.15,  # intraday
    3: 1.30,  # swing stack
    4: 1.50,  # macro gravity
}


@dataclass(frozen=True)
class SetOpportunity:
    set_id: int
    name: str
    entry_tf: str
    htf_dir: Direction
    ltf_dir: Direction
    velocity: VelocityStrength
    aligned: bool
    pullback: bool
    score: float


@dataclass(frozen=True)
class MarkOpportunity:
    """Result of scanning all 4 official sets like Mark would."""
    action_dir: Direction  # BULL / BEAR / NEUTRAL → map to BUY/SELL/HOLD
    bull_score: float
    bear_score: float
    n_aligned_bull: int
    n_aligned_bear: int
    n_htf_bull: int
    n_htf_bear: int
    best: Optional[SetOpportunity]
    opportunities: Tuple[SetOpportunity, ...]
    reason: str


def _vel(v: VelocityStrength) -> int:
    return int(_VEL_RANK.get(v, 0))


def score_set_opportunity(
    s: OfficialSet,
    conf: SetConfluence,
    ltf_dir: Direction,
) -> SetOpportunity:
    """Score one set: HTF bias from conf, LTF from entry TF groups."""
    htf = Direction(conf.direction)
    ltf = Direction(ltf_dir)
    aligned = htf != Direction.NEUTRAL and ltf == htf
    pullback = (
        htf != Direction.NEUTRAL
        and ltf != Direction.NEUTRAL
        and ltf != htf
    )
    w = float(_SET_WEIGHT.get(s.set_id, 1.0))
    v = _vel(conf.velocity)
    score = 0.0
    if htf == Direction.NEUTRAL:
        score = 0.0
    elif aligned:
        # Best opportunity: HTF clear + LTF trigger same side
        score = (v + 1.0) * w * 2.0
    elif ltf == Direction.NEUTRAL:
        # HTF only — soft bias, not enough alone for aggressive ENTJ scalp
        score = (v + 1.0) * w * 0.35
    else:
        # Pullback forming — Mark waits; score near 0 for entry push
        score = 0.0
    return SetOpportunity(
        set_id=s.set_id,
        name=s.name,
        entry_tf=s.entry_tf,
        htf_dir=htf,
        ltf_dir=ltf,
        velocity=conf.velocity,
        aligned=aligned,
        pullback=pullback,
        score=float(score),
    )


def scan_mark_opportunities(
    official: Mapping[int, SetConfluence],
    entry_dirs: Mapping[int, Direction],
    *,
    min_aligned: int = 1,
    # 1.2: allow one medium-aligned set (ENTJ fast scalp) without thrash flip rules
    min_score: float = 1.2,
    macro_permission: bool = True,
) -> MarkOpportunity:
    """Scan all official sets; pick direction Mark would take.

    Parameters
    ----------
    official : set_id → SetConfluence (HTF confirmation stack)
    entry_dirs : set_id → LTF/entry Direction for that set's entry_tf
    min_aligned : need at least this many aligned sets to fire (default 1)
    min_score : minimum total side score to fire (filters noise)
    macro_permission : if Set 4 HTF is clear, zero out opposite-side scores
        (pt5 Law of Dominant Trends — HTF is binary gate on side)
    """
    opps: List[SetOpportunity] = []
    for s in OFFICIAL_SETS:
        conf = official.get(s.set_id)
        if conf is None:
            continue
        ltf = Direction(entry_dirs.get(s.set_id, Direction.NEUTRAL))
        opps.append(score_set_opportunity(s, conf, ltf))

    # pt5: macro Set 4 HTF = permission / gravity when clear
    macro = official.get(4)
    macro_dir = Direction(macro.direction) if macro is not None else Direction.NEUTRAL
    gated: List[SetOpportunity] = list(opps)
    if macro_permission and macro_dir in (Direction.BULL, Direction.BEAR):
        # Drop score for sets fighting macro permission
        gated = []
        for o in opps:
            if o.htf_dir != Direction.NEUTRAL and o.htf_dir != macro_dir:
                gated.append(
                    SetOpportunity(
                        set_id=o.set_id,
                        name=o.name,
                        entry_tf=o.entry_tf,
                        htf_dir=o.htf_dir,
                        ltf_dir=o.ltf_dir,
                        velocity=o.velocity,
                        aligned=False,
                        pullback=o.pullback or True,
                        score=0.0,
                    )
                )
            elif o.aligned and o.htf_dir == macro_dir:
                gated.append(o)
            elif o.htf_dir == macro_dir:
                gated.append(o)
            else:
                gated.append(
                    SetOpportunity(
                        set_id=o.set_id,
                        name=o.name,
                        entry_tf=o.entry_tf,
                        htf_dir=o.htf_dir,
                        ltf_dir=o.ltf_dir,
                        velocity=o.velocity,
                        aligned=o.aligned,
                        pullback=o.pullback,
                        score=0.0 if o.htf_dir != macro_dir else o.score,
                    )
                )
        opps = gated

    bull_score = sum(o.score for o in opps if o.htf_dir == Direction.BULL)
    bear_score = sum(o.score for o in opps if o.htf_dir == Direction.BEAR)
    n_ab = sum(1 for o in opps if o.aligned and o.htf_dir == Direction.BULL)
    n_ar = sum(1 for o in opps if o.aligned and o.htf_dir == Direction.BEAR)
    n_hb = sum(1 for o in opps if o.htf_dir == Direction.BULL)
    n_hr = sum(1 for o in opps if o.htf_dir == Direction.BEAR)

    best: Optional[SetOpportunity] = None
    scored = [o for o in opps if o.score > 0]
    if scored:
        best = max(scored, key=lambda o: (o.score, _vel(o.velocity), o.set_id))

    # ENTJ decision: fire only when aligned opportunity presents (with tide)
    action = Direction.NEUTRAL
    reason = "no_aligned_opportunity"
    if n_ab >= min_aligned and bull_score >= min_score and bull_score > bear_score:
        if macro_permission and macro_dir == Direction.BEAR:
            reason = "macro_permission_blocks_bull"
        else:
            action = Direction.BULL
            reason = f"aligned_bull n={n_ab} score={bull_score:.2f}>{bear_score:.2f}"
    elif n_ar >= min_aligned and bear_score >= min_score and bear_score > bull_score:
        if macro_permission and macro_dir == Direction.BULL:
            reason = "macro_permission_blocks_bear"
        else:
            action = Direction.BEAR
            reason = f"aligned_bear n={n_ar} score={bear_score:.2f}>{bull_score:.2f}"
    elif n_hb > 0 and n_hr > 0 and n_ab == 0 and n_ar == 0:
        reason = "htf_conflict_or_pullback_wait"
    elif n_hb + n_hr > 0 and n_ab + n_ar == 0:
        reason = "htf_bias_no_ltf_trigger"

    return MarkOpportunity(
        action_dir=action,
        bull_score=float(bull_score),
        bear_score=float(bear_score),
        n_aligned_bull=int(n_ab),
        n_aligned_bear=int(n_ar),
        n_htf_bull=int(n_hb),
        n_htf_bear=int(n_hr),
        best=best,
        opportunities=tuple(opps),
        reason=reason,
    )


def mark_dir_to_action(d: Direction) -> int:
    """Map Direction → ACTION_HOLD/BUY/SELL ints (0/1/2)."""
    # local import avoids circular import with policy_stub in some test paths
    from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
        ACTION_BUY,
        ACTION_HOLD,
        ACTION_SELL,
    )

    if d == Direction.BULL:
        return ACTION_BUY
    if d == Direction.BEAR:
        return ACTION_SELL
    return ACTION_HOLD


def official_sets_table() -> List[dict]:
    """Human-readable set table (docs / probes) — MARK SETS LAW."""
    from lineages.adaptive_rl_brain_7_31_26.perception.sets import mark_sets_law_table

    return mark_sets_law_table()
