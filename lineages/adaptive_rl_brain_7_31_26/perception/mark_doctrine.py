"""Mark / ENTJ five-law doctrine → discrete action teacher.

CHANGE LOG:
- 2026-08-04  soft scalp min_score 2.0→1.2 + trend release quality gate —
  WHY: skeptic — hardcoded 2.0 ignored multi-set opportunity dial; BULL/BEAR
  fired any aligned LTF without velocity quality → thrash or missed soft banks.
- 2026-08-04  created — WHY: user sleep mission — clone must think like Mark:
  FORCE(HTF) → REGIME → VELOCITY(LTF) → ENTRY if side matches + risk allows.
  Encodes Laws 1–5 from MARK_DOCTRINE_FIVE_LAWS.md. Parallel lineage only.

Laws (short):
  1 Dominant trends: HTF permission; LTF timing only; never side against HTF.
  2 Acceleration: breath (fast against force) vs launch (fast+slow with force).
  3 Regime: bull/bear/chop/flat → playbook rewrite; unknown = no trade.
  4 Capital: shell owns hard risk; doctrine returns m_conf / m_regime hints.
  5 Speed vs weight: velocity=LTF fast; force=HTF mass.

Official sets — MARK SETS LAW (immutable; MARK_SETS_LAW.md):
  1: 1m | 15m,30m   LTF pullback/cont/add · HTF trend confirm
  2: 5m | 30m,1h
  3: 15m | 1h,4h
  4: 30m | 4h,1d
  Scan ALL four. Never set-2-only for Mark path.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.perception.mark_sets_opportunity import (
    mark_dir_to_action,
    scan_mark_opportunities,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    SetConfluence,
    VelocityStrength,
)

_VEL_RANK = {
    VelocityStrength.NONE: 0,
    VelocityStrength.WEAK: 1,
    VelocityStrength.MEDIUM: 2,
    VelocityStrength.STRONG: 3,
}


class Regime(str, Enum):
    BULL = "bull_trend"
    BEAR = "bear_trend"
    CHOP = "chop"
    FLAT = "flat_undefined"


class PlayState(str, Enum):
    """Law 2 — breath vs launch (relative to force)."""
    NONE = "none"
    BREATHER = "breather"  # pullback / slingshot load
    LAUNCH = "launch"  # acceleration with force
    ALIGNED = "aligned"  # force + velocity same (ready or in trend)


@dataclass(frozen=True)
class DoctrineDecision:
    """Full teacher output for one decision bar."""
    action: int  # 0 hold 1 buy 2 sell
    force_dir: Direction
    regime: Regime
    play: PlayState
    m_conf: float
    m_regime: float
    reason: str
    n_force_bull: int
    n_force_bear: int
    n_aligned: int
    n_breather: int
    best_set_id: Optional[int]


def _dir_sign(d: Direction) -> int:
    if d == Direction.BULL:
        return 1
    if d == Direction.BEAR:
        return -1
    return 0


def regime_from_sets(
    official: Mapping[int, SetConfluence],
    *,
    min_force_sets: int = 2,
) -> Tuple[Regime, Direction, int, int]:
    """Law 3: discrete regime from HTF force across official sets.

    Count sets with clear HTF direction (velocity not NONE preferred).
    Majority with ≥ min_force_sets → bull/bear trend.
    Both sides present → chop.
    Else flat/undefined → no trade.
    """
    n_bull = 0
    n_bear = 0
    for s in OFFICIAL_SETS:
        c = official.get(s.set_id)
        if c is None or c.direction == Direction.NEUTRAL:
            continue
        # Weak single-group votes count less: require at least WEAK
        if _VEL_RANK.get(c.velocity, 0) < 1 and c.direction != Direction.NEUTRAL:
            # still count clear direction from confluence
            pass
        if c.direction == Direction.BULL:
            n_bull += 1
        elif c.direction == Direction.BEAR:
            n_bear += 1

    if n_bull >= min_force_sets and n_bull > n_bear:
        return Regime.BULL, Direction.BULL, n_bull, n_bear
    if n_bear >= min_force_sets and n_bear > n_bull:
        return Regime.BEAR, Direction.BEAR, n_bull, n_bear
    if n_bull > 0 and n_bear > 0:
        return Regime.CHOP, Direction.NEUTRAL, n_bull, n_bear
    if n_bull == 1 and n_bear == 0:
        # Single-set force — soft bull bias but regime still weak
        return Regime.FLAT, Direction.BULL, n_bull, n_bear
    if n_bear == 1 and n_bull == 0:
        return Regime.FLAT, Direction.BEAR, n_bull, n_bear
    return Regime.FLAT, Direction.NEUTRAL, n_bull, n_bear


def classify_set_play(
    force: Direction,
    velocity: Direction,
) -> PlayState:
    """Law 2 + 5: breath vs launch vs aligned for one set."""
    if force == Direction.NEUTRAL:
        return PlayState.NONE
    if velocity == Direction.NEUTRAL:
        return PlayState.NONE
    if velocity == force:
        return PlayState.ALIGNED  # release / ride
    # velocity against force = slingshot load (breather)
    return PlayState.BREATHER


def m_regime_for(regime: Regime) -> float:
    """Law 4: environment size multiplier (≤1). Shell still hard-caps risk."""
    if regime == Regime.BULL or regime == Regime.BEAR:
        return 1.0
    if regime == Regime.CHOP:
        return 0.35
    return 0.0  # flat/undefined → no new risk preferred


def m_conf_for(
    *,
    n_aligned: int,
    n_force: int,
    play: PlayState,
    regime: Regime,
) -> float:
    """Law 4: confidence multiplier in [0, 1.5]."""
    if regime in (Regime.FLAT, Regime.CHOP) and n_aligned < 2:
        return 0.0
    if play == PlayState.BREATHER:
        return 0.0  # loading slingshot — do not enter against on LTF side
    if play not in (PlayState.ALIGNED, PlayState.LAUNCH):
        return 0.0
    base = 0.85
    if n_aligned >= 3:
        base = 1.35
    elif n_aligned == 2:
        base = 1.15
    elif n_aligned == 1 and n_force >= 2:
        base = 1.0
    elif n_aligned == 1:
        base = 0.9
    return min(1.5, base)


# Shared ENTJ opportunity floor (was 2.0 only on soft scalp; ignored on trend)
DEFAULT_OPP_MIN_SCORE = 1.2


def decide_doctrine(
    official: Mapping[int, SetConfluence],
    entry_dirs: Mapping[int, Direction],
    *,
    min_force_sets: int = 2,
    allow_single_set_scalp: bool = True,
    opp_min_score: float = DEFAULT_OPP_MIN_SCORE,
    target_pct: float = 0.0,
    equity_pct: float = 0.0,
) -> DoctrineDecision:
    """Full five-law decision for one bar.

    Returns action HOLD/BUY/SELL with reasons and size multipliers.

    Chart-read law (Mark 2026-08-04): soft single-set scalp appears on ~all
    hard-target MISSES. For target >= 2.5% Mark refuses flat-regime single-set
    noise — only multi-set BULL/BEAR tide releases. Soft scalp only for soft
    daily targets (scalp bank early).
    """
    # Hard target → no flat soft-scalp (measured: 30/31 hard misses had soft_single)
    if float(target_pct) >= 2.5:
        allow_single_set_scalp = False

    regime, force_hint, n_fb, n_fr = regime_from_sets(
        official, min_force_sets=min_force_sets
    )
    m_reg = m_regime_for(regime)

    # Per-set force/velocity plays
    n_aligned = 0
    n_breather = 0
    bull_aligned = 0
    bear_aligned = 0
    best_id: Optional[int] = None
    best_score = -1.0
    best_vel_rank = 0

    for s in OFFICIAL_SETS:
        conf = official.get(s.set_id)
        if conf is None:
            continue
        force = conf.direction
        vel = Direction(entry_dirs.get(s.set_id, Direction.NEUTRAL))
        play = classify_set_play(force, vel)
        if play == PlayState.BREATHER:
            n_breather += 1
        if play == PlayState.ALIGNED:
            n_aligned += 1
            vr = int(_VEL_RANK.get(conf.velocity, 0))
            sc = float(vr + 1) * (1.0 + 0.15 * s.set_id)
            if force == Direction.BULL:
                bull_aligned += 1
            elif force == Direction.BEAR:
                bear_aligned += 1
            if sc > best_score:
                best_score = sc
                best_id = s.set_id
                best_vel_rank = vr

    # Law 3 hard rewrite
    if regime == Regime.CHOP:
        return DoctrineDecision(
            action=0,
            force_dir=Direction.NEUTRAL,
            regime=regime,
            play=PlayState.NONE,
            m_conf=0.0,
            m_regime=m_reg,
            reason="law3_chop_no_breakout_chase",
            n_force_bull=n_fb,
            n_force_bear=n_fr,
            n_aligned=n_aligned,
            n_breather=n_breather,
            best_set_id=best_id,
        )

    if regime == Regime.FLAT and not allow_single_set_scalp:
        return DoctrineDecision(
            action=0,
            force_dir=force_hint,
            regime=regime,
            play=PlayState.NONE,
            m_conf=0.0,
            m_regime=0.0,
            reason="law3_flat_undefined_no_trade",
            n_force_bull=n_fb,
            n_force_bear=n_fr,
            n_aligned=n_aligned,
            n_breather=n_breather,
            best_set_id=best_id,
        )

    # Shared multi-set opportunity score (Mark eyes) — quality gate for releases
    mark_opp = scan_mark_opportunities(
        official,
        entry_dirs,
        min_aligned=1,
        min_score=float(opp_min_score),
        macro_permission=True,
    )

    # Law 1: force gate — only trade with regime force
    if regime == Regime.BULL:
        force_dir = Direction.BULL
        if bull_aligned == 0:
            # slingshot loading or no trigger
            play = PlayState.BREATHER if n_breather > 0 else PlayState.NONE
            return DoctrineDecision(
                action=0,
                force_dir=force_dir,
                regime=regime,
                play=play,
                m_conf=0.0,
                m_regime=m_reg,
                reason="law1_bull_tide_wait_ltf_resume"
                if n_breather
                else "law1_bull_tide_no_ltf_trigger",
                n_force_bull=n_fb,
                n_force_bear=n_fr,
                n_aligned=n_aligned,
                n_breather=n_breather,
                best_set_id=best_id,
            )
        # Quality: LTF aligned + opportunity score / velocity not noise
        if mark_opp.action_dir != Direction.BULL and best_vel_rank < 1:
            return DoctrineDecision(
                action=0,
                force_dir=force_dir,
                regime=regime,
                play=PlayState.ALIGNED,
                m_conf=0.0,
                m_regime=m_reg,
                reason="law1_bull_aligned_but_opp_score_weak",
                n_force_bull=n_fb,
                n_force_bear=n_fr,
                n_aligned=n_aligned,
                n_breather=n_breather,
                best_set_id=best_id,
            )
        # fire long only
        m_c = m_conf_for(
            n_aligned=bull_aligned,
            n_force=n_fb,
            play=PlayState.ALIGNED,
            regime=regime,
        )
        return DoctrineDecision(
            action=1,  # BUY
            force_dir=force_dir,
            regime=regime,
            play=PlayState.ALIGNED,
            m_conf=m_c,
            m_regime=m_reg,
            reason=f"law1_slingshot_release_long n_aligned={bull_aligned} vel={best_vel_rank}",
            n_force_bull=n_fb,
            n_force_bear=n_fr,
            n_aligned=n_aligned,
            n_breather=n_breather,
            best_set_id=best_id,
        )

    if regime == Regime.BEAR:
        force_dir = Direction.BEAR
        if bear_aligned == 0:
            play = PlayState.BREATHER if n_breather > 0 else PlayState.NONE
            return DoctrineDecision(
                action=0,
                force_dir=force_dir,
                regime=regime,
                play=play,
                m_conf=0.0,
                m_regime=m_reg,
                reason="law1_bear_tide_wait_ltf_resume"
                if n_breather
                else "law1_bear_tide_no_ltf_trigger",
                n_force_bull=n_fb,
                n_force_bear=n_fr,
                n_aligned=n_aligned,
                n_breather=n_breather,
                best_set_id=best_id,
            )
        if mark_opp.action_dir != Direction.BEAR and best_vel_rank < 1:
            return DoctrineDecision(
                action=0,
                force_dir=force_dir,
                regime=regime,
                play=PlayState.ALIGNED,
                m_conf=0.0,
                m_regime=m_reg,
                reason="law1_bear_aligned_but_opp_score_weak",
                n_force_bull=n_fb,
                n_force_bear=n_fr,
                n_aligned=n_aligned,
                n_breather=n_breather,
                best_set_id=best_id,
            )
        m_c = m_conf_for(
            n_aligned=bear_aligned,
            n_force=n_fr,
            play=PlayState.ALIGNED,
            regime=regime,
        )
        return DoctrineDecision(
            action=2,  # SELL
            force_dir=force_dir,
            regime=regime,
            play=PlayState.ALIGNED,
            m_conf=m_c,
            m_regime=m_reg,
            reason=f"law1_slingshot_release_short n_aligned={bear_aligned} vel={best_vel_rank}",
            n_force_bull=n_fb,
            n_force_bear=n_fr,
            n_aligned=n_aligned,
            n_breather=n_breather,
            best_set_id=best_id,
        )

    # FLAT with optional single-set ENTJ scalp — same opp_min_score as trend quality
    if allow_single_set_scalp and n_aligned >= 1 and force_hint != Direction.NEUTRAL:
        if (
            mark_opp.action_dir == force_hint
            and mark_opp.action_dir != Direction.NEUTRAL
        ):
            m_c = 0.75
            act = mark_dir_to_action(mark_opp.action_dir)
            return DoctrineDecision(
                action=act,
                force_dir=force_hint,
                regime=regime,
                play=PlayState.ALIGNED,
                m_conf=m_c,
                m_regime=0.5,
                reason=f"law1_soft_single_set_scalp {mark_opp.reason}",
                n_force_bull=n_fb,
                n_force_bear=n_fr,
                n_aligned=n_aligned,
                n_breather=n_breather,
                best_set_id=best_id,
            )

    return DoctrineDecision(
        action=0,
        force_dir=force_hint,
        regime=regime,
        play=PlayState.NONE,
        m_conf=0.0,
        m_regime=m_reg,
        reason="law3_no_permission_or_trigger",
        n_force_bull=n_fb,
        n_force_bear=n_fr,
        n_aligned=n_aligned,
        n_breather=n_breather,
        best_set_id=best_id,
    )


def doctrine_action_from_perception(perc: Mapping) -> DoctrineDecision:
    """Convenience: read official + entry_dirs (+ optional target/equity) from perc."""
    official = perc.get("official") or {}
    entry_dirs = perc.get("entry_dirs") or {}
    return decide_doctrine(
        official,
        entry_dirs,
        target_pct=float(perc.get("target_pct", 0.0) or 0.0),
        equity_pct=float(perc.get("equity_pct", 0.0) or 0.0),
    )
