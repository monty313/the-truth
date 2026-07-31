"""Channel 1 observation block for adaptive_rl_brain_7_31_26 only.

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 3 — WHY: lean lineage-local obs (sets + structure +
  placeholders). Does NOT change PROVEN frame_dim / policy input size.

Layout (fixed length = CHANNEL1_DIM = 32):
  [0:12]  Official Sets 1–4 × (direction, velocity, confluence_score)
  [12:27] Sub-Sets A–E × (direction, velocity, confluence_score)
  [27]    pullback (0/1)
  [28]    scale_conflict (0/1)
  [29]    progress_to_goal  (placeholder input, typically [-1, 1] or [0, 1])
  [30]    danger            (placeholder input, typically [0, 1])
  [31]    session_phase     (placeholder input, typically [0, 1])
"""
from __future__ import annotations

from typing import Dict, Mapping, Optional

import numpy as np

from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    OFFICIAL_SETS,
    SUB_SETS,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    SetConfluence,
    StructureFlags,
    VelocityStrength,
)

N_OFFICIAL = 4
N_SUB = 5
FEATURES_PER_SET = 3  # direction, velocity, confluence_score
CHANNEL1_DIM = N_OFFICIAL * FEATURES_PER_SET + N_SUB * FEATURES_PER_SET + 5  # 12+15+5=32

_VEL_SCORE = {
    VelocityStrength.NONE: 0.0,
    VelocityStrength.WEAK: 1.0 / 3.0,
    VelocityStrength.MEDIUM: 2.0 / 3.0,
    VelocityStrength.STRONG: 1.0,
}


def velocity_to_float(v: VelocityStrength) -> float:
    return float(_VEL_SCORE.get(VelocityStrength(v), 0.0))


def direction_to_float(d: Direction) -> float:
    return float(int(Direction(d)))


def confluence_score(c: SetConfluence) -> float:
    """Signed agreement in [-1, 1]: (n_bull - n_bear) / max(n_groups, 1)."""
    n = max(int(c.n_bull + c.n_bear + c.n_neutral), 1)
    return float(c.n_bull - c.n_bear) / float(n)


def _pack_set(c: Optional[SetConfluence]) -> tuple[float, float, float]:
    if c is None:
        return 0.0, 0.0, 0.0
    return (
        direction_to_float(c.direction),
        velocity_to_float(c.velocity),
        confluence_score(c),
    )


def empty_confluence(set_key: str) -> SetConfluence:
    return SetConfluence(
        set_key=set_key,
        direction=Direction.NEUTRAL,
        velocity=VelocityStrength.NONE,
        votes=(),
        n_bull=0,
        n_bear=0,
        n_neutral=3,
    )


def build_channel1_obs(
    official: Mapping[int, SetConfluence] | None = None,
    subs: Mapping[str, SetConfluence] | None = None,
    structure: StructureFlags | None = None,
    *,
    progress_to_goal: float = 0.0,
    danger: float = 0.0,
    session_phase: float = 0.0,
) -> np.ndarray:
    """Build fixed-length Channel 1 vector (lineage-local only)."""
    off = dict(official or {})
    sub = {str(k).upper(): v for k, v in (subs or {}).items()}
    struct = structure or StructureFlags(pullback=False, scale_conflict=False)

    out = np.zeros(CHANNEL1_DIM, dtype=np.float32)
    # Official 1..4
    for i, s in enumerate(OFFICIAL_SETS):
        d, v, sc = _pack_set(off.get(s.set_id))
        base = i * FEATURES_PER_SET
        out[base] = d
        out[base + 1] = v
        out[base + 2] = sc
    # Sub A..E
    sub_base = N_OFFICIAL * FEATURES_PER_SET
    for j, s in enumerate(SUB_SETS):
        d, v, sc = _pack_set(sub.get(s.sub_id))
        base = sub_base + j * FEATURES_PER_SET
        out[base] = d
        out[base + 1] = v
        out[base + 2] = sc
    out[27] = 1.0 if struct.pullback else 0.0
    out[28] = 1.0 if struct.scale_conflict else 0.0
    out[29] = float(progress_to_goal)
    out[30] = float(danger)
    out[31] = float(session_phase)
    return out


def channel1_layout() -> Dict[str, object]:
    """Human-readable layout map for debugging / docs."""
    return {
        "dim": CHANNEL1_DIM,
        "official_block": "0:12  (4 sets × dir,vel,score)",
        "sub_block": "12:27 (5 subs × dir,vel,score)",
        "pullback": 27,
        "scale_conflict": 28,
        "progress_to_goal": 29,
        "danger": 30,
        "session_phase": 31,
        "note": "lineage-local only — does not alter PROVEN obs_dim",
    }
