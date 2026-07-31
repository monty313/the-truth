"""Reward dials interface for adaptive_rl_brain_7_31_26 (no training loop yet).

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 4 — WHY: searchable dials + pure credit formula.
  MINDLESS is a wall (fixed massive penalty). Parallel lineage only.

Locked dials (bounds inclusive):
  w_with_vector      [0.5, 2.0]
  w_qualified_macro  [0.5, 2.0]
  w_qualified_micro  [0.15, 0.7]
  w_inactivity       [0.0, 1.0]

credit = w_class × clip(realized_pnl / risk_amount, -1, +1)
MINDLESS → fixed penalty (not scaled by PnL luck).
"""
from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Dict, Mapping

from lineages.adaptive_rl_brain_7_31_26.perception.types import TradeTag

# Inclusive bounds: (lo, hi)
DIAL_BOUNDS: Dict[str, tuple[float, float]] = {
    "w_with_vector": (0.5, 2.0),
    "w_qualified_macro": (0.5, 2.0),
    "w_qualified_micro": (0.15, 0.7),
    "w_inactivity": (0.0, 1.0),
}

# Defaults mid-band (not frozen final answers — meta may search later)
DEFAULT_DIALS: Dict[str, float] = {
    "w_with_vector": 1.0,
    "w_qualified_macro": 1.0,
    "w_qualified_micro": 0.4,
    "w_inactivity": 0.0,
}

# Fixed massive penalty — mindless wall (not a searchable "reward for luck")
MINDLESS_PENALTY = -10.0

_TAG_TO_DIAL = {
    TradeTag.WITH_VECTOR: "w_with_vector",
    TradeTag.QUALIFIED_MACRO: "w_qualified_macro",
    TradeTag.QUALIFIED_MICRO: "w_qualified_micro",
}


@dataclass(frozen=True)
class RewardDials:
    w_with_vector: float = 1.0
    w_qualified_macro: float = 1.0
    w_qualified_micro: float = 0.4
    w_inactivity: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {f.name: float(getattr(self, f.name)) for f in fields(self)}


def clip_dials(dials: Mapping[str, float]) -> Dict[str, float]:
    """Clamp every known dial into locked bounds; drop unknowns."""
    out: Dict[str, float] = {}
    for k, (lo, hi) in DIAL_BOUNDS.items():
        v = float(dials.get(k, DEFAULT_DIALS[k]))
        out[k] = float(min(max(v, lo), hi))
    return out


def make_dials(**kwargs: float) -> RewardDials:
    base = dict(DEFAULT_DIALS)
    base.update({k: float(v) for k, v in kwargs.items() if k in DIAL_BOUNDS})
    clipped = clip_dials(base)
    return RewardDials(**clipped)


def pnl_unit(realized_pnl: float, risk_amount: float) -> float:
    """clip(realized_pnl / risk_amount, -1, +1). risk_amount <= 0 → 0."""
    r = float(risk_amount)
    if r <= 0.0 or not (r == r):  # NaN guard
        return 0.0
    x = float(realized_pnl) / r
    if x > 1.0:
        return 1.0
    if x < -1.0:
        return -1.0
    return float(x)


def class_weight(tag: TradeTag, dials: Mapping[str, float]) -> float:
    """Weight for a trade class; inactivity is separate (flat-day path)."""
    d = clip_dials(dials)
    if tag == TradeTag.MINDLESS:
        return 0.0  # wall uses fixed penalty, not class weight
    key = _TAG_TO_DIAL.get(TradeTag(tag))
    if key is None:
        return 0.0
    return float(d[key])


def credit(
    tag: TradeTag,
    realized_pnl: float,
    risk_amount: float,
    dials: Mapping[str, float] | RewardDials | None = None,
    *,
    mindless_penalty: float = MINDLESS_PENALTY,
) -> float:
    """Pure credit for one closed trade (or mindless attempt).

    MINDLESS → fixed massive penalty (ignores lucky PnL).
    Else → w_class × clip(pnl / risk, -1, +1).
    """
    t = TradeTag(tag)
    if t == TradeTag.MINDLESS:
        return float(mindless_penalty)
    if dials is None:
        dmap: Mapping[str, float] = DEFAULT_DIALS
    elif isinstance(dials, RewardDials):
        dmap = dials.as_dict()
    else:
        dmap = dials
    w = class_weight(t, dmap)
    return float(w) * pnl_unit(realized_pnl, risk_amount)


def inactivity_penalty(
    dials: Mapping[str, float] | RewardDials | None = None,
) -> float:
    """Negative pressure when flat / no trade (w_inactivity as magnitude)."""
    if dials is None:
        dmap: Mapping[str, float] = DEFAULT_DIALS
    elif isinstance(dials, RewardDials):
        dmap = dials.as_dict()
    else:
        dmap = dials
    d = clip_dials(dmap)
    return -float(d["w_inactivity"])
