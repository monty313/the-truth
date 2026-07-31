"""Shared enums / dataclasses for adaptive_rl_brain_7_31_26 Phase 1.

CHANGE LOG:
- 2026-07-31  created — WHY: Phase 1 pure perception types for Official Sets,
  confluence votes, structure flags, trade tags, MINDLESS wall inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class Direction(int, Enum):
    """Signed market direction. Neutral is flat / no clear vote."""
    BEAR = -1
    NEUTRAL = 0
    BULL = 1


class VelocityStrength(str, Enum):
    """How many confluence groups agree on the majority direction.

    Mapping (locked):
      3 agree → STRONG
      2 agree → MEDIUM
      1 agree → WEAK
      0 or conflict → NONE
    """
    NONE = "none"
    WEAK = "weak"
    MEDIUM = "medium"
    STRONG = "strong"


class TradeTag(str, Enum):
    """Primary trade classification (priority order in classify.py)."""
    MINDLESS = "MINDLESS"
    WITH_VECTOR = "WITH_VECTOR"
    QUALIFIED_MACRO = "QUALIFIED_MACRO"
    QUALIFIED_MICRO = "QUALIFIED_MICRO"


class ConfluenceGroup(str, Enum):
    CCI = "cci"
    RSI = "rsi"
    PRICE_CHANNEL = "price_channel"


@dataclass(frozen=True)
class OfficialSet:
    """Full-strength set: Entry TF + two Confirmation TFs."""
    set_id: int
    name: str
    entry_tf: str
    confirmation_tfs: Tuple[str, str]  # exactly two, higher TFs

    @property
    def tfs(self) -> Tuple[str, str, str]:
        """All three TFs, LTF/Entry first."""
        return (self.entry_tf, self.confirmation_tfs[0], self.confirmation_tfs[1])

    def __hash__(self) -> int:
        return hash((self.set_id, self.entry_tf, self.confirmation_tfs))


@dataclass(frozen=True)
class SubSet:
    """Weaker pair: first TF = Entry, second = Confirmation."""
    sub_id: str  # "A".."E"
    entry_tf: str
    confirmation_tf: str

    @property
    def tfs(self) -> Tuple[str, str]:
        return (self.entry_tf, self.confirmation_tf)

    def __hash__(self) -> int:
        return hash((self.sub_id, self.entry_tf, self.confirmation_tf))


@dataclass(frozen=True)
class GroupVote:
    """One confluence group's vote on the Confirmation stack."""
    group: ConfluenceGroup
    direction: Direction  # BULL / BEAR / NEUTRAL


@dataclass(frozen=True)
class SetConfluence:
    """Simple-majority Direction + Velocity strength for one set's confirmations."""
    set_key: str  # e.g. "official:1" or "sub:A"
    direction: Direction
    velocity: VelocityStrength
    votes: Tuple[GroupVote, ...]  # length 3 when fully evaluated
    n_bull: int
    n_bear: int
    n_neutral: int


@dataclass(frozen=True)
class StructureFlags:
    pullback: bool
    scale_conflict: bool


@dataclass(frozen=True)
class MindlessInputs:
    """Inputs for the 3-condition MINDLESS wall (relative higher/lower sets)."""
    trade_side: Direction  # BULL or BEAR (proposed trade)
    lower_vector_turned: bool   # (a) lower-set Vector M turned trade direction
    lower_velocity_confirms: bool  # (b) lower-set velocity confirms the turn
    higher_weakening_or_pullback: bool  # (c) higher sets weakening / pullback


@dataclass(frozen=True)
class Classification:
    tag: TradeTag
    mindless: bool
    reasons: Tuple[str, ...]
