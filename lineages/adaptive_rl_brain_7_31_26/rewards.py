"""Reward dials interface for adaptive_rl_brain_7_31_26 (no training loop yet).

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 4 — WHY: searchable dials + pure credit formula.
  MINDLESS is a wall (fixed massive penalty). Parallel lineage only.
- 2026-07-31  anti-hold collapse — WHY: end-of-day did-nothing wall + stronger
  per-step inactivity on WITH_VECTOR / QUALIFIED_MACRO + correct-side entry
  shaping. Stops all-hold free ride. Parallel lineage only.

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
from typing import Dict, Mapping, Optional

from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, TradeTag

# Inclusive bounds: (lo, hi)
DIAL_BOUNDS: Dict[str, tuple[float, float]] = {
    "w_with_vector": (0.5, 2.0),
    "w_qualified_macro": (0.5, 2.0),
    "w_qualified_micro": (0.15, 0.7),
    "w_inactivity": (0.0, 1.0),
}

# Defaults mid-band; w_inactivity raised so flat-on-setup is not free
DEFAULT_DIALS: Dict[str, float] = {
    "w_with_vector": 1.0,
    "w_qualified_macro": 1.0,
    "w_qualified_micro": 0.4,
    "w_inactivity": 0.85,
}

# Fixed massive penalty — mindless wall (not a searchable "reward for luck")
MINDLESS_PENALTY = -10.0

# End-of-day: day finished with no participation (realized ~0, no entries).
# Same order of magnitude as serious failure penalties; fires once per day.
# Top of suggested band so pure hold is not cheaper than careful trading.
DID_NOTHING_EOD_PENALTY = -25.0

# Near-zero PnL threshold after costs / float noise
EOD_PNL_EPS = 1e-6

# Per-step floor when WITH_VECTOR / QUALIFIED_MACRO is active and policy HOLDs
# (ensures pressure even if dial is dialed down mid-search)
INACTIVITY_SETUP_FLOOR = 1.25

# Always-on tax while flat + HOLD (even without setup) — kills free hold
FLAT_HOLD_TAX = 0.20

# Immediate shaping when entry side matches higher / setup direction
# Large enough to offset some inactivity and beat pure hold under REINFORCE
CORRECT_SIDE_ENTRY_BONUS = 0.75

# --- directional structure shaping (bake HOLD-fix into train rewards) ---
# Flat + structure rec BUY/SELL + policy HOLD → stronger than setup inactivity
# (~setup_hold ≈ −1.45). Must dominate pure-hold free ride under REINFORCE.
DIRECTIONAL_HOLD_PENALTY = -3.5
# Flat + policy takes the matching side as structure rec → clear entry bonus
# (stacks with correct_side_entry_bonus when higher TF also agrees).
STRUCTURE_MATCH_ENTRY_BONUS = 2.0

# --- second-best logit regret (missed opportunity when HOLD while flat) ---
# If 2nd-best entry side would have been profitable after fees over a short
# horizon → fixed missed-opportunity penalty. Not larger than MINDLESS wall.
SECOND_BEST_REGRET_PENALTY = -3.0
# Lookahead M1 bars for counterfactual entry (≈ one decide_every=25 window)
SECOND_BEST_HORIZON_BARS = 25
# Round-trip fee in price units (≈ 2× default gold spread 0.25 from equity_day)
SECOND_BEST_FEE_PX = 0.50

# Per-step majority-idle (active agents consensus — see signal_majority.py)
# ≥10 agents active, ≥60% of those on one side, bot HOLDs with <2 trades open
MAJORITY_IDLE_PENALTY = -1.50
MAJORITY_MIN_ACTIVE = 10
MAJORITY_AGREE_FRAC = 0.60  # at least 60% of active (e.g. 6 of 10)
MAJORITY_MIN_OPEN_EXEMPT = 2  # no idle penalty if already ≥2 trades open

# --- thrash control (Phase A) — hard limits inside lineage only ---
# Max concurrent units on one side (1 entry + up to 2 scale-ins)
MAX_OPEN_UNITS = 3
# After a reverse, block another reverse for this many M1 bars
# (~4 decisions when decide_every=25)
REVERSE_COOLDOWN_BARS = 100
# Extra penalty applied on every reverse (discourages flip-flops)
FLIP_FLOP_PENALTY = -1.0

_TAG_TO_DIAL = {
    TradeTag.WITH_VECTOR: "w_with_vector",
    TradeTag.QUALIFIED_MACRO: "w_qualified_macro",
    TradeTag.QUALIFIED_MICRO: "w_qualified_micro",
}

# Tags that demand action when present (not micro-only / weak pullback)
ACTION_TAGS = frozenset({TradeTag.WITH_VECTOR, TradeTag.QUALIFIED_MACRO})


@dataclass(frozen=True)
class RewardDials:
    w_with_vector: float = 1.0
    w_qualified_macro: float = 1.0
    w_qualified_micro: float = 0.4
    w_inactivity: float = 0.85

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


def _dmap(dials: Mapping[str, float] | RewardDials | None) -> Mapping[str, float]:
    if dials is None:
        return DEFAULT_DIALS
    if isinstance(dials, RewardDials):
        return dials.as_dict()
    return dials


def inactivity_penalty(
    dials: Mapping[str, float] | RewardDials | None = None,
) -> float:
    """Negative pressure when flat / no trade (w_inactivity as magnitude)."""
    d = clip_dials(_dmap(dials))
    return -float(d["w_inactivity"])


def setup_hold_penalty(
    tag: TradeTag | None,
    dials: Mapping[str, float] | RewardDials | None = None,
    *,
    flat_tax: float = FLAT_HOLD_TAX,
) -> float:
    """Per-step penalty when policy HOLDs while a setup tag is active.

    WITH_VECTOR / QUALIFIED_MACRO → clear pressure (max of dial and floor)
    plus always-on flat tax.
    QUALIFIED_MICRO / other → dial + flat tax (weaker than action tags).
    MINDLESS / None → flat tax only (still not free to sit).
    """
    tax = -abs(float(flat_tax))
    if tag is None or tag == TradeTag.MINDLESS:
        return float(tax)
    base = inactivity_penalty(dials)
    t = TradeTag(tag)
    if t in ACTION_TAGS:
        # more negative = stronger; ensure at least floor magnitude
        floor = -float(INACTIVITY_SETUP_FLOOR)
        setup_part = float(min(base, floor))  # min because both negative
        return float(setup_part + tax)
    # micro / weak: dial + tax
    return float(base + tax)


def flat_hold_tax(*, tax: float = FLAT_HOLD_TAX) -> float:
    """Always-on per-step cost of HOLD while flat (no position)."""
    return -abs(float(tax))


def correct_side_entry_bonus(
    tag: TradeTag | None,
    trade_side: Optional[Direction],
    higher_direction: Optional[Direction],
    *,
    bonus: float = CORRECT_SIDE_ENTRY_BONUS,
) -> float:
    """Small immediate shaping when entry is the correct side on strong tags.

    Correct = trade_side matches non-neutral higher_direction.
    Only WITH_VECTOR / QUALIFIED_MACRO (and MICRO gets half).
    """
    if tag is None or trade_side is None:
        return 0.0
    if trade_side == Direction.NEUTRAL:
        return 0.0
    if higher_direction is None or higher_direction == Direction.NEUTRAL:
        return 0.0
    if trade_side != higher_direction:
        return 0.0
    t = TradeTag(tag)
    if t in ACTION_TAGS:
        return float(bonus)
    if t == TradeTag.QUALIFIED_MICRO:
        return float(bonus) * 0.5
    return 0.0


def did_nothing_eod_penalty(
    realized_pnl: float,
    n_entries: int,
    *,
    penalty: float = DID_NOTHING_EOD_PENALTY,
    eps: float = EOD_PNL_EPS,
) -> float:
    """Once-per-day penalty when the agent never participated.

    Fires if realized PnL is ~0 AND no entries were taken.
    Does not fire every bar — caller applies once at day end.
    """
    if int(n_entries) > 0:
        return 0.0
    if abs(float(realized_pnl)) < float(eps):
        return float(penalty)
    return 0.0


def flip_flop_penalty(*, penalty: float = FLIP_FLOP_PENALTY) -> float:
    """Small negative shaping on reverse (close + flip). Thrash control."""
    return -abs(float(penalty))


def directional_hold_penalty(
    *,
    is_flat: bool,
    action: int,
    structure_rec: int,
    penalty: float = DIRECTIONAL_HOLD_PENALTY,
) -> float:
    """Strong penalty when flat, structure wants BUY/SELL, policy HOLDs.

    Larger magnitude than setup inactivity so REINFORCE moves HOLD logits down
    on directional bars. Does not hard-ban HOLD in the sampler.
    action / structure_rec use policy_stub codes: 0=HOLD, 1=BUY, 2=SELL.
    """
    if not is_flat:
        return 0.0
    if int(action) != 0:  # ACTION_HOLD
        return 0.0
    rec = int(structure_rec)
    if rec not in (1, 2):  # BUY, SELL
        return 0.0
    return -abs(float(penalty))


def structure_match_entry_bonus(
    *,
    is_flat: bool,
    action: int,
    structure_rec: int,
    bonus: float = STRUCTURE_MATCH_ENTRY_BONUS,
) -> float:
    """Positive shaping when flat entry matches structure BUY/SELL rec.

    Encourages raw argmax to prefer the structure side over HOLD.
    """
    if not is_flat:
        return 0.0
    rec = int(structure_rec)
    act = int(action)
    if rec not in (1, 2):
        return 0.0
    if act != rec:
        return 0.0
    return abs(float(bonus))


def second_best_entry_from_logits(logits) -> Optional[int]:
    """2nd-best action by logit; return BUY/SELL only, else None.

    When HOLD is argmax, this is the natural counterfactual entry side.
    If 2nd-best is HOLD (policy held against a directional argmax), returns None.
    """
    import numpy as np

    arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    order = list(np.argsort(-arr))
    if len(order) < 2:
        return None
    second = int(order[1])
    if second in (1, 2):  # BUY, SELL
        return second
    return None


def missed_opportunity_penalty(
    *,
    cf_pnl_after_fees: float,
    penalty: float = SECOND_BEST_REGRET_PENALTY,
) -> float:
    """If counterfactual 2nd-best entry would have been profitable → regret.

    cf_pnl_after_fees > 0 → fixed penalty in [-4, -2] band (default -3).
    Otherwise 0. Never exceeds MINDLESS wall magnitude by design (caller).
    """
    if float(cf_pnl_after_fees) > 0.0:
        return -abs(float(penalty))
    return 0.0


def majority_agents_idle_penalty(
    *,
    has_majority: bool,
    action_is_hold: bool,
    n_open: int = 0,
    min_open_exempt: int = MAJORITY_MIN_OPEN_EXEMPT,
    penalty: float = MAJORITY_IDLE_PENALTY,
    # legacy alias: is_flat ignored for gate (open-count is the gate now)
    is_flat: bool | None = None,
) -> float:
    """Penalty when active agents consensus and bot does nothing.

    Fires when:
      - has_majority (caller: ≥20 active and >70% agree on one side)
      - action is HOLD
      - fewer than min_open_exempt trades are already open (default 2)

    Stacks with setup-hold / flat-hold tax when those also apply.
    """
    if not has_majority or not action_is_hold:
        return 0.0
    if int(n_open) >= int(min_open_exempt):
        return 0.0
    return -abs(float(penalty))
