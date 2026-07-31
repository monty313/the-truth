"""Day-level runner: multi-TF → perception → Channel 1 → rewards.

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 5 — WHY: first trainable day loop for the lineage.
  Parallel only; no PROVEN.

Flow per decision bar (M1 step or stride):
  1. asof multi-TF bars from pack
  2. official/sub confluence via live indicators
  3. structure + classify for a proposed/active trade side
  4. Channel 1 obs
  5. reward: credit on close; inactivity if setup active & policy flat
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import pandas as pd

from lineages.adaptive_rl_brain_7_31_26.data.mtf import bar_asof, build_mtf_pack
from lineages.adaptive_rl_brain_7_31_26.perception.classify import (
    classify_trade,
    make_mindless_inputs,
)
from lineages.adaptive_rl_brain_7_31_26.perception.live_indicators import (
    confluence_from_confirmation_ohlc,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import build_channel1_obs
from lineages.adaptive_rl_brain_7_31_26.perception.pipeline import (
    direction_from_single_tf_ohlc,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS, SUB_SETS
from lineages.adaptive_rl_brain_7_31_26.perception.structure import structure_flags
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Classification,
    Direction,
    SetConfluence,
    StructureFlags,
    TradeTag,
    VelocityStrength,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    action_to_trade_side,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    DEFAULT_DIALS,
    credit,
    inactivity_penalty,
)


@dataclass
class DayStepResult:
    t: int
    ts: Any
    obs: np.ndarray
    action: int
    trade_side: Optional[Direction]
    tag: TradeTag
    reward: float
    classification: Classification
    structure: StructureFlags
    info: Dict[str, Any] = field(default_factory=dict)


def _slice_to_bar(frame: pd.DataFrame, bar_i: int) -> pd.DataFrame:
    """Prefix of frame through bar_i inclusive (causal)."""
    if bar_i < 0 or len(frame) == 0:
        return frame.iloc[0:0]
    return frame.iloc[: bar_i + 1]


def _neutral_conf(set_key: str) -> SetConfluence:
    return SetConfluence(
        set_key=set_key,
        direction=Direction.NEUTRAL,
        velocity=VelocityStrength.NONE,
        votes=(),
        n_bull=0,
        n_bear=0,
        n_neutral=3,
    )


def _confluence_at(
    pack: Mapping[str, pd.DataFrame],
    conf_a: str,
    conf_b: str,
    ts: pd.Timestamp,
    set_key: str,
) -> SetConfluence:
    fa = pack.get(conf_a)
    fb = pack.get(conf_b)
    if fa is None or fb is None or len(fa) == 0 or len(fb) == 0:
        return _neutral_conf(set_key)
    ia, ib = bar_asof(fa, ts), bar_asof(fb, ts)
    if ia < 0 or ib < 0:
        return _neutral_conf(set_key)
    sa = _slice_to_bar(fa, ia)
    sb = _slice_to_bar(fb, ib)
    if len(sa) < 30 or len(sb) < 30:
        return _neutral_conf(set_key)
    try:
        return confluence_from_confirmation_ohlc(
            set_key, sa, sb, bar_a=-1, bar_b=-1,
        )
    except Exception:
        return _neutral_conf(set_key)


def _entry_dir(
    pack: Mapping[str, pd.DataFrame],
    entry_tf: str,
    ts: pd.Timestamp,
) -> Direction:
    fr = pack.get(entry_tf)
    if fr is None or len(fr) == 0:
        return Direction.NEUTRAL
    i = bar_asof(fr, ts)
    if i < 0:
        return Direction.NEUTRAL
    sl = _slice_to_bar(fr, i)
    if len(sl) < 30:
        return Direction.NEUTRAL
    try:
        return direction_from_single_tf_ohlc(sl, bar=-1)
    except Exception:
        return Direction.NEUTRAL


def build_perception_at(
    pack: Mapping[str, pd.DataFrame],
    ts: pd.Timestamp,
    *,
    trade_side: Optional[Direction] = None,
    progress_to_goal: float = 0.0,
    danger: float = 0.0,
    session_phase: float = 0.0,
) -> Dict[str, Any]:
    """Official/sub confluence + structure + optional classify + Channel 1 obs."""
    official: Dict[int, SetConfluence] = {}
    for s in OFFICIAL_SETS:
        c0, c1 = s.confirmation_tfs
        official[s.set_id] = _confluence_at(
            pack, c0, c1, ts, f"official:{s.set_id}",
        )
    subs: Dict[str, SetConfluence] = {}
    for s in SUB_SETS:
        subs[s.sub_id] = _confluence_at(
            pack, s.entry_tf, s.confirmation_tf, ts, f"sub:{s.sub_id}",
        )

    primary = official.get(2) or next(iter(official.values()))
    lower = _entry_dir(pack, "5m", ts)
    higher = primary.direction
    struct = structure_flags(higher_direction=higher, lower_direction=lower)

    cl: Optional[Classification] = None
    if trade_side is not None and trade_side != Direction.NEUTRAL:
        m = make_mindless_inputs(
            trade_side,
            turned=(lower == trade_side and lower != Direction.NEUTRAL),
            velocity_confirms=True,
            higher_weakening=struct.pullback or primary.velocity in (
                VelocityStrength.WEAK, VelocityStrength.NONE,
            ),
        )
        cl = classify_trade(trade_side, higher, lower, m, pullback=struct.pullback)

    obs = build_channel1_obs(
        official, subs, struct,
        progress_to_goal=progress_to_goal,
        danger=danger,
        session_phase=session_phase,
    )
    return {
        "official": official,
        "subs": subs,
        "structure": struct,
        "higher": higher,
        "lower": lower,
        "classification": cl,
        "obs": obs,
        "primary": primary,
    }


def setup_active(cl: Optional[Classification], struct: StructureFlags) -> bool:
    """True when QUALIFIED_* or WITH_VECTOR is active, or pullback setup present."""
    if cl is not None and cl.tag in (
        TradeTag.WITH_VECTOR,
        TradeTag.QUALIFIED_MACRO,
        TradeTag.QUALIFIED_MICRO,
    ):
        return True
    return bool(struct.pullback)


class DayRunner:
    """Minimal day loop over M1 decision bars."""

    def __init__(
        self,
        m1: pd.DataFrame,
        *,
        decide_every: int = 5,
        dials: Mapping[str, float] | None = None,
        risk_amount: float = 1.0,
        goal_pct: float = 3.0,
    ):
        self.m1 = m1.sort_index()
        self.pack = build_mtf_pack(self.m1)
        self.decide_every = max(1, int(decide_every))
        self.dials = dict(dials or DEFAULT_DIALS)
        self.risk_amount = float(risk_amount)
        self.goal_pct = float(goal_pct)
        self.position: Optional[Direction] = None
        self.entry_price: float = 0.0
        self.realized = 0.0

    def decision_indices(self) -> List[int]:
        n = len(self.m1)
        start = min(max(120, self.decide_every), max(0, n - 1))
        return list(range(start, n, self.decide_every))

    def observe(self, t: int, trade_side: Optional[Direction] = None) -> np.ndarray:
        """Channel 1 obs at bar t without mutating position state."""
        ts = self.m1.index[t]
        phase = float(t) / float(max(len(self.m1) - 1, 1))
        progress = float(np.clip(self.realized / max(self.goal_pct, 1e-6), -1.0, 1.0))
        perc = build_perception_at(
            self.pack, ts,
            trade_side=trade_side if trade_side is not None else self.position,
            progress_to_goal=progress,
            danger=0.0,
            session_phase=phase,
        )
        return perc["obs"]

    def step(self, t: int, action: int) -> DayStepResult:
        ts = self.m1.index[t]
        price = float(self.m1["close"].iloc[t])
        phase = float(t) / float(max(len(self.m1) - 1, 1))
        progress = float(np.clip(self.realized / max(self.goal_pct, 1e-6), -1.0, 1.0))

        side = action_to_trade_side(action)
        perc = build_perception_at(
            self.pack, ts,
            trade_side=side if side is not None else self.position,
            progress_to_goal=progress,
            danger=0.0,
            session_phase=phase,
        )
        struct: StructureFlags = perc["structure"]
        cl: Optional[Classification] = perc["classification"]
        reward = 0.0
        tag = TradeTag.WITH_VECTOR

        if side is not None and self.position is None:
            # open
            if cl is not None and cl.tag == TradeTag.MINDLESS:
                reward = credit(TradeTag.MINDLESS, 0.0, self.risk_amount, self.dials)
                tag = TradeTag.MINDLESS
            else:
                self.position = side
                self.entry_price = price
                tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
                reward = 0.0
        elif int(action) == ACTION_HOLD and self.position is None:
            # flat: inactivity if setup active (decision #5)
            probe_side = (
                perc["higher"]
                if perc["higher"] != Direction.NEUTRAL
                else Direction.BULL
            )
            probe = build_perception_at(
                self.pack, ts, trade_side=probe_side,
                progress_to_goal=progress, danger=0.0, session_phase=phase,
            )
            probe_cl = probe["classification"]
            if setup_active(probe_cl, probe["structure"]):
                reward = inactivity_penalty(self.dials)
                tag = probe_cl.tag if probe_cl is not None else TradeTag.WITH_VECTOR
            else:
                reward = 0.0
        elif int(action) == ACTION_HOLD and self.position is not None:
            reward = 0.0
            tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
        elif side is not None and self.position is not None:
            # close / reverse
            sign = 1.0 if self.position == Direction.BULL else -1.0
            pnl = sign * (price - self.entry_price)
            close_tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
            if cl is not None and cl.mindless:
                reward = credit(TradeTag.MINDLESS, pnl, self.risk_amount, self.dials)
                tag = TradeTag.MINDLESS
            else:
                reward = credit(close_tag, pnl, self.risk_amount, self.dials)
                tag = close_tag
            self.realized += pnl
            if side != self.position:
                self.position = side
                self.entry_price = price
            else:
                # same side again → close only
                self.position = None
                self.entry_price = 0.0

        if cl is None:
            cl = Classification(
                tag=tag,
                mindless=(tag == TradeTag.MINDLESS),
                reasons=("derived",),
            )

        return DayStepResult(
            t=t,
            ts=ts,
            obs=perc["obs"],
            action=int(action),
            trade_side=side,
            tag=tag,
            reward=float(reward),
            classification=cl,
            structure=struct,
            info={
                "price": price,
                "realized": self.realized,
                "position": self.position,
            },
        )
