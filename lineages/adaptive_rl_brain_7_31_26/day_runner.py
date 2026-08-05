"""Day-level runner: multi-TF → perception → Channel 1 → rewards.

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 5 — WHY: first trainable day loop for the lineage.
  Parallel only; no PROVEN.
- 2026-07-31  anti-hold collapse — WHY: setup-hold penalty floor, correct-side
  entry bonus, end-of-day did-nothing penalty. Parallel only; no PROVEN.
- 2026-07-31  signal-agent majority idle — WHY: if >half of signal agents agree
  on one side and policy HOLDs flat, add MAJORITY_IDLE_PENALTY. Parallel only.
- 2026-07-31  active consensus — WHY: ≥20 active agents, >70% agree → idle
  penalty unless ≥2 trades already open. Scale-in same side counts as open.
- 2026-07-31  thrash control — WHY: max open units, reverse cooldown, flip
  tax so greedy cannot open/close every bar. Parallel only; no PROVEN.

Flow per decision bar (M1 step or stride):
  1. asof multi-TF bars from pack
  2. official/sub confluence via live indicators
  3. structure + classify for a proposed/active trade side
  4. Channel 1 obs
  5. reward: credit on close; inactivity if setup active & policy flat
  6. end_day(): large penalty once if no entries and realized ~0
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
from lineages.adaptive_rl_brain_7_31_26.perception.confluence import (
    confluence_from_confirmation_flags,
    majority_direction,
)
from lineages.adaptive_rl_brain_7_31_26.perception.live_indicators import (
    GROUP_KEYS,
    dual_confirmation_flags,
    dual_flags_to_confluence_kwargs,
    group_flags_on_tf,
    indicator_frame,
    snapshot_at,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import build_channel1_obs
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS, SUB_SETS
from lineages.adaptive_rl_brain_7_31_26.perception.mark_doctrine import (
    doctrine_action_from_perception,
)
from lineages.adaptive_rl_brain_7_31_26.perception.mark_sets_opportunity import (
    mark_dir_to_action,
    scan_mark_opportunities,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import OFFICIAL_SETS
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
    FLIP_FLOP_PENALTY,
    MAJORITY_AGREE_FRAC,
    MAJORITY_IDLE_PENALTY,
    MAJORITY_MIN_ACTIVE,
    MAJORITY_MIN_OPEN_EXEMPT,
    MAX_OPEN_UNITS,
    REVERSE_COOLDOWN_BARS,
    SECOND_BEST_FEE_PX,
    SECOND_BEST_HORIZON_BARS,
    correct_side_entry_bonus,
    credit,
    did_nothing_eod_penalty,
    directional_hold_penalty,
    flat_hold_tax,
    flip_flop_penalty,
    majority_agents_idle_penalty,
    missed_opportunity_penalty,
    second_best_entry_from_logits,
    setup_hold_penalty,
    structure_match_entry_bonus,
)
from lineages.adaptive_rl_brain_7_31_26.signal_majority import (
    compute_panel_matrix,
    majority_at,
    panel_summary,
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


def _confluence_from_ind(
    ind: Mapping[str, pd.DataFrame],
    conf_a: str,
    conf_b: str,
    ts: pd.Timestamp,
    set_key: str,
) -> SetConfluence:
    """Fast path: precomputed indicator frames + asof snapshots."""
    fa = ind.get(conf_a)
    fb = ind.get(conf_b)
    if fa is None or fb is None or len(fa) == 0 or len(fb) == 0:
        return _neutral_conf(set_key)
    ia, ib = bar_asof(fa, ts), bar_asof(fb, ts)
    # Warmup: need enough bars for CCI/RSI/channel (was 20; 12 still stable
    # and lets 1h/4h engage on multi-hour curriculum days).
    if ia < 12 or ib < 12:
        return _neutral_conf(set_key)
    try:
        sa = snapshot_at(fa, conf_a, ia)
        sb = snapshot_at(fb, conf_b, ib)
        dual = dual_confirmation_flags(sa, sb)
        kw = dual_flags_to_confluence_kwargs(dual)
        return confluence_from_confirmation_flags(set_key, **kw)
    except Exception:
        return _neutral_conf(set_key)


def _entry_dir_from_ind(
    ind: Mapping[str, pd.DataFrame],
    entry_tf: str,
    ts: pd.Timestamp,
) -> Direction:
    fr = ind.get(entry_tf)
    if fr is None or len(fr) == 0:
        return Direction.NEUTRAL
    i = bar_asof(fr, ts)
    if i < 12:
        return Direction.NEUTRAL
    try:
        snap = snapshot_at(fr, entry_tf, i)
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
    ind: Mapping[str, pd.DataFrame] | None = None,
) -> Dict[str, Any]:
    """Official/sub confluence + structure + optional classify + Channel 1 obs.

    Pass `ind` = precomputed indicator_frame(pack[tf]) for a big speedup.
    """
    if ind is None:
        ind = {tf: indicator_frame(df) for tf, df in pack.items() if len(df)}

    official: Dict[int, SetConfluence] = {}
    for s in OFFICIAL_SETS:
        c0, c1 = s.confirmation_tfs
        official[s.set_id] = _confluence_from_ind(
            ind, c0, c1, ts, f"official:{s.set_id}",
        )
    subs: Dict[str, SetConfluence] = {}
    for s in SUB_SETS:
        subs[s.sub_id] = _confluence_from_ind(
            ind, s.entry_tf, s.confirmation_tf, ts, f"sub:{s.sub_id}",
        )

    # Per-set LTF entry dirs (Mark scans all 4 stacks — not hard-coded 5m only)
    entry_dirs: Dict[int, Direction] = {}
    for s in OFFICIAL_SETS:
        entry_dirs[s.set_id] = _entry_dir_from_ind(ind, s.entry_tf, ts)

    mark_opp = scan_mark_opportunities(official, entry_dirs)

    # Legacy claim path: collapsed to Official Set 2 (5m) only — kept for scoreboards
    primary = official.get(2) or next(iter(official.values()))
    lower_legacy = entry_dirs.get(2, Direction.NEUTRAL)
    higher_legacy = primary.direction
    struct_legacy = structure_flags(
        higher_direction=higher_legacy, lower_direction=lower_legacy
    )

    # Mark eyes: best aligned set's HTF + LTF (or macro if no best)
    if mark_opp.best is not None:
        higher = mark_opp.best.htf_dir
        lower = mark_opp.best.ltf_dir
        primary_mark = official.get(mark_opp.best.set_id, primary)
    else:
        # Fall back to strongest HTF by set weight preference: set4 → set1
        higher = Direction.NEUTRAL
        lower = Direction.NEUTRAL
        primary_mark = primary
        for sid in (4, 3, 2, 1):
            c = official.get(sid)
            if c is not None and c.direction != Direction.NEUTRAL:
                higher = c.direction
                lower = entry_dirs.get(sid, Direction.NEUTRAL)
                primary_mark = c
                break
    struct = structure_flags(higher_direction=higher, lower_direction=lower)

    cl: Optional[Classification] = None
    if trade_side is not None and trade_side != Direction.NEUTRAL:
        m = make_mindless_inputs(
            trade_side,
            turned=(lower == trade_side and lower != Direction.NEUTRAL),
            velocity_confirms=True,
            higher_weakening=struct.pullback or primary_mark.velocity in (
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
        "structure_legacy_set2": struct_legacy,
        "higher": higher,
        "lower": lower,
        "higher_legacy_set2": higher_legacy,
        "lower_legacy_set2": lower_legacy,
        "entry_dirs": entry_dirs,
        "mark_opportunity": mark_opp,
        "classification": cl,
        "obs": obs,
        "primary": primary_mark,
        "primary_legacy_set2": primary,
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
        use_signal_majority: bool = True,
        majority_idle_penalty: float = MAJORITY_IDLE_PENALTY,
        majority_min_active: int = MAJORITY_MIN_ACTIVE,
        majority_agree_frac: float = MAJORITY_AGREE_FRAC,
        majority_min_open_exempt: int = MAJORITY_MIN_OPEN_EXEMPT,
        max_open_units: int = MAX_OPEN_UNITS,
        reverse_cooldown_bars: int = REVERSE_COOLDOWN_BARS,
        flip_flop_penalty_val: float = FLIP_FLOP_PENALTY,
        # Eyes:
        #   legacy_set2   = claim (Official Set 2 only)
        #   mark_all_sets = multi-set aligned scan
        #   mark_doctrine = full five laws (FORCE→REGIME→VELOCITY→ENTRY)
        eyes_mode: str = "legacy_set2",
    ):
        self.m1 = m1.sort_index()
        self.pack = build_mtf_pack(self.m1)
        mode = str(eyes_mode or "legacy_set2").strip().lower()
        if mode not in ("legacy_set2", "mark_all_sets", "mark_doctrine"):
            mode = "legacy_set2"
        self.eyes_mode = mode
        self.last_doctrine = None  # type: ignore
        # Precompute indicators once per TF (huge speedup vs recompute every step)
        self.ind = {
            tf: indicator_frame(df)
            for tf, df in self.pack.items()
            if df is not None and len(df) > 0
        }
        self.decide_every = max(1, int(decide_every))
        self.dials = dict(dials or DEFAULT_DIALS)
        self.risk_amount = float(risk_amount)
        self.goal_pct = float(goal_pct)
        self.position: Optional[Direction] = None
        self.entry_price: float = 0.0
        self.realized = 0.0
        self.n_entries = 0
        self.n_open = 0  # concurrent open trades (scale-in increases this)
        self._eod_applied = False
        # Thrash control
        self.max_open_units = max(1, int(max_open_units))
        self.reverse_cooldown_bars = max(0, int(reverse_cooldown_bars))
        self.flip_flop_penalty_val = float(flip_flop_penalty_val)
        self.cooldown_until_t = -1  # M1 index; reverse blocked while t < this
        self.n_reverses = 0
        self.n_scale_blocks = 0
        self.n_cooldown_blocks = 0
        # Signal-agent majority panel: ALL filled slots from configs/signal_slots.yaml
        # (92 agents). Precomputed once per day with features.engine.
        self.use_signal_majority = bool(use_signal_majority)
        self.majority_idle_penalty = float(majority_idle_penalty)
        self.majority_min_active = int(majority_min_active)
        self.majority_agree_frac = float(majority_agree_frac)
        self.majority_min_open_exempt = int(majority_min_open_exempt)
        if self.use_signal_majority:
            self.agent_matrix, self.agent_names = compute_panel_matrix(
                self.m1,
                only_enabled=False,  # all 92 filled slots
            )
        else:
            self.agent_matrix = np.zeros((len(self.m1), 0), dtype=np.float32)
            self.agent_names = []
        self.agent_panel_info = panel_summary(self.agent_matrix, self.agent_names)

    def _apply_majority_idle(
        self, t: int, reward: float, info_extra: Dict[str, Any]
    ) -> float:
        """Add consensus idle penalty when HOLD and n_open < exempt floor."""
        if not (self.use_signal_majority and self.agent_matrix.size):
            return reward
        maj = majority_at(
            self.agent_matrix,
            t,
            min_active=self.majority_min_active,
            agree_frac=self.majority_agree_frac,
        )
        info_extra["majority_n_bull"] = maj.n_bull
        info_extra["majority_n_bear"] = maj.n_bear
        info_extra["majority_n_agents"] = maj.n_agents
        info_extra["majority_n_active"] = maj.n_active
        info_extra["majority_agree_frac"] = maj.agree_frac
        info_extra["majority_has"] = maj.has_majority
        info_extra["n_open"] = self.n_open
        maj_pen = majority_agents_idle_penalty(
            has_majority=maj.has_majority,
            action_is_hold=True,
            n_open=self.n_open,
            min_open_exempt=self.majority_min_open_exempt,
            penalty=self.majority_idle_penalty,
        )
        if maj_pen != 0.0:
            reward = float(reward) + float(maj_pen)
            info_extra["majority_idle"] = True
            info_extra["majority_dir"] = maj.direction.name
            info_extra["majority_penalty"] = maj_pen
        return float(reward)

    def decision_indices(self) -> List[int]:
        n = len(self.m1)
        # ~12×1h bars of M1 so Official Set 2 (30m/1h) can leave NEUTRAL
        warm = 720
        start = min(max(warm, self.decide_every), max(0, n - 1))
        if start >= n:
            start = max(0, n // 4)
        return list(range(start, n, self.decide_every))

    def observe(self, t: int, trade_side: Optional[Direction] = None) -> np.ndarray:
        """Channel 1 obs at bar t without mutating position state."""
        return self.perceive(t, trade_side=trade_side)["obs"]

    def perceive(
        self, t: int, trade_side: Optional[Direction] = None
    ) -> Dict[str, Any]:
        """Full perception dict at bar t (obs + higher/lower/structure/...)."""
        ts = self.m1.index[t]
        phase = float(t) / float(max(len(self.m1) - 1, 1))
        progress = float(np.clip(self.realized / max(self.goal_pct, 1e-6), -1.0, 1.0))
        return build_perception_at(
            self.pack, ts,
            trade_side=trade_side if trade_side is not None else self.position,
            progress_to_goal=progress,
            danger=0.0,
            session_phase=phase,
            ind=self.ind,
        )

    def structure_action_at(self, t: int) -> int:
        """Structure-only BUY/SELL/HOLD at bar t (ignores open position).

        mark_all_sets: scan Official Sets 1–4 (HTF last two, LTF first) and
        take aligned opportunities only (Mark / ENTJ fast logical scalping).
        legacy_set2: old claim path (set 2 HTF + 5m LTF collapse).
        """
        perc = self.perceive(t)
        if self.eyes_mode == "mark_doctrine":
            # Inject goal context so doctrine can refuse soft-scalp on hard targets
            perc = dict(perc)
            perc["target_pct"] = float(getattr(self, "goal_pct", 0.0) or 0.0)
            perc["equity_pct"] = float(getattr(self, "realized", 0.0) or 0.0)
            dec = doctrine_action_from_perception(perc)
            self.last_doctrine = dec
            perc["doctrine"] = dec
            return int(dec.action)
        if self.eyes_mode == "mark_all_sets":
            mark_opp = perc.get("mark_opportunity")
            if mark_opp is not None:
                return int(mark_dir_to_action(mark_opp.action_dir))
            return ACTION_HOLD
        # legacy claim collapse
        direction = perc.get("higher_legacy_set2", perc["higher"])
        if direction == Direction.NEUTRAL:
            direction = perc.get("lower_legacy_set2", perc["lower"])
        if direction == Direction.BULL:
            return ACTION_BUY
        if direction == Direction.BEAR:
            return ACTION_SELL
        return ACTION_HOLD

    def recommended_action(self, t: int) -> int:
        """Heuristic action from structure direction (for guided train / aux loss).

        Flat + no clear direction → HOLD.
        Already in a position → HOLD (manage via later logic) — DayRunner only.
        GoalEquityDay forces flat perception for reverse-on-flip.
        mark_all_sets: multi-set Mark eyes; legacy_set2: set-2 collapse.
        """
        if self.position is not None:
            return ACTION_HOLD
        return int(self.structure_action_at(t))

    def counterfactual_entry_pnl(
        self,
        t: int,
        entry_action: int,
        *,
        horizon_bars: int = SECOND_BEST_HORIZON_BARS,
        fee_px: float = SECOND_BEST_FEE_PX,
    ) -> float:
        """Deterministic CF: enter at close[t], exit at horizon or opposite structure.

        Returns PnL in price units after round-trip fee. Does not mutate state.
        """
        act = int(entry_action)
        if act == ACTION_BUY:
            sign = 1.0
            opposite = ACTION_SELL
        elif act == ACTION_SELL:
            sign = -1.0
            opposite = ACTION_BUY
        else:
            return 0.0
        n = len(self.m1)
        if n < 2 or t < 0 or t >= n:
            return 0.0
        entry = float(self.m1["close"].iloc[t])
        max_t = min(int(t) + max(1, int(horizon_bars)), n - 1)
        exit_t = max_t
        # Early exit on opposite structure (decision stride only — cheap + deterministic)
        stride = max(1, int(self.decide_every))
        for j in range(int(t) + stride, max_t + 1, stride):
            if int(self.structure_action_at(j)) == opposite:
                exit_t = int(j)
                break
        exit_px = float(self.m1["close"].iloc[exit_t])
        gross = sign * (exit_px - entry)
        return float(gross) - abs(float(fee_px))

    def end_day(self) -> float:
        """Apply once-per-day did-nothing penalty. Returns 0 if already applied."""
        if self._eod_applied:
            return 0.0
        self._eod_applied = True
        return float(
            did_nothing_eod_penalty(
                self.realized,
                self.n_entries,
            )
        )

    def _coerce_thrash(self, t: int, action: int) -> tuple[int, Optional[Direction], Dict[str, Any]]:
        """Hard thrash limits: max units, reverse cooldown. Returns (action, side, flags)."""
        flags: Dict[str, Any] = {}
        side = action_to_trade_side(action)
        # Cap scale-ins: force HOLD when already at max concurrent units
        if (
            side is not None
            and self.position is not None
            and side == self.position
            and int(self.n_open) >= int(self.max_open_units)
        ):
            flags["scale_blocked"] = True
            self.n_scale_blocks += 1
            return ACTION_HOLD, None, flags
        # Reverse cooldown: block opposite entry until cooldown_until_t
        if (
            side is not None
            and self.position is not None
            and side != self.position
            and int(t) < int(self.cooldown_until_t)
        ):
            flags["cooldown_block"] = True
            self.n_cooldown_blocks += 1
            return ACTION_HOLD, None, flags
        # Flat entry also blocked during cooldown (prevent re-flip spam)
        if (
            side is not None
            and self.position is None
            and int(t) < int(self.cooldown_until_t)
        ):
            flags["cooldown_block"] = True
            self.n_cooldown_blocks += 1
            return ACTION_HOLD, None, flags
        return int(action), side, flags

    def step(
        self,
        t: int,
        action: int,
        logits: Optional[np.ndarray] = None,
    ) -> DayStepResult:
        ts = self.m1.index[t]
        price = float(self.m1["close"].iloc[t])
        phase = float(t) / float(max(len(self.m1) - 1, 1))
        progress = float(np.clip(self.realized / max(self.goal_pct, 1e-6), -1.0, 1.0))

        was_flat = self.position is None
        # Structure rec while flat (BUY/SELL/HOLD). Used for directional shaping.
        # recommended_action returns HOLD when already in a position — skip then.
        structure_rec = int(self.recommended_action(t)) if was_flat else ACTION_HOLD

        action, side, thrash_flags = self._coerce_thrash(t, int(action))
        perc = build_perception_at(
            self.pack, ts,
            trade_side=side if side is not None else self.position,
            progress_to_goal=progress,
            danger=0.0,
            session_phase=phase,
            ind=self.ind,
        )
        struct: StructureFlags = perc["structure"]
        cl: Optional[Classification] = perc["classification"]
        higher: Direction = perc["higher"]
        reward = 0.0
        tag = TradeTag.WITH_VECTOR
        info_extra: Dict[str, Any] = dict(thrash_flags)
        info_extra["structure_rec"] = int(structure_rec)

        if side is not None and self.position is None:
            # open first trade
            if cl is not None and cl.tag == TradeTag.MINDLESS:
                reward = credit(TradeTag.MINDLESS, 0.0, self.risk_amount, self.dials)
                tag = TradeTag.MINDLESS
            else:
                self.position = side
                self.entry_price = price
                self.n_entries += 1
                self.n_open = 1
                tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
                bonus = correct_side_entry_bonus(tag, side, higher)
                reward = float(bonus)
                info_extra["entry_bonus"] = bonus
                info_extra["entry"] = True
                info_extra["scale_in"] = False
                # Bake HOLD-fix: matching structure side gets clear positive
                smb = structure_match_entry_bonus(
                    is_flat=True,
                    action=int(action),
                    structure_rec=structure_rec,
                )
                if smb > 0.0:
                    reward = float(reward) + float(smb)
                    info_extra["structure_match_bonus"] = float(smb)
        elif int(action) == ACTION_HOLD:
            # idle: base hold tax when flat; consensus idle unless ≥2 open
            if self.position is None:
                probe_side = (
                    perc["higher"]
                    if perc["higher"] != Direction.NEUTRAL
                    else Direction.BULL
                )
                probe = build_perception_at(
                    self.pack, ts, trade_side=probe_side,
                    progress_to_goal=progress, danger=0.0, session_phase=phase,
                    ind=self.ind,
                )
                probe_cl = probe["classification"]
                if setup_active(probe_cl, probe["structure"]):
                    probe_tag = (
                        probe_cl.tag if probe_cl is not None else TradeTag.WITH_VECTOR
                    )
                    reward = setup_hold_penalty(probe_tag, self.dials)
                    tag = probe_tag
                    info_extra["inactivity"] = True
                else:
                    reward = flat_hold_tax()
                    tag = TradeTag.QUALIFIED_MICRO
                    info_extra["flat_hold_tax"] = True
                # Bake HOLD-fix: structure wants a side but policy HOLDs
                dhp = directional_hold_penalty(
                    is_flat=True,
                    action=ACTION_HOLD,
                    structure_rec=structure_rec,
                )
                if dhp < 0.0:
                    # Take the more severe (more negative) of inactivity vs directional
                    reward = float(min(float(reward), float(dhp)))
                    info_extra["directional_hold_penalty"] = float(dhp)
                # Second-best logit regret: CF 2nd-best entry; penalize if profitable
                if logits is not None:
                    sb = second_best_entry_from_logits(logits)
                    if sb is not None:
                        cf_pnl = float(
                            self.counterfactual_entry_pnl(t, int(sb))
                        )
                        mop = missed_opportunity_penalty(cf_pnl_after_fees=cf_pnl)
                        info_extra["second_best_action"] = int(sb)
                        info_extra["cf_pnl_after_fees"] = cf_pnl
                        info_extra["second_best_profitable"] = bool(cf_pnl > 0.0)
                        if mop < 0.0:
                            reward = float(reward) + float(mop)
                            info_extra["second_best_regret"] = float(mop)
            else:
                reward = 0.0
                tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
            reward = self._apply_majority_idle(t, reward, info_extra)
        elif side is not None and self.position is not None:
            if side == self.position:
                # same side → scale-in (another open trade), keep average entry
                prev_n = max(int(self.n_open), 1)
                self.entry_price = (
                    float(self.entry_price) * prev_n + price
                ) / float(prev_n + 1)
                self.n_open = prev_n + 1
                self.n_entries += 1
                tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
                if cl is not None and cl.tag == TradeTag.MINDLESS:
                    reward = credit(TradeTag.MINDLESS, 0.0, self.risk_amount, self.dials)
                    tag = TradeTag.MINDLESS
                else:
                    bonus = correct_side_entry_bonus(tag, side, higher)
                    reward = float(bonus)
                    info_extra["entry_bonus"] = bonus
                info_extra["entry"] = True
                info_extra["scale_in"] = True
            else:
                # opposite side → close all open trades, then open 1 reverse
                sign = 1.0 if self.position == Direction.BULL else -1.0
                # PnL on average entry × notional units (n_open)
                units = max(int(self.n_open), 1)
                pnl = sign * (price - self.entry_price) * float(units)
                close_tag = cl.tag if cl is not None else TradeTag.WITH_VECTOR
                if cl is not None and cl.mindless:
                    reward = credit(TradeTag.MINDLESS, pnl, self.risk_amount, self.dials)
                    tag = TradeTag.MINDLESS
                else:
                    reward = credit(close_tag, pnl, self.risk_amount, self.dials)
                    tag = close_tag
                self.realized += pnl
                self.position = side
                self.entry_price = price
                self.n_entries += 1
                self.n_open = 1
                if tag != TradeTag.MINDLESS:
                    bonus = correct_side_entry_bonus(tag, side, higher)
                    reward = float(reward) + float(bonus)
                    info_extra["entry_bonus"] = bonus
                # thrash: flip tax + arm cooldown before next reverse/entry
                flip = flip_flop_penalty(penalty=self.flip_flop_penalty_val)
                reward = float(reward) + float(flip)
                info_extra["flip_flop_penalty"] = flip
                self.cooldown_until_t = int(t) + int(self.reverse_cooldown_bars)
                self.n_reverses += 1
                info_extra["reverse"] = True
                info_extra["cooldown_until_t"] = self.cooldown_until_t

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
                "n_entries": self.n_entries,
                "n_open": self.n_open,
                **info_extra,
            },
        )
