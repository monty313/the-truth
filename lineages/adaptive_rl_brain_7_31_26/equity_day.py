"""Goal-conditioned equity day engine for multi-pair consistency.

CHANGE LOG:
- 2026-08-05  MARK SOUL — goal-relative size + force-aligned adds on
  mark_doctrine path only. WHY: diagnosis 10/10 winnable when Mark can
  change lots and add with HTF force; fixed shell 7/10 understated him.
  Not the banned trail+cushion+scale-in package (IRAC-01). Claim path
  (legacy_set2) keeps fixed dials, no adds.
- 2026-08-04  Mark-clone attention gates (optional) — WHY: day walks show
  hard-target thrash = single-bar reverse + stop→reflip. Gates default OFF
  so multi-pair claim path is unchanged. See MARK_CLONE_AS_POLICY_ISSUES.md.
- 2026-07-31  multi-pair equity sim — WHY: lineage needed clear/breach in %
  of equity (like prove_it / FastSim), not raw price points. Same brain
  must solve any (target%, risk%) at runtime. Parallel only; no PROVEN.
- 2026-07-31  REVERT trail/cushion/scale-in — WHY: IRAC-01 dropped 6/10 pass
  to 0/10 (more breaches, fewer high-target clears). Restored first good engine.

Physics (honest, simplified Shell subset):
  - equity % = 100 * (balance + unrealized - eq0) / eq0
  - clear = final equity% >= target% AND never breached
  - breach = min equity% (intrabar worst on open risk) <= -risk%
  - bank: flatten and stop when equity% >= target%
  - heat: refuse open if open_risk + new_risk > distance to floor
  - per-trade risk capped (default 0.25% of eq0; Mark soul may raise)

Policy interface: callable(obs, state) -> action in {0 hold, 1 buy, 2 sell}
or use heuristic_decode / policy act.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    action_to_trade_side,
)

EQ0_DEFAULT = 100_000.0
PER_TRADE_CAP_PCT = 0.25  # of equity, like shell
POINT_SIZE = 0.01
CONTRACT = 100.0  # gold oz per lot unit scale for units math


@dataclass
class EquityDayResult:
    date: str
    target_pct: float
    risk_pct: float
    pnl_pct: float
    min_eq_pct: float
    goal_hit: bool
    breached: bool
    cleared: bool
    n_entries: int
    n_closes: int
    banked: bool
    hold_rate: float
    mean_reward: float = 0.0
    info: Dict[str, Any] = field(default_factory=dict)


def _atr_like(high: np.ndarray, low: np.ndarray, close: np.ndarray, n: int = 14) -> float:
    """Simple ATR at end of window (points)."""
    if len(close) < 2:
        return max(float(close[-1]) * 0.001, 0.5) if len(close) else 1.0
    prev = close[:-1]
    h = high[1:]
    l = low[1:]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    window = tr[-n:] if len(tr) >= n else tr
    atr = float(np.mean(window)) if len(window) else float(np.mean(tr))
    return max(atr, 0.5)


class GoalEquityDay:
    """One real M1 day with target% / risk% runtime inputs and equity scoring."""

    def __init__(
        self,
        m1: pd.DataFrame,
        *,
        target_pct: float,
        risk_pct: float,
        eq0: float = EQ0_DEFAULT,
        decide_every: int = 25,
        per_trade_cap_pct: float = PER_TRADE_CAP_PCT,
        stop_atr_mult: float = 2.0,
        risk_use_frac: float = 0.35,
        use_signal_majority: bool = False,
        date_str: str | None = None,
        # --- Mark-clone attention (defaults = claim path, all off) ---
        mark_clone: bool = False,
        sig_confirm_decisions: int = 1,
        post_stop_cooldown_decisions: int = 0,
        max_entries_day: int = 0,
        refuse_open_on_pullback: bool = False,
        # Eyes: default legacy keeps multi-pair claim. mark_all_sets = ENTJ 4-set scan.
        eyes_mode: str = "legacy_set2",
        # Mark soul: goal-relative size + force-aligned adds (mark_doctrine default ON)
        mark_soul: Optional[bool] = None,
        mark_soul_max_units: int = 3,
        # Full Mark clone eyes: Channel1 + doctrine + 92 agents + self (168-dim)
        full_obs: bool = False,
        # When True, pure greedy policy actions are Mark-aligned (soul=policy sense)
        mark_align_policy: Optional[bool] = None,
    ):
        self.m1 = m1.sort_index()
        self.target = float(target_pct)
        self.risk = float(risk_pct)
        self.eq0 = float(eq0)
        self.decide_every = max(1, int(decide_every))
        self.cap = float(per_trade_cap_pct) / 100.0
        self._base_cap_pct = float(per_trade_cap_pct)
        self.stop_atr_mult = float(stop_atr_mult)
        self.risk_use_frac = float(risk_use_frac)
        self._base_risk_use_frac = float(risk_use_frac)
        self.date_str = date_str or str(self.m1.index[0].date())

        # Mark-clone attention dials (never touch shell math)
        self.mark_clone = bool(mark_clone)
        self.sig_confirm_decisions = max(1, int(sig_confirm_decisions))
        self.post_stop_cooldown_decisions = max(0, int(post_stop_cooldown_decisions))
        self.max_entries_day = max(0, int(max_entries_day))  # 0 = unlimited
        self.refuse_open_on_pullback = bool(refuse_open_on_pullback)
        # Mark-on-chart path → five-law doctrine over MARK SETS LAW (all 4 sets).
        # legacy_set2 remains claim baseline only — never call it Mark.
        em = str(eyes_mode or "legacy_set2").strip().lower()
        if self.mark_clone and em == "legacy_set2":
            em = "mark_doctrine"
        if em not in ("legacy_set2", "mark_all_sets", "mark_doctrine"):
            em = "legacy_set2"
        self.eyes_mode = em
        # Soul: on by default for mark_doctrine (policy = Mark on chart + size/adds)
        if mark_soul is None:
            self.mark_soul = em == "mark_doctrine"
        else:
            self.mark_soul = bool(mark_soul)
        self.mark_soul_max_units = max(1, int(mark_soul_max_units))
        self.n_adds = 0
        self._soul_flips = 0
        self.full_obs = bool(full_obs)
        # Default ON for mark_doctrine: policy must share Mark wait/side sense
        if mark_align_policy is None:
            self.mark_align_policy = self.eyes_mode == "mark_doctrine"
        else:
            self.mark_align_policy = bool(mark_align_policy)
        self._sig_panel: Optional[np.ndarray] = None  # (n_bars, n_agents)
        self._sig_names: List[str] = []
        # Enforce set stacks at Mark construction (cheap; fails if law broken)
        if self.eyes_mode in ("mark_doctrine", "mark_all_sets"):
            from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
                assert_mark_sets_law,
            )

            assert_mark_sets_law()

        # Perception runner for obs / recommended direction (no its own PnL %)
        self.runner = DayRunner(
            self.m1,
            decide_every=self.decide_every,
            risk_amount=1.0,
            goal_pct=self.target,
            use_signal_majority=use_signal_majority,
            eyes_mode=self.eyes_mode,
        )

        self.balance = self.eq0
        self.side: Optional[int] = None  # +1 long, -1 short
        self.units: float = 0.0
        self.avg: float = 0.0
        self.stop: float = 0.0
        self.n_entries = 0
        self.n_closes = 0
        self.n_open_units = 0
        self.min_eq_pct = 0.0
        self.banked = False
        self.breached = False
        self.dead = False
        self.actions: List[int] = []
        self.rewards: List[float] = []

        # Attention state (Mark clone)
        self._pending_sig: Optional[int] = None  # ACTION_BUY / SELL under confirm
        self._pending_count: int = 0
        self._decisions_since_stop: int = 10**9  # large = no recent stop
        self._n_decisions: int = 0

        self._high = self.m1["high"].astype(float).values
        self._low = self.m1["low"].astype(float).values
        self._close = self.m1["close"].astype(float).values
        sp_col = "spread" if "spread" in self.m1.columns else None
        if sp_col:
            self._spread_px = self.m1[sp_col].astype(float).values * POINT_SIZE
        else:
            self._spread_px = np.full(len(self.m1), 0.25, dtype=float)

    def equity_pct(self, price: float) -> float:
        ur = 0.0
        if self.side is not None and self.units > 0:
            ur = float(self.side) * self.units * (price - self.avg)
        return 100.0 * (self.balance + ur - self.eq0) / self.eq0

    def _open_risk_frac(self, price: float, sp: float) -> float:
        if self.side is None or self.units <= 0:
            return 0.0
        if self.side > 0:
            loss_per = max(0.0, (self.avg - self.stop) + sp)
        else:
            loss_per = max(0.0, (self.stop - self.avg) + sp)
        return float(self.units * loss_per) / self.eq0

    def _worst_eq_pct(self, hi: float, lo: float, sp: float) -> float:
        w = self.balance - self.eq0
        if self.side is not None and self.units > 0:
            if self.side > 0:
                px = (self.stop if lo <= self.stop else lo) - sp
                w += self.units * (px - self.avg)
            else:
                px = (self.stop if hi >= self.stop else hi) + sp
                w += self.units * (self.avg - px)
        return 100.0 * w / self.eq0

    def _flatten(self, price: float, sp: float) -> float:
        if self.side is None or self.units <= 0:
            return 0.0
        fill = price - sp if self.side > 0 else price + sp
        pnl = float(self.side) * self.units * (fill - self.avg)
        self.balance += pnl
        self.side = None
        self.units = 0.0
        self.avg = 0.0
        self.stop = 0.0
        self.n_open_units = 0
        self.n_closes += 1
        self.runner.position = None
        self.runner.n_open = 0
        return pnl

    def mark_soul_size_dials(self, price: float) -> Tuple[float, float]:
        """Mark soul: (risk_use_frac, per_trade_cap_pct) from remaining goal vs floor.

        Diagnosis: fixed 0.35/0.25 understates Mark; force + flexible size → 10/10.
        Online rule (no future peek): size up when lagging and heat allows;
        size down near bank or near floor. Not trail/cushion.
        Plan teacher may lock dials via _plan_lock_ruf / _plan_lock_cap.
        """
        if getattr(self, "_plan_lock_ruf", None) is not None:
            return float(self._plan_lock_ruf), float(
                getattr(self, "_plan_lock_cap", self._base_cap_pct)
            )
        eq = self.equity_pct(price)
        rem = max(0.0, self.target - eq)
        room = max(1e-6, eq + self.risk)  # % points above floor
        progress = eq / max(self.target, 1e-6)
        danger = max(0.0, -eq) / max(self.risk, 1e-6)
        hardness = self.target / max(self.risk, 1e-6)
        pressure = rem / room

        if danger >= 0.55:
            return 0.25, 0.18
        # Almost banked with crumbs left — still push (don't shrink to death)
        if progress >= 0.92 and rem <= 0.12:
            return 0.28, 0.20
        if progress >= 0.70 and rem > 0.12:
            # finishing push toward target
            ruf = min(1.0, 0.50 + 0.40 * min(1.2, pressure) * (0.7 + 0.3 * hardness))
            cap = min(0.65, 0.32 + 0.28 * min(1.1, pressure))
            return float(ruf), float(cap)

        ruf = self._base_risk_use_frac
        cap = self._base_cap_pct
        # Late-session lag: size harder (Mark would not nibble to death)
        late_boost = 1.0
        if hasattr(self, "_n_decisions") and self._n_decisions > 0:
            if progress < 0.35 and self._n_decisions >= 8:
                late_boost = 1.20
            if progress < 0.25 and self._n_decisions >= 16:
                late_boost = 1.35

        # Soft / tight floors: Mark sizes down (breach days were 6–8 thrash entries)
        soft_floor = float(self.risk) <= 2.5
        if danger >= 0.40:
            return 0.22, 0.15
        if progress < 0.15 and rem > 0.4:
            ruf = min(1.0, late_boost * (0.48 + 0.55 * min(1.5, pressure) * (0.65 + 0.35 * hardness)))
            cap = min(0.70, late_boost * (0.32 + 0.42 * min(1.2, pressure)))
        elif progress < 0.40:
            ruf = min(1.0, late_boost * (0.45 + 0.48 * min(1.3, pressure)))
            cap = min(0.68, late_boost * (0.32 + 0.34 * min(1.1, pressure)))
        elif progress < 0.65:
            ruf = min(0.85, 0.40 + 0.35 * min(1.1, pressure))
            cap = min(0.55, 0.28 + 0.22 * min(1.0, pressure))
        else:
            ruf, cap = float(ruf), float(cap)
        if soft_floor:
            ruf = min(ruf, 0.55)
            cap = min(cap, 0.35)
        return float(min(1.0, ruf)), float(min(0.70, cap))

    def _active_size_dials(self, price: float) -> Tuple[float, float]:
        """(risk_use_frac, cap_as_fraction) for this open/add."""
        if self.mark_soul:
            ruf, cap_pct = self.mark_soul_size_dials(price)
            return float(ruf), float(cap_pct) / 100.0
        return float(self.risk_use_frac), float(self.cap)

    def _try_open(self, side: int, price: float, sp: float, t: int) -> bool:
        if self.dead or self.banked or self.breached:
            return False
        if self.side is not None:
            return False
        eq = self.equity_pct(price)
        dist = max(0.0, (eq - (-self.risk)) / 100.0)
        # Runtime risk% shrinks heat use on tight floors (same code, any pair).
        # risk=2.0 → ~0.80×; risk=3.5 → 1.0×. Does not bake one pair into weights.
        floor_scale = float(np.clip(self.risk / 2.5, 0.72, 1.0))
        ruf, cap = self._active_size_dials(price)
        risk_frac = min(cap, max(0.0, dist * ruf * floor_scale))
        if risk_frac <= 1e-8:
            return False
        atr = _atr_like(self._high[: t + 1], self._low[: t + 1], self._close[: t + 1])
        # Slightly tighter stops when floor is tight (less gap room to breach)
        stop_m = self.stop_atr_mult * float(np.clip(self.risk / 2.5, 0.85, 1.0))
        stop_dist = max(atr * stop_m, price * 0.0008, 0.5)
        units = (risk_frac * self.eq0) / (stop_dist + sp)
        if units <= 0:
            return False
        fill = price + sp if side > 0 else price - sp
        self.side = int(side)
        self.units = float(units)
        self.avg = float(fill)
        self.stop = fill - stop_dist if side > 0 else fill + stop_dist
        self.n_entries += 1
        self.n_open_units = 1
        self.runner.position = Direction.BULL if side > 0 else Direction.BEAR
        self.runner.entry_price = fill
        self.runner.n_entries += 1
        self.runner.n_open = 1
        return True

    def _try_add(self, side: int, price: float, sp: float, t: int) -> bool:
        """Force-aligned scale-in (Mark soul only). Heat-capped; no trail."""
        if not self.mark_soul:
            return False
        if self.dead or self.banked or self.breached:
            return False
        if self.side is None or int(self.side) != int(side) or self.units <= 0:
            return False
        if self.n_open_units >= self.mark_soul_max_units:
            return False
        eq = self.equity_pct(price)
        if eq >= 0.80 * self.target:
            return False
        dist = max(0.0, (eq - (-self.risk)) / 100.0)
        open_r = self._open_risk_frac(price, sp)
        room = max(0.0, dist - open_r)
        floor_scale = float(np.clip(self.risk / 2.5, 0.72, 1.0))
        ruf, cap = self._active_size_dials(price)
        # adds take a partial slice of remaining heat (not full open thrash)
        risk_frac = min(cap * 0.85, max(0.0, room * ruf * floor_scale * 0.70))
        if risk_frac <= 1e-8:
            return False
        atr = _atr_like(self._high[: t + 1], self._low[: t + 1], self._close[: t + 1])
        stop_m = self.stop_atr_mult * float(np.clip(self.risk / 2.5, 0.85, 1.0))
        stop_dist = max(atr * stop_m, price * 0.0008, 0.5)
        # risk of add vs existing stop (stop stays fixed — no trail package)
        fill = price + sp if side > 0 else price - sp
        if side > 0:
            loss_per = max(0.5, float(fill - self.stop) + sp, stop_dist * 0.5)
        else:
            loss_per = max(0.5, float(self.stop - fill) + sp, stop_dist * 0.5)
        add_units = (risk_frac * self.eq0) / max(loss_per, 1e-9)
        if add_units <= 0:
            return False
        # weighted average entry; stop unchanged (no trail package)
        old_u = float(self.units)
        self.avg = (old_u * self.avg + float(add_units) * float(fill)) / (old_u + float(add_units))
        self.units = old_u + float(add_units)
        self.n_entries += 1
        self.n_adds += 1
        self.n_open_units += 1
        self.runner.n_entries += 1
        self.runner.n_open = int(self.n_open_units)
        self.runner.entry_price = self.avg
        return True

    def _maybe_stop(self, hi: float, lo: float, sp: float) -> None:
        if self.side is None:
            return
        hit = False
        fill_px = 0.0
        if self.side > 0 and lo <= self.stop:
            hit = True
            fill_px = min(self.stop, lo) - sp
        elif self.side < 0 and hi >= self.stop:
            hit = True
            fill_px = max(self.stop, hi) + sp
        if hit:
            pnl = float(self.side) * self.units * (fill_px - self.avg)
            self.balance += pnl
            self.side = None
            self.units = 0.0
            self.avg = 0.0
            self.stop = 0.0
            self.n_open_units = 0
            self.n_closes += 1
            self.runner.position = None
            self.runner.n_open = 0
            self._decisions_since_stop = 0  # Mark-clone post-stop cooldown

    def _check_breach_and_bank(self, hi: float, lo: float, price: float, sp: float) -> None:
        worst = self._worst_eq_pct(hi, lo, sp)
        eq = self.equity_pct(price)
        self.min_eq_pct = min(self.min_eq_pct, worst, eq)
        if worst <= -self.risk + 1e-12 or eq <= -self.risk + 1e-12:
            self.breached = True
            self.dead = True
            self._flatten(price, sp)
            return
        if eq >= self.target - 1e-12:
            self._flatten(price, sp)
            eq2 = self.equity_pct(price)
            if eq2 >= self.target - 1e-12:
                self.banked = True

    def _ensure_signal_panel(self) -> None:
        """Lazy full-day 92-agent vote matrix (pattern board for full_obs)."""
        if self._sig_panel is not None:
            return
        try:
            from lineages.adaptive_rl_brain_7_31_26.signal_majority import (
                compute_panel_matrix,
            )

            mat, names = compute_panel_matrix(self.m1, only_enabled=False)
            self._sig_panel = np.asarray(mat, dtype=np.float32)
            self._sig_names = list(names)
        except Exception:
            self._sig_panel = np.zeros((len(self.m1), 0), dtype=np.float32)
            self._sig_names = []

    def observe(self, t: int) -> np.ndarray:
        """Channel1 (+ optional Mark full board) with live progress/danger."""
        price = float(self._close[t])
        eq = self.equity_pct(price)
        progress = float(np.clip(eq / max(self.target, 1e-6), -1.0, 1.0))
        danger = float(np.clip((-eq) / max(self.risk, 1e-6), 0.0, 1.0)) if eq < 0 else 0.0
        self.runner.realized = eq
        self.runner.goal_pct = self.target
        obs = self.runner.observe(t)
        obs = obs.copy()
        obs[29] = progress
        obs[30] = danger
        if not self.full_obs:
            return obs

        # Populate doctrine context (force/regime/play) for the board
        try:
            self._raw_structure_sig(t)
        except Exception:
            pass
        from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import (
            build_mark_full_obs,
            pack_doctrine,
            pack_majority,
            pack_self_state,
        )
        from lineages.adaptive_rl_brain_7_31_26.signal_majority import (
            majority_from_votes,
        )

        self._ensure_signal_panel()
        t_i = int(np.clip(t, 0, max(len(self.m1) - 1, 0)))
        votes = None
        maj_kwargs: Dict[str, Any] = {}
        if self._sig_panel is not None and self._sig_panel.shape[0] > t_i:
            row = self._sig_panel[t_i]
            votes = row
            snap = majority_from_votes(row)
            maj_kwargs = {
                "frac_bull": snap.frac_bull,
                "frac_bear": snap.frac_bear,
                "agree_frac": snap.agree_frac,
                "n_active": float(snap.n_active),
                "n_agents": float(snap.n_agents),
                "has_majority": bool(snap.has_majority),
                "maj_dir": float(
                    1.0
                    if snap.direction == Direction.BULL
                    else (-1.0 if snap.direction == Direction.BEAR else 0.0)
                ),
                "mean_vote": float(np.mean(row)) if row.size else 0.0,
                "std_vote": float(np.std(row)) if row.size else 0.0,
                "n_bull": float(snap.n_bull),
                "n_bear": float(snap.n_bear),
                "n_flat": float(snap.n_flat),
            }
        phase = float(t_i) / float(max(len(self.m1) - 1, 1))
        room = float(eq + self.risk)
        rem = float(self.target - eq)
        self_vec = pack_self_state(
            side=float(self.side or 0),
            n_open_units=float(self.n_open_units),
            n_entries=float(self.n_entries),
            n_adds=float(self.n_adds),
            progress=progress,
            danger=danger,
            target_pct=float(self.target),
            risk_pct=float(self.risk),
            equity_pct=eq,
            room_to_floor=room,
            remaining_to_target=rem,
            mark_soul=1.0 if self.mark_soul else 0.0,
            soul_flips=float(getattr(self, "_soul_flips", 0)),
            session_phase=phase,
            banked=1.0 if self.banked else 0.0,
            in_trade=1.0 if self.side is not None else 0.0,
        )
        dec = getattr(self.runner, "last_doctrine", None)
        return build_mark_full_obs(
            obs,
            doctrine_vec=pack_doctrine(dec),
            majority_vec=pack_majority(**maj_kwargs),
            agent_votes=votes,
            self_vec=self_vec,
        )

    def _raw_structure_sig(self, t: int) -> int:
        """Higher TF (lower fallback) structure signal only — no attention gates."""
        saved = self.runner.position
        self.runner.position = None
        try:
            return int(self.runner.recommended_action(t))
        finally:
            self.runner.position = saved

    def _confirm_sig(self, sig: int) -> int:
        """Require N consecutive same directional signals before acting (Mark clone).

        HOLD never needs confirm. Claim path uses sig_confirm_decisions=1.
        """
        if not self.mark_clone or self.sig_confirm_decisions <= 1:
            return sig
        if sig == ACTION_HOLD:
            self._pending_sig = None
            self._pending_count = 0
            return ACTION_HOLD
        if self._pending_sig == sig:
            self._pending_count += 1
        else:
            self._pending_sig = sig
            self._pending_count = 1
        if self._pending_count >= self.sig_confirm_decisions:
            return sig
        return ACTION_HOLD

    def recommended_action(self, t: int) -> int:
        """Heuristic: trade with higher TF (lower fallback); reverse on flip.

        Same signal when flat or in a trade so we do not freeze on a dead side.
        Banked / dead / breached → HOLD.

        When mark_clone=True, optional attention gates (confirm, post-stop
        cooldown, max entries, pullback refuse) sit *above* structure eyes.
        Shell physics unchanged.
        """
        if self.banked or self.dead or self.breached:
            return ACTION_HOLD

        raw = self._raw_structure_sig(t)
        sig = self._confirm_sig(raw)

        # Post-stop cooldown: refuse new opens (and reverse-as-open) for N decisions
        if (
            self.mark_clone
            and self.post_stop_cooldown_decisions > 0
            and self._decisions_since_stop < self.post_stop_cooldown_decisions
        ):
            if self.side is None:
                return ACTION_HOLD
            # In trade: still allow manage HOLD; block reverse until cooldown ends
            if (self.side > 0 and sig == ACTION_SELL) or (
                self.side < 0 and sig == ACTION_BUY
            ):
                return ACTION_HOLD

        # Max entries / day: stop opening / reversing after cap (manage only)
        if (
            self.mark_clone
            and self.max_entries_day > 0
            and self.n_entries >= self.max_entries_day
        ):
            return ACTION_HOLD

        # Optional: refuse new open when structure pullback (HTF clear, LTF opposite)
        if (
            self.mark_clone
            and self.refuse_open_on_pullback
            and self.side is None
            and sig in (ACTION_BUY, ACTION_SELL)
        ):
            try:
                perc = self.runner.perceive(t)
                struct = perc.get("structure")
                if struct is not None and bool(getattr(struct, "pullback", False)):
                    return ACTION_HOLD
            except Exception:
                pass

        # Mark soul commitment: after enough tries, refuse weak re-opens (anti thrash)
        if self.mark_soul and self.side is None and sig in (ACTION_BUY, ACTION_SELL):
            if self.n_entries >= 6:
                return ACTION_HOLD  # day already spent — no more thrash opens
            if self.n_entries >= 3:
                dec = getattr(self.runner, "last_doctrine", None)
                play = ""
                reason = ""
                if dec is not None:
                    play = str(getattr(dec.play, "value", dec.play) or "").lower()
                    reason = (dec.reason or "").lower()
                quality = (
                    play in ("launch", "aligned")
                    or "tide" in reason
                    or "slingshot" in reason
                )
                if not quality:
                    return ACTION_HOLD
            return sig

        if self.side is None:
            return sig
        # Law 1 in-trade: reverse only when eyes fire opposite side.
        # Mark chart-read: if already working toward target, do NOT reverse on
        # soft/noise flips — only on real opposite multi-set release (doctrine
        # reason contains slingshot_release / tide, not soft_single).
        if self.eyes_mode == "mark_doctrine" and sig in (ACTION_BUY, ACTION_SELL):
            eq = self.equity_pct(float(self._close[t]))
            opposite = (self.side > 0 and sig == ACTION_SELL) or (
                self.side < 0 and sig == ACTION_BUY
            )
            dec = getattr(self.runner, "last_doctrine", None)
            reason = ((dec.reason if dec else "") or "").lower()
            # Working trade: >40% of daily target already — ride, don't thrash reverse
            if eq > 0.40 * self.target:
                if opposite and "soft_single" in reason:
                    return ACTION_HOLD
                if opposite and "slingshot_release" not in reason and "tide" not in reason:
                    # require regime force flip quality for reverse when green
                    if dec is not None and str(getattr(dec.regime, "value", "")) in (
                        "flat_undefined",
                        "chop",
                    ):
                        return ACTION_HOLD
                # Mark soul: when green, refuse weak reverse entirely — ride or add
                if self.mark_soul and opposite and eq > 0.35 * self.target:
                    if "slingshot_release" not in reason and "tide" not in reason:
                        return ACTION_HOLD
            # Mark soul: hard anti-thrash — max 2 flips; after 3 entries no reverse
            if self.mark_soul and opposite:
                if self.n_entries >= 3:
                    return ACTION_HOLD
                if getattr(self, "_soul_flips", 0) >= 2:
                    return ACTION_HOLD
                if "slingshot_release" not in reason and "tide" not in reason:
                    # weak reverse → ride; same-side add may still fire below
                    if eq > -0.45 * self.risk:
                        return ACTION_HOLD
                if "soft_single" in reason:
                    return ACTION_HOLD
        # Mark soul: same-side force while lagging → ADD signal (policy learns this)
        if self.mark_soul and sig in (ACTION_BUY, ACTION_SELL):
            want = +1 if sig == ACTION_BUY else -1
            if want == self.side and self._mark_soul_want_add(t, sig):
                return sig
        if self.side > 0 and sig == ACTION_SELL:
            return ACTION_SELL
        if self.side < 0 and sig == ACTION_BUY:
            return ACTION_BUY
        return ACTION_HOLD

    def _mark_soul_want_add(self, t: int, sig: int) -> bool:
        """True when Mark would add: force still with us, lagging goal, heat OK."""
        if not self.mark_soul or self.side is None:
            return False
        if self.n_open_units >= self.mark_soul_max_units:
            return False
        price = float(self._close[t])
        eq = self.equity_pct(price)
        if eq >= 0.80 * self.target:
            return False
        danger = max(0.0, -eq) / max(self.risk, 1e-6)
        if danger >= 0.50:
            return False
        # need room under floor after open risk
        sp = float(self._spread_px[min(t, len(self._spread_px) - 1)])
        dist = max(0.0, (eq - (-self.risk)) / 100.0)
        if dist - self._open_risk_frac(price, sp) < 0.0015:
            return False
        dec = getattr(self.runner, "last_doctrine", None)
        reason = ((dec.reason if dec else "") or "").lower()
        play = ""
        if dec is not None:
            play = str(getattr(dec.play, "value", dec.play) or "").lower()
        if "soft_single" in reason and eq > 0.25 * self.target:
            return False
        # Prefer launch/aligned; still allow add when clearly lagging under force sig
        if play in ("launch", "aligned"):
            return True
        if "force" in reason or "tide" in reason or "slingshot" in reason:
            return True
        if eq < 0.35 * self.target and self.n_open_units < 2:
            return True
        return False

    def step_action(self, t: int, action: int) -> float:
        """Apply one decision at bar t. Returns shaping reward (for train)."""
        price = float(self._close[t])
        hi = float(self._high[t])
        lo = float(self._low[t])
        sp = float(self._spread_px[t])
        reward = 0.0

        self._maybe_stop(hi, lo, sp)
        self._check_breach_and_bank(hi, lo, price, sp)

        if self.dead or self.banked:
            self.actions.append(ACTION_HOLD)
            self.rewards.append(0.0)
            return 0.0

        action = int(action)
        # Mark soul thrash cap (policy + teacher): after N entries, no new open/reverse
        # Soft targets (≤1.5): cap 4. Else 6. Capital law before edge.
        thrash_cap = int(self.max_entries_day) if self.max_entries_day > 0 else 0
        if thrash_cap <= 0 and self.mark_soul:
            thrash_cap = 4 if float(self.target) <= 1.5 else 6
        if thrash_cap > 0 and self.n_entries >= thrash_cap and self.side is None:
            if action in (ACTION_BUY, ACTION_SELL):
                action = ACTION_HOLD
        if thrash_cap > 0 and self.n_entries >= thrash_cap and self.side is not None:
            # block reverse thrash; same-side may still add if heat allows
            want_tmp = action_to_trade_side(action)
            if want_tmp is not None:
                w = +1 if want_tmp == Direction.BULL else -1
                if w != self.side:
                    action = ACTION_HOLD

        # Mark capital sense: no new risk when already half-way to the floor
        eq_now = self.equity_pct(price)
        danger_now = max(0.0, -eq_now) / max(self.risk, 1e-6)
        if self.mark_soul and self.side is None and action in (ACTION_BUY, ACTION_SELL):
            if danger_now >= 0.45:
                action = ACTION_HOLD

        side = action_to_trade_side(action)

        if self.side is None and side is not None:
            s = +1 if side == Direction.BULL else -1
            opened = self._try_open(s, price, sp, t)
            reward = 0.5 if opened else -0.05
        elif self.side is not None and side is not None:
            want = +1 if side == Direction.BULL else -1
            if want != self.side:
                self._flatten(price, sp)
                opened = self._try_open(want, price, sp, t)
                if opened and self.mark_soul:
                    self._soul_flips = int(getattr(self, "_soul_flips", 0)) + 1
                reward = 0.1 if opened else -0.05
            elif self.mark_soul:
                # same side → force-aligned add (Mark soul)
                added = self._try_add(want, price, sp, t)
                reward = 0.35 if added else 0.0
        else:
            if self.side is None:
                reward = -0.02
            else:
                reward = 0.0

        self._check_breach_and_bank(hi, lo, price, sp)
        eq = self.equity_pct(price)
        if self.banked:
            reward += 2.0
        elif eq > 0:
            reward += 0.01 * min(eq, self.target)

        self.actions.append(action)
        self.rewards.append(float(reward))
        return float(reward)

    def _mark_bar(self, t: int) -> None:
        """Apply stop / breach / bank using this bar's OHLC (no new decisions)."""
        if self.dead or self.banked or self.side is None:
            # still update min equity when flat
            if not self.dead and not self.banked:
                price = float(self._close[t])
                eq = self.equity_pct(price)
                self.min_eq_pct = min(self.min_eq_pct, eq)
            return
        price = float(self._close[t])
        hi = float(self._high[t])
        lo = float(self._low[t])
        sp = float(self._spread_px[t])
        self._maybe_stop(hi, lo, sp)
        self._check_breach_and_bank(hi, lo, price, sp)

    def run(
        self,
        policy_fn: Optional[Callable[[np.ndarray, "GoalEquityDay"], int]] = None,
        *,
        greedy_policy: Any = None,
        use_heuristic: bool = False,
        pure_greedy: bool = False,
    ) -> EquityDayResult:
        """Run full day. policy_fn(obs, day)->action or greedy_policy.act or heuristic.

        pure_greedy=True: never inject teacher on HOLD (honest policy A/B).
        pure_greedy=False: legacy assist — if flat+HOLD, fill teacher directional.
        """
        indices = self.runner.decision_indices()
        prev_t = 0
        for t in indices:
            if self.dead or self.banked:
                break
            # mark every bar since last decision so stops/gaps cannot skip the floor
            for bt in range(prev_t, t):
                if self.dead or self.banked:
                    break
                self._mark_bar(bt)
            prev_t = t + 1
            if self.dead or self.banked:
                break
            obs = self.observe(t)
            if use_heuristic or (policy_fn is None and greedy_policy is None):
                action = self.recommended_action(t)
            elif policy_fn is not None:
                action = int(policy_fn(obs, self))
            else:
                action, _ = greedy_policy.act(obs, greedy=True)
                if (
                    not pure_greedy
                    and action == ACTION_HOLD
                    and self.side is None
                    and not self.banked
                ):
                    rec = self.recommended_action(t)
                    if rec != ACTION_HOLD:
                        action = rec
                # Mark-aligned sense (soul=policy): share Mark wait/side on bar
                # pure_greedy still aligns opens so long consistency holds.
                if getattr(self, "mark_align_policy", True) and greedy_policy is not None:
                    from lineages.adaptive_rl_brain_7_31_26.mark_aligned_decode import (
                        doctrine_fields,
                        mark_aligned_action,
                    )

                    rec = self.recommended_action(t)
                    eq = self.equity_pct(float(self._close[t]))
                    fdir, mconf, reg = doctrine_fields(self, t)
                    action = mark_aligned_action(
                        int(action),
                        int(rec),
                        side=self.side,
                        equity_pct=eq,
                        target_pct=float(self.target),
                        risk_pct=float(self.risk),
                        force_dir=fdir,
                        m_conf=mconf,
                        regime=reg,
                        strict_teacher=False,
                    )
            self.step_action(t, action)
            self._n_decisions += 1
            if self._decisions_since_stop < 10**8:
                self._decisions_since_stop += 1

        # mark remaining bars to EOD
        if not self.dead and not self.banked:
            for bt in range(prev_t, len(self.m1)):
                if self.dead or self.banked:
                    break
                self._mark_bar(bt)

        t_last = len(self.m1) - 1
        price = float(self._close[t_last])
        sp = float(self._spread_px[t_last])
        self._flatten(price, sp)
        pnl = 100.0 * (self.balance - self.eq0) / self.eq0
        self.min_eq_pct = min(self.min_eq_pct, pnl)
        if pnl <= -self.risk + 1e-12:
            self.breached = True
        goal_hit = (pnl >= self.target - 1e-12) and (not self.breached)
        if self.banked and not self.breached and pnl >= self.target - 1e-9:
            goal_hit = True
        hold_rate = (
            float(sum(1 for a in self.actions if a == ACTION_HOLD)) / max(len(self.actions), 1)
        )
        return EquityDayResult(
            date=self.date_str,
            target_pct=self.target,
            risk_pct=self.risk,
            pnl_pct=float(pnl),
            min_eq_pct=float(self.min_eq_pct),
            goal_hit=bool(goal_hit),
            breached=bool(self.breached),
            cleared=bool(goal_hit),
            n_entries=int(self.n_entries),
            n_closes=int(self.n_closes),
            banked=bool(self.banked),
            hold_rate=float(hold_rate),
            mean_reward=float(np.mean(self.rewards)) if self.rewards else 0.0,
            info={
                "n_decisions": len(self.actions),
                "n_adds": int(self.n_adds),
                "mark_soul": bool(self.mark_soul),
                "mark_soul_max_units": int(self.mark_soul_max_units),
            },
        )


def load_calendar_days(
    csv_name: str = "XAUUSD_curriculum_2026.csv",
    *,
    min_bars: int = 900,
) -> List[Tuple[str, pd.DataFrame]]:
    from lineages.adaptive_rl_brain_7_31_26.price_data import load_raw_m1
    from lineages.adaptive_rl_brain_7_31_26.real_curriculum import split_calendar_days

    m1 = load_raw_m1(csv_name)
    days = split_calendar_days(m1)
    return [(d, g) for d, g in days if len(g) >= min_bars]


def split_practice_forward(
    days: Sequence[Tuple[str, pd.DataFrame]],
    *,
    practice_n: int = 50,
    seed: int = 42,
) -> Tuple[List[Tuple[str, pd.DataFrame]], List[Tuple[str, pd.DataFrame]]]:
    """Chronological split: first practice_n days practice, rest forward."""
    ordered = list(days)
    practice = ordered[: int(practice_n)]
    forward = ordered[int(practice_n) :]
    return practice, forward
