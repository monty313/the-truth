"""Reward engine v2 — Monty's doctrine (ADR-0005), post-audit rebuild.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty; weights from configs/rewards.yaml, changes only
       via meta-optimizer PROPOSALS approved by Monty.
WHAT:  Closed-trades-only payments (Monty verbatim: "the trade has to be
       closed in order to get any rewards"). v2 fixes from the 2026-07-19
       audit round 2: pyramid + no-drawdown bonuses pay ONLY on FULL
       stack closes (T2 farm: 231 half-closes once paid 169.6 reward);
       pullback bonus moved from entry-queue time to CLOSE time via
       stack tags (T3: was paid for phantom never-filled orders);
       idleness hunger only when FLAT (T4: was taxing bank-and-ride);
       day-DD extra + consistency target scale from configs and the
       EPISODE's own floor (T9/S8, any-X correctness); update_idx and
       streak/record persist via state_dict (T8: wheels now actually
       dissolve across capped runs).
WHEN:  2026-07-19 (v2, same day as audit).
WHERE: training/env.py calls on_step/on_day_end; ppo.save/load carries
       state_dict.
WHY:   The reward IS the personality; every audit hole here was a
       personality disorder waiting to be learned.
INTERCONNECTED WITH: backtesting/simulator (closed-trade records incl.
       full/tags), configs/rewards.yaml, training/trophy_case.py,
       training/ppo.py (state persistence), ADR-0005.
----------------------------------------------------------------------
"""
from __future__ import annotations

from core.configs import load as _load_cfg


def load_weights() -> dict:
    return dict(_load_cfg("rewards"))


class RewardEngine:
    """Stateful across a training run (streak memory, record book, decay)."""

    def __init__(self, weights: dict | None = None):
        self.w = weights or load_weights()
        assert self.w.get("pay_only_on_close", True), \
            "ADR-0005: pay_only_on_close is a ruled invariant"
        self.streak_days = 0
        self.record_win_pct = 0.0
        self.update_idx = 0

    # ---------- persistence (audit T8) ----------
    def state_dict(self) -> dict:
        return {"streak_days": self.streak_days,
                "record_win_pct": self.record_win_pct,
                "update_idx": self.update_idx}

    def load_state(self, d: dict) -> None:
        self.streak_days = int(d.get("streak_days", 0))
        self.record_win_pct = float(d.get("record_win_pct", 0.0))
        self.update_idx = int(d.get("update_idx", 0))

    # ---------- per step ----------
    def on_step(self, closed_now: list[dict], acted: bool, anti_gravity: bool,
                flat: bool) -> float:
        """closed_now: trades closed at THIS bar. Closed-only payments.
        FULL closes carry the stack bonuses; partials pay pure P/L only."""
        w = self.w
        r = 0.0
        for tr in closed_now:
            r += w["w_net_profit"] * tr["pnl_pct"]
            if tr.get("full"):
                if tr["pnl"] > 0 and tr["max_adverse"] <= w.get(
                        "no_drawdown_tolerance", 0.0):
                    r += w["w_no_drawdown_close"]
                if tr["adds"] > 0 and tr["stack_green"]:
                    r += w["w_pyramid_stack_green"] * min(tr["adds"], 5)
                if tr.get("tags", {}).get("pullback") and not tr.get("probe"):
                    r += w["w_pullback_with_htf"]          # paid at close (T3)
        if flat and not acted and not closed_now:          # T4: only when flat
            r += w["w_idleness_hunger"]
        if anti_gravity:
            decay = max(0.0, 1.0 - self.update_idx /
                        max(1, w["antigravity_decay_updates"]))
            r += w["w_antigravity_penalty"] * decay
        return r

    # ---------- per day ----------
    def on_day_end(self, day, floor: float) -> tuple[float, dict]:
        """day: DayResult; floor: the EPISODE's floor (any-X correct, T9)."""
        w = self.w
        r, info = 0.0, {}
        pnls = [t["pnl_pct"] for t in day.closed_trades if not t["probe"]]
        if day.breached:
            r += w["w_death_penalty"]
            info["death"] = True
        if day.goal_hit:
            r += w["w_day_goal_hit"]
            min_eq = min(day.equity_curve) if day.equity_curve else 0.0
            r += (w["w_day_goal_hit"] * max(0.0, 1.0 + min_eq / max(floor, 1e-6))
                  * w.get("day_dd_extra_scale", 0.5))
            self.streak_days += 1
            r += w["w_streak_per_day"] * self.streak_days   # climbs FOREVER
        else:
            self.streak_days = 0
        if len(pnls) > 1:
            import statistics
            spread = statistics.pstdev(pnls)
            r += w["w_trade_consistency"] * max(
                0.0, w.get("trade_consistency_target", 0.3) - spread)
        best = max((t["pnl_pct"] for t in day.closed_trades), default=0.0)
        if day.goal_hit and best > self.record_win_pct:
            self.record_win_pct = best
            r += w["w_record_win"]
            info["new_record_win_pct"] = best
        info["streak_days"] = self.streak_days
        return r, info
