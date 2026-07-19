"""Reward engine — Monty's doctrine (ADR-0005), every weight in configs.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty; weights from configs/rewards.yaml, changes only
       via meta-optimizer PROPOSALS approved by Monty.
WHAT:  Two layers: (1) per-step: closed-trade payments (pay ONLY on
       close), idleness hunger, anti-gravity training wheels (decaying);
       (2) day-end: goal/consistency/streak/death terms.
WHEN:  2026-07-19 overnight build.
WHERE: training/env.py calls on_step/on_day_end each episode.
WHY:   The reward IS the bot's personality: capitalist, steady, hates
       losing days, streak-hungry, pays for discipline not paper riches.
INTERCONNECTED WITH: backtesting/simulator (closed_trades records),
       configs/rewards.yaml, training/trophy_case.py, ADR-0005.
----------------------------------------------------------------------
"""
from __future__ import annotations
import yaml, os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_weights() -> dict:
    with open(os.path.join(ROOT, "configs", "rewards.yaml")) as f:
        return {k: v for k, v in yaml.safe_load(f).items()}


class RewardEngine:
    """Stateful across a training run (streak memory, record book, decay)."""

    def __init__(self, weights: dict | None = None):
        self.w = weights or load_weights()
        self.streak_days = 0          # consecutive goal-met-no-breach days
        self.record_win_pct = 0.0     # trophy-case ladder (per run)
        self.update_idx = 0           # PPO update counter (anti-gravity decay)

    # ---------- per step ----------
    def on_step(self, closed_now: list[dict], acted: bool,
                anti_gravity: bool) -> float:
        """closed_now: trades closed at THIS bar (from DayResult.closed_trades).
        Pay ONLY on close (ADR-0005). Idleness hunger when no position and
        no action. Anti-gravity wheels decay to zero."""
        w = self.w
        r = 0.0
        for tr in closed_now:
            r += w["w_net_profit"] * tr["pnl_pct"]
            if tr["pnl"] > 0 and tr["max_adverse"] <= 0:
                r += w["w_no_drawdown_close"]
            if tr["adds"] > 0 and tr["stack_green"]:
                r += w["w_pyramid_stack_green"] * min(tr["adds"], 5)
        if not acted and not closed_now:
            r += w["w_idleness_hunger"]
        if anti_gravity:
            decay = max(0.0, 1.0 - self.update_idx / max(1, w["antigravity_decay_updates"]))
            r += w["w_antigravity_penalty"] * decay
        return r

    def on_entry_quality(self, pullback_with_htf: bool) -> float:
        """Small immediate shaping for the Gravity-framework entry Monty
        rewards explicitly (pullback on LTF while HTF trend is strong)."""
        return self.w["w_pullback_with_htf"] if pullback_with_htf else 0.0

    # ---------- per day ----------
    def on_day_end(self, day) -> tuple[float, dict]:
        """day: DayResult. Returns (reward, info). Updates streak + records."""
        w = self.w
        r, info = 0.0, {}
        pnls = [t["pnl_pct"] for t in day.closed_trades if not t["probe"]]
        if day.breached:
            r += w["w_death_penalty"]
            info["death"] = True
        if day.goal_hit:
            r += w["w_day_goal_hit"]
            # profit-with-less-intraday-drawdown extra: goal days paid more
            # when the equity path never went deep (ADR-0005)
            min_eq = min(day.equity_curve) if day.equity_curve else 0.0
            r += w["w_day_goal_hit"] * max(0.0, 1.0 + min_eq / 4.0) * 0.5
            self.streak_days += 1
            r += w["w_streak_per_day"] * self.streak_days   # climbs FOREVER
        else:
            self.streak_days = 0
        if len(pnls) > 1:
            import statistics
            spread = statistics.pstdev(pnls)
            r += w["w_trade_consistency"] * max(0.0, 0.3 - spread)
        # trophy: record win pays ONLY on a won day (critic round, ADR record)
        best = max((t["pnl_pct"] for t in day.closed_trades), default=0.0)
        if day.goal_hit and best > self.record_win_pct:
            self.record_win_pct = best
            r += w["w_record_win"]
            info["new_record_win_pct"] = best
        info["streak_days"] = self.streak_days
        return r, info
