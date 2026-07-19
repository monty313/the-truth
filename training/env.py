"""Trading environment — the arena the brain lives in (Gym-style API).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0004/0005/0006; pillar 1+2 of §14 handoff).
WHAT:  Wraps DaySim + feature table into reset()/step(). Observation =
       frame-stacked market features + goal-conditioned self-state.
       Action = (op categorical, size in (0,1] of the 0.25% cap).
       v1 simplification (documented): add/close ops act on the LARGEST
       stack of that side; per-stack addressing arrives in v2.
WHEN:  2026-07-19 overnight build.
WHERE: training/ppo.py rolls episodes; canary + smoke tests use it too.
WHY:   One env for boot camp and expansion; the Shell inside DaySim is
       the same law live — the brain can never learn an illegal move.
INTERCONNECTED WITH: backtesting/simulator.DaySim, features/engine
       (obs_columns), training/rewards.RewardEngine, configs/*.yaml.
OPS:   0 hold · 1 open_long · 2 open_short · 3 add_long · 4 add_short ·
       5 close_half_long · 6 close_long · 7 close_half_short ·
       8 close_short · 9 probe_long · 10 probe_short
----------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np

from backtesting.simulator import DaySim
from features.engine import obs_columns
from training.rewards import RewardEngine

N_OPS = 11
FRAME = 10                     # configs/features.yaml frame_stack
SELF_DIM = 12


class TradingEnv:
    """One episode = one trading day (ADR-0005: day-life)."""

    def __init__(self, days: list, goal: float = 2.5, floor: float = 4.0,
                 reward_engine: RewardEngine | None = None,
                 shell_cfg: dict | None = None, rng: np.random.Generator | None = None,
                 goal_ranges: tuple | None = None):
        self.days = days                       # [(date, F_day)]
        self.goal0, self.floor0 = goal, floor
        self.goal_ranges = goal_ranges         # ((gmin,gmax),(fmin,fmax)) any-X
        self.re = reward_engine or RewardEngine()
        self.shell_cfg = shell_cfg or {}
        self.rng = rng or np.random.default_rng(0)
        self._cols = None
        self.day_idx = -1

    # ---------- helpers ----------
    def _obs_matrix(self, F_day):
        if self._cols is None:
            self._cols = obs_columns(F_day)
        M = F_day[self._cols].to_numpy(dtype=np.float32)
        return np.nan_to_num(M, nan=0.0, posinf=5.0, neginf=-5.0)

    def _self_state(self) -> np.ndarray:
        s = self.sim
        eq = s.equity_pct()
        closed = [t for t in s.res.closed_trades if not t["probe"]]
        wins = sum(1 for t in closed if t["pnl"] > 0)
        wr = wins / len(closed) if closed else 0.5
        longs = [st for st in s.stacks if st.side > 0]
        shorts = [st for st in s.stacks if st.side < 0]
        mark = float(s.row["close"])
        sp = s._sp(s.t)
        return np.array([
            self.goal / 5.0, self.floor / 6.0,
            (self.goal - eq) / max(self.goal, 1e-6),        # distance to goal
            (eq + self.floor) / max(self.floor, 1e-6),      # distance to floor
            max(0.0, s.ratchet_floor) / 5.0,                # win-lock level
            wr,                                             # today's win rate
            min(self.re.streak_days, 50) / 50.0,            # streak sense
            s.open_risk_frac(mark, sp) * 100 / 4.0,         # open heat
            (sum(st.units for st in longs) - sum(st.units for st in shorts))
            / (1.0 + sum(st.units for st in s.stacks)),     # net direction
            min(max((st.bars_open for st in s.stacks), default=0), 240) / 240.0,
            s.trades_used / 400.0,                          # budget used
            s.unrealized(mark) / s.eq0 * 100 / 4.0,         # floating P/L
        ], dtype=np.float32)

    @property
    def obs_dim(self):
        return FRAME * (len(self._cols) + SELF_DIM)

    def _obs(self):
        t = self.sim.t
        rows = []
        for k in range(FRAME - 1, -1, -1):
            i = max(0, t - k)
            rows.append(np.concatenate([self.M[i], self._self_state()]))
        return np.concatenate(rows)

    # ---------- gym-style ----------
    def reset(self, day_idx: int | None = None):
        self.day_idx = (self.day_idx + 1) % len(self.days) if day_idx is None else day_idx
        date, F_day = self.days[self.day_idx]
        if self.goal_ranges:                          # any-X conditioning
            (gmin, gmax), (fmin, fmax) = self.goal_ranges
            self.goal = float(self.rng.uniform(gmin, gmax))
            self.floor = float(self.rng.uniform(fmin, fmax))
        else:
            self.goal, self.floor = self.goal0, self.floor0
        self.sim = DaySim(F_day, self.goal, self.floor, self.shell_cfg)
        self.M = self._obs_matrix(F_day)
        self._closed_seen = 0
        return self._obs()

    def step(self, op: int, size: float):
        """op in [0,11); size in (0,1] of the per-trade cap."""
        sim = self.sim
        risk = float(np.clip(size, 0.05, 1.0)) * sim.cap
        row = sim.row
        acted, anti_grav, entry_bonus = False, False, 0.0

        def biggest(side):
            c = [st for st in sim.stacks if st.side == side and not st.is_probe]
            return max(c, key=lambda st: st.units) if c else None

        if op in (1, 2, 9, 10):
            side = +1 if op in (1, 9) else -1
            ok, _ = sim.try_open(side, risk, probe=op >= 9)
            acted = ok
            if ok:
                tag = "buy" if side > 0 else "sell"
                pull = any(row.get(f"set{k}::pull_{tag}", 0) > 0 for k in (1, 2, 3, 4))
                entry_bonus = self.re.on_entry_quality(bool(pull))
                # anti-gravity: acting against the highest set showing gravity
                s4 = row.get("set4::cont_buy", 0) - row.get("set4::cont_sell", 0)
                anti_grav = (s4 > 0 and side < 0) or (s4 < 0 and side > 0)
        elif op in (3, 4):
            side = +1 if op == 3 else -1
            tgt = biggest(side)
            if tgt is not None:
                ok, _ = sim.try_open(side, risk, add_to=tgt)
                acted = ok
        elif op in (5, 6, 7, 8):
            side = +1 if op in (5, 6) else -1
            tgt = biggest(side)
            if tgt is not None:
                acted = sim.try_close(tgt, 0.5 if op in (5, 7) else 1.0)

        alive = sim.step()
        closed_now = sim.res.closed_trades[self._closed_seen:]
        self._closed_seen = len(sim.res.closed_trades)
        r = self.re.on_step(closed_now, acted, anti_grav) + entry_bonus

        done = not alive
        info = {}
        if done:
            day = sim.finish()
            day_r, info = self.re.on_day_end(day)
            r += day_r
            info.update({"pnl_pct": day.pnl_pct, "goal_hit": day.goal_hit,
                         "breached": day.breached, "trades": day.trades,
                         "rejected": day.rejected})
        return (None if done else self._obs()), r, done, info
