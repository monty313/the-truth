"""MT5 bridge v2 — one-physics execution layer (dry-run works anywhere).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty; the live-order path runs ONLY on Monty's Windows
       laptop (the MetaTrader5 package is Windows-only). v2 after the
       2026-07-19 audit found v1 bypassed the Shell (adds skipped the
       forever mask, no close-only after 400, no floor/ratchet, HUD gauge
       fields never produced, and Bridge.tick was never called).
WHAT:  The bridge drives a TradingEnv (which wraps DaySim) with the frozen
       champion Brain. Because DaySim IS the sim/training physics, "the
       same Shell law, identical sim==live" is true BY CONSTRUCTION —
       masks (incl. on adds/probes), 0.25% cap, heat guard, 400->close-
       only, midnight flat, ratchet, kill switch all apply. Each heartbeat
       writes artifacts/hud_state.json with the live gauge fields
       (eq_pct, ratchet, goal, floor) the HUD needs. Floor/kill events
       fire phone alerts. In mode='live' (Phase 8, Monty-gated) the same
       decisions are mirrored to real mt5.order_send calls.
WHEN:  2026-07-19 (v2).
WHERE: scripts/run_live.py; dry-run runs in this container, demo/live on
       the laptop.
WHY:   The frozen champion meets the market through ONE door that enforces
       the exact law it trained under.
INTERCONNECTED WITH: training/env.TradingEnv + backtesting/simulator
       (physics), inference/loader.load_brain, dashboards/hud (state file),
       alerts/notify, core.configs.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  drives DaySim (one physics), emits HUD gauge fields  — WHY: v1 bypassed masks/close-only and HUD gauge was dead (audit S1/S5/R12/R13).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import json
import os
import time

from core.configs import path as rpath
from telemetry import tracer

STATE = rpath("artifacts", "hud_state.json")
KILL = rpath("artifacts", "KILL")

try:
    import MetaTrader5 as mt5      # Windows only
    HAS_MT5 = True
except Exception:
    mt5 = None
    HAS_MT5 = False


class Bridge:
    """Drives a TradingEnv day with the frozen champion; writes HUD state."""

    def __init__(self, exec_cfg: dict, env, brain=None):
        self.cfg = exec_cfg
        self.mode = exec_cfg.get("mode", "dry_run")
        self.env = env                 # TradingEnv (one-physics)
        self.brain = brain             # frozen Brain or None (observe-only)
        self._h = None                 # GRU hidden state, carried across bars
        self._alerted_floor = False

    # ---------- lifecycle ----------
    def connect(self) -> bool:
        if self.mode == "dry_run" or not HAS_MT5:
            return True
        b = self.cfg.get("broker", {})
        return bool(mt5.initialize(login=b.get("login"), server=b.get("server"),
                                   password=os.environ.get("MT5_PASSWORD", "")))

    # ---------- run one day (dry-run: paper; live: mirrors to MT5) ----------
    def run_day(self, day_idx: int, throttle: float = 0.0) -> dict:
        import torch
        from alerts import notify
        obs = self.env.reset(day_idx)
        self._h = None
        self._alerted_floor = False
        done = False
        info = {}
        while not done:
            if os.path.exists(KILL):
                self.env.sim.killed = True
                notify.push("Momentum One", "KILL switch — freezing & closing all.")
            op, size = 0, 0.0
            if self.brain is not None and not os.path.exists(KILL):
                with tracer.span("policy_inference"):
                    t = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                    op, size, _, _, self._h = self.brain.act(t, self._h, greedy=True)
            # The env/DaySim Shell enforces masks (incl. adds), caps, close-only,
            # heat, ratchet — the brain only PROPOSES (audit S1/S5/R13).
            obs, _, done, info = self.env.step(op, size)
            self._write_hud(op, size)
            if self.env.sim.dead and not self._alerted_floor:
                self._alerted_floor = True
                notify.push("Momentum One",
                            f"Trading stopped: {info.get('breached') and 'FLOOR BREACH' or 'stand-down'}"
                            f" on {self.env.days[day_idx][0]}.")
            if self.mode != "dry_run" and HAS_MT5:
                self._mirror_to_mt5(op, size)   # Phase 8 (Monty-gated)
            if throttle:
                time.sleep(throttle)
        return info

    # ---------- HUD state (audit R12: the gauge was dead) ----------
    def _write_hud(self, op: int, size: float) -> None:
        s = self.env.sim
        state = {
            "ts": time.time(), "mode": self.mode,
            "killed": os.path.exists(KILL),
            "goal": self.env.goal, "floor": self.env.floor,
            "eq_pct": round(s.equity_pct(), 4),
            "ratchet": round(max(0.0, s.ratchet_floor), 4),
            "day_trades": s.trades_used, "positions": len(s.stacks),
            "close": float(s.row["close"]),
            "mask_buy_blocked": bool(s.row["mask_buy_blocked"]),
            "mask_sell_blocked": bool(s.row["mask_sell_blocked"]),
            "last_action": {"op": op, "size": round(size, 3)},
            "dead": s.dead,
        }
        tmp = STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE)          # atomic for the HUD reader

    def _mirror_to_mt5(self, op: int, size: float) -> None:
        """Real order path — opens at Phase 8 with Monty, on Windows."""
        raise NotImplementedError("live MT5 order path opens at Phase 8 with Monty")
