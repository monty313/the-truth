"""MT5 live bridge — Windows-only execution layer (Phase 6 skeleton).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty; runs ONLY on Monty's Windows laptop (the
       MetaTrader5 package does not exist on Linux/macOS).
WHAT:  Connect -> sync/resume -> loop: pull latest M1 window, rebuild
       features, ask the frozen Brain, pass EVERY intent through the
       same Shell math as the simulator, place/manage orders, write HUD
       state, honor kill switch, flatten at broker midnight, close-only
       after 400. DRY-RUN mode (default) fakes orders against live
       prices — the Phase-7 gate (48h clean) runs in this mode.
WHEN:  2026-07-19 overnight build (skeleton — NOT yet run against MT5).
WHERE: Monty's laptop; started by scripts/run_live.py.
WHY:   The frozen champion meets the market through one door, and that
       door enforces the identical physics it trained under.
INTERCONNECTED WITH: backtesting/simulator (Shell math source of truth),
       features/engine, training/policy.Brain, dashboards/hud (state
       file), alerts/notify, configs/execution.yaml.
SAFETY: mode 'live' refuses to start unless configs/execution.yaml has
       been explicitly edited by Monty (broker login set + mode: live).
----------------------------------------------------------------------
"""
from __future__ import annotations
import json, os, time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATE = os.path.join(ROOT, "artifacts", "hud_state.json")
KILL = os.path.join(ROOT, "artifacts", "KILL")

try:
    import MetaTrader5 as mt5      # Windows only
    HAS_MT5 = True
except Exception:
    HAS_MT5 = False


class Bridge:
    def __init__(self, cfg: dict, brain=None):
        self.cfg = cfg
        self.mode = cfg.get("mode", "dry_run")
        self.brain = brain                       # frozen champion (None = observe)
        self.day_trades = 0
        self.positions: list[dict] = []          # dry-run book

    # ---------- lifecycle ----------
    def connect(self) -> bool:
        if self.mode == "dry_run" or not HAS_MT5:
            return True
        b = self.cfg["broker"]
        ok = mt5.initialize(login=b["login"], server=b["server"],
                            password=os.environ.get("MT5_PASSWORD", ""))
        return bool(ok)

    def resume(self) -> None:
        """Crash recovery (ADR-0002): re-read open positions and continue
        managing them — never orphan a position."""
        if self.mode != "dry_run" and HAS_MT5:
            self.positions = [p._asdict() for p in (mt5.positions_get() or [])]

    # ---------- one heartbeat ----------
    def tick(self, F_row, obs_vec) -> dict:
        """One decision heartbeat. Returns the HUD state written."""
        killed = os.path.exists(KILL)
        actions = {"op": 0, "size": 0.0}
        if killed:
            self._close_all("kill_switch")
        elif self.brain is not None and self.day_trades < 400:
            import torch
            t = torch.tensor(obs_vec, dtype=torch.float32).unsqueeze(0)
            op, size, *_ = self.brain.act(t, greedy=True)
            # Shell gate — identical law as sim (masks re-checked here)
            if op in (1, 9) and F_row["mask_buy_blocked"] > 0:
                op = 0
            if op in (2, 10) and F_row["mask_sell_blocked"] > 0:
                op = 0
            actions = {"op": op, "size": size}
            self._execute(op, size, F_row)
        state = {
            "ts": time.time(), "mode": self.mode, "killed": killed,
            "day_trades": self.day_trades, "positions": len(self.positions),
            "last_action": actions,
            "close": float(F_row.get("close", 0.0)),
            "mask_buy_blocked": bool(F_row.get("mask_buy_blocked", 0)),
            "mask_sell_blocked": bool(F_row.get("mask_sell_blocked", 0)),
        }
        tmp = STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, STATE)                   # atomic for the HUD reader
        return state

    # ---------- execution ----------
    def _execute(self, op: int, size: float, row) -> None:
        if op == 0:
            return
        self.day_trades += 1
        if self.mode == "dry_run" or not HAS_MT5:
            if op in (1, 3, 9):
                self.positions.append({"side": +1, "px": row["close"]})
            elif op in (2, 4, 10):
                self.positions.append({"side": -1, "px": row["close"]})
            elif op in (5, 6) and any(p["side"] > 0 for p in self.positions):
                self.positions = [p for p in self.positions if p["side"] < 0]
            elif op in (7, 8) and any(p["side"] < 0 for p in self.positions):
                self.positions = [p for p in self.positions if p["side"] > 0]
            return
        # real order path (Phase 8+, after Monty's gates): mt5.order_send(...)
        raise NotImplementedError("live order path opens at Phase 8 with Monty")

    def _close_all(self, why: str) -> None:
        self.positions.clear()
        # real mode: iterate mt5.positions_get() and close each — Phase 8.
