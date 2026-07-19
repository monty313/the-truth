"""Event-driven M1 simulator — paranoid fills + the full Shell (v2).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0002 Shell, ADR-0003 masks, ADR-0009 fills).
WHAT:  One trading-day episode, bar by bar. v2 after 3-agent adversarial
       review (2026-07-19): pending-aware Shell + fill-time re-validation,
       stop-anchored true risk, effective floor (ratchet-aware) heat
       guard, intrabar worst-case equity check, gap-through-stop fills
       at bar extremes, kill-switch cancels pending, true 0.01-lot
       probes, winning-stack-only adds, validated close fractions.
WHEN:  2026-07-19 overnight build (v2 same night, post-review).
WHERE: Gauntlet + training/env.py; the SAME Shell logic ships to the
       live bridge — one physics.
WHY:   Review proved v1's law was breakable (batched orders, unrealized-
       profit heat anchor, wick-through-floor invisibility). A reward
       signal inherits every Shell hole, so the Shell must be airtight
       BEFORE training exists.
INTERCONNECTED WITH: features/engine (mask/ATR columns), configs/
       masks_shell.yaml + goals.yaml, telemetry spans, tests/test_shell.py.
UNITS: prices in quote units; spread column = POINTS * POINT_SIZE.
       Sizing in 'units' (P/L = units * Δprice). 0.01-lot probe =
       0.01 * CONTRACT_SIZE units (gold: 1 oz). Lot conversion for real
       orders happens only at the live bridge.
----------------------------------------------------------------------
"""
from __future__ import annotations
from dataclasses import dataclass, field
import os

import numpy as np
import pandas as pd

from telemetry import tracer

POINT_SIZE = 0.01          # gold; per-symbol via configs/data.yaml at expansion
CONTRACT_SIZE = 100.0      # gold: 100 oz per 1.0 lot -> 0.01 lot = 1 unit
ATR_STOP_MULT = 6.0        # wide stop (configs/masks_shell.yaml broker_stop)
KILL_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "artifacts", "KILL")


@dataclass
class Stack:
    """One position stack (entry + adds). Hedging = separate stacks."""
    side: int
    entries: list = field(default_factory=list)   # [(price, units)]
    stop: float = 0.0
    bars_open: int = 0
    is_probe: bool = False
    max_adverse: float = 0.0

    @property
    def units(self): return sum(u for _, u in self.entries)

    @property
    def avg_price(self):
        t = self.units
        return sum(p * u for p, u in self.entries) / t if t else 0.0

    def true_risk(self, mark: float, sp: float, eq0: float) -> float:
        """Fraction of eq0 lost if the stop fills from here (incl. exit spread).
        Stop-anchored — the ONLY honest risk number (review finding R3#4/5)."""
        if self.units <= 0:
            return 0.0
        loss = self.side * (mark - self.stop) + sp          # per unit, >=0 normally
        return max(0.0, self.units * loss) / eq0


@dataclass
class DayResult:
    date: str = ""
    pnl_pct: float = 0.0
    goal_hit: bool = False
    breached: bool = False
    trades: int = 0
    rejected: int = 0
    closed_trades: list = field(default_factory=list)
    equity_curve: list = field(default_factory=list)


class DaySim:
    """One 00:00->00:00 day episode over precomputed features F (M1 rows)."""

    def __init__(self, F_day: pd.DataFrame, goal: float, floor: float,
                 shell_cfg: dict | None = None, start_equity: float = 100_000.0):
        if len(F_day) == 0:
            raise ValueError("empty day (holiday/data gap) — filter upstream")
        cfg = shell_cfg or {}
        self.F = F_day
        self.n = len(F_day)
        self.goal, self.floor = goal, floor
        self.cap = cfg.get("per_trade_risk_cap_pct", 0.25) / 100.0
        self.max_adds = cfg.get("max_adds_per_stack", 5)
        self.max_trades = cfg.get("max_trades_per_day", 400)
        self.heat_on = cfg.get("heat_guard", {}).get("enabled", True)
        self.eq0 = start_equity
        self.balance = start_equity
        self.stacks: list[Stack] = []
        self.t = 0
        self.trades_used = 0
        self.dead = False
        self.flat_done = False
        self.ratchet_floor = -floor
        self.res = DayResult(date=str(F_day.index[0].date()))
        self.killed = False
        self._pending: list[dict] = []

    # ---------- marks & risk ----------
    @property
    def row(self): return self.F.iloc[self.t]

    def _sp(self, t): return float(self.F["spread"].iloc[t]) * POINT_SIZE

    def unrealized(self, price: float) -> float:
        return sum(s.side * s.units * (price - s.avg_price) for s in self.stacks)

    def equity_pct(self, price: float | None = None) -> float:
        p = float(self.row["close"]) if price is None else price
        return 100.0 * (self.balance + self.unrealized(p) - self.eq0) / self.eq0

    def open_risk_frac(self, mark: float, sp: float) -> float:
        return sum(s.true_risk(mark, sp, self.eq0) for s in self.stacks)

    def _eff_floor(self) -> float:
        """Effective floor: ratchet-aware (R3#6)."""
        return max(-self.floor, self.ratchet_floor)

    def worst_eq_pct(self, hi: float, lo: float, sp: float) -> float:
        """Exact intrabar worst-case equity: every long marked at max(stop, lo)
        (a gap fills at lo), every short at min(stop, hi); exit spread paid.
        Linear per stack, so endpoint evaluation is exact (R2#1, R3#9)."""
        w = self.balance - self.eq0
        for s in self.stacks:
            if s.side > 0:
                px = (s.stop if lo <= s.stop else lo) - sp
                w += s.units * (px - s.avg_price)
            else:
                px = (s.stop if hi >= s.stop else hi) + sp
                w += s.units * (s.avg_price - px)
        return 100.0 * w / self.eq0

    # ---------- Shell (pending-aware; run at decision AND at fill) ----------
    def _shell_check(self, side: int, risk_frac: float, add_to: Stack | None,
                     row) -> tuple[bool, str]:
        if self.killed:                         return False, "kill_switch"
        if self.dead:                           return False, "day_dead"
        if risk_frac <= 0:                      return False, "bad_risk"
        if risk_frac > self.cap + 1e-12:        return False, "per_trade_cap"
        pend_opens = [p for p in self._pending if p["kind"] == "open"]
        if self.trades_used + len(pend_opens) >= self.max_trades:
            return False, "budget_close_only"
        if side > 0 and row["mask_buy_blocked"] > 0:   return False, "forever_mask_buy"
        if side < 0 and row["mask_sell_blocked"] > 0:  return False, "forever_mask_sell"
        if add_to is not None:
            if add_to not in self.stacks:       return False, "stack_gone"
            if len(add_to.entries) >= 1 + self.max_adds:
                return False, "max_adds"
            mark = float(row["close"])
            if add_to.side * (mark - add_to.avg_price) <= 0:
                return False, "add_to_loser"     # winners only (ADR-0002/R3#16)
        if self.heat_on:
            mark, sp = float(row["close"]), self._sp(self.t)
            open_r = self.open_risk_frac(mark, sp)
            pend_r = sum(p["risk"] for p in pend_opens)
            dist = max(0.0, (self.equity_pct(mark) - self._eff_floor()) / 100.0)
            if open_r + pend_r + risk_frac > dist + 1e-12:
                return False, "heat_guard"
        return True, "ok"

    # ---------- intents ----------
    def try_open(self, side: int, risk_frac: float, add_to: Stack | None = None,
                 probe: bool = False) -> tuple[bool, str]:
        ok, why = self._shell_check(side, risk_frac, add_to, self.row)
        with tracer.span("mask_check", side="buy" if side > 0 else "sell",
                         ok=ok, why=why):
            pass
        if not ok:
            self.res.rejected += 1
            return False, why
        self._pending.append({"kind": "open", "side": side, "risk": risk_frac,
                              "add_to": add_to, "probe": probe})
        return True, "queued"

    def try_close(self, stack: Stack, fraction: float = 1.0) -> bool:
        if fraction <= 0:                       # R3#2: no negative-close adds
            self.res.rejected += 1
            return False
        fraction = 1.0 if fraction >= 0.999 else fraction
        self._pending.append({"kind": "close", "stack": stack, "frac": fraction})
        return True

    # ---------- one bar ----------
    def step(self) -> bool:
        if self.t >= self.n - 1:
            self._flatten("midnight_flat")
            return False
        self.t += 1
        row = self.row
        hi, lo, close = float(row["high"]), float(row["low"]), float(row["close"])
        sp = self._sp(self.t)

        # kill switch: cancel pending OPENS, keep closes (R3#10)
        self.killed = os.path.exists(KILL_FILE)
        if self.killed:
            dropped = [p for p in self._pending if p["kind"] == "open"]
            for _ in dropped:
                tracer.event("order_submission", ok=False, why="kill_cancel")
            self._pending = [p for p in self._pending if p["kind"] != "open"]

        # execute pending (decided at t-1) with FILL-TIME re-validation (R3#1)
        for order in self._pending:
            if order["kind"] == "open":
                ok, why = self._shell_check(order["side"], order["risk"],
                                            order["add_to"], row)
                if not ok:
                    self.res.rejected += 1
                    tracer.event("order_submission", ok=False, why=f"fill_{why}")
                    continue
                side, risk_frac = order["side"], order["risk"]
                fill = (hi + sp) if side > 0 else (lo - sp)
                atr = float(row.get("15min::atr14", np.nan))
                if not np.isfinite(atr) or atr <= 0:
                    self.res.rejected += 1
                    tracer.event("order_submission", ok=False, why="atr_warmup")
                    continue
                tgt = order["add_to"]
                if tgt is not None and tgt in self.stacks:
                    # adds sized off the ACTUAL stack stop (R3#5)
                    dist = tgt.side * (fill - tgt.stop) + sp
                    if dist <= 0:
                        self.res.rejected += 1
                        tracer.event("order_submission", ok=False, why="add_stop_side")
                        continue
                    units = (risk_frac * self.eq0) / dist
                    tgt.entries.append((fill, units))
                else:
                    stop_dist = ATR_STOP_MULT * atr
                    units = (risk_frac * self.eq0) / (stop_dist + sp)
                    if order["probe"]:            # true 0.01-lot probe (R3#11)
                        units = min(units, 0.01 * CONTRACT_SIZE)
                    s = Stack(side=side, is_probe=order["probe"])
                    s.entries.append((fill, units))
                    s.stop = fill - side * stop_dist
                    self.stacks.append(s)
                self.trades_used += 1
                with tracer.span("order_submission", side=side, ok=True,
                                 probe=order["probe"], risk=risk_frac):
                    pass
            else:
                st, frac = order["stack"], order["frac"]
                if st in self.stacks:
                    px = (lo - sp) if st.side > 0 else (hi + sp)
                    self._close(st, frac, px, "policy_close")
        self._pending.clear()

        # broker stops — gap-aware fills (R3#8)
        for s in list(self.stacks):
            if s.side > 0 and lo <= s.stop:
                px = (s.stop if hi >= s.stop else lo) - sp
                self._close(s, 1.0, px, "broker_stop")
            elif s.side < 0 and hi >= s.stop:
                px = (s.stop if lo <= s.stop else hi) + sp
                self._close(s, 1.0, px, "broker_stop")

        # marks + intrabar worst-case floor law (R2#1 / R3#9)
        eq = self.equity_pct(close)
        self.res.equity_curve.append(eq)
        for s in self.stacks:
            s.bars_open += 1
            s.max_adverse = max(s.max_adverse, s.side * (s.avg_price - close))

        worst = self.worst_eq_pct(hi, lo, sp)
        eff = self._eff_floor()
        if worst <= eff or eq <= eff:
            self._flatten("floor_stand_down")
            self.dead = True
            self.res.breached = min(worst, self.equity_pct(close)) <= -self.floor
            tracer.event("state_update", event="stand_down", eq=eq, worst=worst)
            return False

        # ratchet (ADR-0001): activation needs heat-aware clearance so the
        # stand-down flatten still realizes >= goal (R3#7)
        flat_cost = 100.0 * sum(s.units * ((close - lo) + sp) if s.side > 0
                                else s.units * ((hi - close) + sp)
                                for s in self.stacks) / self.eq0
        if eq - flat_cost >= self.goal or self.ratchet_floor >= self.goal:
            peak = max(self.res.equity_curve)
            trail = peak - (self.floor * 0.25)
            self.ratchet_floor = max(self.ratchet_floor, self.goal, trail)

        if self.killed:
            self._flatten("kill_switch")
            self.dead = True
            return False
        return True

    # ---------- closing ----------
    def _close(self, s: Stack, fraction: float, price: float, why: str) -> None:
        full = fraction >= 0.999
        units = s.units if full else s.units * fraction
        pnl = s.side * units * (price - s.avg_price)
        self.balance += pnl
        self.res.closed_trades.append(
            {"side": s.side, "units": units, "pnl": pnl,
             "pnl_pct": 100.0 * pnl / self.eq0, "bars": s.bars_open, "why": why,
             "probe": s.is_probe, "adds": len(s.entries) - 1,
             "max_adverse": s.max_adverse, "stack_green": pnl > 0})
        with tracer.span("fill_handling", why=why, pnl_pct=100.0 * pnl / self.eq0):
            pass
        if full:
            self.stacks.remove(s)
        else:
            keep = 1.0 - fraction
            s.entries = [(p, u * keep) for p, u in s.entries]

    def _flatten(self, why: str) -> None:
        if self.flat_done and why == "midnight_flat":
            return
        row = self.row
        sp = self._sp(self.t)
        for s in list(self.stacks):
            px = (float(row["low"]) - sp) if s.side > 0 else (float(row["high"]) + sp)
            self._close(s, 1.0, px, why)
        if why == "midnight_flat":
            self.flat_done = True

    def finish(self) -> DayResult:
        self._flatten("midnight_flat")
        self.res.pnl_pct = 100.0 * (self.balance - self.eq0) / self.eq0
        self.res.goal_hit = (self.res.pnl_pct >= self.goal) and not self.res.breached
        self.res.trades = self.trades_used
        return self.res
