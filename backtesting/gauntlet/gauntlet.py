"""Feasibility Gauntlet — evidence BEFORE training spends anything.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (Phase 4 of the plan; ADR-0009/0010).
WHAT:  1) data audit  2) ORACLE (bounded future sight, still obeying the
       Shell/masks/paranoid fills): the physical ceiling per day
       3) BASELINE (S1-S4 entries, dumb exits): the raw edge floor
       4) CANARY lives in training/canary.py (learning plumbing proof).
WHEN:  2026-07-19 overnight build.
WHERE: scripts/run_gauntlet.py drives; reports to artifacts/gauntlet/.
WHY:   If the oracle can't reach 2x goal, no brain ever will — Monty
       rules on the bar with evidence, not hope (HARD GATE).
INTERCONNECTED WITH: simulator.DaySim (same physics as training/live),
       features/engine, experiments/tracker (run cards).
----------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from backtesting.simulator import DaySim, POINT_SIZE


def data_audit(m1: pd.DataFrame) -> dict:
    """Gaps, bad candles, spread behavior — the classroom inspection."""
    idx = m1.index
    gaps = idx.to_series().diff().dt.total_seconds().div(60).fillna(1)
    weekend = (idx.dayofweek >= 5)
    bad = ((m1["high"] < m1["low"]) | (m1["close"] > m1["high"]) |
           (m1["close"] < m1["low"])).sum()
    sp = m1["spread"]
    return {
        "rows": int(len(m1)), "start": str(idx[0]), "end": str(idx[-1]),
        "bad_candles": int(bad),
        "gaps_over_120min_weekday": int(((gaps > 120) & ~weekend).sum()),
        "spread_median_pts": float(sp.median()),
        "spread_p95_pts": float(sp.quantile(0.95)),
        "spread_max_pts": float(sp.max()),
    }


def oracle_day(F_day: pd.DataFrame, goal: float, floor: float,
               lookahead: int = 30, shell_cfg: dict | None = None,
               cost_mult: float = 4.0):
    """Perfect-foresight trader: sees `lookahead` bars ahead, takes the best
    mask-legal, cap-legal move when its edge clears round-trip costs.
    Still pays paranoid fills. NOTE (review R2#6): this is a GREEDY
    FORESIGHT PROBE — a LOWER BOUND on the true ceiling, not the ceiling
    itself. Report it as such; sweep cost_mult for sensitivity."""
    sim = DaySim(F_day, goal, floor, shell_cfg)
    close = F_day["close"].values
    spread_px = F_day["spread"].values * POINT_SIZE
    n = len(F_day)
    while True:
        t = sim.t
        if t + 2 >= n:
            break
        look = min(lookahead, n - t - 2)
        future = close[t + 2: t + 2 + look] if look > 0 else close[t:t]
        if len(future) and not sim.dead:
            up = future.max() - close[t]
            dn = close[t] - future.min()
            cost = cost_mult * spread_px[t]           # round trip, paranoid both ways
            side = +1 if up >= dn else -1
            edge = max(up, dn)
            if edge > cost:                            # stack up while future pays
                sim.try_open(side, risk_frac=0.0025)   # heat guard limits the pile
        for s in list(sim.stacks):
            if len(future) == 0:
                break
            # exit when the future stops improving for our side
            if s.side > 0 and future.max() <= close[t]:
                sim.try_close(s)
            if s.side < 0 and future.min() >= close[t]:
                sim.try_close(s)
        if not sim.step():
            break
    return sim.finish()


def baseline_day(F_day: pd.DataFrame, goal: float, floor: float,
                 shell_cfg: dict | None = None):
    """S1-S4 entries (any set fires -> take it), dumb exits: opposite LTF
    trigger of the same set, or stop, or midnight. The raw-edge floor."""
    sim = DaySim(F_day, goal, floor, shell_cfg)
    buy_cols = [c for c in F_day.columns if c.endswith("_buy_event")
                and "reload" not in c]
    sell_cols = [c.replace("_buy_event", "_sell_event") for c in buy_cols]
    B = F_day[buy_cols].values.sum(axis=1)
    S = F_day[sell_cols].values.sum(axis=1)
    while True:
        t = sim.t
        if not sim.dead:
            if B[t] > 0 and not any(s.side > 0 for s in sim.stacks):
                sim.try_open(+1, risk_frac=0.0025)
            if S[t] > 0 and not any(s.side < 0 for s in sim.stacks):
                sim.try_open(-1, risk_frac=0.0025)
        for s in list(sim.stacks):
            if (s.side > 0 and S[t] > 0) or (s.side < 0 and B[t] > 0):
                sim.try_close(s)
        if not sim.step():
            break
    return sim.finish()


def run_over_days(days, fn, goal, floor, label, shell_cfg=None) -> pd.DataFrame:
    rows = []
    for date, F_day in days:
        r = fn(F_day, goal, floor, shell_cfg=shell_cfg)
        rows.append({"date": date, "pnl_pct": round(r.pnl_pct, 3),
                     "goal_hit": r.goal_hit, "breached": r.breached,
                     "trades": r.trades,
                     "closed": len(r.closed_trades)})
    df = pd.DataFrame(rows)
    df.attrs["label"] = label
    return df
