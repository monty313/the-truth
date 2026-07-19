"""Phase-4 driver: audit -> oracle -> baseline -> evidence + VERDICT.
5W+I: WHO Claude/Monty gate. WHAT runs the Gauntlet (real data if present,
synthetic else — labeled), writes evidence_report.json + VERDICT.json; boot
camp refuses to run without a verdict (audit R14: the HARD GATE gated
nothing). WHEN 2026-07-19 v2. WHERE any cwd (ROOT-anchored, audit R3).
WHY evidence before training money. INTERCONNECTED: gauntlet.py, tracker,
artifacts/gauntlet/, scripts/train_bootcamp.py (reads VERDICT).
"""
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd                                              # noqa: E402
from core.configs import goals_cfg, load as cfg, path as rpath   # noqa: E402
from data_io.loader import read_mt5_m1, synthetic_m1, trading_days  # noqa: E402
from features.engine import build_features                       # noqa: E402
from backtesting.gauntlet.gauntlet import (                      # noqa: E402
    data_audit, oracle_day, baseline_day, run_over_days)
from experiments.tracker import Run                              # noqa: E402
from telemetry.logging_setup import setup                        # noqa: E402


def main():
    log = setup("gauntlet")
    g = goals_cfg()
    GOAL, FLOOR = float(g["goal_pct"]), float(g["floor_pct"])
    BAR = 2.0 * GOAL
    OUT = rpath("artifacts", "gauntlet")
    os.makedirs(OUT, exist_ok=True)

    pats = [os.path.join(rpath("..", "data"), "XAUUSD_M1_*.csv"),
            rpath("data", "XAUUSD_M1_*.csv")]
    real = sorted(sum((glob.glob(p) for p in pats), []))
    if real:
        src = "REAL_XAUUSD"
        m1 = read_mt5_m1(real[0])
        m1 = m1.loc[m1.index >= m1.index.max() - pd.Timedelta("30D")]  # audit R1
    else:
        src = "SYNTHETIC (gold-like — ADR-0010, replace when zip lands)"
        m1 = synthetic_m1(days=10, seed=7)
    log.info("source: %s rows=%d", src, len(m1))

    run = Run("gauntlet", symbols=["XAUUSD" if real else "SYNTH"],
              timeframes=["all-matrix"],
              data_window=f"{m1.index[0]}..{m1.index[-1]}", seed=7,
              assumptions={"fills": "paranoid(ADR-0009)",
                           "spread": "recorded column", "source": src})
    audit = data_audit(m1)
    json.dump(audit, open(os.path.join(OUT, "audit.json"), "w"), indent=2)
    log.info("audit: %s", audit)

    F = build_features(m1)
    days = trading_days(F)[1:]
    oracle = run_over_days(days, oracle_day, GOAL, FLOOR, "oracle")
    base = run_over_days(days, baseline_day, GOAL, FLOOR, "baseline")
    oracle.to_csv(os.path.join(OUT, "oracle_days.csv"), index=False)
    base.to_csv(os.path.join(OUT, "baseline_days.csv"), index=False)

    def summary(df):
        return {"days": len(df), "mean_pnl_pct": round(df.pnl_pct.mean(), 3),
                "min": round(df.pnl_pct.min(), 3),
                "max": round(df.pnl_pct.max(), 3),
                "goal_hit_rate": round(df.goal_hit.mean(), 3),
                "double_goal_days": int((df.pnl_pct >= BAR).sum()),
                "breaches": int(df.breached.sum())}

    rep = {"source": src, "goal": GOAL, "floor": FLOOR,
           "bar": f"+{BAR}% EVERY day, zero -{FLOOR}% touches",
           "oracle": summary(oracle), "baseline": summary(base),
           "audit": audit}
    json.dump(rep, open(os.path.join(OUT, "evidence_report.json"), "w"),
              indent=2)

    # ---- VERDICT (audit R14): the gate the bootcamp actually reads ----
    o = rep["oracle"]
    oracle_clears_bar = (o["double_goal_days"] == o["days"]
                         and o["breaches"] == 0 and o["days"] > 0)
    verdict = {
        "source": src, "bar": rep["bar"],
        "oracle_clears_bar_every_day": oracle_clears_bar,
        "oracle_mean_pnl_pct": o["mean_pnl_pct"],
        "ruling_required_from_monty": not oracle_clears_bar,
        "note": ("Oracle (a LOWER-BOUND probe) cleared the bar every day."
                 if oracle_clears_bar else
                 "Oracle could NOT clear the bar every day — Monty must rule "
                 "on the bar before boot camp is trusted (the bar bends only "
                 "by his hand). Boot camp will run but stamps this warning "
                 "on every report."),
    }
    json.dump(verdict, open(os.path.join(OUT, "VERDICT.json"), "w"), indent=2)
    run.log(**{f"oracle_{k}": v for k, v in o.items()})
    run.log(**{f"base_{k}": v for k, v in rep["baseline"].items()})
    run.artifact(os.path.join(OUT, "evidence_report.json"))
    run.finish(f"gauntlet on {src}: oracle {o['mean_pnl_pct']}%/day, "
               f"clears bar={oracle_clears_bar}")
    print(json.dumps(rep, indent=2))
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__":
    main()
