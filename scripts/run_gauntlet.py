"""Phase-4 driver: audit -> oracle -> baseline -> evidence report.
5W+I: WHO Claude/Monty gate. WHAT runs the Gauntlet on real data if present,
else synthetic (ADR-0010, clearly labeled). WHEN 2026-07-19. WHY hard gate
before training. INTERCONNECTED: gauntlet.py, tracker, artifacts/gauntlet/.
"""
import sys, os, json, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd
from data_io.loader import read_mt5_m1, synthetic_m1, trading_days
from features.engine import build_features
from backtesting.gauntlet.gauntlet import data_audit, oracle_day, baseline_day, run_over_days
from experiments.tracker import Run
from telemetry.logging_setup import setup

log = setup("gauntlet")
GOAL, FLOOR = 2.5, 4.0     # configs/goals.yaml boot-camp values
OUT = "artifacts/gauntlet"; os.makedirs(OUT, exist_ok=True)

real = sorted(glob.glob("../data/XAUUSD_M1_*.csv") + glob.glob("data/XAUUSD_M1_*.csv"))
if real:
    src, m1 = "REAL_XAUUSD", read_mt5_m1(real[0])
    m1 = m1.last("30D")   # audit window; week picking uses full file separately
else:
    src, m1 = "SYNTHETIC (gold-like — ADR-0010, replace when zip lands)", synthetic_m1(days=10, seed=7)
log.info("source: %s rows=%d", src, len(m1))

run = Run("gauntlet", symbols=["XAUUSD" if real else "SYNTH"], timeframes=["all-matrix"],
          data_window=f"{m1.index[0]}..{m1.index[-1]}", seed=7,
          assumptions={"fills": "paranoid(ADR-0009)", "spread": "recorded column",
                       "source": src})
audit = data_audit(m1)
json.dump(audit, open(f"{OUT}/audit.json", "w"), indent=2)
log.info("audit: %s", audit)

F = build_features(m1)
days = trading_days(F)[1:]           # drop warmup day
oracle = run_over_days(days, oracle_day, GOAL, FLOOR, "oracle")
base   = run_over_days(days, baseline_day, GOAL, FLOOR, "baseline")
oracle.to_csv(f"{OUT}/oracle_days.csv", index=False)
base.to_csv(f"{OUT}/baseline_days.csv", index=False)

def summary(df):
    return {"days": len(df), "mean_pnl_pct": round(df.pnl_pct.mean(), 3),
            "min": round(df.pnl_pct.min(), 3), "max": round(df.pnl_pct.max(), 3),
            "goal_hit_rate": round(df.goal_hit.mean(), 3),
            "double_goal_days": int((df.pnl_pct >= 2*GOAL).sum()),
            "breaches": int(df.breached.sum())}
report = {"source": src, "goal": GOAL, "floor": FLOOR, "bar": f"+{2*GOAL}% EVERY day, zero -{FLOOR}% touches",
          "oracle": summary(oracle), "baseline": summary(base), "audit": audit}
json.dump(report, open(f"{OUT}/evidence_report.json", "w"), indent=2)
run.log(**{f"oracle_{k}": v for k, v in report["oracle"].items() if isinstance(v,(int,float))})
run.log(**{f"base_{k}": v for k, v in report["baseline"].items() if isinstance(v,(int,float))})
run.artifact(f"{OUT}/evidence_report.json")
run.finish(f"gauntlet on {src}: oracle mean {report['oracle']['mean_pnl_pct']}%/day, "
           f"baseline mean {report['baseline']['mean_pnl_pct']}%/day")
print(json.dumps(report, indent=2))
