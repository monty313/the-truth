"""Replay a saved brain on the drill week and print the evidence.
Recreate the demonstrated performance ANY TIME:
    python scripts/replay_best.py                       (replays the live best_trading)
    python scripts/replay_best.py --ckpt PROVEN_2x_2026-07-19   (the frozen proof)
WHY (Monty): keep the evidence + the exact bot state so the 2x day is reproducible
on demand. Loads the checkpoint (identified by its hash) and re-runs it.
"""
import argparse
import glob
import hashlib
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd                                              # noqa: E402
from core.configs import goals_cfg, path as rpath               # noqa: E402
from data_io.loader import read_mt5_m1, trading_days            # noqa: E402
from features.engine import build_features                      # noqa: E402
from training.env import TradingEnv                             # noqa: E402
from training.ppo import PPO                                    # noqa: E402
from training.rewards import RewardEngine                       # noqa: E402
from evaluation.champion import bench                           # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", default="best_trading",
                help="checkpoint name in artifacts/checkpoints (no .pt)")
a = ap.parse_args()

g = goals_cfg(); GOAL = float(g["goal_pct"]); FLOOR = float(g["floor_pct"]); BAR = 2 * GOAL
ckpt = rpath("artifacts", "checkpoints", a.ckpt + ".pt")
digest = (hashlib.sha256(open(ckpt, "rb").read()).hexdigest()[:16]
          if os.path.exists(ckpt) else "MISSING")

m1 = read_mt5_m1(sorted(glob.glob(rpath("data", "XAUUSD_M1_*.csv")))[0])
m1 = m1.loc[m1.index >= m1.index.max() - pd.Timedelta("21D")]
F = build_features(m1); week = trading_days(F)[1:][:5]
env = TradingEnv(week, GOAL, FLOOR, reward_engine=RewardEngine(), goal_ranges=None)
ppo = PPO(env); ppo.load(a.ckpt)
env.goal_ranges = None; env.goal0, env.floor0 = GOAL, FLOOR
b = bench(ppo, len(week))

print("STATE  %s.pt  sha256[:16] = %s" % (a.ckpt, digest))
print("WEEK   %s .. %s   target +%.1f%%/day   floor -%.1f%%"
      % (week[0][0], week[-1][0], BAR, FLOOR))
best = ("", -1e9)
for i, d in enumerate(b.get("day_details", [])):
    p = float(d.get("pnl_pct", 0.0))
    flag = "   <== 2x+ CLEARED" if p >= BAR else ""
    print("  %s   pnl %+6.2f%%   trades %3d   breach %s%s"
          % (week[i][0], p, d.get("trades", 0), d.get("breached"), flag))
    if p > best[1]:
        best = (week[i][0], p)
print("BEST DAY  %s = %+.2f%%    total breaches: %d"
      % (best[0], best[1], int(b.get("breaches", 0))))
