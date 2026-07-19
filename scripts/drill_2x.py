"""Drill ONE hot week to 2x — resumable trainer.
5W+I: WHO Claude trains, Monty's mandate. WHAT trains the bot on one fixed hot
week at the fixed goal/floor until it makes >= 2x goal EVERY day with 0 floor
breaches, using its own lot-size head to create edge. WHEN 2026-07-19. WHERE
container (torch). WHY Monty: "don't stop until the bot learns to make 2x the
goal without breaching the daily DD — nothing more, nothing less." Resumable via
the 'drill2x' checkpoint so training accumulates across runs.
INTERCONNECTED: training.env/ppo/rewards, evaluation.champion.bench, data_io.

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-19  created — WHY: focused fixed-target drill loop (boot camp trains
  randomized any-X + rebuilds features each call; this builds once, trains the
  exact 2x target, checkpoints, and loops until success — Monty's mandate).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                               # noqa: E402
import pandas as pd                                             # noqa: E402
from core.configs import goals_cfg, path as rpath              # noqa: E402
from data_io.loader import read_mt5_m1, trading_days           # noqa: E402
from features.engine import build_features                     # noqa: E402
from training.env import TradingEnv                            # noqa: E402
from training.ppo import PPO                                   # noqa: E402
from training.rewards import RewardEngine                      # noqa: E402
from evaluation.champion import bench                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=8.0)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--ckpt", default="drill2x")
    a = ap.parse_args()

    g = goals_cfg()
    GOAL, FLOOR = float(g["goal_pct"]), float(g["floor_pct"])
    BAR = 2.0 * GOAL

    paths = sorted(glob.glob(rpath("data", "XAUUSD_M1_*.csv")))
    m1 = read_mt5_m1(paths[0])
    m1 = m1.loc[m1.index >= m1.index.max() - pd.Timedelta("21D")]   # mirror boot camp
    F = build_features(m1)
    week = trading_days(F)[1:][:5]
    print("DRILL week = %s .. %s (%d days) | BAR=+%.1f%%/day, floor -%.1f%%"
          % (week[0][0], week[-1][0], len(week), BAR, FLOOR), flush=True)

    re_engine = RewardEngine()
    env = TradingEnv(week, GOAL, FLOOR, reward_engine=re_engine, goal_ranges=None)
    ppo = PPO(env)
    try:
        ppo.load(a.ckpt)
        print("resumed checkpoint:", a.ckpt, flush=True)
    except Exception as e:
        print("fresh start (%s)" % type(e).__name__, flush=True)

    def evaluate():
        env.goal_ranges = None
        env.goal0, env.floor0 = GOAL, FLOOR
        b = bench(ppo, len(week))
        dd = b.get("day_details", [])
        pnls = [round(float(d.get("pnl_pct", 0.0)), 3) for d in dd]
        br = int(b.get("breaches", 1))
        ok = (br == 0 and len(pnls) == len(week) and all(p >= BAR for p in pnls))
        return pnls, br, ok, b

    prog = rpath("artifacts", "drill2x_progress.json")
    t0, u, best = time.time(), 0, -1e9
    # best_trade = the checkpoint "where the bot was trading well" (Monty 2026-07-19).
    # Judged by REAL trading, not reward: most days>=2x, then biggest week %, 0 breaches.
    best_trade = (-1, -1e9)
    hit = False
    while time.time() - t0 < a.minutes * 60:
        batch = [ppo.play_day(i) for i in range(len(week))]
        mean_r = float(np.mean([b[5].sum() for b in batch]))
        if mean_r > best:
            best = mean_r
        ppo.update(batch)
        u += 1
        if u % a.eval_every == 0:
            pnls, br, ok, b = evaluate()
            hits = sum(1 for p in pnls if p >= BAR)
            wk = round(float(sum(pnls)), 3)
            print("upd %4d | reward %8.2f | week pnl%% %s | >=2x %d/%d | week %+.2f%% | breaches %d"
                  % (u, mean_r, pnls, hits, len(week), wk, br), flush=True)
            ppo.save(a.ckpt)                                   # latest — resume next chunk here
            if br == 0 and (hits, wk) > best_trade:
                best_trade = (hits, wk)
                ppo.save("best_trading")                       # <-- the good-trading checkpoint
                print("   ^ NEW best-trading checkpoint: %d/%d days at 2x, week %+.2f%%, 0 breaches"
                      % (hits, len(week), wk), flush=True)
            json.dump({"updates_run": u, "week_pnl_pct": pnls, "days_at_2x": hits,
                       "of": len(week), "week_pct": wk, "breaches": br, "BAR": BAR,
                       "hit_2x_everyday": ok,
                       "best_trading_days_at_2x": best_trade[0],
                       "best_trading_week_pct": best_trade[1]}, open(prog, "w"), indent=2)
            if ok:
                print(">>> SUCCESS: 2x every day, 0 breaches", flush=True)
                ppo.save("best_trading")
                hit = True
                break

    ppo.save(a.ckpt)
    pnls, br, ok, b = evaluate()
    hits = sum(1 for p in pnls if p >= BAR)
    out = {"updates_this_run": u, "drill_week": [week[0][0], week[-1][0]],
           "week_pnl_pct": pnls, "days_at_2x": hits, "of": len(week),
           "breaches": br, "BAR": BAR, "hit_2x_everyday": ok,
           "best_reward": round(best, 2)}
    json.dump(out, open(prog, "w"), indent=2)
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
