"""Continuous META training — the any-X brain (Monty's original ask).
5W+I: WHO Claude trains, Monty's spec. WHAT trains ONE brain across RANDOMIZED
goal/floor ranges (goals.yaml goal_conditioning), so it handles whatever numbers
Monty types WITHOUT retraining. Evals + checkpoints at Monty's headline target
(2.5/4). Warm-starts from the proof brain (best_trading) so nothing is lost.
WHEN 2026-07-19. WHERE container or PC. WHY: the deliverable is goal-conditioned
meta-RL, not a single-setting specialist. Resumable in chunks.
USAGE:  python scripts/meta_train.py --minutes 600

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-19  created — WHY: switch training from the fixed-target drill (drill_2x,
  goal_ranges=None) to the META objective (randomized any-X). Separate best_meta
  lineage so the fixed 2x proof (best_trading) stays intact.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import statistics
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                              # noqa: E402
import pandas as pd                                            # noqa: E402
from core.configs import goals_cfg, path as rpath              # noqa: E402
from data_io.loader import read_mt5_m1, trading_days           # noqa: E402
from features.engine import build_features                     # noqa: E402
from training.env import TradingEnv                            # noqa: E402
from training.ppo import PPO                                   # noqa: E402
from training.rewards import RewardEngine                      # noqa: E402
from evaluation.champion import bench                          # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=600.0)
    ap.add_argument("--eval-every", type=int, default=8)
    ap.add_argument("--ckpt", default="meta_live")     # working checkpoint
    ap.add_argument("--best", default="best_meta")     # best any-X brain
    a = ap.parse_args()

    g = goals_cfg(); GOAL = float(g["goal_pct"]); FLOOR = float(g["floor_pct"]); BAR = 2 * GOAL
    gc = g.get("goal_conditioning", {})
    RANGES = ((tuple(gc["goal_range"]), tuple(gc["floor_range"]))
              if gc.get("randomize_in_training") else None)
    if RANGES is None:
        print("goals.yaml: randomize_in_training is off — nothing to meta-train.", flush=True)
        return

    m1 = read_mt5_m1(sorted(glob.glob(rpath("data", "XAUUSD_M1_*.csv")))[0])
    m1 = m1.loc[m1.index >= m1.index.max() - pd.Timedelta("21D")]
    F = build_features(m1); week = trading_days(F)[1:][:5]
    print("META training | goal in %s, floor in %s | %d days | headline exam +%.1f%%/day"
          % (RANGES[0], RANGES[1], len(week), BAR), flush=True)

    re_engine = RewardEngine()
    env = TradingEnv(week, GOAL, FLOOR, reward_engine=re_engine, goal_ranges=RANGES)  # TRAIN randomized
    ppo = PPO(env)

    def evaluate():
        env.goal_ranges = None                       # exam at Monty's actual numbers
        env.goal0, env.floor0 = GOAL, FLOOR
        b = bench(ppo, len(week))
        env.goal_ranges = RANGES                     # RESTORE randomized meta training
        dd = b.get("day_details", [])
        pnls = [round(float(d.get("pnl_pct", 0.0)), 3) for d in dd]
        return pnls, int(b.get("breaches", 1)), b

    def cons_of(pnls, br):
        if br != 0 or not pnls:
            return (-1, -1, -1, -1e9)
        sd = round(statistics.pstdev(pnls), 3) if len(pnls) > 1 else 0.0
        return (sum(1 for p in pnls if p >= BAR), sum(1 for p in pnls if p >= GOAL),
                sum(1 for p in pnls if p >= 0), -sd)

    # seed the bar from the existing best_meta (never overwrite it with a worse brain)
    best_cons = (-1, -1, -1, -1e9)
    try:
        ppo.load(a.best); p0, br0, _ = evaluate(); best_cons = cons_of(p0, br0)
        print("existing %s bar: 2x=%d goal=%d +=%d" % (a.best, best_cons[0], best_cons[1], best_cons[2]), flush=True)
    except Exception:
        print("no existing %s yet — bar starts at zero" % a.best, flush=True)
    # warm-start weights: working -> best_meta -> the proof brain best_trading
    loaded = None
    for name in (a.ckpt, a.best, "best_trading"):
        try:
            ppo.load(name); loaded = name; break
        except Exception:
            continue
    print(("warm-started from: " + loaded) if loaded else "fresh start", flush=True)

    prog = rpath("artifacts", "meta_progress.json")
    t0, u, best = time.time(), 0, -1e9
    while time.time() - t0 < a.minutes * 60:
        batch = [ppo.play_day(i) for i in range(len(week))]
        mean_r = float(np.mean([b[5].sum() for b in batch]))
        if mean_r > best:
            best = mean_r
        ppo.update(batch); u += 1
        if u % a.eval_every == 0:
            pnls, br, b = evaluate()
            hits = sum(1 for p in pnls if p >= BAR)
            dgoal = sum(1 for p in pnls if p >= GOAL)
            dpos = sum(1 for p in pnls if p >= 0)
            wk = round(float(sum(pnls)), 3)
            cons = cons_of(pnls, br)
            print("upd %4d | reward %7.2f | @target pnl%% %s | 2x %d/%d goal %d/%d +%d/%d | wk %+.2f%% | br %d"
                  % (u, mean_r, pnls, hits, len(week), dgoal, len(week), dpos, len(week), wk, br), flush=True)
            ppo.save(a.ckpt)
            if cons > best_cons:
                best_cons = cons
                ppo.save(a.best)
                # Versioned, dated, hashed snapshot + full-detail history so we can
                # ALWAYS revert to any good state (Monty 2026-07-19).
                bestpath = rpath("artifacts", "checkpoints", a.best + ".pt")
                hh = hashlib.sha256(open(bestpath, "rb").read()).hexdigest()[:12]
                stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
                histdir = rpath("artifacts", "checkpoints", "history")
                os.makedirs(histdir, exist_ok=True)
                frozen = "meta_%s_%s.pt" % (stamp, hh)
                shutil.copy2(bestpath, os.path.join(histdir, frozen))
                histmd = rpath("artifacts", "CHECKPOINT_HISTORY.md")
                first = not os.path.exists(histmd)
                with open(histmd, "a") as fh:
                    if first:
                        fh.write("# Checkpoint history - every improvement, newest at bottom\n\n"
                                 "Each entry is a FROZEN, revertible brain. Inspect one:\n"
                                 "  python scripts/replay_best.py --ckpt history/<name-without-.pt>\n"
                                 "Revert: copy that file over artifacts/checkpoints/best_meta.pt\n\n")
                    fh.write("## %s  (update %d, mode META any-X)\n" % (stamp, u))
                    fh.write("- frozen checkpoint: artifacts/checkpoints/history/%s\n" % frozen)
                    fh.write("- sha256[:12]: %s\n" % hh)
                    fh.write("- trained goal-range %s, floor-range %s\n" % (RANGES[0], RANGES[1]))
                    fh.write("- at your target (%.1f%%/%.1f%%): per-day %s\n" % (GOAL, FLOOR, pnls))
                    fh.write("- 2x %d/%d | goal %d/%d | positive %d/%d | breaches %d | week %+.2f%%\n\n"
                             % (hits, len(week), dgoal, len(week), dpos, len(week), br, wk))
                print("   ^ NEW best any-X brain - frozen history/%s + logged to CHECKPOINT_HISTORY.md"
                      % frozen, flush=True)
            json.dump({"updates_run": u, "at_target_pnl_pct": pnls, "days_at_2x": hits,
                       "days_at_goal": dgoal, "breaches": br, "mode": "meta_any_X",
                       "goal_range": list(RANGES[0]), "floor_range": list(RANGES[1]),
                       "best_meta_2x": best_cons[0], "best_meta_goal": best_cons[1]},
                      open(prog, "w"), indent=2)
    ppo.save(a.ckpt)
    print("META chunk done: %d updates (meta_live saved; best any-X = %s)" % (u, a.best), flush=True)


if __name__ == "__main__":
    main()
