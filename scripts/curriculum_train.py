"""Curriculum trainer — master days ONE AT A TIME (Monty's spec, 2026-07-20).
Ladder: start at the good week; add ONE day only when ALL accumulated days make
>= GOAL (2.5%) with ZERO -FLOOR (4%) breach, held for `mastery` clean evals in a
row. FAIL any day -> RESTART at day 1 (the brain KEEPS its learning). Climb until
TARGET (100) days or the data runs out. Masks stay on; the bot is free on
everything else (its own strategy + lot size). At every NEW highest level it
freezes a dated+hashed checkpoint and logs full detail so we can always revert.
USAGE:  python scripts/curriculum_train.py --minutes 600

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  created — WHY: Monty's day-by-day curriculum (restart-on-fail, up to
  100 days) + big do-nothing penalty in the reward. Warm-starts from best_trading.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import argparse
import datetime
import glob
import hashlib
import json
import os
import shutil
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd                                             # noqa: E402  (kept for parity/deps)
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
    ap.add_argument("--target-days", type=int, default=100)
    ap.add_argument("--mastery", type=int, default=2)          # clean evals in a row to advance
    ap.add_argument("--updates-per-eval", type=int, default=3)
    ap.add_argument("--start", default="2026-01-27")           # the good week (first day)
    ap.add_argument("--week-end", default="2026-02-02")        # good week last day (ordered first)
    ap.add_argument("--ckpt", default="curric_live")
    a = ap.parse_args()

    g = goals_cfg(); GOAL = float(g["goal_pct"]); FLOOR = float(g["floor_pct"])
    files = (sorted(glob.glob(rpath("data", "XAUUSD_curriculum_*.csv")))
             or sorted(glob.glob(rpath("data", "XAUUSD_M1_*.csv"))))
    m1 = read_mt5_m1(files[0]); F = build_features(m1)
    alldays = trading_days(F)
    pool = [d for d in alldays if str(d[0])[:10] >= a.start] or alldays
    # Order: the GOOD WEEK first, then the rest — each easiest-first (a day that moved
    # more is easier to make +2.5% on). Starts with "the week we did well" (Monty) and
    # climbs into harder days. Restart-on-fail still applies at every level.
    _m = m1.copy(); _m["_d"] = _m.index.date
    _rng = {}
    for _dt, _grp in _m.groupby("_d"):
        _o = float(_grp["open"].iloc[0]) or 1.0
        _rng[str(_dt)] = float((_grp["high"].max() - _grp["low"].min()) / _o * 100.0)
    _byrange = lambda d: _rng.get(str(d[0])[:10], 0.0)
    _wk = sorted([d for d in pool if str(d[0])[:10] <= a.week_end], key=_byrange, reverse=True)
    _rest = sorted([d for d in pool if str(d[0])[:10] > a.week_end], key=_byrange, reverse=True)
    pool = _wk + _rest
    TARGET = min(a.target_days, len(pool))
    print("CURRICULUM | pool %d days from %s | target %d | day goal +%.1f%%, floor -%.1f%% | masks ON, all other actions free"
          % (len(pool), str(pool[0][0])[:10], TARGET, GOAL, FLOOR), flush=True)

    re_engine = RewardEngine()

    def make_env(days):
        e = TradingEnv(days, GOAL, FLOOR, reward_engine=re_engine, goal_ranges=None)
        e.goal0, e.floor0 = GOAL, FLOOR
        return e

    level = 1
    env = make_env(pool[:level]); ppo = PPO(env)
    loaded = None
    for name in (a.ckpt, "best_meta", "best_trading"):     # resume, else warm-start from the proof
        if ppo.load(name):                                # ppo.load returns True only on success
            loaded = name; break
    print(("warm-start from: " + loaded) if loaded else "fresh start (no checkpoint)", flush=True)

    def passes(n):
        b = bench(ppo, n)
        dd = b.get("day_details", [])
        ok = (len(dd) == n
              and all(d.get("pnl_pct", 0.0) >= GOAL and not d.get("breached") for d in dd))
        return ok, dd

    histdir = rpath("artifacts", "checkpoints", "history"); os.makedirs(histdir, exist_ok=True)
    histmd = rpath("artifacts", "CURRICULUM_HISTORY.md")
    prog = rpath("artifacts", "curriculum_progress.json")

    def milestone(lvl, dd):
        bp = rpath("artifacts", "checkpoints", a.ckpt + ".pt")
        hh = hashlib.sha256(open(bp, "rb").read()).hexdigest()[:12]
        stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        frozen = "curric_L%03d_%s_%s.pt" % (lvl, stamp, hh)
        shutil.copy2(bp, os.path.join(histdir, frozen))
        first = not os.path.exists(histmd)
        with open(histmd, "a") as fh:
            if first:
                fh.write("# Curriculum history - new highest level, newest at bottom\n"
                         "Revert: copy artifacts/checkpoints/history/<file> over "
                         "artifacts/checkpoints/curric_live.pt\n\n")
            fh.write("## LEVEL %d days clean  -  %s\n" % (lvl, stamp))
            fh.write("- frozen: artifacts/checkpoints/history/%s  (sha256[:12] %s)\n" % (frozen, hh))
            fh.write("- per-day pnl%%: %s\n" % [round(d.get("pnl_pct", 0), 2) for d in dd])
            fh.write("- every day >= %.1f%%, 0 breaches, %d consecutive days\n\n" % (GOAL, lvl))
        print("   *** MILESTONE: %d days clean -> frozen history/%s" % (lvl, frozen), flush=True)

    t0 = time.time(); consec = 0; maxlevel = 0; evals = 0
    while time.time() - t0 < a.minutes * 60 and level <= TARGET:
        for _ in range(a.updates_per_eval):
            batch = [ppo.play_day(i) for i in range(level)]
            ppo.update(batch)
        ppo.save(a.ckpt)
        ok, dd = passes(level); evals += 1
        perday = [round(d.get("pnl_pct", 0), 2) for d in dd]
        if ok:
            consec += 1
            print("level %3d/%d | clean %d/%d | per-day %s"
                  % (level, TARGET, consec, a.mastery, perday), flush=True)
            if consec >= a.mastery:
                if level > maxlevel:
                    maxlevel = level; milestone(level, dd)
                level += 1; consec = 0
                env = make_env(pool[:level]); ppo.env = env
                print(">>> ADVANCE -> %d days" % level, flush=True)
        else:
            print("level %3d/%d | FAIL per-day %s -> RESTART at day 1"
                  % (level, TARGET, perday), flush=True)
            level = 1; consec = 0
            env = make_env(pool[:level]); ppo.env = env
        json.dump({"level": level, "max_level": maxlevel, "consec_clean": consec,
                   "target": TARGET, "last_per_day_pnl": perday, "evals": evals},
                  open(prog, "w"), indent=2)
    print("CURRICULUM chunk done | highest level reached: %d days (target %d)" % (maxlevel, TARGET), flush=True)


if __name__ == "__main__":
    main()
