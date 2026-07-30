#!/usr/bin/env python3
"""Batch Perception scoreboard — pull present vs policy acted across curriculum days.

5W+I -----------------------------------------------------------------
WHO:   Fable 5 for Monty.
WHAT:  Runs Mind Probe over many days and aggregates bread-and-butter recognition:
       n_pull bars, acted vs held, mean alt-style signal from hold-on-pull rate.
WHEN:  2026-07-24 Phase 1 residual + Phase 4.
WHERE: python scripts/perception_scoreboard.py [brain] [goal] [floor] [max_days]
WHY:   Consistency requires the policy to see chart patterns that produce clears.
       This scoreboard is the Perception metric for IRAC.
INTERCONNECTED: telemetry/mind_probe.py, doctrine/STANDING_LAWS.md, scripts/diagnose_day.py
----------------------------------------------------------------------

CHANGE LOG:
- 2026-07-24 created — WHY: batch Perception scoreboard for self-heal plan completion.
# NEXT EDITOR: append dated WHY; keep this line.
"""
from __future__ import annotations

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np

from core.configs import path as rpath
from telemetry.mind_probe import load_and_probe
from telemetry.ghost_trades import build_ghosts


def main():
    brain = sys.argv[1] if len(sys.argv) > 1 else "PROVEN_SPRINT_row04_clear24_2026-07-20"
    goal = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
    max_days = int(sys.argv[4]) if len(sys.argv) > 4 else 90

    src = rpath("data", "raw", "XAUUSD_curriculum_2026.csv")
    if not os.path.exists(src):
        src = rpath("data", "raw", "XAUUSD_M1_drill.csv")
    tag = os.path.splitext(os.path.basename(src))[0]
    cache = rpath("artifacts", f"gpu_cache_{tag}.npz")

    from training.gpu_data import build_day_tensors
    days_obs, days_phys, day_lens, dates, cols = build_day_tensors(
        src, cache_path=cache if os.path.exists(os.path.dirname(cache)) else None
    )

    n = min(int(days_obs.shape[0]), max_days)
    rows = []
    tot_pull = tot_acted = tot_held = tot_high_miss = 0
    tot_wrong_bull = tot_wrong_bear = 0
    side_bulls, side_bears = [], []

    print(f"Perception scoreboard brain={brain} days=0..{n-1} goal={goal} floor={floor}")
    for i in range(n):
        L = int(day_lens[i])
        day_obs = np.asarray(days_obs[i, :L], dtype=np.float32)
        day_phys = np.asarray(days_phys[i, :L], dtype=np.float32)
        label = str(dates[i]) if dates is not None and i < len(dates) else str(i)
        try:
            dump = load_and_probe(
                brain_name=brain, day_obs=day_obs, day_phys=day_phys,
                cols=list(cols), goal_pct=goal, floor_pct=floor,
                day_index=i, day_label=label,
            )
        except FileNotFoundError as e:
            print(f"FAIL load brain: {e}")
            sys.exit(1)
        ghosts = build_ghosts(dump)
        pull = dump.n_pull_buy_bars + dump.n_pull_sell_bars
        acted = dump.pull_buy_seen_and_acted + dump.pull_sell_seen_and_acted
        held = dump.pull_buy_seen_and_held + dump.pull_sell_seen_and_held
        tot_pull += pull
        tot_acted += acted
        tot_held += held
        tot_high_miss += ghosts.n_high_miss_pull
        tot_wrong_bull += int(getattr(dump, "n_wrong_side_under_bull", 0))
        tot_wrong_bear += int(getattr(dump, "n_wrong_side_under_bear", 0))
        side_bulls.append(float(getattr(dump, "side_bias_bull", 0.0)))
        side_bears.append(float(getattr(dump, "side_bias_bear", 0.0)))
        rows.append({
            "day": label, "idx": i, "pull": pull, "acted": acted, "held": held,
            "high_miss": ghosts.n_high_miss_pull, "entropy": dump.mean_op_entropy,
            "side_bias_bull": float(getattr(dump, "side_bias_bull", 0.0)),
            "side_bias_bear": float(getattr(dump, "side_bias_bear", 0.0)),
            "wrong_side_bull": int(getattr(dump, "n_wrong_side_under_bull", 0)),
            "wrong_side_bear": int(getattr(dump, "n_wrong_side_under_bear", 0)),
            "forward_ok": int(getattr(dump, "forward_ok", 0)),
            "forward_fail": int(getattr(dump, "forward_fail", 0)),
        })
        if (i + 1) % 10 == 0:
            print(f"  ... {i+1}/{n} days")

    act_rate = (tot_acted / tot_pull) if tot_pull else 0.0
    hold_rate = (tot_held / tot_pull) if tot_pull else 0.0
    msb = float(np.mean(side_bulls)) if side_bulls else 0.0
    mss = float(np.mean(side_bears)) if side_bears else 0.0
    summary = {
        "brain": brain, "goal": goal, "floor": floor, "n_days": n,
        "total_pull_bars": tot_pull,
        "total_acted_on_pull": tot_acted,
        "total_held_on_pull": tot_held,
        "total_high_miss_pull": tot_high_miss,
        "act_rate_on_pull": round(act_rate, 4),
        "hold_rate_on_pull": round(hold_rate, 4),
        "perception_flag": hold_rate > 0.5 and tot_pull > 50,
        "mean_side_bias_bull": round(msb, 4),
        "mean_side_bias_bear": round(mss, 4),
        "total_wrong_side_under_bull": tot_wrong_bull,
        "total_wrong_side_under_bear": tot_wrong_bear,
        "wrong_side_flag": tot_wrong_bull > 50 and msb < 0.05,
        "days": rows,
    }

    out_dir = rpath("artifacts", "mind_dumps")
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"perception_scoreboard_{brain}.json")
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print("=== PERCEPTION / SIDE SCOREBOARD ===")
    print(f"Days: {n}")
    print(f"Pull bars: {tot_pull}")
    print(f"Acted on pull: {tot_acted} ({act_rate*100:.1f}%)")
    print(f"Held on pull: {tot_held} ({hold_rate*100:.1f}%)")
    print(f"High-miss pull (Ghost): {tot_high_miss}")
    print(f"Side-bias bull/bear: {msb:+.4f} / {mss:+.4f}")
    print(f"Wrong-side under bull/bear cont: {tot_wrong_bull} / {tot_wrong_bear}")
    if summary["wrong_side_flag"]:
        print("FLAG: wrong-side under bull — candidate WrongSide class for self-heal dials.")
    if summary["perception_flag"]:
        print("FLAG: high hold-on-pull rate — candidate Perception/Policy issue for IRAC.")
    if not summary["perception_flag"] and not summary["wrong_side_flag"]:
        print("No strong hold or wrong-side aggregate flag.")
    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
