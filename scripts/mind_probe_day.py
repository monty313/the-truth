#!/usr/bin/env python3
"""Run Mind Probe on one day — entry point for conversational diagnosis.

5W+I: WHO Fable 5. WHAT load brain+day, run Mind Probe, write JSON dump.
WHEN 2026-07-24 Phase 1. WHERE python scripts/mind_probe_day.py <brain> [day] [goal] [floor]
WHY talk to the policy about chart patterns without changing weights/obs.
INTERCONNECTED: telemetry/mind_probe.py, inference/loader, training/gpu_data, doctrine/STANDING_LAWS.md

CHANGE LOG:
- 2026-07-24 created — WHY Phase 1 MRI entry point.
# NEXT EDITOR: append dated WHY; keep this line.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
import numpy as np
from core.configs import path as rpath
from telemetry.mind_probe import load_and_probe

def main():
    brain = sys.argv[1] if len(sys.argv) > 1 else "PROVEN_SPRINT_row04_clear24_2026-07-20"
    day_i = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    goal = float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
    floor = float(sys.argv[4]) if len(sys.argv) > 4 else 3.5
    src = rpath("data", "raw", "XAUUSD_curriculum_2026.csv")
    if not os.path.exists(src):
        src = rpath("data", "raw", "XAUUSD_M1_drill.csv")
    tag = os.path.splitext(os.path.basename(src))[0]
    cache = rpath("artifacts", f"gpu_cache_{tag}.npz")
    from training.gpu_data import build_day_tensors
    days_obs, days_phys, day_lens, dates, cols = build_day_tensors(
        src, cache_path=cache if os.path.exists(os.path.dirname(cache)) else None)
    if day_i < 0 or day_i >= days_obs.shape[0]:
        print(f"day_index {day_i} out of range [0, {days_obs.shape[0]-1}]")
        sys.exit(1)
    L = int(day_lens[day_i])
    day_obs = np.asarray(days_obs[day_i, :L], dtype=np.float32)
    day_phys = np.asarray(days_phys[day_i, :L], dtype=np.float32)
    label = str(dates[day_i]) if dates is not None and day_i < len(dates) else str(day_i)
    print(f"Probing brain={brain} day={label} (idx {day_i}) goal={goal} floor={floor} ...")
    dump = load_and_probe(brain_name=brain, day_obs=day_obs, day_phys=day_phys,
                          cols=list(cols), goal_pct=goal, floor_pct=floor,
                          day_index=day_i, day_label=label)
    out_dir = rpath("artifacts", "mind_dumps")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"mind_{brain}_{label}.json")
    dump.write(out_path)
    print()
    print("=== MIND PROBE SUMMARY ===")
    print(dump.summary)
    print()
    print(f"Full dump: {out_path}")
    print(f"Decisions: {dump.n_decisions}")
    print(f"Pull buy: {dump.n_pull_buy_bars} acted={dump.pull_buy_seen_and_acted} held={dump.pull_buy_seen_and_held}")
    print(f"Pull sell: {dump.n_pull_sell_bars} acted={dump.pull_sell_seen_and_acted} held={dump.pull_sell_seen_and_held}")
    print(f"Cont buy/sell: {dump.n_cont_buy_bars}/{dump.n_cont_sell_bars}")
    print(f"Rev buy/sell: {dump.n_rev_buy_bars}/{dump.n_rev_sell_bars}")
    print(f"Mean op entropy: {dump.mean_op_entropy:.4f}")

if __name__ == "__main__":
    main()
