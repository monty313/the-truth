#!/usr/bin/env python3
"""Give the Diagnostic LLM what it asked for to learn the hesitation principles.

1) Rebuild feature cache under locked Gravity sets
2) Multi-day Mind Probe + Ghosts (policy_hold under trend+pull/cont)
3) Aggregate IRAC
4) prove_it baseline
5) Print host GPU climb command (frontier training hours)

Usage:
  python scripts/give_llm_what_it_needs.py [brain] [goal] [floor] [--days 12] [--sprint-minutes 0]
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np

from core.configs import path as rpath, load
from training.gpu_data import build_day_tensors
from telemetry.mind_probe import load_and_probe
from telemetry.ghost_trades import build_ghosts


def main():
    brain = sys.argv[1] if len(sys.argv) > 1 else "PROVEN_SPRINT_row04_clear24_2026-07-20"
    goal = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
    floor = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5
    n_days = 12
    sprint_min = 0.0
    if "--days" in sys.argv:
        n_days = int(sys.argv[sys.argv.index("--days") + 1])
    if "--sprint-minutes" in sys.argv:
        sprint_min = float(sys.argv[sys.argv.index("--sprint-minutes") + 1])

    rw = load("rewards")
    print("=" * 64)
    print("GIVE LLM WHAT IT NEEDS — hesitation extinction curriculum")
    print(f"brain={brain}  goal={goal}%  floor={floor}%")
    print(f"w_pullback_with_htf={rw.get('w_pullback_with_htf')}")
    print("=" * 64)

    src = rpath("data", "XAUUSD_curriculum_2026.csv")
    cache = rpath("artifacts", "gpu_cache_XAUUSD_curriculum_2026.npz")
    print("\n[1] Feature cache under locked Gravity sets...")
    t0 = time.time()
    days_obs, days_phys, day_lens, dates, cols = build_day_tensors(
        src, cache_path=cache, verbose=True
    )
    cols = list(cols)
    D = int(days_obs.shape[0])
    print(f"    done in {time.time()-t0:.1f}s | days={D} cols={len(cols)}")

    print(f"\n[2] Mind Probe + Ghosts on {n_days} days (policy_hold focus)...")
    indices = list(range(0, D, max(1, D // n_days)))[:n_days]
    skip_total = Counter()
    ph_total = 0
    high_miss_total = 0
    ph_ghosts_total = 0
    day_rows = []
    out_dir = rpath("artifacts", "llm_curriculum")
    os.makedirs(out_dir, exist_ok=True)

    for di in indices:
        L = int(day_lens[di])
        label = str(dates[di])
        dump = load_and_probe(
            brain_name=brain,
            day_obs=np.asarray(days_obs[di, :L], np.float32),
            day_phys=np.asarray(days_phys[di, :L], np.float32),
            cols=cols,
            goal_pct=goal,
            floor_pct=floor,
            day_index=di,
            day_label=label,
        )
        ghosts = build_ghosts(dump)
        sc = getattr(dump, "skip_counts", {}) or {}
        skip_total.update(sc)
        ph = int(getattr(dump, "n_policy_hold_on_setup", 0))
        ph_total += ph
        high_miss_total += int(ghosts.n_high_miss_pull)
        ph_ghosts_total += int(getattr(ghosts, "n_policy_hold_ghosts", 0))
        day_rows.append({
            "day": label,
            "pull": dump.n_pull_buy_bars + dump.n_pull_sell_bars,
            "policy_hold_on_setup": ph,
            "high_miss_pull": ghosts.n_high_miss_pull,
            "policy_hold_ghosts": getattr(ghosts, "n_policy_hold_ghosts", 0),
            "skip_counts": sc,
            "summary": dump.summary,
        })
        print(
            f"    {label}: pull={day_rows[-1]['pull']} "
            f"policy_hold={ph} high_miss={ghosts.n_high_miss_pull}"
        )

    print("\n[3] Aggregate IRAC...")
    irac = {
        "issue": (
            "Policy hesitates when Gravity setup is visible: firm HTF momentum + "
            "LTF pullback/continuation, but skip_reason=policy_hold dominates."
        ),
        "rule": (
            "Bread-and-butter (LLM_REGIME_DEFINITIONS): pullback under trend on "
            "locked sets A/B/C must be acted unless mask_veto or risk floor."
        ),
        "application": {
            "days_probed": len(day_rows),
            "sum_policy_hold_on_setup": ph_total,
            "sum_high_miss_pull": high_miss_total,
            "sum_policy_hold_ghosts": ph_ghosts_total,
            "skip_counts_total": dict(skip_total),
            "w_pullback_with_htf": rw.get("w_pullback_with_htf"),
            "days": day_rows,
        },
        "conclusion": {
            "class": "Policy",
            "cure_in_config": "w_pullback_with_htf=0.25 (already set)",
            "cure_in_training": (
                "GPU consistency_sprint hours on THIS cache; adopt only if "
                "prove_it clear rate rises and breach stays 0."
            ),
            "not_perception": "pull/cont flags present; ghosts show held setups.",
        },
    }
    irac_path = os.path.join(out_dir, f"irac_{brain}.json")
    with open(irac_path, "w") as f:
        json.dump(irac, f, indent=2)
    print(f"    wrote {irac_path}")
    print(f"    policy_hold_on_setup total={ph_total}  high_miss_pull={high_miss_total}")

    print("\n[4] prove_it baseline...")
    os.system(f"{sys.executable} scripts/prove_it.py {brain} {goal} {floor}")

    if sprint_min > 0:
        print(f"\n[5] consistency_sprint --minutes {sprint_min}...")
        os.system(
            f"{sys.executable} scripts/consistency_sprint.py "
            f"--minutes {sprint_min} --envs 64"
        )
    else:
        print("\n[5] Frontier training (host GPU — required for 80% climb):")
        print("    python scripts/consistency_sprint.py --minutes 600 --envs 256")
        print(f"    python scripts/prove_it.py <sprint_record_brain> {goal} {floor}")
        print(
            f"    python scripts/give_llm_what_it_needs.py "
            f"<sprint_record_brain> {goal} {floor}"
        )

    print("\n[6] Multi-symbol: configs/data.yaml lists expansion EURUSD, GBPUSD, US30")
    print("    but only XAUUSD CSVs are in data/. Provide those M1 curricula to unlock.")
    print("\nDONE — LLM curriculum artifacts under artifacts/llm_curriculum/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
