#!/usr/bin/env python3
"""IRAC diagnostic for one day — Mind Probe + Ghost Trades + reward proposal.

5W+I -----------------------------------------------------------------
WHO:   Fable 5 for Monty.
WHAT:  Load brain+day, run Mind Probe, build Ghost Trades, emit IRAC-style
       diagnosis and a bounded rewards.yaml delta proposal (never applied
       automatically — must pass meta_tuner adopt gate).
WHEN:  2026-07-24 Phase 3.
WHERE: python scripts/diagnose_day.py <brain> [day_index] [goal] [floor]
WHY:   Conversational Diagnostic LLM surface: Perception vs Policy vs
       Generalization, evidence-backed, SkillOpt-style.
INTERCONNECTED: telemetry/mind_probe, telemetry/ghost_trades,
       doctrine/STANDING_LAWS.md, doctrine/policy_skill.md, configs/rewards.yaml.
----------------------------------------------------------------------

CHANGE LOG:
- 2026-07-24 created — WHY Phase 3 IRAC entry point.
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

from core.configs import path as rpath, load as load_cfg
from telemetry.mind_probe import load_and_probe
from telemetry.ghost_trades import build_ghosts


def propose_reward_delta(ghost_report, mind_dump) -> dict:
    """Bounded dial search suggestions. Not applied here — self_heal/meta gate.

    Classes: WrongSide | Policy | Perception | Shell | none
    """
    proposal = {
        "issue": None,
        "rule": (
            "With firm HTF cont, prefer matching side; bread-and-butter pull under trend. "
            "Meta searches dials; humans do not freeze the answer."
        ),
        "application": ghost_report.summary,
        "conclusion": None,
        "rewards_delta": {},
        "diagnosis_class": None,
        "side_bias_bull": float(getattr(mind_dump, "side_bias_bull", 0.0)),
        "side_bias_bear": float(getattr(mind_dump, "side_bias_bear", 0.0)),
        "wrong_side_under_bull": int(getattr(mind_dump, "n_wrong_side_under_bull", 0)),
        "wrong_side_under_bear": int(getattr(mind_dump, "n_wrong_side_under_bear", 0)),
    }
    wb = proposal["wrong_side_under_bull"]
    sb = proposal["side_bias_bull"]
    ph = int(getattr(mind_dump, "n_policy_hold_on_setup", 0))
    mask_v = int(getattr(mind_dump, "n_mask_veto", 0))

    if mask_v > ph and mask_v > 5:
        proposal["diagnosis_class"] = "Shell"
        proposal["issue"] = f"mask_veto={mask_v} dominates — not a reward-dial problem alone."
        proposal["conclusion"] = "Inspect forever masks; do not only raise pullback."
        return proposal

    if wb > 10 and sb < 0.05:
        proposal["diagnosis_class"] = "WrongSide"
        proposal["issue"] = (
            f"Under bull cont: wrong_side={wb}, side_bias_bull={sb:+.3f} — "
            "prefers short (or weak long mass), not just hold."
        )
        proposal["rewards_delta"] = {
            "w_with_trend_close": "+0.05 search (default 0)",
            "w_against_trend_close": "-0.05 search (more negative)",
        }
        proposal["conclusion"] = (
            "Propose searching with/against-trend close dials. Gate via self_heal/meta + prove_it."
        )
        return proposal

    if ghost_report.n_high_miss_pull <= 5 and ph < 10:
        proposal["issue"] = "No repeated high-miss pull or wrong-side pattern on this day."
        proposal["conclusion"] = "No reward change proposed from this day alone."
        proposal["diagnosis_class"] = "none"
        return proposal

    high = [g for g in ghost_report.ghosts if g.high_miss]
    mean_alt = float(np.mean([g.alt_prob for g in high])) if high else 0.0

    if mean_alt < 0.08 and ghost_report.n_high_miss_pull > 5:
        proposal["diagnosis_class"] = "Perception"
        proposal["issue"] = (
            f"Pull present on {ghost_report.n_high_miss_pull} bars but policy "
            f"mean alt_prob={mean_alt:.3f} — weak mass on aligned open."
        )
        proposal["rewards_delta"] = {
            "w_pullback_with_htf": "+0.02 to +0.08 (search)",
            "w_with_trend_close": "optional +0.05 if side also weak",
        }
        proposal["conclusion"] = (
            "Propose raising pull / with-trend dials. Must pass adopt_gate + prove_it."
        )
    else:
        proposal["diagnosis_class"] = "Policy"
        proposal["issue"] = (
            f"Setup visible; hold or mis-side under incentives "
            f"(high_miss={ghost_report.n_high_miss_pull}, alt_prob={mean_alt:.3f})."
        )
        proposal["rewards_delta"] = {
            "w_pullback_with_htf": "modest increase (search)",
            "w_setup_skip": "optional small negative (default 0)",
            "w_with_trend_close": "if side_bias weak, search +",
        }
        proposal["conclusion"] = "Propose dial search; gate via meta_tuner / self_heal."
    return proposal

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
        src, cache_path=cache if os.path.exists(os.path.dirname(cache)) else None
    )
    if day_i < 0 or day_i >= days_obs.shape[0]:
        print(f"day_index {day_i} out of range")
        sys.exit(1)

    L = int(day_lens[day_i])
    day_obs = np.asarray(days_obs[day_i, :L], dtype=np.float32)
    day_phys = np.asarray(days_phys[day_i, :L], dtype=np.float32)
    label = str(dates[day_i]) if dates is not None and day_i < len(dates) else str(day_i)

    print(f"=== DIAGNOSE day={label} brain={brain} goal={goal} floor={floor} ===\n")
    dump = load_and_probe(
        brain_name=brain, day_obs=day_obs, day_phys=day_phys,
        cols=list(cols), goal_pct=goal, floor_pct=floor,
        day_index=day_i, day_label=label,
    )
    ghosts = build_ghosts(dump)
    proposal = propose_reward_delta(ghosts, dump)

    out_dir = rpath("artifacts", "mind_dumps")
    os.makedirs(out_dir, exist_ok=True)
    base = f"diag_{brain}_{label}"
    dump.write(os.path.join(out_dir, base + "_mind.json"))
    with open(os.path.join(out_dir, base + "_ghosts.json"), "w") as f:
        json.dump(ghosts.to_dict(), f, indent=2)
    with open(os.path.join(out_dir, base + "_irac.json"), "w") as f:
        json.dump(proposal, f, indent=2)

    print("MIND:", dump.summary)
    print()
    print("GHOSTS:", ghosts.summary)
    print()
    print("=== IRAC ===")
    print("Issue:", proposal["issue"])
    print("Rule:", proposal["rule"])
    print("Application:", proposal["application"])
    print("Conclusion:", proposal["conclusion"])
    print("Class:", proposal["diagnosis_class"])
    print("Proposed rewards delta:", proposal["rewards_delta"])
    print()
    print(f"Artifacts: {out_dir}/{base}_*")


if __name__ == "__main__":
    main()
