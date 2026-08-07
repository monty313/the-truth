"""Combine CHILD floor + TEEN fire_skill by weight blend; KEEP first alpha with same↑.

PRE = score CHILD (35). Candidate_α = (1-α)·CHILD + α·TEEN. Score each α.
KEEP first α with same>PRE and breach0. Writes source climb_35_combined_knowledge_KEEP_blend_teen.

This is pack-safe knowledge fusion without thrashy BC (which cratered 35→29).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.climb_35_combined_knowledge import (
    BASELINE,
    CKPT,
    OUT,
    REPORT,
    dated_backup,
    keep_gate,
    meters,
)
from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import score_policy
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials

CHILD = os.path.join(_HERE, "checkpoints", "CHILD_STAGE_same35_mark_clone_full_obs.pt")
TEEN = os.path.join(_HERE, "checkpoints", "TEEN_STAGE_same36_fable_kag_fire_skill.pt")
BEST_JSON = os.path.join(OUT, "BEST__latest.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def blend_state(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
    alpha: float,
) -> Dict[str, torch.Tensor]:
    """out = (1-α)·a + α·b"""
    out = {}
    for k in a:
        if k not in b:
            out[k] = a[k].detach().clone()
            continue
        if a[k].shape != b[k].shape:
            out[k] = b[k].detach().clone() if alpha >= 0.5 else a[k].detach().clone()
            continue
        out[k] = (1.0 - alpha) * a[k].float() + alpha * b[k].float()
    return out


def main() -> int:
    scratch = os.environ.get("CLIMB_SCRATCH", os.path.join(OUT, "scratch_climb"))
    os.makedirs(scratch, exist_ok=True)

    print("=== climb blend CHILD(35)+TEEN(36) weight fusion ===", flush=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}

    child = load_policy(CHILD)
    teen = load_policy(TEEN)
    child_sd = {k: v.detach().clone() for k, v in child.state_dict().items()}
    teen_sd = {k: v.detach().clone() for k, v in teen.state_dict().items()}

    print("PRE score CHILD…", flush=True)
    pre = score_policy(child, day_map, mark_rows)
    pre_m = meters(pre)
    print("PRE", pre_m, flush=True)
    with open(os.path.join(scratch, "climb_pre.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "PRE_TRAIN",
                "meters": pre_m,
                "warm": CHILD,
                "teacher": TEEN,
                "path": "climb_35_blend_teen",
                "sources": [
                    "child_pack_floor",
                    "teen_fire_skill_KEEP",
                    "weight_blend_fusion",
                ],
                "ts": _utcnow(),
            },
            f,
            indent=2,
        )
    assert pre_m["same_outcome"] == 35 and pre_m["n_breach"] == 0

    alphas = [0.25, 0.4, 0.55, 0.7, 0.85, 1.0]
    cycles = []
    kept = False
    best_post = pre_m
    best_pol = None
    best_alpha = None
    keep_backup = None

    for alpha in alphas:
        print(f"\n===== blend α={alpha:.2f} =====", flush=True)
        pol = Channel1Policy(obs_dim=MARK_FULL_DIM, hidden=128, multi_head=False)
        pol.load_state_dict(blend_state(child_sd, teen_sd, alpha))
        pol.eval()
        post = score_policy(pol, day_map, mark_rows)
        pm = meters(post)
        print("POST", pm, flush=True)
        ok, reason = keep_gate(post, pre, baseline_clear=27)
        cycles.append({"alpha": alpha, "meters": pm, "ok": ok, "reason": reason})
        print(f"  gate={'KEEP' if ok else 'REJECT'} ({reason})", flush=True)
        if ok:
            kept = True
            best_post = pm
            best_pol = pol
            best_alpha = alpha
            break

    report = {
        "path": "climb_35_combined_knowledge via blend_teen",
        "source_tag": "climb_35_combined_knowledge_KEEP_blend_teen",
        "pre": pre_m,
        "post": best_post,
        "kept": kept,
        "alpha": best_alpha,
        "cycles": cycles,
        "ts": _utcnow(),
        "sources": [
            "child_pack_floor_35",
            "teen_fire_skill_KEEP_36",
            "weight_space_blend",
        ],
        "ban": "strategy_only_multi_head_full_replace",
    }

    if kept and best_pol is not None:
        dials = clip_streak_dials(default_streak_dials())
        save_policy(
            best_pol,
            note=f"climb_combined_blend_teen KEEP α={best_alpha} same {pre_m['same_outcome']}→{best_post['same_outcome']}",
            dials=dials,
        )
        keep_backup = dated_backup(
            best_pol,
            best_post,
            note=f"climb_35_combined_knowledge KEEP blend α={best_alpha}",
        )
        report["backup"] = keep_backup
        with open(BEST_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "same_outcome": best_post["same_outcome"],
                    "policy_clear": best_post["policy_clear"],
                    "mwt": best_post["mark_would_take"],
                    "breach": best_post["n_breach"],
                    "source": f"climb_35_combined_knowledge_KEEP_blend_teen_a{best_alpha}",
                    "stage": "teen",
                    "growth_method": "weight_blend_child_floor_teen_fire_skill",
                    "core_skill": "fire_skill multi-day (teen) fused with child pack floor",
                    "prior_same": pre_m["same_outcome"],
                    "alpha": best_alpha,
                    "backup": keep_backup,
                    "ts": _utcnow(),
                },
                f,
                indent=2,
            )
        print(f"KEEP α={best_alpha} backup={keep_backup}", flush=True)
    else:
        print("REJECT all alphas — live BEST untouched", flush=True)

    with open(os.path.join(scratch, "climb_post.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "POST_TRAIN_KEEP" if kept else "POST_TRAIN_NO_KEEP",
                "meters": best_post,
                "pre": pre_m,
                "kept": kept,
                "alpha": best_alpha,
                "path": "climb_35_blend_teen",
                "ts": _utcnow(),
            },
            f,
            indent=2,
        )
    with open(os.path.join(scratch, "climb_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    try:
        with open(os.path.join(scratch, "climb_run.log"), "w", encoding="utf-8") as f:
            f.write(json.dumps(report, indent=2) + "\n")
    except OSError:
        pass
    print(f"DONE kept={kept} pre={pre_m} post={best_post}", flush=True)
    return 0 if kept else 1


if __name__ == "__main__":
    raise SystemExit(main())
