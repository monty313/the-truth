"""Distill TEEN (same36 fire_skill KEEP) into CHILD floor — combined knowledge climb.

PRE: score CHILD (same35). Collect labels = TEEN greedy acts on frozen days.
Train CHILD→TEEN with high KL to CHILD (pack protect) + action match to TEEN.
KEEP only if same↑ and breach0. Writes source=climb_35_combined_knowledge_distill_teen.

This fuses: child pack floor + teen fire_skill KEEP knowledge into one update.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
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
from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import score_policy
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

CHILD = os.path.join(_HERE, "checkpoints", "CHILD_STAGE_same35_mark_clone_full_obs.pt")
TEEN = os.path.join(_HERE, "checkpoints", "TEEN_STAGE_same36_fable_kag_fire_skill.pt")
BEST_JSON = os.path.join(OUT, "BEST__latest.json")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def collect_teacher_labels(
    teacher,
    day_map: Dict[str, Any],
    mark_rows: List[dict],
    *,
    subsample_hold: int = 2,
    student=None,
) -> tuple:
    """If student given: DAgger — walk student path, label with teacher (fixes covariate shift)."""
    xs, ys, ws = [], [], []
    for i, row in enumerate(mark_rows):
        if i % 10 == 0:
            print(f"  label day {i+1}/{len(mark_rows)}", flush=True)
        date = str(row["date"])
        day = GoalEquityDay(
            day_map[date],
            target_pct=float(row["target_pct"]),
            risk_pct=float(row["risk_pct"]),
            date_str=date,
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=True,
        )
        step = 0
        for tb in day.runner.decision_indices():
            if day.dead or day.banked:
                break
            obs = day.observe(tb)
            with torch.no_grad():
                ta, _ = teacher.act(obs, greedy=True)
                ta = int(ta)
                if student is not None:
                    sa, _ = student.act(obs, greedy=True)
                    sa = int(sa)
                else:
                    sa = ta
            # keep all teacher dirs; HOLD sparsely; boost disagrees
            if ta == ACTION_HOLD and (step % subsample_hold != 0) and sa == ta:
                day.step_action(tb, sa if student is not None else ta)
                step += 1
                continue
            w = 2.2 if ta != ACTION_HOLD else 1.0
            if student is not None and sa != ta:
                w *= 3.0
                n_copy = 3
            else:
                n_copy = 1
            for _ in range(n_copy):
                xs.append(np.asarray(obs, np.float32).reshape(-1))
                ys.append(ta)
                ws.append(w)
            # DAgger: student acts (own path); pure teacher path if no student
            day.step_action(tb, sa if student is not None else ta)
            step += 1
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    w = np.asarray(ws, np.float32)
    w = w / float(w.mean())
    return X, y, w


def main() -> int:
    scratch = os.environ.get(
        "CLIMB_SCRATCH",
        os.path.join(OUT, "scratch_climb"),
    )
    # prefer goal implementer scratch if set via CLI env from caller
    if len(sys.argv) > 1 and sys.argv[1].startswith("--scratch="):
        scratch = sys.argv[1].split("=", 1)[1]
    os.makedirs(scratch, exist_ok=True)

    print("=== climb distill TEEN(36) knowledge into CHILD(35) ===", flush=True)
    assert os.path.isfile(CHILD) and os.path.isfile(TEEN)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}

    child = load_policy(CHILD)
    teen = load_policy(TEEN)
    assert not child.multi_head and not teen.multi_head

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
                "path": "climb_35_distill_teen",
                "sources": [
                    "child_pack_floor",
                    "teen_fire_skill_KEEP",
                    "kl_anchor_child",
                    "distill_teacher_acts",
                ],
                "ts": _utcnow(),
            },
            f,
            indent=2,
        )
    assert pre_m["same_outcome"] == 35 and pre_m["n_breach"] == 0, pre_m

    print("TEEN reference score…", flush=True)
    teen_s = score_policy(teen, day_map, mark_rows)
    teen_m = meters(teen_s)
    print("TEEN", teen_m, flush=True)
    assert teen_m["same_outcome"] >= 36 and teen_m["n_breach"] == 0, teen_m

    child_state = {k: v.detach().clone() for k, v in child.state_dict().items()}
    pol2 = child
    post = pre
    post_m = pre_m
    ok, reason = False, "not_run"
    # Multi-iter DAgger distill: student path + teacher labels (covariate-shift fix)
    for it in range(1, 4):
        print(f"Collect TEEN labels DAgger iter {it}/3…", flush=True)
        X, y, w = collect_teacher_labels(
            teen, day_map, mark_rows, student=pol2 if it > 1 else child
        )
        print(f"  n={len(y)} dir={int((y!=0).sum())} hold={int((y==0).sum())}", flush=True)
        print(f"BC distill TEEN → CHILD iter {it}…", flush=True)
        warm = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
        pol2, losses = train_bc(
            X,
            y,
            sample_weights=w,
            epochs=18 if it == 1 else 12,
            lr=1.2e-4,
            hidden=128,
            seed=40 + it,
            warm_state=warm,
            kl_anchor_state=child_state,  # always protect child floor identity
            kl_coef=0.55 if it == 1 else 0.40,
            freeze_trunk=False,
            multi_head=False,
            obs_dim=MARK_FULL_DIM,
        )
        print(
            f"  match={match_rate(pol2, X, y)} loss={losses[-1] if losses else None}",
            flush=True,
        )
        print(f"POST score iter {it}…", flush=True)
        post = score_policy(pol2, day_map, mark_rows)
        post_m = meters(post)
        print("POST", post_m, flush=True)
        ok, reason = keep_gate(post, pre, baseline_clear=27)
        print(f"DECISION {'KEEP' if ok else 'REJECT'} ({reason})", flush=True)
        if ok:
            break
        # if crater, restore child and stop early
        if post_m["n_breach"] > 0 or post_m["same_outcome"] < pre_m["same_outcome"] - 3:
            print("  pack crater — restore child, stop DAgger", flush=True)
            pol2 = load_policy(CHILD)
            post = pre
            post_m = pre_m
            ok, reason = False, "crater_restore"
            break

    report = {
        "path": "climb_35_combined_knowledge via distill_teen",
        "source_tag": "climb_35_combined_knowledge_KEEP_distill_teen",
        "pre": pre_m,
        "teen_ref": teen_m,
        "post": post_m,
        "kept": ok,
        "reason": reason,
        "ts": _utcnow(),
        "sources": [
            "child_pack_floor_35",
            "teen_fire_skill_KEEP_36",
            "distill_teacher_greedy_acts",
            "kl_anchor_child",
        ],
        "ban": "strategy_only_multi_head_full_replace",
    }

    if ok:
        dials = clip_streak_dials(default_streak_dials())
        save_policy(
            pol2,
            note=f"climb_combined_distill_teen KEEP same {pre_m['same_outcome']}→{post_m['same_outcome']}",
            dials=dials,
        )
        bak = dated_backup(
            pol2,
            post_m,
            note="climb_35_combined_knowledge KEEP distill TEEN fire_skill into CHILD",
        )
        report["backup"] = bak
        with open(BEST_JSON, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "same_outcome": post_m["same_outcome"],
                    "policy_clear": post_m["policy_clear"],
                    "mwt": post_m["mark_would_take"],
                    "breach": post_m["n_breach"],
                    "source": "climb_35_combined_knowledge_KEEP_distill_teen",
                    "stage": "teen",
                    "growth_method": "distill_teen_fire_skill_into_child",
                    "core_skill": "fire_skill multi-day pattern (teen) + child pack floor",
                    "prior_same": pre_m["same_outcome"],
                    "backup": bak,
                    "ts": _utcnow(),
                },
                f,
                indent=2,
            )
        print(f"KEEP backup={bak}", flush=True)
    else:
        print("REJECT — live BEST untouched", flush=True)

    with open(os.path.join(scratch, "climb_post.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "POST_TRAIN_KEEP" if ok else "POST_TRAIN_NO_KEEP",
                "meters": post_m,
                "pre": pre_m,
                "kept": ok,
                "reason": reason,
                "path": "climb_35_distill_teen",
                "ts": _utcnow(),
            },
            f,
            indent=2,
        )
    with open(os.path.join(scratch, "climb_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_path = os.path.join(scratch, "climb_run.log")
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
    except OSError:
        with open(os.path.join(scratch, "climb_run_distill.log"), "w", encoding="utf-8") as f:
            f.write(json.dumps(report) + "\n")
    print(f"DONE kept={ok} pre={pre_m} post={post_m}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
