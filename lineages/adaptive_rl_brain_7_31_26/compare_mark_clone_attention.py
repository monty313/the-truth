"""A/B claim heuristic vs Mark teacher eyes and/or trained policy weights.

Lineage only. Does not touch PROVEN.

Usage (repo root, PYTHONPATH=.;code):
  # heuristic eyes only
  python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 3.0 3.5 --mode forward --eyes-only

  # TRAINED policy greedy vs claim baseline (required for skeptic A/B)
  python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 3.0 3.5 --mode forward --policy-ckpt checkpoints/mark_clone_doctrine_v1.pt
  python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 1.0 2.0 --mode forward --policy-ckpt checkpoints/mark_clone_doctrine_v1.pt

  # combined hard+soft report
  python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --combined-ab --policy-ckpt checkpoints/mark_clone_doctrine_v1.pt
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

REPORT_DIR = os.path.join(_HERE, "checkpoints")
DEFAULT_CKPT = os.path.join(REPORT_DIR, "mark_clone_doctrine_v1.pt")


def _summarize(rows: List[dict], label: str) -> Dict[str, Any]:
    n = len(rows)
    cleared = sum(1 for x in rows if x["cleared"])
    breached = sum(1 for x in rows if x["breached"])
    entries = [x["n_entries"] for x in rows]
    misses = [x for x in rows if not x["cleared"]]
    clears = [x for x in rows if x["cleared"]]
    return {
        "label": label,
        "n_days": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "breach_pct": 100.0 * breached / max(n, 1),
        "mean_entries": float(np.mean(entries)) if entries else 0.0,
        "mean_entries_miss": float(np.mean([x["n_entries"] for x in misses]))
        if misses
        else 0.0,
        "mean_entries_clear": float(np.mean([x["n_entries"] for x in clears]))
        if clears
        else 0.0,
        "mean_pnl": float(np.mean([x["pnl_pct"] for x in rows])) if rows else 0.0,
        "day_rows": rows,
    }


def _row_from_result(r) -> dict:
    return {
        "date": r.date,
        "pnl_pct": round(r.pnl_pct, 4),
        "min_eq_pct": round(r.min_eq_pct, 4),
        "cleared": r.cleared,
        "breached": r.breached,
        "n_entries": r.n_entries,
        "banked": r.banked,
    }


def run_baseline_heuristic(
    days: Sequence[Tuple[str, Any]],
    target: float,
    risk: float,
) -> Dict[str, Any]:
    rows = []
    for date_str, m1 in days:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            eyes_mode="legacy_set2",
        )
        r = day.run(use_heuristic=True)
        rows.append(_row_from_result(r))
    return _summarize(rows, "claim_baseline_legacy_set2")


def run_mark_teacher_heuristic(
    days: Sequence[Tuple[str, Any]],
    target: float,
    risk: float,
    *,
    eyes_mode: str = "mark_doctrine",
) -> Dict[str, Any]:
    rows = []
    for date_str, m1 in days:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            eyes_mode=eyes_mode,
            mark_clone=False,
        )
        r = day.run(use_heuristic=True)
        rows.append(_row_from_result(r))
    return _summarize(rows, f"mark_teacher_{eyes_mode}")


def load_policy(ckpt_path: str) -> Channel1Policy:
    path = ckpt_path
    if not os.path.isabs(path):
        # allow relative to lineage or repo
        cand = [
            path,
            os.path.join(_HERE, path),
            os.path.join(_HERE, "checkpoints", os.path.basename(path)),
            os.path.join(_ROOT, path),
        ]
        path = next((p for p in cand if os.path.isfile(p)), path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"policy ckpt not found: {ckpt_path}")
    blob = torch.load(path, map_location="cpu", weights_only=False)
    hidden = int(blob.get("hidden", 64))
    obs_dim = int(blob.get("obs_dim", CHANNEL1_DIM))
    pol = Channel1Policy(obs_dim=obs_dim, hidden=hidden)
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol


def run_policy_greedy(
    days: Sequence[Tuple[str, Any]],
    target: float,
    risk: float,
    policy: Channel1Policy,
    *,
    eyes_mode: str = "mark_doctrine",
    full_obs: bool | None = None,
    mark_soul: bool = True,
) -> Dict[str, Any]:
    """Pure greedy policy through shell. eyes_mode only for teacher-match obs path."""
    if full_obs is None:
        full_obs = int(getattr(policy, "obs_dim", CHANNEL1_DIM)) >= 100
    rows = []
    for date_str, m1 in days:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            eyes_mode=eyes_mode,
            mark_soul=mark_soul,
            full_obs=bool(full_obs),
        )
        r = day.run(greedy_policy=policy, use_heuristic=False, pure_greedy=True)
        rows.append(_row_from_result(r))
    return _summarize(rows, "mark_policy_greedy")


def _print_block(tag: str, s: Dict[str, Any]) -> None:
    print(
        f"{tag} clear {s['cleared']}/{s['n_days']} ({s['clear_pct']:.1f}%)  "
        f"breach {s['breached']}  mean_entries {s['mean_entries']:.2f}  "
        f"miss_entries {s['mean_entries_miss']:.2f}",
        flush=True,
    )


def ab_one_pair(
    days: Sequence[Tuple[str, Any]],
    target: float,
    risk: float,
    *,
    policy: Optional[Channel1Policy],
    eyes_mode: str,
    mode_label: str,
    full_obs: bool | None = None,
) -> Dict[str, Any]:
    base = run_baseline_heuristic(days, target, risk)
    _print_block("BASE ", base)
    teacher = run_mark_teacher_heuristic(days, target, risk, eyes_mode=eyes_mode)
    _print_block("TEACH", teacher)
    out: Dict[str, Any] = {
        "mode": mode_label,
        "target_pct": target,
        "risk_pct": risk,
        "eyes_mode": eyes_mode,
        "baseline": {k: v for k, v in base.items() if k != "day_rows"},
        "teacher": {k: v for k, v in teacher.items() if k != "day_rows"},
        "baseline_day_rows": base["day_rows"],
        "teacher_day_rows": teacher["day_rows"],
        "delta_teacher_clear": teacher["cleared"] - base["cleared"],
        "delta_teacher_mean_entries": teacher["mean_entries"] - base["mean_entries"],
        "breach_ok_teacher": base["breached"] == 0 and teacher["breached"] == 0,
    }
    if policy is not None:
        pol = run_policy_greedy(
            days, target, risk, policy, eyes_mode=eyes_mode, full_obs=full_obs
        )
        _print_block("POL  ", pol)
        out["policy"] = {k: v for k, v in pol.items() if k != "day_rows"}
        out["policy_day_rows"] = pol["day_rows"]
        out["delta_policy_clear"] = pol["cleared"] - base["cleared"]
        out["delta_policy_mean_entries"] = pol["mean_entries"] - base["mean_entries"]
        out["breach_ok_policy"] = base["breached"] == 0 and pol["breached"] == 0
        # soft collapse flag: soft targets should not drop more than 20pp clear
        out["soft_clear_collapse"] = (
            target <= 1.5
            and (base["clear_pct"] - pol["clear_pct"]) > 20.0
        )
        out["thrash_improved"] = pol["mean_entries"] < base["mean_entries"]
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="A/B claim vs Mark teacher / policy")
    ap.add_argument("--mode", choices=["all", "practice", "forward"], default="forward")
    ap.add_argument("--pair", nargs=2, type=float, metavar=("TARGET", "RISK"), default=[3.0, 3.5])
    ap.add_argument("--date", default=None)
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    ap.add_argument("--eyes-mode", choices=["mark_doctrine", "mark_all_sets"], default="mark_doctrine")
    ap.add_argument(
        "--policy-ckpt",
        default=None,
        help="Load Channel1Policy weights and score pure greedy vs baseline",
    )
    ap.add_argument(
        "--combined-ab",
        action="store_true",
        help="Run hard 3.0/3.5 + soft 1.0/2.0 forward A/B into one JSON",
    )
    ap.add_argument("--out", default=None, help="override output JSON path")
    # legacy flags kept for CLI compatibility (unused when --policy-ckpt set)
    ap.add_argument("--eyes-only", action="store_true")
    ap.add_argument("--confirm", type=int, default=2)
    ap.add_argument("--cooldown", type=int, default=2)
    ap.add_argument("--max-entries", type=int, default=8)
    ap.add_argument("--pullback-refuse", action="store_true")
    args = ap.parse_args(argv)

    all_days = load_calendar_days(args.data, min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=50)

    policy = None
    ckpt_used = None
    if args.policy_ckpt or args.combined_ab:
        ckpt_used = args.policy_ckpt or DEFAULT_CKPT
        print(f"loading policy {ckpt_used}", flush=True)
        policy = load_policy(ckpt_used)

    if args.combined_ab:
        if args.mode == "practice":
            days = practice
            mode_label = "practice"
        elif args.mode == "all":
            days = all_days
            mode_label = "all"
        else:
            days = forward
            mode_label = "forward"
        print(f"COMBINED A/B mode={mode_label} n_days={len(days)}", flush=True)
        hard = ab_one_pair(
            days, 3.0, 3.5, policy=policy, eyes_mode=args.eyes_mode, mode_label=mode_label
        )
        soft = ab_one_pair(
            days, 1.0, 2.0, policy=policy, eyes_mode=args.eyes_mode, mode_label=mode_label
        )
        out = {
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "kind": "policy_weight_ab_hard_soft",
            "policy_ckpt": ckpt_used,
            "eyes_mode": args.eyes_mode,
            "mode": mode_label,
            "hard_3_0_3_5": hard,
            "soft_1_0_2_0": soft,
            "pass_gates": {
                "hard_breach_0": hard.get("breach_ok_policy", False),
                "soft_breach_0": soft.get("breach_ok_policy", False),
                "soft_no_collapse": not soft.get("soft_clear_collapse", True),
                "hard_thrash_improved": hard.get("thrash_improved", False),
            },
            "proven_touched": False,
        }
        path = args.out or os.path.join(REPORT_DIR, "mark_clone_policy_ab_hard_soft.json")
        os.makedirs(REPORT_DIR, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {path}", flush=True)
        print("pass_gates", out["pass_gates"], flush=True)
        return 0

    target, risk = float(args.pair[0]), float(args.pair[1])
    if args.date:
        days = [(d, m) for d, m in all_days if str(d) == str(args.date)]
        if not days:
            print(f"date {args.date} not found", flush=True)
            return 2
        mode_label = f"single:{args.date}"
    else:
        if args.mode == "practice":
            days = practice
        elif args.mode == "forward":
            days = forward
        else:
            days = all_days
        mode_label = args.mode

    print(
        f"A/B pair {target}/{risk}  mode={mode_label}  n_days={len(days)}  "
        f"policy={bool(policy)}",
        flush=True,
    )
    block = ab_one_pair(
        days,
        target,
        risk,
        policy=policy,
        eyes_mode=args.eyes_mode,
        mode_label=mode_label,
    )
    out = {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "kind": "policy_weight_ab" if policy else "teacher_eyes_ab",
        "policy_ckpt": ckpt_used,
        **block,
        "proven_touched": False,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = args.out or os.path.join(
        REPORT_DIR,
        f"mark_clone_ab_{target:g}_{risk:g}_{mode_label.replace(':', '_')}.json",
    )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
