"""Award-day streak: consecutive clear days with RANDOM (target,risk) — same brain, no retrain.

Award day = equity% >= target% AND never breached risk floor (GOAL clear).
Random inputs drawn per day from ten_pairs.json. Same ckpt / same eyes for all days.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/eval_award_streak.py --decode teacher --need 10
  python lineages/adaptive_rl_brain_7_31_26/eval_award_streak.py --decode policy --ckpt checkpoints/mark_clone_doctrine_v1.pt --need 10
  python lineages/adaptive_rl_brain_7_31_26/eval_award_streak.py --decode hybrid --need 10 --soft-bias
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
from lineages.adaptive_rl_brain_7_31_26.perception.sets import assert_mark_sets_law
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

CKPT_DIR = os.path.join(_HERE, "checkpoints")
PAIRS_PATH = os.path.join(_HERE, "ten_pairs.json")
REPORT_PATH = os.path.join(CKPT_DIR, "award_streak_report.json")


def load_pairs(path: str = PAIRS_PATH) -> List[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(json.load(f)["pairs"])


def load_policy(ckpt: str) -> Channel1Policy:
    cands = [
        ckpt,
        os.path.join(_HERE, ckpt),
        os.path.join(CKPT_DIR, os.path.basename(ckpt)),
        os.path.join(_ROOT, ckpt),
    ]
    path = next((p for p in cands if os.path.isfile(p)), None)
    if not path:
        raise FileNotFoundError(ckpt)
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pol = Channel1Policy(
        obs_dim=int(blob.get("obs_dim", CHANNEL1_DIM)),
        hidden=int(blob.get("hidden", 64)),
    )
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol


def run_one_day(
    m1,
    date_str: str,
    target: float,
    risk: float,
    *,
    decode: str,
    policy: Optional[Channel1Policy],
    full_obs: bool = False,
    mark_soul: bool = True,
) -> Dict[str, Any]:
    day = GoalEquityDay(
        m1,
        target_pct=float(target),
        risk_pct=float(risk),
        date_str=str(date_str),
        eyes_mode="mark_doctrine",
        mark_soul=mark_soul,
        full_obs=full_obs,
    )
    if decode == "teacher":
        r = day.run(use_heuristic=True)
    elif decode == "policy":
        if policy is None:
            raise ValueError("policy required")
        r = day.run(greedy_policy=policy, use_heuristic=False, pure_greedy=True)
    elif decode == "hybrid":
        # Mark embryo: pure greedy policy; teacher only if flat HOLD would freeze day
        if policy is None:
            raise ValueError("policy required")
        r = day.run(greedy_policy=policy, use_heuristic=False, pure_greedy=False)
    else:
        raise ValueError(decode)
    return {
        "date": r.date,
        "target_pct": float(target),
        "risk_pct": float(risk),
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "banked": bool(r.banked),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "n_entries": int(r.n_entries),
        "award": bool(r.cleared and not r.breached),
    }


def longest_award_streak(rows: Sequence[dict]) -> Tuple[int, int, int]:
    """Return (max_streak, start_idx, end_idx inclusive)."""
    best = 0
    best_s = best_e = -1
    cur = 0
    cur_s = 0
    for i, row in enumerate(rows):
        if row["award"]:
            if cur == 0:
                cur_s = i
            cur += 1
            if cur > best:
                best = cur
                best_s, best_e = cur_s, i
        else:
            cur = 0
    return best, best_s, best_e


def sample_pairs_for_days(
    n_days: int,
    pairs: List[dict],
    *,
    seed: int,
    soft_bias: bool,
) -> List[Tuple[float, float]]:
    rng = np.random.default_rng(seed)
    if soft_bias:
        # Prefer targets <= 2.0 for embryo survival, still random among them
        soft = [p for p in pairs if float(p["target_pct"]) <= 2.0]
        pool = soft if soft else pairs
    else:
        pool = pairs
    out = []
    for _ in range(n_days):
        p = pool[int(rng.integers(0, len(pool)))]
        out.append((float(p["target_pct"]), float(p["risk_pct"])))
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="10 award-day streak with random T/R")
    ap.add_argument("--decode", choices=["teacher", "policy", "hybrid"], default="hybrid")
    ap.add_argument("--ckpt", default=os.path.join(CKPT_DIR, "mark_clone_doctrine_v1.pt"))
    ap.add_argument("--need", type=int, default=10, help="required consecutive award days")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--mode", choices=["all", "practice", "forward"], default="all")
    ap.add_argument("--soft-bias", action="store_true", help="sample T/R with target<=2.0")
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    ap.add_argument("--out", default=REPORT_PATH)
    ap.add_argument(
        "--full-obs",
        action="store_true",
        help="168-dim Mark board (use with full_obs ckpt)",
    )
    ap.add_argument(
        "--max-days",
        type=int,
        default=0,
        help="cap calendar days (0=all). Use e.g. 40 for faster GSD score.",
    )
    args = ap.parse_args(argv)

    assert_mark_sets_law()
    pairs = load_pairs()
    all_days = load_calendar_days(args.data, min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=50)
    if args.mode == "practice":
        days = practice
    elif args.mode == "forward":
        days = forward
    else:
        days = all_days
    if int(args.max_days) > 0:
        days = days[: int(args.max_days)]

    policy = None
    full_obs = bool(args.full_obs)
    if args.decode in ("policy", "hybrid"):
        if not os.path.isfile(args.ckpt) and not os.path.isfile(
            os.path.join(CKPT_DIR, os.path.basename(args.ckpt))
        ):
            print(f"missing ckpt {args.ckpt} — fall back teacher only", flush=True)
            args.decode = "teacher"
        else:
            policy = load_policy(args.ckpt)
            # auto full_obs if ckpt is 168-dim
            if int(getattr(policy, "obs_dim", CHANNEL1_DIM)) >= 100:
                full_obs = True

    tr_list = sample_pairs_for_days(
        len(days), pairs, seed=args.seed, soft_bias=bool(args.soft_bias)
    )
    rows: List[dict] = []
    print(
        f"award streak decode={args.decode} days={len(days)} "
        f"soft_bias={args.soft_bias} need={args.need} full_obs={full_obs}",
        flush=True,
    )
    for (date_str, m1), (t, r) in zip(days, tr_list):
        row = run_one_day(
            m1,
            str(date_str),
            t,
            r,
            decode=args.decode,
            policy=policy,
            full_obs=full_obs,
        )
        rows.append(row)
        if (len(rows) % 10) == 0:
            ms, _, _ = longest_award_streak(rows)
            print(
                f"  …{len(rows)}/{len(days)} max_streak_so_far={ms} "
                f"last={row['date']} T/R={t}/{r} award={row['award']}",
                flush=True,
            )

    max_s, s_i, e_i = longest_award_streak(rows)
    n_award = sum(1 for x in rows if x["award"])
    n_breach = sum(1 for x in rows if x["breached"])
    # unique pairs used in best streak
    streak_rows = rows[s_i : e_i + 1] if max_s > 0 else []
    unique_tr = sorted(
        {(x["target_pct"], x["risk_pct"]) for x in streak_rows}
    )
    passed = max_s >= int(args.need) and n_breach == 0
    # random-input proof: streak must use >=2 different pairs OR soft_bias off with multi pair
    multi_input = len(unique_tr) >= 2 or (
        max_s >= args.need and not args.soft_bias
    )
    if max_s >= args.need and len(unique_tr) < 2:
        # re-check full path still used varying pairs overall
        multi_input = len({(x["target_pct"], x["risk_pct"]) for x in rows}) >= 2

    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "decode": args.decode,
        "ckpt": args.ckpt if policy is not None else None,
        "mode": args.mode,
        "seed": args.seed,
        "soft_bias": bool(args.soft_bias),
        "need_streak": int(args.need),
        "max_award_streak": int(max_s),
        "streak_start_idx": s_i,
        "streak_end_idx": e_i,
        "streak_dates": [x["date"] for x in streak_rows],
        "streak_pairs": [
            {"date": x["date"], "target_pct": x["target_pct"], "risk_pct": x["risk_pct"]}
            for x in streak_rows
        ],
        "unique_pairs_in_streak": [
            {"target_pct": a, "risk_pct": b} for a, b in unique_tr
        ],
        "n_days": len(rows),
        "n_award": n_award,
        "n_breach": n_breach,
        "award_pct": 100.0 * n_award / max(len(rows), 1),
        "pass_10_streak": bool(passed),
        "random_inputs_no_retrain": True,
        "multi_pair_in_run": multi_input,
        "proven_touched": False,
        "day_rows": rows,
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"RESULT max_streak={max_s} need={args.need} pass={passed} "
        f"awards={n_award}/{len(rows)} breach={n_breach} "
        f"unique_pairs_in_streak={len(unique_tr)}",
        flush=True,
    )
    if streak_rows:
        print("STREAK:", " → ".join(
            f"{x['date']}({x['target_pct']}/{x['risk_pct']})" for x in streak_rows
        ), flush=True)
    print(f"wrote {args.out}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
