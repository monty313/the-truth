"""Long-term consistency: Mark soul labels → BC → Mark-aligned policy → long streak.

Goal: soul = policy sense lasting hundreds of days.
1. Collect soul-plan + HITL auto-corrections (Mark HOLD when policy early-fired)
2. BC full-obs clone
3. Score 10d + N-day award streak under random T/R (no retrain)

Usage:
  python lineages/adaptive_rl_brain_7_31_26/mark_consistency_loop.py
  python lineages/adaptive_rl_brain_7_31_26/mark_consistency_loop.py --streak-days 100 --epochs 40
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import (
    longest_award_streak,
    sample_pairs_for_days,
    load_pairs,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import collect_soul_plan_labels
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc, match_rate

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT = os.path.join(CKPT_DIR, "mark_consistency")
CKPT = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
HITL_JSON = os.path.join(CKPT_DIR, "mark_chart_hitl", "HITL__latest.json")


def auto_hitl_hold_labels(
    days: List[Tuple[str, Any]],
    pairs: List[Tuple[float, float]],
    *,
    seed: int,
    n_days: int = 10,
    start_idx: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """Where policy historically disagreed and Mark held — emit HOLD labels.

    Without live MarkOS, codified Mark soul plan + doctrine HOLD is ground truth
    (HITL pack pattern: Mark HOLD, policy early BUY/SELL).
    """
    rng = np.random.default_rng(seed)
    window = days[start_idx : start_idx + n_days]
    xs: List[np.ndarray] = []
    ys: List[int] = []
    # Prefer full-obs policy if present to find wrong opens
    pol = None
    if os.path.isfile(CKPT):
        blob = torch.load(CKPT, map_location="cpu", weights_only=False)
        pol = Channel1Policy(
            obs_dim=int(blob.get("obs_dim", MARK_FULL_DIM)),
            hidden=int(blob.get("hidden", 128)),
        )
        pol.load_state_dict(blob["state_dict"])
        pol.eval()

    for date, m1 in window:
        t, r = pairs[int(rng.integers(0, len(pairs)))]
        day = GoalEquityDay(
            m1,
            target_pct=t,
            risk_pct=r,
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=False,  # measure raw policy
        )
        for tb in day.runner.decision_indices():
            if day.dead or day.banked:
                break
            obs = day.observe(tb)
            mark = int(day.recommended_action(tb))
            pol_a = mark
            if pol is not None:
                with torch.no_grad():
                    pol_a, _ = pol.act(obs, greedy=True)
                    pol_a = int(pol_a)
            # Mark sense correction: if Mark HOLD and policy wants dir → HOLD label
            if mark == ACTION_HOLD and pol_a in (ACTION_BUY, ACTION_SELL):
                xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                ys.append(ACTION_HOLD)
            # Mark side is truth when Mark directional
            if mark in (ACTION_BUY, ACTION_SELL):
                xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                ys.append(mark)
            day.step_action(tb, mark)  # walk Mark path for context

    if not xs:
        return (
            np.zeros((0, MARK_FULL_DIM), np.float32),
            np.zeros((0,), np.int64),
        )
    return np.stack(xs), np.asarray(ys, dtype=np.int64)


def run_streak(
    policy: Channel1Policy,
    days: List[Tuple[str, Any]],
    pairs_raw: List[dict],
    *,
    seed: int,
    n_days: int,
    aligned: bool,
) -> Dict[str, Any]:
    window = days[:n_days]
    tr = sample_pairs_for_days(len(window), pairs_raw, seed=seed, soft_bias=False)
    rows = []
    for (date, m1), (t, r) in zip(window, tr):
        day = GoalEquityDay(
            m1,
            target_pct=t,
            risk_pct=r,
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=aligned,
        )
        r_out = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        rows.append(
            {
                "date": str(date),
                "target_pct": t,
                "risk_pct": r,
                "cleared": bool(r_out.cleared),
                "breached": bool(r_out.breached),
                "pnl_pct": round(float(r_out.pnl_pct), 4),
                "n_entries": int(r_out.n_entries),
                "award": bool(r_out.cleared and not r_out.breached),
            }
        )
    max_s, s_i, e_i = longest_award_streak(rows)
    n_breach = sum(1 for x in rows if x["breached"])
    n_award = sum(1 for x in rows if x["award"])
    return {
        "n_days": len(rows),
        "max_award_streak": int(max_s),
        "n_award": n_award,
        "n_breach": n_breach,
        "award_pct": 100.0 * n_award / max(len(rows), 1),
        "pass_10_streak_breach0": bool(max_s >= 10 and n_breach == 0),
        "streak_dates": [rows[i]["date"] for i in range(s_i, e_i + 1)] if max_s else [],
        "rows": rows,
        "mark_align_policy": aligned,
    }


def score_10d(policy: Channel1Policy, days, pairs, *, seed: int = 7, start: int = 40):
    from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day

    rng = np.random.default_rng(seed)
    window = days[start : start + 10]
    mark_c = pol_c = breach = 0
    for date, m1 in window:
        t, r = pairs[int(rng.integers(0, len(pairs)))]
        m = execute_mark_soul_day(m1, str(date), t, r)
        mark_c += int(m["cleared"] and not m["breached"])
        day = GoalEquityDay(
            m1,
            target_pct=t,
            risk_pct=r,
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=True,
        )
        res = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        pol_c += int(res.cleared and not res.breached)
        breach += int(res.breached)
    return {
        "mark_clear": mark_c,
        "policy_clear": pol_c,
        "breach": breach,
        "n": 10,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--max-train-days", type=int, default=30)
    ap.add_argument("--streak-days", type=int, default=100)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=201)
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    print("Loading days…", flush=True)
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    practice = all_days[:50]
    pairs_dicts = load_pairs()
    pairs = [(float(p["target_pct"]), float(p["risk_pct"])) for p in pairs_dicts]

    print("Soul-plan labels (full_obs)…", flush=True)
    X1, y1, counts, meta = collect_soul_plan_labels(
        practice,
        max_days=args.max_train_days,
        multi_pair=True,
        seed=args.seed,
        pairs=pairs,
        full_obs=True,
        max_entry_samples=36,
    )
    print(f"  plan samples={len(y1)} meta={meta} counts={counts}", flush=True)

    print("Mark HOLD sense corrections (HITL auto)…", flush=True)
    X2, y2 = auto_hitl_hold_labels(
        all_days, pairs, seed=7, n_days=15, start_idx=40
    )
    print(f"  correction samples={len(y2)}", flush=True)

    if len(y1) + len(y2) < 50:
        print("too few labels", flush=True)
        return 2
    if len(y2):
        # oversample corrections 3x — Mark wait sense is the gap
        X = np.concatenate([X1, X2, X2, X2], axis=0)
        y = np.concatenate([y1, y2, y2, y2], axis=0)
    else:
        X, y = X1, y1

    warm = None
    if os.path.isfile(CKPT):
        try:
            blob0 = torch.load(CKPT, map_location="cpu", weights_only=False)
            if int(blob0.get("obs_dim", 0)) == MARK_FULL_DIM:
                warm = blob0["state_dict"]
                print("  warm-start full_obs embryo", flush=True)
        except Exception as e:
            print(f"  warm skip {e}", flush=True)

    print("BC…", flush=True)
    policy, losses = train_bc(
        X,
        y,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
        warm_state=warm,
        obs_dim=MARK_FULL_DIM,
        lr=8e-4,
    )
    metrics = match_rate(policy, X, y)
    print(f"  match={metrics}", flush=True)

    blob = {
        "tag": "mark_clone_full_obs_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state_dict": policy.state_dict(),
        "hidden": args.hidden,
        "obs_dim": MARK_FULL_DIM,
        "eyes_mode": "mark_doctrine",
        "teacher": "soul_plan_plus_mark_hold_sense",
        "full_obs": True,
        "mark_align_policy": True,
        "long_consistency": True,
        "proven_touched": False,
    }
    torch.save(blob, CKPT)
    torch.save(blob, os.path.join(CKPT_DIR, "mark_clone_latest.pt"))
    print(f"wrote {CKPT}", flush=True)

    print("10d score (Mark-aligned decode)…", flush=True)
    s10 = score_10d(policy, all_days, pairs, seed=7, start=40)
    print(f"  {s10}", flush=True)

    # Forward window for streak (after practice)
    forward = all_days[50:]
    n_st = min(int(args.streak_days), len(forward))
    print(f"Award streak forward n={n_st} Mark-aligned…", flush=True)
    streak = run_streak(
        policy, forward, pairs_dicts, seed=42, n_days=n_st, aligned=True
    )
    print(
        f"  max_streak={streak['max_award_streak']} awards={streak['n_award']}/{streak['n_days']} "
        f"breach={streak['n_breach']} pass10={streak['pass_10_streak_breach0']}",
        flush=True,
    )

    # Also pure (no align) diagnostic on same window (shorter)
    n_diag = min(30, n_st)
    print(f"Diagnostic pure greedy (no align) n={n_diag}…", flush=True)
    pure = run_streak(
        policy, forward, pairs_dicts, seed=42, n_days=n_diag, aligned=False
    )
    print(
        f"  pure max_streak={pure['max_award_streak']} breach={pure['n_breach']}",
        flush=True,
    )

    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "goal": "soul=policy long consistency (100s of days)",
        "train_match": metrics,
        "bc_losses_tail": losses[-5:] if losses else [],
        "score_10d": s10,
        "streak": {k: v for k, v in streak.items() if k != "rows"},
        "pure_diag_30": {k: v for k, v in pure.items() if k != "rows"},
        "ckpt": CKPT,
        "proven_touched": False,
        "mark_sense_rules": [
            "Mark HOLD → no open",
            "no opposite side to Mark force",
            "danger>=0.45 no new open",
            "soft target thrash cap 4",
            "Mark-aligned decode default ON for mark_doctrine",
        ],
    }
    # keep streak rows in separate file (large)
    with open(os.path.join(OUT, "CONSISTENCY__latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
        json.dump(streak.get("rows", []), f, indent=2)

    md = [
        "# Mark long-term consistency",
        "",
        f"**When:** {report['saved_at']}",
        "",
        "## Soul = policy sense",
        "- Mark HOLD → policy cannot open",
        "- Opposite side to Mark force blocked",
        "- Danger freeze + soft thrash caps",
        "- BC on soul plans + auto HITL HOLD corrections",
        "",
        "## Scores",
        f"- 10d Mark clear: **{s10['mark_clear']}/10** · policy (aligned): **{s10['policy_clear']}/10** · breach {s10['breach']}",
        f"- Streak {n_st}d: max **{streak['max_award_streak']}** · awards {streak['n_award']}/{streak['n_days']} · breach **{streak['n_breach']}** · pass10={streak['pass_10_streak_breach0']}",
        f"- Pure greedy diag {n_diag}d: max {pure['max_award_streak']} · breach {pure['n_breach']}",
        f"- BC dir_match: **{metrics.get('dir_match')}** · match **{metrics.get('match')}**",
        "",
        "## MARK HERE",
        "1. Open `MARK HERE!.lnk`",
        "2. Review `checkpoints/mark_chart_hitl/HITL__latest.md`",
        "3. Any bar Mark still rejects → add to corrections and re-run this loop",
        "",
        f"Ckpt: `{CKPT}`",
        "",
    ]
    with open(os.path.join(OUT, "CONSISTENCY__latest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    print(f"wrote {OUT}/CONSISTENCY__latest.md", flush=True)
    return 0 if streak["n_breach"] == 0 and s10["policy_clear"] >= 9 else 1


if __name__ == "__main__":
    raise SystemExit(main())
