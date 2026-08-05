"""Day-by-day autopsy of non-award streak gaps.

For each day in STREAK_ROWS (or a fresh forward run):
  1) Policy result already known (award / miss / breach)
  2) Mark soul plan search: is the day winnable under force + risk shell?
  3) Classify gap:

     NO_OPPORTUNITY     — Mark soul plan cannot clear (physics / force shell)
     MARK_WOULD_TAKE    — Mark soul clears; policy missed → learnable misread
     BOTH_MISS          — Mark soul also fails (plan + online fallback)
     POLICY_BREACH      — policy hit floor (should be rare; shell check)
     AWARD              — not a gap

Usage (repo root):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/autopsy_streak_gaps.py
  python lineages/adaptive_rl_brain_7_31_26/autopsy_streak_gaps.py --from-rows
  python lineages/adaptive_rl_brain_7_31_26/autopsy_streak_gaps.py --n-days 40 --seed 42
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
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import (
    load_pairs,
    longest_award_streak,
    sample_pairs_for_days,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import (
    execute_mark_soul_day,
    search_mark_soul_plan,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    STREAK_REWARD_DIALS,
    apply_autopsy_to_streak_dials,
    default_streak_dials,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT_DIR = os.path.join(CKPT_DIR, "mark_consistency")
STREAK_ROWS = os.path.join(OUT_DIR, "STREAK_ROWS__latest.json")
CKPT = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
AUTOPSY_JSON = os.path.join(OUT_DIR, "AUTOPSY_GAPS__latest.json")
AUTOPSY_MD = os.path.join(OUT_DIR, "AUTOPSY_GAPS__latest.md")
STREAK_DIALS_PATH = os.path.join(OUT_DIR, "STREAK_REWARD_DIALS__latest.json")


def _load_policy(path: str) -> Optional[Channel1Policy]:
    if not os.path.isfile(path):
        return None
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pol = Channel1Policy(
        obs_dim=int(blob.get("obs_dim", MARK_FULL_DIM)),
        hidden=int(blob.get("hidden", 128)),
    )
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol


def _run_policy_day(
    m1,
    date: str,
    target: float,
    risk: float,
    policy: Optional[Channel1Policy],
) -> Dict[str, Any]:
    day = GoalEquityDay(
        m1,
        target_pct=float(target),
        risk_pct=float(risk),
        date_str=str(date),
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=True,
        mark_align_policy=True,
    )
    if policy is None:
        r = day.run(use_heuristic=True)
    else:
        r = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
    return {
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "n_entries": int(r.n_entries),
        "award": bool(r.cleared and not r.breached),
        "min_eq_pct": round(float(r.min_eq_pct), 4),
    }


def classify_gap(
    *,
    policy: Dict[str, Any],
    mark_plan_winnable: bool,
    mark_cleared: bool,
    mark_source: str,
    mark_pnl: float,
    mark_entries: int,
) -> Tuple[str, str]:
    """Return (gap_class, sub_class)."""
    if policy.get("breached"):
        return "POLICY_BREACH", "floor"
    if policy.get("award"):
        return "AWARD", "clear"

    # Non-award
    if mark_plan_winnable and mark_cleared:
        # Mark would have taken / could clear with soul plan
        sub = "misread_valid_opportunity"
        pe = int(policy.get("n_entries", 0) or 0)
        me = int(mark_entries)
        pp = float(policy.get("pnl_pct", 0.0) or 0.0)
        if pe == 0 and me > 0:
            sub = "policy_froze_mark_acted"
        elif pe > me + 1:
            sub = "policy_thrash_vs_sparse_mark"
        elif pe > 0 and pp < 0 and mark_pnl >= 0:
            sub = "policy_wrong_size_or_timing"
        elif pe > 0 and me > 0 and pe < me:
            sub = "policy_undersize_or_wait"
        return "MARK_WOULD_TAKE", sub

    if not mark_plan_winnable and not mark_cleared:
        # No force-aligned plan hits target under shell
        sub = "physics_or_force_impossible"
        t = float(policy.get("target_pct", 0) or 0)
        if t >= 2.5:
            sub = "hard_target_no_force_path"
        elif int(policy.get("n_entries", 0) or 0) == 0:
            sub = "dead_day_no_setup"
        return "NO_OPPORTUNITY", sub

    # Plan not found but online mark cleared, or plan found but didn't clear (edge)
    if mark_cleared and not policy.get("award"):
        return "MARK_WOULD_TAKE", f"online_mark_clear_src={mark_source}"
    return "BOTH_MISS", f"src={mark_source}"


def autopsy_one(
    m1,
    date: str,
    target: float,
    risk: float,
    policy: Optional[Channel1Policy],
    *,
    policy_row: Optional[Dict[str, Any]] = None,
    max_entry_samples: int = 36,
) -> Dict[str, Any]:
    if policy_row is not None:
        pol = {
            "cleared": bool(policy_row.get("cleared")),
            "breached": bool(policy_row.get("breached")),
            "pnl_pct": float(policy_row.get("pnl_pct", 0.0)),
            "n_entries": int(policy_row.get("n_entries", 0)),
            "award": bool(policy_row.get("award")),
            "min_eq_pct": float(policy_row.get("min_eq_pct", 0.0) or 0.0),
            "target_pct": float(target),
            "risk_pct": float(risk),
        }
    else:
        pol = _run_policy_day(m1, date, target, risk, policy)
        pol["target_pct"] = float(target)
        pol["risk_pct"] = float(risk)

    found = search_mark_soul_plan(
        m1,
        date,
        float(target),
        float(risk),
        require_force=True,
        max_entry_samples=int(max_entry_samples),
    )
    mark_plan_winnable = bool(found.get("winnable"))
    mark_exec = execute_mark_soul_day(
        m1,
        date,
        float(target),
        float(risk),
        max_entry_samples=int(max_entry_samples),
    )
    gap_class, sub = classify_gap(
        policy=pol,
        mark_plan_winnable=mark_plan_winnable,
        mark_cleared=bool(mark_exec.get("cleared") and not mark_exec.get("breached")),
        mark_source=str(mark_exec.get("source", "")),
        mark_pnl=float(mark_exec.get("pnl_pct", 0.0) or 0.0),
        mark_entries=int(mark_exec.get("n_entries", 0) or 0),
    )
    return {
        "date": str(date),
        "target_pct": float(target),
        "risk_pct": float(risk),
        "gap_class": gap_class,
        "sub_class": sub,
        "policy": {
            "cleared": pol["cleared"],
            "breached": pol["breached"],
            "pnl_pct": pol["pnl_pct"],
            "n_entries": pol["n_entries"],
            "award": pol["award"],
        },
        "mark": {
            "plan_winnable": mark_plan_winnable,
            "source": mark_exec.get("source"),
            "mode": mark_exec.get("mode"),
            "cleared": bool(mark_exec.get("cleared")),
            "breached": bool(mark_exec.get("breached")),
            "pnl_pct": mark_exec.get("pnl_pct"),
            "n_entries": mark_exec.get("n_entries"),
            "side": mark_exec.get("side"),
            "risk_use_frac": mark_exec.get("risk_use_frac"),
            "per_trade_cap_pct": mark_exec.get("per_trade_cap_pct"),
        },
        # Reward-update hint only (not shell change)
        "reward_hint": {
            "NO_OPPORTUNITY": "reward patient HOLD; cut inactivity tax when force neutral",
            "MARK_WOULD_TAKE": "penalty misread; boost soul-side entry; EOD streak-break penalty",
            "BOTH_MISS": "neutral / research; do not force thrash for awards",
            "POLICY_BREACH": "keep mindless/floor walls; never loosen shell",
            "AWARD": "EOD streak award bonus",
        }.get(gap_class, "none"),
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts: Dict[str, int] = {}
    sub_counts: Dict[str, int] = {}
    gaps = [r for r in rows if r["gap_class"] not in ("AWARD",)]
    for r in rows:
        counts[r["gap_class"]] = counts.get(r["gap_class"], 0) + 1
        key = f"{r['gap_class']}:{r['sub_class']}"
        sub_counts[key] = sub_counts.get(key, 0) + 1
    n = len(rows)
    n_award = counts.get("AWARD", 0)
    n_mark_take = counts.get("MARK_WOULD_TAKE", 0)
    n_no_opp = counts.get("NO_OPPORTUNITY", 0)
    # streak from policy awards
    award_rows = [
        {"award": bool(r["policy"]["award"])} for r in rows
    ]
    max_s, s_i, e_i = longest_award_streak(award_rows)
    return {
        "n_days": n,
        "n_award": n_award,
        "n_gaps": len(gaps),
        "counts": counts,
        "sub_counts": sub_counts,
        "mark_would_take_pct_of_gaps": (
            round(100.0 * n_mark_take / max(len(gaps), 1), 1)
        ),
        "no_opportunity_pct_of_gaps": (
            round(100.0 * n_no_opp / max(len(gaps), 1), 1)
        ),
        "max_award_streak": int(max_s),
        "learnable_fraction": round(
            n_mark_take / max(n_mark_take + n_no_opp, 1), 3
        ),
        "headline": (
            f"Of {len(gaps)} non-award days: "
            f"{n_mark_take} Mark-would-take (learnable), "
            f"{n_no_opp} no-opportunity (do not thrash)."
        ),
    }


def write_md(rows: List[Dict[str, Any]], summary: Dict[str, Any], dials: Dict[str, float]) -> str:
    lines = [
        "# Streak gap autopsy — no-opp vs Mark-would-take",
        "",
        f"**When:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        "",
        "## Headline",
        "",
        summary.get("headline", ""),
        "",
        f"- Max award streak (this window): **{summary.get('max_award_streak')}**",
        f"- Learnable fraction of (Mark-take + no-opp): **{summary.get('learnable_fraction')}**",
        f"- Counts: `{json.dumps(summary.get('counts', {}))}`",
        "",
        "## Rule for rewards (only)",
        "",
        "| Gap class | Reward / penalty action |",
        "|-----------|-------------------------|",
        "| **NO_OPPORTUNITY** | Reward patient HOLD; **cut** inactivity / majority-idle tax when force neutral. Do **not** force entries for awards. |",
        "| **MARK_WOULD_TAKE** | **Penalize** misread; boost soul-side entry; EOD **streak-break** penalty. BC labels from Mark soul plan. |",
        "| **BOTH_MISS** | Neutral research; no thrash for score. |",
        "| **POLICY_BREACH** | Keep floor walls; never loosen shell. |",
        "| **AWARD** | EOD **streak award** bonus (longer streak → larger bonus). |",
        "",
        "## Day-by-day",
        "",
        "| Date | T/R | Policy PnL / n | Mark plan? / PnL / n | Class | Sub |",
        "|------|-----|----------------|----------------------|-------|-----|",
    ]
    for r in rows:
        p = r["policy"]
        m = r["mark"]
        lines.append(
            f"| {r['date']} | {r['target_pct']}/{r['risk_pct']} | "
            f"{p['pnl_pct']}% / {p['n_entries']} | "
            f"{'Y' if m['plan_winnable'] else 'N'} / {m['pnl_pct']}% / {m['n_entries']} | "
            f"**{r['gap_class']}** | {r['sub_class']} |"
        )
    lines += [
        "",
        "## Streak reward dials (updated from this autopsy)",
        "",
        "```json",
        json.dumps(dials, indent=2),
        "```",
        "",
        "**Forbidden:** shell heat/bank/breach · trail package · PROVEN overwrite · entry-rule thrash for awards.",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Autopsy non-award streak gaps")
    ap.add_argument("--from-rows", action="store_true", help="use STREAK_ROWS__latest.json")
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    ap.add_argument("--max-entry-samples", type=int, default=36)
    ap.add_argument("--ckpt", default=CKPT)
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_days = load_calendar_days(args.data, min_bars=900)
    _, forward = split_practice_forward(all_days, practice_n=50)
    day_map = {str(d): m1 for d, m1 in all_days}
    policy = _load_policy(args.ckpt)

    work: List[Tuple[str, Any, float, float, Optional[dict]]] = []
    if args.from_rows and os.path.isfile(STREAK_ROWS):
        with open(STREAK_ROWS, "r", encoding="utf-8") as f:
            rows_in = json.load(f)
        for row in rows_in:
            d = str(row["date"])
            if d not in day_map:
                print(f"skip missing day {d}", flush=True)
                continue
            work.append(
                (
                    d,
                    day_map[d],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    row,
                )
            )
        print(f"autopsy from STREAK_ROWS n={len(work)}", flush=True)
    else:
        window = forward[: int(args.n_days)]
        pairs = load_pairs()
        tr = sample_pairs_for_days(len(window), pairs, seed=args.seed, soft_bias=False)
        for (date, m1), (t, r) in zip(window, tr):
            work.append((str(date), m1, float(t), float(r), None))
        print(f"autopsy fresh forward n={len(work)} seed={args.seed}", flush=True)

    out_rows: List[Dict[str, Any]] = []
    for date, m1, t, r, prow in work:
        print(f"  autopsy {date} T/R={t}/{r} ...", flush=True)
        rec = autopsy_one(
            m1,
            date,
            t,
            r,
            policy,
            policy_row=prow,
            max_entry_samples=int(args.max_entry_samples),
        )
        out_rows.append(rec)
        print(
            f"    → {rec['gap_class']}/{rec['sub_class']} "
            f"pol={rec['policy']['pnl_pct']}% mark_win={rec['mark']['plan_winnable']}",
            flush=True,
        )

    summary = summarize(out_rows)
    dials = apply_autopsy_to_streak_dials(summary, base=default_streak_dials())

    payload = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "source": "STREAK_ROWS" if args.from_rows else f"forward_seed_{args.seed}",
        "summary": summary,
        "rows": out_rows,
        "streak_reward_dials": dials,
        "proven_touched": False,
        "shell_touched": False,
        "only_rewards_penalties": True,
    }
    with open(AUTOPSY_JSON, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(STREAK_DIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": payload["saved_at"],
                "dials": dials,
                "from_autopsy": AUTOPSY_JSON,
                "bounds": STREAK_REWARD_DIALS,
                "note": "Update rewards/penalties only — longer award streaks. No shell/PROVEN.",
            },
            f,
            indent=2,
        )
    md = write_md(out_rows, summary, dials)
    with open(AUTOPSY_MD, "w", encoding="utf-8") as f:
        f.write(md)

    print(summary.get("headline", ""), flush=True)
    print(f"wrote {AUTOPSY_MD}", flush=True)
    print(f"wrote {STREAK_DIALS_PATH}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
