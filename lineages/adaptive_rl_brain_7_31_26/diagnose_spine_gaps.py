"""Diagnose WHY MWT days fail — size lock vs timing vs thrash.

For each practice MARK_WOULD_TAKE day, compare:
  A) policy pure greedy mark_align (baseline score path)
  B) same policy but with Mark soul plan size dials locked (_plan_lock)
  C) gold spine run_plan (oracle ceiling)

If B >> A: gap is size dials (spine size head / lock), not side.
If B ≈ A and C wins: gap is timing/path under live eyes.
Writes KAG-readable card.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import get_plan, load_oracle, score_policy
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import run_plan
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
REPORT = os.path.join(OUT, "SPINE_GAP_DIAGNOSIS__latest.json")
REPORT_MD = os.path.join(OUT, "SPINE_GAP_DIAGNOSIS__latest.md")


def run_policy(day_map, date, t, r, policy, *, lock_ruf=None, lock_cap=None):
    day = GoalEquityDay(
        day_map[date],
        target_pct=float(t),
        risk_pct=float(r),
        date_str=date,
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=True,
        mark_align_policy=True,
    )
    if lock_ruf is not None:
        day._plan_lock_ruf = float(lock_ruf)
        day._plan_lock_cap = float(lock_cap)
    res = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
    return {
        "award": bool(res.cleared and not res.breached),
        "breached": bool(res.breached),
        "pnl": round(float(res.pnl_pct), 4),
        "n_entries": int(res.n_entries),
    }


def main() -> int:
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    policy = load_policy(CKPT)
    oracle = load_oracle()

    print("Full pack score…", flush=True)
    pack = score_policy(policy, day_map, mark_rows)
    mwt = [r for r in pack["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
    print(
        f"pack same={pack['same_outcome']} mwt={len(mwt)} breach={pack['n_breach']}",
        flush=True,
    )

    rows = []
    n_a = n_b = n_c = 0
    n_b_only = 0  # size lock converts where baseline fails
    for i, r in enumerate(mwt):
        date, t, rr = r["date"], float(r["target_pct"]), float(r["risk_pct"])
        print(f"  diag {i+1}/{len(mwt)} {date}", flush=True)
        mark = get_plan(oracle, day_map, date, t, rr)
        a = run_policy(day_map, date, t, rr, policy)
        lock_ruf = lock_cap = None
        if mark.get("risk_use_frac") not in (None, "dynamic"):
            lock_ruf = float(mark["risk_use_frac"])
            lock_cap = float(mark["per_trade_cap_pct"])
        b = run_policy(day_map, date, t, rr, policy, lock_ruf=lock_ruf, lock_cap=lock_cap)
        c = {"award": False, "breached": False, "pnl": 0.0, "n_entries": 0}
        if mark.get("plan"):
            plan = {int(k): int(v) for k, v in mark["plan"].items()}
            if lock_ruf is not None:
                res = run_plan(
                    day_map[date],
                    date,
                    t,
                    rr,
                    risk_use_frac=lock_ruf,
                    per_trade_cap_pct=lock_cap,
                    plan=plan,
                )
                res.pop("day", None)
                c = {
                    "award": bool(res["cleared"] and not res["breached"]),
                    "breached": bool(res["breached"]),
                    "pnl": res["pnl_pct"],
                    "n_entries": res["n_entries"],
                }
        n_a += int(a["award"])
        n_b += int(b["award"])
        n_c += int(c["award"])
        if b["award"] and not a["award"]:
            n_b_only += 1
        gap = "timing_or_side"
        if b["award"] and not a["award"]:
            gap = "size_lock_converts"
        elif not b["award"] and c["award"]:
            gap = "timing_vs_gold_plan"
        elif a["n_entries"] == 0:
            gap = "false_hold"
        elif a["n_entries"] >= 4:
            gap = "thrash"
        rows.append(
            {
                "date": date,
                "t": t,
                "r": rr,
                "policy": a,
                "policy_size_lock": b,
                "gold_plan": c,
                "gap_class": gap,
                "mark_ruf": lock_ruf,
                "mark_cap": lock_cap,
                "mark_side": mark.get("side"),
                "mark_mode": mark.get("mode"),
            }
        )

    summary = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "pack_same": pack["same_outcome"],
        "pack_mwt": len(mwt),
        "mwt_policy_award": n_a,
        "mwt_size_lock_award": n_b,
        "mwt_gold_award": n_c,
        "size_lock_extra_converts": n_b_only,
        "implication": (
            "SIZE is primary gap — add size head / soul dial prior"
            if n_b_only >= max(3, len(mwt) // 4)
            else (
                "TIMING/PATH is primary — spine event recall + DAgger"
                if n_c > n_b + 2
                else "mixed — per-day class surgery"
            )
        ),
        "rows": rows,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    lines = [
        "# Spine gap diagnosis",
        "",
        f"- pack same={pack['same_outcome']} mwt={len(mwt)}",
        f"- MWT policy awards: {n_a}/{len(mwt)}",
        f"- MWT + size-lock awards: {n_b}/{len(mwt)} (extra converts vs plain: {n_b_only})",
        f"- MWT gold plan awards: {n_c}/{len(mwt)}",
        f"- **Implication:** {summary['implication']}",
        "",
        "| date | pol | size_lock | gold | gap | ruf |",
        "|------|----:|----------:|-----:|-----|-----|",
    ]
    for row in rows:
        lines.append(
            f"| {row['date']} | {int(row['policy']['award'])} | "
            f"{int(row['policy_size_lock']['award'])} | {int(row['gold_plan']['award'])} | "
            f"{row['gap_class']} | {row['mark_ruf']} |"
        )
    with open(REPORT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(json.dumps({k: summary[k] for k in summary if k != "rows"}, indent=2), flush=True)
    print(f"wrote {REPORT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
