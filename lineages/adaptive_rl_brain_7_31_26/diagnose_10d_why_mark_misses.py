"""Diagnose: if Mark sees the full chart + pt5 principles, why miss award days?

For each day in the 10d test pack:
  1) Baseline mark_doctrine result
  2) Physics: is there ANY single-entry plan that banks target without floor breach?
  3) Principles filter: same search but only entries with live HTF force agreeing (pt5.1)
  4) Classify ISSUE:
     - A_physics_impossible: no plan hits target under risk shell
     - B_principles_block_winning_entry: physics win exists but only against HTF force
     - C_teacher_missed_valid_win: force-aligned winning entry exists; doctrine didn't take it
     - D_teacher_won: already clear

Usage:
  python lineages/adaptive_rl_brain_7_31_26/diagnose_10d_why_mark_misses.py
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from lineages.adaptive_rl_brain_7_31_26.perception.sets import assert_mark_sets_law

CKPT_DIR = os.path.join(_HERE, "checkpoints")
PACK = os.path.join(CKPT_DIR, "test_run_10d_mark_vs_policy", "COMPARISON__latest.json")
OUT_DIR = os.path.join(CKPT_DIR, "test_run_10d_mark_vs_policy")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def run_day_heuristic(m1, date: str, target: float, risk: float, eyes: str = "mark_doctrine"):
    day = GoalEquityDay(
        m1, target_pct=target, risk_pct=risk, date_str=date, eyes_mode=eyes
    )
    r = day.run(use_heuristic=True)
    return {
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "n_entries": int(r.n_entries),
        "min_eq_pct": round(float(r.min_eq_pct), 4),
        "banked": bool(r.banked),
    }


def force_at_bars(m1, date: str, target: float, risk: float) -> Dict[int, int]:
    day = GoalEquityDay(
        m1, target_pct=target, risk_pct=risk, date_str=date, eyes_mode="mark_doctrine"
    )
    out = {}
    for t in day.runner.decision_indices():
        try:
            out[int(t)] = int(day.recommended_action(t))
        except Exception:
            out[int(t)] = ACTION_HOLD
    return out


def simulate_single_entry(
    m1, date: str, target: float, risk: float, entry_t: int, side: int
) -> Dict[str, Any]:
    """Enter once at decision bar entry_t, then HOLD manage until bank/stop/eod."""
    day = GoalEquityDay(
        m1, target_pct=target, risk_pct=risk, date_str=date, eyes_mode="mark_doctrine"
    )
    indices = day.runner.decision_indices()
    prev_t = 0
    for t in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev_t, t):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = t + 1
        if day.dead or day.banked:
            break
        a = side if int(t) == int(entry_t) else ACTION_HOLD
        day.step_action(t, a)
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    day.min_eq_pct = min(day.min_eq_pct, pnl)
    if pnl <= -day.risk + 1e-12:
        day.breached = True
    cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
        day.banked and not day.breached
    )
    return {
        "entry_t": int(entry_t),
        "side": NAMES[side],
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "pnl_pct": round(float(pnl), 4),
        "n_entries": int(day.n_entries),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
        "banked": bool(day.banked),
    }


def search_wins(
    m1, date: str, target: float, risk: float, force: Dict[int, int]
) -> Dict[str, Any]:
    day0 = GoalEquityDay(
        m1, target_pct=target, risk_pct=risk, date_str=date, eyes_mode="mark_doctrine"
    )
    indices = list(day0.runner.decision_indices())
    any_win = []
    principle_win = []  # force agrees
    for t in indices:
        for side in (ACTION_BUY, ACTION_SELL):
            sim = simulate_single_entry(m1, date, target, risk, t, side)
            if sim["cleared"] and not sim["breached"]:
                any_win.append(sim)
                f = force.get(int(t), ACTION_HOLD)
                if f == side:
                    principle_win.append(sim)
    return {
        "n_decision_bars": len(indices),
        "n_physics_wins": len(any_win),
        "n_principle_wins": len(principle_win),
        "best_physics": max(any_win, key=lambda x: x["pnl_pct"]) if any_win else None,
        "best_principle": (
            max(principle_win, key=lambda x: x["pnl_pct"]) if principle_win else None
        ),
        "sample_principle_wins": principle_win[:5],
        "sample_physics_only": [w for w in any_win if w not in principle_win][:5],
    }


def classify(baseline: Dict[str, Any], search: Dict[str, Any]) -> str:
    if baseline["cleared"] and not baseline["breached"]:
        return "D_teacher_already_won"
    if search["n_physics_wins"] == 0:
        return "A_physics_impossible_under_shell"
    if search["n_principle_wins"] == 0:
        return "B_principles_block_all_winning_entries"
    return "C_teacher_missed_valid_principle_win"


def main() -> int:
    assert_mark_sets_law()
    if not os.path.isfile(PACK):
        print(f"missing pack {PACK} — run test_run_10d_mark_vs_policy.py first", flush=True)
        return 2
    pack = json.loads(open(PACK, encoding="utf-8").read())
    day_map = {str(d): m for d, m in load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)}

    rows = []
    print("=" * 72, flush=True)
    print("DIAGNOSE: full chart + pt5 principles → why not 10/10?", flush=True)
    print("=" * 72, flush=True)

    for i, day in enumerate(pack["days"], 1):
        date = day["date"]
        t, r = float(day["target_pct"]), float(day["risk_pct"])
        if date not in day_map:
            print(f"skip {date}", flush=True)
            continue
        m1 = day_map[date]
        print(f"\n[{i}/10] {date} T/R={t}/{r}", flush=True)
        base = run_day_heuristic(m1, date, t, r, "mark_doctrine")
        force = force_at_bars(m1, date, t, r)
        n_force = sum(1 for v in force.values() if v != ACTION_HOLD)
        search = search_wins(m1, date, t, r, force)
        issue = classify(base, search)
        print(
            f"  baseline clear={base['cleared']} pnl={base['pnl_pct']} ent={base['n_entries']}",
            flush=True,
        )
        print(
            f"  force directional bars={n_force}/{search['n_decision_bars']}  "
            f"physics_wins={search['n_physics_wins']} principle_wins={search['n_principle_wins']}",
            flush=True,
        )
        print(f"  ISSUE → {issue}", flush=True)
        if search["best_principle"]:
            bp = search["best_principle"]
            print(
                f"  best principle win: {bp['side']} @t={bp['entry_t']} pnl={bp['pnl_pct']}",
                flush=True,
            )
        elif search["best_physics"]:
            bp = search["best_physics"]
            print(
                f"  best physics-only win: {bp['side']} @t={bp['entry_t']} pnl={bp['pnl_pct']} "
                f"(AGAINST or WITHOUT force)",
                flush=True,
            )
        rows.append(
            {
                "date": date,
                "target_pct": t,
                "risk_pct": r,
                "baseline": base,
                "pack_mark_cleared": day["mark_cleared"],
                "pack_policy_cleared": day["policy_cleared"],
                "force_directional_bars": n_force,
                "decision_bars": search["n_decision_bars"],
                "physics_wins": search["n_physics_wins"],
                "principle_wins": search["n_principle_wins"],
                "best_principle": search["best_principle"],
                "best_physics": search["best_physics"],
                "issue": issue,
            }
        )

    # tallies
    from collections import Counter

    tallies = Counter(r["issue"] for r in rows)
    n = len(rows)

    md = []
    md.append("# Diagnosis — why Mark doesn’t win all 10 (with full chart + pt5)")
    md.append("")
    md.append(f"**When:** {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("## Question")
    md.append(
        "If Mark **sees the same 10 charts** and uses **pt5 principles**, "
        "he should know how to win. What’s the issue when he doesn’t?"
    )
    md.append("")
    md.append("## Issue classes")
    md.append("")
    md.append("| Code | Meaning |")
    md.append("|------|---------|")
    md.append(
        "| **A_physics_impossible_under_shell** | No single-entry plan banks target without hitting floor. Day may need multi-leg or be too tight for target/risk. |"
    )
    md.append(
        "| **B_principles_block_all_winning_entries** | A win exists only by trading **against** live HTF force (breaks pt5.1). |"
    )
    md.append(
        "| **C_teacher_missed_valid_principle_win** | At least one **force-aligned** entry would have won — codified Mark didn’t take it (timing/selectivity bug). |"
    )
    md.append("| **D_teacher_already_won** | Baseline doctrine already cleared. |")
    md.append("")
    md.append("## Tallies")
    md.append("")
    md.append("| Issue | Count |")
    md.append("|-------|------:|")
    for k, v in sorted(tallies.items()):
        md.append(f"| `{k}` | **{v}** |")
    md.append(f"| **Total days** | **{n}** |")
    md.append("")
    md.append("## Day-by-day")
    md.append("")
    md.append(
        "| Date | T/R | Base clear | Physics wins | Principle wins | Issue |"
    )
    md.append("|------|----:|:----------:|-------------:|---------------:|-------|")
    for r in rows:
        md.append(
            f"| {r['date']} | {r['target_pct']}/{r['risk_pct']} | "
            f"{'Y' if r['baseline']['cleared'] else 'n'} | "
            f"{r['physics_wins']} | {r['principle_wins']} | `{r['issue']}` |"
        )
    md.append("")
    md.append("## Root-cause read (for the lab)")
    md.append("")
    a = tallies.get("A_physics_impossible_under_shell", 0)
    b = tallies.get("B_principles_block_all_winning_entries", 0)
    c = tallies.get("C_teacher_missed_valid_principle_win", 0)
    d = tallies.get("D_teacher_already_won", 0)
    md.append(f"- Already won under doctrine: **{d}/{n}**")
    md.append(f"- Teacher missed a **legal** (pt5 force-aligned) win: **{c}/{n}** ← fix encoding/timing")
    md.append(
        f"- Win only by **violating** HTF permission: **{b}/{n}** ← principles vs greed conflict"
    )
    md.append(
        f"- **No** single-entry win under shell risk: **{a}/{n}** ← target/risk too hard for that day path, or need multi-leg / different size rules"
    )
    md.append("")
    md.append("### What “Mark sees the chart” does NOT mean here")
    md.append(
        "- It does **not** mean ignore pt5 and buy the perfect hindsight trade against HTF."
    )
    md.append(
        "- It **does** mean: with full price path known offline, find the best plan **allowed by principles + risk shell**."
    )
    md.append("")
    if c > 0:
        md.append("### Primary bug if C > 0")
        md.append(
            "Doctrine is **too timid or mistimed**: a force-aligned single entry would bank, but the live teacher didn’t take it (or overtraded and gave it back)."
        )
    if b > 0 and c == 0:
        md.append("### Primary tension if only B")
        md.append(
            "Physics wants a trade **against** the force vote that day. Strict Mark **refuses**; thrash bot might take it and luck into target."
        )
    if a > 0:
        md.append("### Primary wall if A")
        md.append(
            "Even omniscient single-entry under this shell cannot bank that T/R. Need multi-entry skill, different stops, or accept no-award day."
        )
    md.append("")
    md.append("## Reproduce")
    md.append("```powershell")
    md.append("cd C:\\Users\\user\\Fable5_Foundation\\MOMENTUM_ONE\\the-truth")
    md.append("$env:PYTHONPATH = \".;code\"")
    md.append(
        "python lineages/adaptive_rl_brain_7_31_26/diagnose_10d_why_mark_misses.py"
    )
    md.append("```")
    md.append("")

    md_path = os.path.join(OUT_DIR, "DIAGNOSIS_WHY_MARK_MISSES__latest.md")
    json_path = os.path.join(OUT_DIR, "DIAGNOSIS_WHY_MARK_MISSES__latest.json")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "tallies": dict(tallies),
                "days": rows,
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 72, flush=True)
    print("TALLIES", dict(tallies), flush=True)
    print(f"WROTE {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
