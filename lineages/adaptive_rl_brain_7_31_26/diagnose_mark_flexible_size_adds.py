"""Re-diagnose 10 days: Mark with FULL chart + flexible lots + adds (pt5 force).

Prior diagnosis only allowed:
  - one entry
  - fixed shell dials (risk_use_frac=0.35, cap=0.25)

User point: Mark would change lot sizes and ADD into opportunities to meet
that day's goal. Then "Mark failed a day" often stops making sense.

This search (offline, chart known):
  - force-aligned entries only (pt5.1) when require_force=True
  - size mult grid on risk_use_frac and per_trade_cap
  - same-side adds up to max_adds
  - bank at target / die on floor still sacred

Usage:
  python lineages/adaptive_rl_brain_7_31_26/diagnose_mark_flexible_size_adds.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
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

PACK = os.path.join(
    _HERE, "checkpoints", "test_run_10d_mark_vs_policy", "COMPARISON__latest.json"
)
OUT_DIR = os.path.join(_HERE, "checkpoints", "test_run_10d_mark_vs_policy")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def force_map(m1, date: str, target: float, risk: float) -> Dict[int, int]:
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


def run_plan(
    m1,
    date: str,
    target: float,
    risk: float,
    *,
    risk_use_frac: float,
    per_trade_cap_pct: float,
    plan: Dict[int, int],
) -> Dict[str, Any]:
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
        risk_use_frac=risk_use_frac,
        per_trade_cap_pct=per_trade_cap_pct,
    )
    # allow more scale-ins via runner thrash limits for this offline study
    day.runner.max_open_units = 8
    indices = day.runner.decision_indices()
    prev_t = 0
    n_add_signals = 0
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
        a = int(plan.get(int(t), ACTION_HOLD))
        # count adds: same side while already in trade
        if day.side is not None and a in (ACTION_BUY, ACTION_SELL):
            want = 1 if a == ACTION_BUY else -1
            if want == day.side:
                n_add_signals += 1
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
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "pnl_pct": round(float(pnl), 4),
        "n_entries": int(day.n_entries),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
        "banked": bool(day.banked),
        "risk_use_frac": risk_use_frac,
        "per_trade_cap_pct": per_trade_cap_pct,
        "n_add_signals": n_add_signals,
    }


def search_flexible(
    m1,
    date: str,
    target: float,
    risk: float,
    *,
    require_force: bool,
) -> Dict[str, Any]:
    """Grid: size dials × entry times × side × optional one same-side add."""
    force = force_map(m1, date, target, risk)
    day0 = GoalEquityDay(
        m1, target_pct=target, risk_pct=risk, date_str=date, eyes_mode="mark_doctrine"
    )
    indices = list(day0.runner.decision_indices())

    # size flexibility Mark would use relative to the day goal
    size_grid = [
        (0.25, 0.20),
        (0.35, 0.25),  # default shell
        (0.50, 0.35),
        (0.65, 0.45),
        (0.80, 0.55),
        (1.00, 0.70),
    ]

    wins = []
    # subsample entry bars for speed (every decision bar still OK for 10 days)
    entry_ts = indices  # all

    for ruf, cap in size_grid:
        for i, t1 in enumerate(entry_ts):
            for side in (ACTION_BUY, ACTION_SELL):
                f1 = force.get(int(t1), ACTION_HOLD)
                if require_force and f1 != side:
                    continue
                # plan A: single entry
                plan = {int(tt): ACTION_HOLD for tt in indices}
                plan[int(t1)] = side
                res = run_plan(
                    m1, date, target, risk,
                    risk_use_frac=ruf,
                    per_trade_cap_pct=cap,
                    plan=plan,
                )
                if res["cleared"] and not res["breached"]:
                    wins.append(
                        {
                            **res,
                            "mode": "single",
                            "t1": int(t1),
                            "side": NAMES[side],
                            "require_force": require_force,
                        }
                    )
                    # early return first win is enough to prove winnable
                    return {"winnable": True, "best": wins[0], "n_wins_found": 1}

                # plan B: entry + one same-side add later (scale-in)
                for t2 in entry_ts[i + 1 :]:
                    f2 = force.get(int(t2), ACTION_HOLD)
                    if require_force and f2 not in (side, ACTION_HOLD):
                        # add only if force still allows or neutral
                        if f2 != side and f2 != ACTION_HOLD:
                            continue
                    plan2 = {int(tt): ACTION_HOLD for tt in indices}
                    plan2[int(t1)] = side
                    plan2[int(t2)] = side  # add
                    res2 = run_plan(
                        m1, date, target, risk,
                        risk_use_frac=ruf,
                        per_trade_cap_pct=cap,
                        plan=plan2,
                    )
                    if res2["cleared"] and not res2["breached"]:
                        wins.append(
                            {
                                **res2,
                                "mode": "entry_plus_add",
                                "t1": int(t1),
                                "t2": int(t2),
                                "side": NAMES[side],
                                "require_force": require_force,
                            }
                        )
                        return {"winnable": True, "best": wins[0], "n_wins_found": 1}

    return {"winnable": False, "best": None, "n_wins_found": 0}


def baseline(m1, date, target, risk):
    day = GoalEquityDay(
        m1, target_pct=target, risk_pct=risk, date_str=date, eyes_mode="mark_doctrine"
    )
    r = day.run(use_heuristic=True)
    return {
        "cleared": bool(r.cleared),
        "breached": bool(r.breached),
        "pnl_pct": round(float(r.pnl_pct), 4),
        "n_entries": int(r.n_entries),
    }


def main() -> int:
    assert_mark_sets_law()
    if not os.path.isfile(PACK):
        print("missing COMPARISON__latest.json — run 10d test first", flush=True)
        return 2
    pack = json.loads(open(PACK, encoding="utf-8").read())
    day_map = {
        str(d): m for d, m in load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    }

    rows = []
    print("=" * 72, flush=True)
    print("Mark + flexible lots + adds (full chart) — can he win each day?", flush=True)
    print("=" * 72, flush=True)

    for i, day in enumerate(pack["days"], 1):
        date = day["date"]
        t, r = float(day["target_pct"]), float(day["risk_pct"])
        m1 = day_map[date]
        print(f"\n[{i}/10] {date} T/R={t}/{r}", flush=True)
        base = baseline(m1, date, t, r)
        print(
            f"  fixed-shell doctrine: clear={base['cleared']} pnl={base['pnl_pct']} ent={base['n_entries']}",
            flush=True,
        )
        # principle-first flexible
        prin = search_flexible(m1, date, t, r, require_force=True)
        print(
            f"  flexible + force-aligned: winnable={prin['winnable']} best={prin['best']}",
            flush=True,
        )
        anyp = {"winnable": False, "best": None}
        if not prin["winnable"]:
            anyp = search_flexible(m1, date, t, r, require_force=False)
            print(
                f"  flexible ANY side (size+adds): winnable={anyp['winnable']} best={anyp['best']}",
                flush=True,
            )

        if base["cleared"]:
            issue = "D_already_won_fixed_shell"
        elif prin["winnable"]:
            issue = "C_fixed_shell_missed_flexible_principle_win"
        elif anyp["winnable"]:
            issue = "B_need_size_add_or_side_stretch"
        else:
            issue = "A_still_impossible_even_flexible"

        print(f"  ISSUE → {issue}", flush=True)
        rows.append(
            {
                "date": date,
                "target_pct": t,
                "risk_pct": r,
                "baseline": base,
                "principle_flexible": prin,
                "any_flexible": anyp,
                "issue": issue,
            }
        )

    tallies = Counter(r["issue"] for r in rows)
    n = len(rows)
    n_flex_prin = sum(
        1
        for r in rows
        if r["baseline"]["cleared"] or r["principle_flexible"]["winnable"]
    )
    n_flex_any = sum(
        1
        for r in rows
        if r["baseline"]["cleared"]
        or r["principle_flexible"]["winnable"]
        or r["any_flexible"]["winnable"]
    )

    md = []
    md.append("# Diagnosis v2 — Mark with flexible lots + adds (full chart)")
    md.append("")
    md.append(f"**When:** {datetime.now(timezone.utc).isoformat()}")
    md.append("")
    md.append("## Your point")
    md.append(
        "Mark would **change lot sizes** and **add** into opportunities relative to "
        "**that day's goal**. Then it often **doesn't make sense** for him to fail a day "
        "if he already sees the chart."
    )
    md.append("")
    md.append("## What we wrong-footed before")
    md.append(
        "v1 diagnosis only allowed **one entry** + **fixed** risk_use_frac=0.35 / cap=0.25. "
        "That understates Mark."
    )
    md.append("")
    md.append("## What we search now (offline, chart known)")
    md.append("- Size grid: risk_use_frac × per_trade_cap (small → aggressive)")
    md.append("- Single entry **or** entry + one same-side **add**")
    md.append("- Prefer **force-aligned** (pt5.1); only then try any-side if needed")
    md.append("- Floor / bank shell still on")
    md.append("")
    md.append("## Tallies")
    md.append("")
    md.append("| Issue | Count |")
    md.append("|-------|------:|")
    for k, v in sorted(tallies.items()):
        md.append(f"| `{k}` | **{v}** |")
    md.append("")
    md.append(f"| Days winnable with **force + flexible size/add** (incl. baseline wins) | **{n_flex_prin}/{n}** |")
    md.append(f"| Days winnable if also allow side stretch | **{n_flex_any}/{n}** |")
    md.append("")
    md.append("## Day-by-day")
    md.append("")
    md.append(
        "| Date | T/R | Fixed clear | Flex+force win | Flex any win | Issue |"
    )
    md.append("|------|----:|:-----------:|:--------------:|:------------:|-------|")
    for r in rows:
        md.append(
            f"| {r['date']} | {r['target_pct']}/{r['risk_pct']} | "
            f"{'Y' if r['baseline']['cleared'] else 'n'} | "
            f"{'Y' if r['principle_flexible']['winnable'] or r['baseline']['cleared'] else 'n'} | "
            f"{'Y' if r['any_flexible']['winnable'] or r['principle_flexible']['winnable'] or r['baseline']['cleared'] else 'n'} | "
            f"`{r['issue']}` |"
        )
    md.append("")
    md.append("## Conclusion")
    md.append("")
    if n_flex_prin >= 9:
        md.append(
            f"**You were right for almost all days:** with flexible lots/adds under force, "
            f"**{n_flex_prin}/{n}** are winnable. Prior “Mark failed” was mostly "
            f"**rigid size + no adds + timid teacher**, not “Mark can’t see the chart.”"
        )
    else:
        md.append(
            f"Flexible size/adds lifts winnability to **{n_flex_prin}/{n}** under force "
            f"(**{n_flex_any}/{n}** if side stretch). Remaining gaps need more adds, "
            f"more size grid, or multi-leg reverses."
        )
    md.append("")
    md.append("### What the policy/teacher must learn next")
    md.append("1. **Size relative to remaining distance to target** (not fixed 0.35 forever)")
    md.append("2. **Adds** on continuation with HTF force (not banned thrash reverse package)")
    md.append("3. **Stop thrashing** when one good sized entry would bank")
    md.append("4. Keep floor sacred while sizing up when heat allows")
    md.append("")
    md.append("## Reproduce")
    md.append("```powershell")
    md.append("cd C:\\Users\\user\\Fable5_Foundation\\MOMENTUM_ONE\\the-truth")
    md.append("$env:PYTHONPATH = \".;code\"")
    md.append(
        "python lineages/adaptive_rl_brain_7_31_26/diagnose_mark_flexible_size_adds.py"
    )
    md.append("```")
    md.append("")

    path_md = os.path.join(OUT_DIR, "DIAGNOSIS_FLEXIBLE_SIZE_ADDS__latest.md")
    path_js = os.path.join(OUT_DIR, "DIAGNOSIS_FLEXIBLE_SIZE_ADDS__latest.json")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md))
    with open(path_js, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "tallies": dict(tallies),
                "n_flex_principle": n_flex_prin,
                "n_flex_any": n_flex_any,
                "days": rows,
            },
            f,
            indent=2,
            default=str,
        )
    print("\nTALLIES", dict(tallies), flush=True)
    print(f"force-flexible wins (incl base): {n_flex_prin}/{n}", flush=True)
    print(f"any-flexible wins (incl base): {n_flex_any}/{n}", flush=True)
    print(f"WROTE {path_md}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
