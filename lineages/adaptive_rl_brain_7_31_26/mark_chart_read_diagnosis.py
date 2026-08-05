"""Mark reads the chart the policy trades on — diagnose clear vs miss DNA."""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
for _p in (_ROOT, os.path.join(_ROOT, "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
    split_practice_forward,
)


def main() -> int:
    p = Path(_HERE) / "checkpoints" / "FORWARD_MARK_POLICY_TEST.json"
    if p.exists():
        r = json.loads(p.read_text(encoding="utf-8"))
        print("=== PRIOR FORWARD REPORT ===")
        print(json.dumps(r.get("bottom_line"), indent=2))
        for k, v in r.get("pairs", {}).items():
            te, po = v.get("teacher", {}), v.get("policy", {})
            print(
                f"pair {k}: teacher clear={te.get('clear_pct')} ent={te.get('mean_entries')} | "
                f"policy clear={po.get('clear_pct')} match={po.get('step_match')}"
            )

    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=50)
    print(f"\nFORWARD n={len(forward)} {forward[0][0]} -> {forward[-1][0]}")

    print("\n=== CHART DNA: hard 3.0/3.5 teacher ===")
    reason_on_clear: Counter = Counter()
    reason_on_miss: Counter = Counter()
    entry_clear, entry_miss, pnl_miss = [], [], []
    regime_clear, regime_miss = Counter(), Counter()
    soft_scalp_miss = 0
    multi_aligned_clear = 0
    near = []

    for date_str, m1 in forward:
        day = GoalEquityDay(
            m1,
            target_pct=3.0,
            risk_pct=3.5,
            date_str=str(date_str),
            eyes_mode="mark_doctrine",
        )
        reasons, regimes = [], []
        n_soft = n_release = 0
        prev = 0
        for t in day.runner.decision_indices():
            if day.banked or day.dead:
                break
            for bt in range(prev, t):
                day._mark_bar(bt)
            prev = t + 1
            act = day.recommended_action(t)
            dec = getattr(day.runner, "last_doctrine", None)
            if dec:
                reasons.append(dec.reason)
                regimes.append(str(dec.regime.value))
                if "soft_single" in dec.reason:
                    n_soft += 1
                if "slingshot_release" in dec.reason:
                    n_release += 1
            day.step_action(t, act)

        day2 = GoalEquityDay(
            m1,
            target_pct=3.0,
            risk_pct=3.5,
            date_str=str(date_str),
            eyes_mode="mark_doctrine",
        )
        res = day2.run(use_heuristic=True)
        bucket = reason_on_clear if res.cleared else reason_on_miss
        for rr in set(reasons):
            bucket[rr] += 1
        if res.cleared:
            entry_clear.append(res.n_entries)
            for rg in set(regimes):
                regime_clear[rg] += 1
            if n_release >= 1:
                multi_aligned_clear += 1
        else:
            entry_miss.append(res.n_entries)
            pnl_miss.append(res.pnl_pct)
            for rg in set(regimes):
                regime_miss[rg] += 1
            if n_soft >= 2:
                soft_scalp_miss += 1
            if res.pnl_pct >= 1.5:
                near.append(
                    (str(date_str), res.pnl_pct, res.n_entries, res.min_eq_pct, n_soft, n_release)
                )

    print(
        "clears",
        len(entry_clear),
        "mean_ent",
        float(np.mean(entry_clear)) if entry_clear else None,
    )
    print(
        "misses",
        len(entry_miss),
        "mean_ent",
        float(np.mean(entry_miss)) if entry_miss else None,
        "mean_pnl",
        float(np.mean(pnl_miss)) if pnl_miss else None,
    )
    print("reasons on CLEAR days:", reason_on_clear.most_common(10))
    print("reasons on MISS days:", reason_on_miss.most_common(10))
    print("regimes on clear:", regime_clear.most_common())
    print("regimes on miss:", regime_miss.most_common())
    print("miss days with >=2 soft_single decisions:", soft_scalp_miss)
    print("clear days with slingshot_release:", multi_aligned_clear)

    print("\n=== NEAR MISS pnl>=1.5 not clear (Mark almost banked) ===")
    print("n", len(near))
    for row in sorted(near, key=lambda z: -z[1])[:12]:
        print(row)

    print("\n=== SAME DAYS: legacy thrash vs Mark ===")
    for date_str, pnl, ent, mn, ns, nr in sorted(near, key=lambda z: -z[1])[:6]:
        m1 = [m for d, m in forward if str(d) == date_str][0]
        leg = GoalEquityDay(
            m1,
            target_pct=3.0,
            risk_pct=3.5,
            date_str=date_str,
            eyes_mode="legacy_set2",
        )
        rl = leg.run(use_heuristic=True)
        print(
            f"{date_str} MARK pnl={pnl:.2f} ent={ent} soft={ns} rel={nr} | "
            f"LEGACY clear={rl.cleared} pnl={rl.pnl_pct:.2f} ent={rl.n_entries}"
        )

    # Opportunity: days legacy cleared hard that Mark missed
    print("\n=== LEGACY CLEAR / MARK MISS (left money Mark refused thrash for?) ===")
    left = 0
    for date_str, m1 in forward:
        mk = GoalEquityDay(
            m1, target_pct=3.0, risk_pct=3.5, date_str=str(date_str), eyes_mode="mark_doctrine"
        )
        lg = GoalEquityDay(
            m1, target_pct=3.0, risk_pct=3.5, date_str=str(date_str), eyes_mode="legacy_set2"
        )
        rm, rl = mk.run(use_heuristic=True), lg.run(use_heuristic=True)
        if rl.cleared and not rm.cleared:
            left += 1
            if left <= 8:
                print(
                    f"{date_str} LEGACY banked pnl={rl.pnl_pct:.2f} ent={rl.n_entries} | "
                    f"MARK pnl={rm.pnl_pct:.2f} ent={rm.n_entries}"
                )
    print("total legacy-clear mark-miss:", left)

    print("\n=== MARK CLEAR / LEGACY MISS (Mark skill days) ===")
    skill = 0
    for date_str, m1 in forward:
        mk = GoalEquityDay(
            m1, target_pct=3.0, risk_pct=3.5, date_str=str(date_str), eyes_mode="mark_doctrine"
        )
        lg = GoalEquityDay(
            m1, target_pct=3.0, risk_pct=3.5, date_str=str(date_str), eyes_mode="legacy_set2"
        )
        rm, rl = mk.run(use_heuristic=True), lg.run(use_heuristic=True)
        if rm.cleared and not rl.cleared:
            skill += 1
            if skill <= 8:
                print(
                    f"{date_str} MARK banked pnl={rm.pnl_pct:.2f} ent={rm.n_entries} | "
                    f"LEGACY pnl={rl.pnl_pct:.2f} ent={rl.n_entries}"
                )
    print("total mark-clear legacy-miss:", skill)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
