"""First-person day walk for the multi-pair tutor (heuristic + equity shell).

Runs the real GoalEquityDay path and prints how "I" think each decision bar.
Lineage only. No PROVEN.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/tutor_day_walk.py --date 2026-04-21 --target 2.0 --risk 3.0
  python lineages/adaptive_rl_brain_7_31_26/tutor_day_walk.py --date 2026-01-20 --target 1.0 --risk 2.0 --max-decisions 15
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)

NAMES = {ACTION_HOLD: "HOLD", ACTION_BUY: "BUY", ACTION_SELL: "SELL"}


def _side_name(side: Optional[int]) -> str:
    if side is None:
        return "FLAT"
    return "LONG" if side > 0 else "SHORT"


def walk_day(
    date: str,
    target: float,
    risk: float,
    *,
    max_decisions: int = 40,
    data: str = "XAUUSD_curriculum_2026.csv",
) -> int:
    days = load_calendar_days(data, min_bars=900)
    match = [(d, m1) for d, m1 in days if str(d) == str(date)]
    if not match:
        print(f"Date {date} not found in {data} (need ≥900 bars).", flush=True)
        print("Available sample:", [d for d, _ in days[:5]], "...", flush=True)
        return 2

    date_str, m1 = match[0]
    day = GoalEquityDay(
        m1,
        target_pct=float(target),
        risk_pct=float(risk),
        date_str=str(date_str),
    )

    print("=" * 64, flush=True)
    print("MULTI-PAIR TUTOR — day walk (first person)", flush=True)
    print(f"date={date_str}  target%={target}  risk%={risk}", flush=True)
    print(
        "dials: risk_use_frac=0.35 stop_atr_mult=2.0 per_trade_cap=0.25 decode=heuristic",
        flush=True,
    )
    print("=" * 64, flush=True)
    print(
        f"I wake with target {target}% and floor -{risk}%. "
        f"Equity starts at 0%. I will bank if I hit {target}%, die if I touch -{risk}%.",
        flush=True,
    )

    indices = day.runner.decision_indices()
    prev_t = 0
    n_shown = 0
    for t in indices:
        if day.dead or day.banked:
            print(
                f"\n[stop] I am {'BANKED' if day.banked else 'DEAD/BREACHED'} — no more decisions.",
                flush=True,
            )
            break
        # mark intervening bars
        for bt in range(prev_t, t):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = t + 1
        if day.dead or day.banked:
            print(
                f"\n[mark] Between decisions I was stopped out or hit floor/bank.",
                flush=True,
            )
            break

        price = float(day._close[t])
        eq = day.equity_pct(price)
        heat_dist = max(0.0, (eq - (-day.risk)) / 100.0)
        rec = int(day.recommended_action(t))
        pos_before = day.side
        action = rec  # heuristic decode
        day.step_action(t, action)
        n_shown += 1

        ts = str(day.m1.index[t])
        print(f"\n--- decision #{n_shown}  bar={t}  time={ts} ---", flush=True)
        print(
            f"I see equity%={eq:+.3f}  heat_to_floor≈{100*heat_dist:.2f}% pts  "
            f"position_before={_side_name(pos_before)}",
            flush=True,
        )
        print(f"Structure eye (same flat/in-trade path): {NAMES[rec]}", flush=True)
        if pos_before is None:
            if action == ACTION_HOLD:
                print(
                    "I stay FLAT — either structure is neutral/HOLD, or heat/size refused me.",
                    flush=True,
                )
            else:
                print(
                    f"I OPEN {NAMES[action]} — structure agreed and heat allowed size "
                    f"(risk_use_frac={day.risk_use_frac}).",
                    flush=True,
                )
        else:
            if action == ACTION_HOLD:
                print(
                    "I MANAGE HOLD — signal not opposite; I keep the trade unless stop/bank fires.",
                    flush=True,
                )
            else:
                print(
                    f"I REVERSE toward {NAMES[action]} — opposite structure while in trade.",
                    flush=True,
                )
        print(
            f"After step: position={_side_name(day.side)}  entries={day.n_entries}  "
            f"banked={day.banked}  breached={day.breached}  "
            f"equity%≈{day.equity_pct(float(day._close[t])):+.3f}",
            flush=True,
        )

        if n_shown >= int(max_decisions):
            print(f"\n(max-decisions={max_decisions} reached; stopping narration)", flush=True)
            break

    # finish remaining marks + flatten like run()
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    price = float(day._close[t_last])
    sp = float(day._spread_px[t_last])
    day._flatten(price, sp)
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    day.min_eq_pct = min(day.min_eq_pct, pnl)
    cleared = bool(pnl >= day.target - 1e-9 and not day.breached)
    print("\n" + "=" * 64, flush=True)
    print("EOD verdict (how I score the day)", flush=True)
    print(
        f"pnl%={pnl:+.3f}  min_eq%={day.min_eq_pct:+.3f}  "
        f"target={day.target}  floor=-{day.risk}",
        flush=True,
    )
    print(f"entries={day.n_entries}  banked={day.banked}  breached={day.breached}", flush=True)
    if day.breached:
        print("I BREACHED the floor. This day is not clear.", flush=True)
    elif cleared:
        print(
            "I CLEARED: hit target and never touched the floor. "
            + ("I banked early." if day.banked else "I finished above target."),
            flush=True,
        )
    else:
        print(
            "I did NOT clear: either missed target or (if breached) hit floor. "
            "Low clear is hesitation/path — lid is off.",
            flush=True,
        )
    print("=" * 64, flush=True)
    return 0


def main() -> None:
    p = argparse.ArgumentParser(description="Multi-pair tutor first-person day walk")
    p.add_argument("--date", required=True, help="YYYY-MM-DD calendar day in curriculum CSV")
    p.add_argument("--target", type=float, default=2.0)
    p.add_argument("--risk", type=float, default=3.0)
    p.add_argument("--max-decisions", type=int, default=40)
    p.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    args = p.parse_args()
    raise SystemExit(
        walk_day(
            args.date,
            args.target,
            args.risk,
            max_decisions=args.max_decisions,
            data=args.data,
        )
    )


if __name__ == "__main__":
    main()
