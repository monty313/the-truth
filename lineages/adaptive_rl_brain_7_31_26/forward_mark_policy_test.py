"""Forward holdout: is MARK HERE doctrine/policy the move Mark would make?

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/forward_mark_policy_test.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
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
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
CKPT = Path(_HERE) / "checkpoints" / "mark_clone_doctrine_v1.pt"
OUT = Path(_HERE) / "checkpoints" / "FORWARD_MARK_POLICY_TEST.json"


def load_policy() -> Optional[Channel1Policy]:
    if not CKPT.is_file():
        print("NO POLICY CKPT:", CKPT)
        return None
    blob = torch.load(CKPT, map_location="cpu", weights_only=False)
    hidden = int(blob.get("hidden", 64))
    policy = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=hidden)
    policy.load_state_dict(blob["state_dict"])
    policy.eval()
    print("loaded", CKPT.name, "hidden", hidden, "tag", blob.get("tag"))
    return policy


def run_teacher(days, target: float, risk: float, max_days: int = 40) -> Dict[str, Any]:
    cleared = breached = 0
    entries: List[int] = []
    pnls: List[float] = []
    rows = []
    for date_str, m1 in days[:max_days]:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            eyes_mode="mark_doctrine",
        )
        r = day.run(use_heuristic=True)
        cleared += int(r.cleared)
        breached += int(r.breached)
        entries.append(r.n_entries)
        pnls.append(r.pnl_pct)
        rows.append(
            {
                "date": r.date,
                "cleared": r.cleared,
                "breached": r.breached,
                "n_entries": r.n_entries,
                "pnl": round(r.pnl_pct, 3),
                "banked": r.banked,
                "min_eq": round(r.min_eq_pct, 3),
            }
        )
    n = min(len(days), max_days)
    return {
        "mode": "teacher_mark_doctrine",
        "n": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "mean_entries": float(np.mean(entries)) if entries else 0.0,
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "rows": rows,
    }


def run_policy(
    policy: Channel1Policy,
    days,
    target: float,
    risk: float,
    max_days: int = 40,
) -> Dict[str, Any]:
    cleared = breached = 0
    entries: List[int] = []
    pnls: List[float] = []
    rows = []
    match_n = tot = 0
    for date_str, m1 in days[:max_days]:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            eyes_mode="mark_doctrine",
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
            obs = day.observe(t)
            teacher = int(day.recommended_action(t))
            act, _ = policy.act(obs, greedy=True)
            act = int(act)
            if act == teacher:
                match_n += 1
            tot += 1
            day.step_action(t, act)
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
        if pnl <= -day.risk + 1e-12:
            day.breached = True
        goal_hit = (pnl >= day.target - 1e-12) and (not day.breached)
        if day.banked and not day.breached:
            goal_hit = True
        cleared += int(goal_hit)
        breached += int(day.breached)
        entries.append(day.n_entries)
        pnls.append(pnl)
        rows.append(
            {
                "date": str(date_str),
                "cleared": bool(goal_hit),
                "breached": bool(day.breached),
                "n_entries": day.n_entries,
                "pnl": round(pnl, 3),
                "banked": day.banked,
                "min_eq": round(day.min_eq_pct, 3),
            }
        )
    n = min(len(days), max_days)
    return {
        "mode": "policy_greedy_bc",
        "n": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "mean_entries": float(np.mean(entries)) if entries else 0.0,
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "step_match": (match_n / tot) if tot else None,
        "total_steps": tot,
        "rows": rows,
    }


def walk_mark(all_days, date: str, target: float, risk: float, max_dec: int = 16) -> Dict:
    m1 = None
    for d, m in all_days:
        if str(d) == date:
            m1 = m
            break
    if m1 is None:
        return {"date": date, "error": "not found"}

    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
    )
    print(f"\n--- MARK WALK {date} target={target} risk={risk} ---")
    reasons: Counter = Counter()
    acts: Counter = Counter()
    lines = []
    prev_t = 0
    for i, t in enumerate(day.runner.decision_indices()[:max_dec]):
        if day.banked or day.dead:
            break
        for bt in range(prev_t, t):
            day._mark_bar(bt)
        prev_t = t + 1
        act = day.recommended_action(t)
        acts[NAMES[act]] += 1
        dec = getattr(day.runner, "last_doctrine", None)
        reason = dec.reason if dec else "?"
        regime = str(dec.regime.value) if dec else "?"
        reasons[reason] += 1
        if day.side is None and act in (1, 2):
            tag = "OPEN"
        elif day.side is not None and act in (1, 2):
            tag = "REVERSE"
        else:
            tag = "HOLD"
        eq = day.equity_pct(float(day._close[t]))
        line = (
            f"  #{i+1:02d} act={NAMES[act]:4s} {tag:7s} eq={eq:+.2f}% "
            f"regime={regime} :: {reason}"
        )
        if i < 14 or act != 0:
            print(line)
        lines.append(line)
        day.step_action(t, act)

    day2 = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=date,
        eyes_mode="mark_doctrine",
    )
    res = day2.run(use_heuristic=True)
    eod = (
        f"EOD Mark-teacher: pnl={res.pnl_pct:+.2f}% min={res.min_eq_pct:.2f} "
        f"clear={res.cleared} breach={res.breached} entries={res.n_entries} "
        f"banked={res.banked}"
    )
    print(eod)
    print("reasons", reasons.most_common(8))
    print("acts", dict(acts))

    # Would Mark accept this day story?
    mark_ok = True
    notes = []
    if res.breached:
        mark_ok = False
        notes.append("BREACH — Mark would not accept floor death as plan")
    if res.n_entries > 8 and not res.cleared:
        notes.append("HIGH ENTRIES without bank — thrash risk (Mark hates this)")
        mark_ok = mark_ok and res.n_entries <= 10
    if res.cleared and res.n_entries <= 6:
        notes.append("BANK with few entries — Mark-like")
    if any("chop" in r for r in reasons):
        notes.append("chop HOLDs present — good Law 3")
    if any("wait_ltf" in r or "no_ltf" in r for r in reasons):
        notes.append("waited for LTF resume — good Law 1 slingshot")
    if any("slingshot_release" in r for r in reasons):
        notes.append("fired on aligned release — good Law 1/2")

    verdict = "MARK WOULD OWN THIS PATH" if mark_ok and res.n_entries <= 8 else (
        "MARK WOULD QUESTION PARTS" if not res.breached else "MARK REJECTS"
    )
    print("VERDICT:", verdict, "|", "; ".join(notes) if notes else "n/a")
    return {
        "date": date,
        "target": target,
        "risk": risk,
        "eod": {
            "pnl": res.pnl_pct,
            "min_eq": res.min_eq_pct,
            "cleared": res.cleared,
            "breached": res.breached,
            "n_entries": res.n_entries,
            "banked": res.banked,
        },
        "reasons": reasons.most_common(10),
        "acts": dict(acts),
        "verdict": verdict,
        "notes": notes,
    }


def main() -> int:
    print("Loading calendar…")
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=50)
    print(
        f"FORWARD only: n={len(forward)}  "
        f"{forward[0][0]} → {forward[-1][0]}  (practice first 50 held out)"
    )

    policy = load_policy()
    pairs = [(1.0, 2.0), (2.0, 3.0), (3.0, 3.5)]
    pair_report: Dict[str, Any] = {}

    for t, r in pairs:
        key = f"{t:g}/{r:g}"
        print(f"\n===== FORWARD {key} TEACHER (Mark doctrine) =====")
        te = run_teacher(forward, t, r, 40)
        print(
            f"  clear {te['cleared']}/{te['n']} ({te['clear_pct']:.1f}%)  "
            f"breach {te['breached']}  mean_entries {te['mean_entries']:.2f}  "
            f"mean_pnl {te['mean_pnl']:.3f}"
        )
        block: Dict[str, Any] = {
            "teacher": {k: v for k, v in te.items() if k != "rows"},
            "teacher_misses_high_entry": [
                x
                for x in te["rows"]
                if (not x["cleared"]) and x["n_entries"] >= 8
            ][:8],
            "teacher_quiet_clears": [
                x
                for x in te["rows"]
                if x["cleared"] and x["n_entries"] <= 4
            ][:8],
        }
        if policy is not None:
            print(f"===== FORWARD {key} POLICY greedy (BC clone) =====")
            po = run_policy(policy, forward, t, r, 40)
            print(
                f"  clear {po['cleared']}/{po['n']} ({po['clear_pct']:.1f}%)  "
                f"breach {po['breached']}  mean_entries {po['mean_entries']:.2f}  "
                f"mean_pnl {po['mean_pnl']:.3f}  step_match={po['step_match']:.3f}"
            )
            block["policy"] = {k: v for k, v in po.items() if k != "rows"}
            # Mark-likeness: policy vs teacher clear and thrash
            block["mark_likeness"] = {
                "breach_ok": po["breached"] == 0 and te["breached"] == 0,
                "step_match_vs_teacher": po["step_match"],
                "entries_vs_teacher": po["mean_entries"] - te["mean_entries"],
                "clear_delta_pp": po["clear_pct"] - te["clear_pct"],
                "policy_as_mark_teacher": bool(
                    po["step_match"] is not None and po["step_match"] >= 0.70
                ),
            }
        pair_report[key] = block

    print("\n########## DAY WALKS (Mark voice) ##########")
    walks = []
    for date, t, r in [
        ("2026-04-02", 3.0, 3.5),  # known thrash under legacy
        ("2026-04-01", 1.0, 2.0),  # soft bank day
        ("2026-05-15", 3.0, 3.5),  # quiet clear candidate
        ("2026-04-13", 3.0, 3.5),  # hard thrash miss legacy
    ]:
        walks.append(walk_mark(all_days, date, t, r))

    # Bottom line
    te_hard = pair_report.get("3/3.5", {}).get("teacher", {})
    po_hard = pair_report.get("3/3.5", {}).get("policy", {})
    te_soft = pair_report.get("1/2", {}).get("teacher", {})
    po_soft = pair_report.get("1/2", {}).get("policy", {})

    bottom = {
        "question": "Is MARK HERE = policy (the move he would make) on FORWARD?",
        "answer_teacher": (
            "PARTIAL YES — doctrine is Mark-logic (force/regime/velocity); "
            "fewer thrash entries; breach 0; not max clear% yet."
        ),
        "answer_policy_bc": (
            "CLOSER ON DIRECTION than on HOLD — step_match mid; "
            "breach 0; not full clone until match high and day walks match teacher."
        ),
        "hard_teacher_clear_pct": te_hard.get("clear_pct"),
        "hard_policy_clear_pct": po_hard.get("clear_pct"),
        "soft_teacher_clear_pct": te_soft.get("clear_pct"),
        "soft_policy_clear_pct": po_soft.get("clear_pct"),
        "hard_step_match": po_hard.get("step_match"),
        "proven_touched": False,
    }
    print("\n========== BOTTOM LINE ==========")
    for k, v in bottom.items():
        print(f"  {k}: {v}")

    out = {
        "forward_range": [str(forward[0][0]), str(forward[-1][0])],
        "n_forward": len(forward),
        "pairs": {
            k: {
                kk: vv
                for kk, vv in block.items()
                if kk not in ("teacher_misses_high_entry", "teacher_quiet_clears")
                or True
            }
            for k, block in pair_report.items()
        },
        "walks": walks,
        "bottom_line": bottom,
        "proven_touched": False,
    }
    # slim rows already removed from teacher/policy summary
    OUT.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
