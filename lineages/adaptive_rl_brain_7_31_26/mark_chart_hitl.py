"""Mark chart HITL pack — Fable exports; Mark reviews via MARK HERE!.lnk.

Loop:
  1. Run Mark soul plans vs pure policy on N days
  2. Collect disagree decision bars + day outcomes
  3. Write markdown Mark can open while looking at the chart
  4. Mark answers on MarkOS; corrections feed next BC

Usage:
  python lineages/adaptive_rl_brain_7_31_26/mark_chart_hitl.py
  python lineages/adaptive_rl_brain_7_31_26/mark_chart_hitl.py --seed 7 --start-idx 40 --full-obs
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

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.test_run_10d_mark_vs_policy import load_pairs

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT_DIR = os.path.join(CKPT_DIR, "mark_chart_hitl")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}


def _load_policy(full_obs: bool) -> Optional[Channel1Policy]:
    prefer = (
        "mark_clone_full_obs_v1.pt" if full_obs else "mark_clone_doctrine_v1.pt"
    )
    for name in (prefer, "mark_clone_latest.pt", "mark_clone_doctrine_v1.pt"):
        path = os.path.join(CKPT_DIR, name)
        if not os.path.isfile(path):
            continue
        blob = torch.load(path, map_location="cpu", weights_only=False)
        dim = int(blob.get("obs_dim", CHANNEL1_DIM))
        want = MARK_FULL_DIM if full_obs else CHANNEL1_DIM
        if full_obs and dim != MARK_FULL_DIM:
            # try next ckpt
            if name != prefer:
                continue
        hidden = int(blob.get("hidden", 128 if full_obs else 64))
        pol = Channel1Policy(obs_dim=dim, hidden=hidden)
        try:
            pol.load_state_dict(blob["state_dict"])
        except Exception:
            continue
        pol.eval()
        return pol
    return None


def walk_policy_labels(
    m1,
    date: str,
    target: float,
    risk: float,
    policy: Channel1Policy,
    *,
    full_obs: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=str(date),
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=full_obs,
    )
    rows = []
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
        with torch.no_grad():
            act, _ = policy.act(obs, greedy=True)
        act = int(act)
        ts = str(m1.index[t])
        if act != teacher:
            rows.append(
                {
                    "time": ts,
                    "bar": int(t),
                    "mark_would": NAMES.get(teacher, str(teacher)),
                    "policy_did": NAMES.get(act, str(act)),
                    "equity_pct": round(float(day.equity_pct(float(day._close[t]))), 4),
                }
            )
        day.step_action(t, act)
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    if pnl <= -day.risk + 1e-12:
        day.breached = True
    cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
        day.banked and not day.breached
    )
    summary = {
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "pnl_pct": round(float(pnl), 4),
        "n_entries": int(day.n_entries),
        "n_disagree": len(rows),
    }
    return rows, summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Export Mark chart HITL pack")
    ap.add_argument("--n-days", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--start-idx", type=int, default=40)
    ap.add_argument("--full-obs", action="store_true")
    ap.add_argument("--max-disagree-per-day", type=int, default=8)
    args = ap.parse_args(argv)

    os.makedirs(OUT_DIR, exist_ok=True)
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    pairs = load_pairs()
    rng = np.random.default_rng(args.seed)
    start = max(0, int(args.start_idx))
    window = all_days[start : start + int(args.n_days)]
    policy = _load_policy(bool(args.full_obs))

    lines: List[str] = []
    pack: Dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "start_idx": start,
        "full_obs": bool(args.full_obs),
        "days": [],
    }

    lines.append("# Mark chart HITL — review pack")
    lines.append("")
    lines.append("**Fable built this. Mark owns the chart.**")
    lines.append("")
    lines.append("## How to use")
    lines.append("1. Double-click repo root **`MARK HERE!.lnk`** (MarkOS).")
    lines.append("2. Open XAUUSD on that calendar day (TradingView or your chart).")
    lines.append("3. For each **disagree** bar below: what would **you** do — HOLD / BUY / SELL?")
    lines.append("4. Reply in MarkOS; Fable will BC corrections into the clone.")
    lines.append("")
    lines.append(f"- seed={args.seed} start={start} n={len(window)} full_obs={args.full_obs}")
    lines.append(f"- policy loaded: **{policy is not None}**")
    lines.append("")

    for i, (date, m1) in enumerate(window, 1):
        t, r = pairs[int(rng.integers(0, len(pairs)))]
        mark = execute_mark_soul_day(m1, str(date), t, r)
        lines.append(f"## Day {i}: {date} · target {t}% · risk {r}%")
        lines.append("")
        lines.append(
            f"- **Mark soul plan:** clear={mark['cleared']} pnl={mark['pnl_pct']} "
            f"entries={mark['n_entries']} mode={mark.get('mode')} src={mark['source']}"
        )
        day_row: Dict[str, Any] = {
            "date": str(date),
            "target_pct": t,
            "risk_pct": r,
            "mark": {
                "cleared": mark["cleared"],
                "pnl_pct": mark["pnl_pct"],
                "n_entries": mark["n_entries"],
                "mode": mark.get("mode"),
            },
        }
        if policy is None:
            lines.append("- policy: _not loaded — train full-obs BC first_")
            lines.append("")
            pack["days"].append(day_row)
            continue
        disagrees, pol_sum = walk_policy_labels(
            m1, str(date), t, r, policy, full_obs=bool(args.full_obs)
        )
        day_row["policy"] = pol_sum
        day_row["disagrees"] = disagrees[: int(args.max_disagree_per_day)]
        lines.append(
            f"- **Policy:** clear={pol_sum['cleared']} pnl={pol_sum['pnl_pct']} "
            f"entries={pol_sum['n_entries']} disagrees={pol_sum['n_disagree']}"
        )
        lines.append("")
        if not disagrees:
            lines.append("_No bar disagreements (or day ended early). Still confirm EOD with chart._")
        else:
            lines.append("### Disagreements (Mark: correct me)")
            for d in disagrees[: int(args.max_disagree_per_day)]:
                lines.append(
                    f"- **{d['time']}** bar={d['bar']} · Mark would **{d['mark_would']}** · "
                    f"policy **{d['policy_did']}** · eq≈{d['equity_pct']}%"
                )
                lines.append(f"  - Your call: `HOLD` / `BUY` / `SELL` / `OK_MARK` ________")
        lines.append("")
        pack["days"].append(day_row)

    lines.append("## Mark sign-off")
    lines.append("- [ ] I reviewed the charts for miss days")
    lines.append("- [ ] Corrections noted for BC")
    lines.append("- [ ] Clone may continue / stop thrash days: ________")
    lines.append("")
    lines.append("_Fable method: evidence → act → verify. You are the soul._")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    md_path = os.path.join(OUT_DIR, f"HITL__{stamp}.md")
    latest = os.path.join(OUT_DIR, "HITL__latest.md")
    json_path = os.path.join(OUT_DIR, "HITL__latest.json")
    body = "\n".join(lines)
    for p in (md_path, latest):
        with open(p, "w", encoding="utf-8") as f:
            f.write(body)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(pack, f, indent=2)
    print(f"wrote {latest}", flush=True)
    print(f"json  {json_path}", flush=True)
    print("Open MARK HERE!.lnk and review HITL__latest.md with the chart.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
