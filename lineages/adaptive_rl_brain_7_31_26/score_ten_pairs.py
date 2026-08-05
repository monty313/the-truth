"""Score the multi-pair consistent brain on real XAUUSD days.

CHANGE LOG:
- 2026-07-31  multi-pair scorer — WHY: GOAL.md consistency = same brain,
  10 (target, risk) pairs, clear days + 0 breach. Lineage only; no PROVEN.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py
  python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all
  python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode forward
  python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode practice
  python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --pair 3.0 3.5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
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
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

PAIRS_PATH = os.path.join(_HERE, "ten_pairs.json")
CKPT_DEFAULT = os.path.join(_HERE, "checkpoints", "multi_pair_consistent_v1.pt")
REPORT_DIR = os.path.join(_HERE, "checkpoints")


def load_pairs_config(path: str = PAIRS_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_policy(ckpt_path: Optional[str]) -> Tuple[Optional[Channel1Policy], dict]:
    meta: dict = {"mode": "heuristic"}
    if not ckpt_path or not os.path.isfile(ckpt_path):
        return None, meta
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hidden = int(blob.get("hidden", 48))
    policy = Channel1Policy(hidden=hidden)
    if "state_dict" in blob:
        policy.load_state_dict(blob["state_dict"])
    elif "model" in blob:
        policy.load_state_dict(blob["model"])
    policy.eval()
    meta = {
        "mode": "checkpoint",
        "path": ckpt_path,
        "hidden": hidden,
        "saved_at": blob.get("saved_at"),
        "tag": blob.get("tag", ""),
        "dials": blob.get("dials", {}),
    }
    return policy, meta


def score_pair_on_days(
    days: Sequence[Tuple[str, Any]],
    target: float,
    risk: float,
    *,
    policy: Optional[Channel1Policy] = None,
    use_heuristic: bool = False,
    decide_every: int = 25,
    eq0: float = 100_000.0,
    risk_use_frac: float = 0.35,
    stop_atr_mult: float = 2.0,
    per_trade_cap_pct: float = 0.25,
) -> Dict[str, Any]:
    results = []
    for date_str, m1 in days:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            eq0=eq0,
            decide_every=decide_every,
            risk_use_frac=risk_use_frac,
            stop_atr_mult=stop_atr_mult,
            per_trade_cap_pct=per_trade_cap_pct,
            use_signal_majority=False,
            date_str=date_str,
        )
        if use_heuristic or policy is None:
            r = day.run(use_heuristic=True)
        else:
            r = day.run(greedy_policy=policy, use_heuristic=False)
        results.append(r)

    n = len(results)
    cleared = sum(1 for r in results if r.cleared)
    breached = sum(1 for r in results if r.breached)
    pnls = [r.pnl_pct for r in results]
    return {
        "target_pct": float(target),
        "risk_pct": float(risk),
        "n_days": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "breach_pct": 100.0 * breached / max(n, 1),
        "mean_pnl": float(np.mean(pnls)) if pnls else 0.0,
        "median_pnl": float(np.median(pnls)) if pnls else 0.0,
        "pass_30_clear": cleared >= 30,
        "pass_0_breach": breached == 0,
        "pass": cleared >= 30 and breached == 0,
        "day_rows": [
            {
                "date": r.date,
                "pnl_pct": round(r.pnl_pct, 4),
                "min_eq_pct": round(r.min_eq_pct, 4),
                "cleared": r.cleared,
                "breached": r.breached,
                "n_entries": r.n_entries,
                "banked": r.banked,
            }
            for r in results
        ],
    }


def score_all_pairs(
    pairs: Sequence[dict],
    days: Sequence[Tuple[str, Any]],
    *,
    policy: Optional[Channel1Policy] = None,
    use_heuristic: bool = False,
    decide_every: int = 25,
    eq0: float = 100_000.0,
    dials: Optional[dict] = None,
) -> Dict[str, Any]:
    dials = dials or {}
    risk_use_frac = float(dials.get("risk_use_frac", 0.35))
    stop_atr_mult = float(dials.get("stop_atr_mult", 2.0))
    per_trade_cap_pct = float(dials.get("per_trade_cap_pct", 0.25))
    pair_results = []
    for p in pairs:
        pr = score_pair_on_days(
            days,
            float(p["target_pct"]),
            float(p["risk_pct"]),
            policy=policy,
            use_heuristic=use_heuristic,
            decide_every=decide_every,
            eq0=eq0,
            risk_use_frac=risk_use_frac,
            stop_atr_mult=stop_atr_mult,
            per_trade_cap_pct=per_trade_cap_pct,
        )
        pr["id"] = p.get("id")
        pair_results.append(pr)
        print(
            "pair %2s  target %4.1f / risk %4.1f  |  clear %3d/%3d (%5.1f%%)  breach %3d (%5.1f%%)  %s"
            % (
                pr.get("id", "?"),
                pr["target_pct"],
                pr["risk_pct"],
                pr["cleared"],
                pr["n_days"],
                pr["clear_pct"],
                pr["breached"],
                pr["breach_pct"],
                "PASS" if pr["pass"] else "FAIL",
            ),
            flush=True,
        )
    n_pass = sum(1 for p in pair_results if p["pass"])
    return {
        "n_pairs": len(pair_results),
        "n_pass": n_pass,
        "all_pass": n_pass == len(pair_results) and len(pair_results) == 10,
        "pairs": pair_results,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score 10 (target,risk) pairs on real days")
    ap.add_argument("--mode", choices=["all", "practice", "forward"], default="all")
    ap.add_argument("--pairs", default=PAIRS_PATH)
    ap.add_argument("--ckpt", default=CKPT_DEFAULT)
    ap.add_argument("--heuristic", action="store_true", help="force perception heuristic")
    ap.add_argument("--pair", nargs=2, type=float, metavar=("TARGET", "RISK"))
    ap.add_argument("--decide-every", type=int, default=25)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)

    cfg = load_pairs_config(args.pairs)
    all_days = load_calendar_days(cfg.get("data_source", "XAUUSD_curriculum_2026.csv").split("/")[-1])
    practice_n = int(cfg.get("practice_day_count", 50))
    practice, forward = split_practice_forward(all_days, practice_n=practice_n)
    if args.mode == "practice":
        days = practice
        label = "practice"
    elif args.mode == "forward":
        days = forward
        label = "forward"
    else:
        days = all_days
        label = "all"

    print("=" * 72)
    print("MULTI-PAIR SCORE  |  mode=%s  days=%d  (practice=%d forward=%d)" % (
        label, len(days), len(practice), len(forward)))
    print("data: %s" % cfg.get("data_source"))
    print("seed: %s" % cfg.get("seed"))
    print("-" * 72)

    policy = None
    meta = {"mode": "heuristic"}
    use_heuristic = bool(args.heuristic)
    # Always try to load dials from checkpoint (even for heuristic decode)
    if os.path.isfile(args.ckpt):
        policy_try, meta = load_policy(args.ckpt)
        decode = str((meta.get("dials") or {}).get("decode", "policy"))
        if use_heuristic or decode == "heuristic":
            use_heuristic = True
            policy = None
            print(
                "brain: perception heuristic + dials from %s  tag=%s"
                % (args.ckpt, meta.get("tag"))
            )
        else:
            policy = policy_try
            print("brain: %s  tag=%s" % (meta.get("path"), meta.get("tag")))
    elif use_heuristic:
        print("brain: perception heuristic (no .pt)")
    else:
        print("no checkpoint at %s → heuristic decode" % args.ckpt)
        use_heuristic = True

    dials = meta.get("dials") or {}
    eq0 = float(cfg.get("eq0", 100000.0))

    if args.pair:
        pairs = [{"id": 0, "target_pct": args.pair[0], "risk_pct": args.pair[1]}]
    else:
        pairs = list(cfg["pairs"])

    report = score_all_pairs(
        pairs,
        days,
        policy=policy,
        use_heuristic=use_heuristic,
        decide_every=args.decide_every,
        eq0=eq0,
        dials=dials,
    )
    report["mode"] = label
    report["n_days_eval"] = len(days)
    report["day_first"] = days[0][0] if days else None
    report["day_last"] = days[-1][0] if days else None
    report["brain"] = meta
    report["scored_at"] = datetime.now(timezone.utc).isoformat()
    report["config"] = {
        "pairs_path": args.pairs,
        "seed": cfg.get("seed"),
        "data_source": cfg.get("data_source"),
        "practice_day_count": practice_n,
    }

    print("-" * 72)
    print(
        "SUMMARY  pairs_pass=%d/%d  all_pass=%s"
        % (report["n_pass"], report["n_pairs"], report["all_pass"])
    )
    print("=" * 72)

    out = args.out
    if not out:
        out = os.path.join(REPORT_DIR, "ten_pair_score_%s.json" % label)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    # strip heavy day_rows for compact default? keep them for repro
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print("wrote %s" % out)
    return 0 if report.get("all_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
