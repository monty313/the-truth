"""Held-out / forward 50d dual score for Spine Shadow finish gate.

Train/fit window: first 50 loadable days (practice, seed=42 T/R from baseline).
Held-out: next 50 loadable calendar days AFTER practice last date, same recipe
(seed=43 for pair sample so T/R not identical sequence, soft_bias=false,
pure_greedy mark_align). Day-set intersection with practice must be empty.

Runs score twice; both must report same_outcome==50 and breach==0 to pass.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import load_pairs, sample_pairs_for_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
HELDOUT_ORACLE = os.path.join(OUT, "MARK_ORACLE_CACHE__heldout50.json")
MAX_ES = 16


def _load_hcache() -> Dict[str, dict]:
    if not os.path.isfile(HELDOUT_ORACLE):
        return {}
    raw = json.load(open(HELDOUT_ORACLE, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {int(a): int(b) for a, b in vv["plan"].items()}
        out[k] = vv
    return out


def _save_hcache(cache: Dict[str, dict]) -> None:
    serial = {}
    for k, v in cache.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {str(a): int(b) for a, b in vv["plan"].items()}
        serial[k] = {kk: vv[kk] for kk in vv if kk != "day"}
    with open(HELDOUT_ORACLE, "w", encoding="utf-8") as f:
        json.dump(serial, f)


def build_heldout_days(
    practice_dates: Sequence[str],
    *,
    n: int = 50,
) -> List:
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    prac = set(str(d) for d in practice_dates)
    last_prac = max(prac) if prac else ""
    held = []
    for date, m1 in all_days:
        ds = str(date)
        if ds in prac:
            continue
        if last_prac and ds <= last_prac:
            continue
        held.append((ds, m1))
        if len(held) >= n:
            break
    return held


def score_heldout(
    policy: Channel1Policy,
    held_days: List,
    pairs_raw: List[dict],
    *,
    seed: int,
    mark_cache: Dict[str, dict],
    max_entry_samples: int = MAX_ES,
    max_days: Optional[int] = None,
) -> Dict[str, Any]:
    window = held_days[: max_days or len(held_days)]
    tr = sample_pairs_for_days(len(window), pairs_raw, seed=seed, soft_bias=False)
    rows = []
    for i, ((date, m1), (t, r)) in enumerate(zip(window, tr)):
        ckey = f"{date}|{t}|{r}"
        if ckey not in mark_cache:
            print(f"    heldout mark {i+1}/{len(window)} {date}…", flush=True)
            mark = execute_mark_soul_day(
                m1, str(date), float(t), float(r), max_entry_samples=max_entry_samples
            )
            mark_cache[ckey] = {k: v for k, v in mark.items() if k != "day"}
            if mark_cache[ckey].get("plan"):
                mark_cache[ckey]["plan"] = {
                    int(a): int(b) for a, b in mark_cache[ckey]["plan"].items()
                }
            _save_hcache(mark_cache)
        else:
            if (i + 1) % 10 == 1:
                print(f"    heldout score {i+1}/{len(window)} {date} (cached mark)", flush=True)
        mark = mark_cache[ckey]
        mark_award = bool(mark.get("cleared") and not mark.get("breached"))
        day = GoalEquityDay(
            m1,
            target_pct=float(t),
            risk_pct=float(r),
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=True,
        )
        res = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        pol_award = bool(res.cleared and not res.breached)
        if mark_award and not pol_award and not res.breached:
            mclass = "MARK_WOULD_TAKE"
        elif pol_award:
            mclass = "AWARD"
        elif res.breached:
            mclass = "POLICY_BREACH"
        else:
            mclass = "NO_OPPORTUNITY"
        rows.append(
            {
                "date": str(date),
                "target_pct": float(t),
                "risk_pct": float(r),
                "mark_award": mark_award,
                "policy_award": pol_award,
                "policy_breached": bool(res.breached),
                "policy_pnl": round(float(res.pnl_pct), 4),
                "policy_n_entries": int(res.n_entries),
                "same_outcome": bool(mark_award == pol_award),
                "miss_class": mclass,
            }
        )
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["miss_class"]] = counts.get(r["miss_class"], 0) + 1
    return {
        "n_days": len(rows),
        "seed": seed,
        "soft_bias": False,
        "decode": "policy_full_obs_mark_align_pure_greedy",
        "day_set": [r["date"] for r in rows],
        "mark_clear": sum(1 for r in rows if r["mark_award"]),
        "policy_clear": sum(1 for r in rows if r["policy_award"]),
        "same_outcome": sum(1 for r in rows if r["same_outcome"]),
        "n_breach": sum(1 for r in rows if r["policy_breached"]),
        "miss_class_counts": counts,
        "mark_would_take": counts.get("MARK_WOULD_TAKE", 0),
        "rows": rows,
        "scored_at": datetime.now(timezone.utc).isoformat(),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=50)
    ap.add_argument("--seed1", type=int, default=43)
    ap.add_argument("--seed2", type=int, default=43)  # same recipe twice
    ap.add_argument("--ckpt", type=str, default=CKPT)
    ap.add_argument("--out1", type=str, default="")
    ap.add_argument("--out2", type=str, default="")
    ap.add_argument("--max-entry-samples", type=int, default=MAX_ES)
    args = ap.parse_args(list(argv) if argv is not None else None)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    practice_dates = [str(r["date"]) for r in baseline["rows"]]
    held = build_heldout_days(practice_dates, n=args.n_days)
    inter = set(practice_dates) & set(d for d, _ in held)
    print(
        f"practice n={len(practice_dates)} heldout n={len(held)} "
        f"intersection={len(inter)} first={held[0][0] if held else None} "
        f"last={held[-1][0] if held else None}",
        flush=True,
    )
    if inter:
        print(f"LEAKAGE: {sorted(inter)[:5]}", flush=True)
        return 3
    if len(held) < args.n_days:
        print(f"WARN: only {len(held)} heldout days available", flush=True)

    pairs_raw = load_pairs()
    policy = load_policy(args.ckpt)
    cache = _load_hcache()

    print("=== HELDOUT RUN 1 ===", flush=True)
    s1 = score_heldout(
        policy,
        held,
        pairs_raw,
        seed=args.seed1,
        mark_cache=cache,
        max_entry_samples=args.max_entry_samples,
    )
    s1["practice_day_set"] = practice_dates
    s1["intersection_with_practice"] = []
    out1 = args.out1 or os.path.join(OUT, "HELDOUT_50D_RUN1__latest.json")
    with open(out1, "w", encoding="utf-8") as f:
        json.dump(s1, f, indent=2)
    print(
        f"RUN1 same={s1['same_outcome']}/{s1['n_days']} breach={s1['n_breach']} "
        f"mwt={s1['mark_would_take']}",
        flush=True,
    )

    print("=== HELDOUT RUN 2 (same frozen recipe) ===", flush=True)
    s2 = score_heldout(
        policy,
        held,
        pairs_raw,
        seed=args.seed2,
        mark_cache=cache,
        max_entry_samples=args.max_entry_samples,
    )
    s2["practice_day_set"] = practice_dates
    s2["intersection_with_practice"] = []
    out2 = args.out2 or os.path.join(OUT, "HELDOUT_50D_RUN2__latest.json")
    with open(out2, "w", encoding="utf-8") as f:
        json.dump(s2, f, indent=2)
    print(
        f"RUN2 same={s2['same_outcome']}/{s2['n_days']} breach={s2['n_breach']} "
        f"mwt={s2['mark_would_take']}",
        flush=True,
    )

    ok = (
        s1["same_outcome"] == s1["n_days"]
        and s2["same_outcome"] == s2["n_days"]
        and s1["n_breach"] == 0
        and s2["n_breach"] == 0
        and s1["n_days"] >= args.n_days
    )
    print(f"PASS={ok} wrote {out1} {out2}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
