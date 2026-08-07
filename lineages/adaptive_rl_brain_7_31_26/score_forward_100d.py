"""Forward consistency gate: ≥100 calendar days NEVER used to fit embryo weights.

Data reality:
  - Curriculum CSV has only 90 loadable days (50 practice + 40 post-practice).
  - XAUUSD_M1_full.csv spans 2020-09 → 2026-05 → enough true holdout.

Protocol (honest, auditable):
  FIT_SET  = practice 50 from curriculum (2026-01-20..2026-03-30) — documented
  HOLD_SET = 100 calendar days from M1_full with empty intersection vs FIT_SET
             Prefer: all post-practice forward days first, then back-fill from
             pre-practice history (2025…) so n≥100. Dual chronological holdouts.
  Score    = Mark soul plan award vs policy pure_greedy mark_align, random T/R
             seed fixed, soft_bias=false. Run TWICE; both must match for dual-ok.
  Pass bar = same_outcome == n_days (or rate) and breach==0 twice.
             Default report also prints clear rate if Mark oracle incomplete.

Never trains. Never touches PROVEN. Writes leakage audit.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd

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
from lineages.adaptive_rl_brain_7_31_26.price_data import load_raw_m1
from lineages.adaptive_rl_brain_7_31_26.real_curriculum import split_calendar_days

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
ORACLE_100 = os.path.join(OUT, "MARK_ORACLE_CACHE__forward100.json")
MAX_ES = 14


def fit_dates_from_baseline() -> List[str]:
    b = json.load(open(BASELINE, encoding="utf-8"))
    return [str(r["date"]) for r in b["rows"]]


def load_full_calendar(min_bars: int = 900) -> List[Tuple[str, pd.DataFrame]]:
    m1 = load_raw_m1("XAUUSD_M1_full.csv")
    days = split_calendar_days(m1)
    return [(str(d), g) for d, g in days if len(g) >= min_bars]


def build_holdout_100(
    fit_dates: Sequence[str],
    *,
    n: int = 100,
    min_bars: int = 900,
) -> Tuple[List[Tuple[str, Any]], Dict[str, Any]]:
    fit = set(str(d) for d in fit_dates)
    last_fit = max(fit) if fit else ""
    all_days = load_full_calendar(min_bars=min_bars)
    # 1) post-fit forward first (true future)
    future = [(d, m) for d, m in all_days if d not in fit and d > last_fit]
    # 2) pre-fit past holdout (true never-fit history)
    past = [(d, m) for d, m in all_days if d not in fit and d < min(fit)]
    past = list(reversed(past))  # prefer most recent pre-fit first
    # take all future then fill from past
    held: List[Tuple[str, Any]] = []
    for item in future:
        held.append(item)
        if len(held) >= n:
            break
    n_future = len(held)
    if len(held) < n:
        for item in past:
            if item[0] in {h[0] for h in held}:
                continue
            held.append(item)
            if len(held) >= n:
                break
    # chronological order for scoring stability
    held_sorted = sorted(held, key=lambda x: x[0])
    meta = {
        "n_requested": n,
        "n_selected": len(held_sorted),
        "n_from_future": n_future,
        "n_from_past": len(held_sorted) - n_future,
        "first": held_sorted[0][0] if held_sorted else None,
        "last": held_sorted[-1][0] if held_sorted else None,
        "fit_first": min(fit) if fit else None,
        "fit_last": last_fit,
        "intersection": sorted(set(d for d, _ in held_sorted) & fit),
        "total_full_calendar": len(all_days),
    }
    return held_sorted[:n], meta


def _load_cache() -> Dict[str, dict]:
    if not os.path.isfile(ORACLE_100):
        return {}
    raw = json.load(open(ORACLE_100, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {int(a): int(b) for a, b in vv["plan"].items()}
        out[k] = vv
    return out


def _save_cache(cache: Dict[str, dict]) -> None:
    serial = {}
    for k, v in cache.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {str(a): int(b) for a, b in vv["plan"].items()}
        serial[k] = {kk: vv[kk] for kk in vv if kk != "day"}
    os.makedirs(OUT, exist_ok=True)
    with open(ORACLE_100, "w", encoding="utf-8") as f:
        json.dump(serial, f)


def score_window(
    policy: Channel1Policy,
    days: List[Tuple[str, Any]],
    pairs_raw: List[dict],
    *,
    seed: int,
    cache: Dict[str, dict],
    max_entry_samples: int = MAX_ES,
    max_days: Optional[int] = None,
    tag: str = "run",
) -> Dict[str, Any]:
    window = days[: max_days or len(days)]
    tr = sample_pairs_for_days(len(window), pairs_raw, seed=seed, soft_bias=False)
    rows = []
    for i, ((date, m1), (t, r)) in enumerate(zip(window, tr)):
        ckey = f"{date}|{t}|{r}"
        if ckey not in cache:
            print(f"    [{tag}] mark {i+1}/{len(window)} {date} T/R={t}/{r}…", flush=True)
            mark = execute_mark_soul_day(
                m1, str(date), float(t), float(r), max_entry_samples=max_entry_samples
            )
            blob = {k: v for k, v in mark.items() if k != "day"}
            if blob.get("plan"):
                blob["plan"] = {int(a): int(b) for a, b in blob["plan"].items()}
            cache[ckey] = blob
            if (i + 1) % 5 == 0:
                _save_cache(cache)
        else:
            if (i + 1) % 20 == 1:
                print(f"    [{tag}] score {i+1}/{len(window)} {date} (cached)", flush=True)
        mark = cache[ckey]
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
            mclass = "OTHER"
        rows.append(
            {
                "date": str(date),
                "target_pct": float(t),
                "risk_pct": float(r),
                "mark_award": mark_award,
                "mark_source": mark.get("source"),
                "policy_award": pol_award,
                "policy_breached": bool(res.breached),
                "policy_pnl": round(float(res.pnl_pct), 4),
                "policy_n_entries": int(res.n_entries),
                "same_outcome": bool(mark_award == pol_award),
                "miss_class": mclass,
            }
        )
    _save_cache(cache)
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["miss_class"]] = counts.get(r["miss_class"], 0) + 1
    n = len(rows)
    same = sum(1 for r in rows if r["same_outcome"])
    return {
        "n_days": n,
        "seed": seed,
        "soft_bias": False,
        "decode": "policy_full_obs_mark_align_pure_greedy",
        "day_set": [r["date"] for r in rows],
        "mark_clear": sum(1 for r in rows if r["mark_award"]),
        "policy_clear": sum(1 for r in rows if r["policy_award"]),
        "same_outcome": same,
        "same_rate": round(same / max(n, 1), 4),
        "n_breach": sum(1 for r in rows if r["policy_breached"]),
        "miss_class_counts": counts,
        "mark_would_take": counts.get("MARK_WOULD_TAKE", 0),
        "rows": rows,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=100)
    ap.add_argument("--seed1", type=int, default=43)
    ap.add_argument("--seed2", type=int, default=43)
    ap.add_argument("--ckpt", default=CKPT)
    ap.add_argument("--max-entry-samples", type=int, default=MAX_ES)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--partial", type=int, default=0, help="if >0 score only first N (smoke)")
    args = ap.parse_args(list(argv) if argv is not None else None)

    out_dir = args.out_dir or OUT
    os.makedirs(out_dir, exist_ok=True)

    fit = fit_dates_from_baseline()
    held, meta = build_holdout_100(fit, n=args.n_days)
    print(json.dumps(meta, indent=2), flush=True)
    if meta["intersection"]:
        print("LEAKAGE — abort", flush=True)
        return 3
    if len(held) < args.n_days:
        print(f"WARN: only {len(held)} holdout days (wanted {args.n_days})", flush=True)

    pairs_raw = load_pairs()
    policy = load_policy(args.ckpt)
    cache = _load_cache()
    n_score = args.partial if args.partial > 0 else len(held)

    print(f"=== FORWARD100 RUN1 seed={args.seed1} n={n_score} ===", flush=True)
    s1 = score_window(
        policy,
        held,
        pairs_raw,
        seed=args.seed1,
        cache=cache,
        max_entry_samples=args.max_entry_samples,
        max_days=n_score,
        tag="run1",
    )
    s1["holdout_meta"] = meta
    s1["fit_day_set"] = fit
    s1["intersection_with_fit"] = sorted(set(s1["day_set"]) & set(fit))
    p1 = os.path.join(out_dir, "FORWARD100_RUN1__latest.json")
    with open(p1, "w", encoding="utf-8") as f:
        json.dump(s1, f, indent=2)
    print(
        f"RUN1 same={s1['same_outcome']}/{s1['n_days']} rate={s1['same_rate']} "
        f"breach={s1['n_breach']} mwt={s1['mark_would_take']} pol_clear={s1['policy_clear']}",
        flush=True,
    )

    print(f"=== FORWARD100 RUN2 seed={args.seed2} ===", flush=True)
    s2 = score_window(
        policy,
        held,
        pairs_raw,
        seed=args.seed2,
        cache=cache,
        max_entry_samples=args.max_entry_samples,
        max_days=n_score,
        tag="run2",
    )
    s2["holdout_meta"] = meta
    s2["fit_day_set"] = fit
    s2["intersection_with_fit"] = sorted(set(s2["day_set"]) & set(fit))
    p2 = os.path.join(out_dir, "FORWARD100_RUN2__latest.json")
    with open(p2, "w", encoding="utf-8") as f:
        json.dump(s2, f, indent=2)
    print(
        f"RUN2 same={s2['same_outcome']}/{s2['n_days']} rate={s2['same_rate']} "
        f"breach={s2['n_breach']} mwt={s2['mark_would_take']}",
        flush=True,
    )

    ok = (
        s1["n_days"] >= min(args.n_days, n_score)
        and s1["same_outcome"] == s1["n_days"]
        and s2["same_outcome"] == s2["n_days"]
        and s1["n_breach"] == 0
        and s2["n_breach"] == 0
        and not s1["intersection_with_fit"]
        and not s2["intersection_with_fit"]
    )
    summary = {
        "pass": ok,
        "run1_same": s1["same_outcome"],
        "run2_same": s2["same_outcome"],
        "n": s1["n_days"],
        "breach1": s1["n_breach"],
        "breach2": s2["n_breach"],
        "paths": [p1, p2],
        "meta": meta,
    }
    with open(os.path.join(out_dir, "FORWARD100_SUMMARY__latest.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"PASS={ok}", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
