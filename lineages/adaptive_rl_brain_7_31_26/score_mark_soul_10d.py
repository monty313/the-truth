"""Score Mark soul teacher vs fixed shell on the 10-day test window."""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.test_run_10d_mark_vs_policy import load_pairs


def score_plan(window, pairs, rng_seed: int):
    """Full-chart Mark soul plan teacher (what Mark does when he sees the day)."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    for date, m1 in window:
        t, r = pairs[int(rng.integers(0, len(pairs)))]
        out = execute_mark_soul_day(m1, str(date), t, r)
        rows.append(
            {
                "date": str(date),
                "target_pct": t,
                "risk_pct": r,
                "cleared": bool(out["cleared"]),
                "breached": bool(out["breached"]),
                "pnl_pct": out["pnl_pct"],
                "n_entries": out["n_entries"],
                "n_adds": out["n_adds"],
                "min_eq_pct": out["min_eq_pct"],
                "source": out["source"],
                "mode": out.get("mode"),
                "side": out.get("side"),
                "risk_use_frac": out.get("risk_use_frac"),
                "per_trade_cap_pct": out.get("per_trade_cap_pct"),
            }
        )
    n = len(rows)
    return {
        "kind": "soul_plan",
        "n": n,
        "cleared": sum(1 for x in rows if x["cleared"]),
        "breached": sum(1 for x in rows if x["breached"]),
        "rows": rows,
    }


def score_walk(window, pairs, rng_seed: int, *, mark_soul: bool):
    rng = np.random.default_rng(rng_seed)
    rows = []
    for date, m1 in window:
        t, r = pairs[int(rng.integers(0, len(pairs)))]
        day = GoalEquityDay(
            m1,
            target_pct=t,
            risk_pct=r,
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=mark_soul,
        )
        res = day.run(use_heuristic=True)
        rows.append(
            {
                "date": str(date),
                "target_pct": t,
                "risk_pct": r,
                "cleared": bool(res.cleared),
                "breached": bool(res.breached),
                "pnl_pct": round(float(res.pnl_pct), 4),
                "n_entries": int(res.n_entries),
                "n_adds": int(res.info.get("n_adds", 0)),
                "min_eq_pct": round(float(res.min_eq_pct), 4),
            }
        )
    n = len(rows)
    return {
        "mark_soul": mark_soul,
        "n": n,
        "cleared": sum(1 for x in rows if x["cleared"]),
        "breached": sum(1 for x in rows if x["breached"]),
        "rows": rows,
    }


def main() -> int:
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    pairs = load_pairs()
    start, n, seed = 40, 10, 7
    window = days[start : start + n]
    print(f"window start={start} n={n} seed={seed}", flush=True)
    plan = score_plan(window, pairs, seed)
    fixed = score_walk(window, pairs, seed, mark_soul=False)
    print(
        f"SOUL PLAN clear={plan['cleared']}/{plan['n']} breach={plan['breached']}",
        flush=True,
    )
    for r in plan["rows"]:
        print(
            f"  {r['date']} {r['target_pct']}/{r['risk_pct']} "
            f"clear={int(r['cleared'])} breach={int(r['breached'])} "
            f"pnl={r['pnl_pct']} entries={r['n_entries']} adds={r['n_adds']} "
            f"src={r['source']} mode={r.get('mode')} size={r.get('risk_use_frac')}/{r.get('per_trade_cap_pct')}",
            flush=True,
        )
    print(
        f"FIXED WALK clear={fixed['cleared']}/{fixed['n']} breach={fixed['breached']}",
        flush=True,
    )
    out = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "start_idx": start,
        "soul_plan": plan,
        "fixed_walk": fixed,
    }
    out_path = os.path.join(
        _HERE, "checkpoints", "test_run_10d_mark_vs_policy", "MARK_SOUL_SCORE__latest.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
