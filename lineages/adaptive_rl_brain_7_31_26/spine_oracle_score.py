"""S1 — Oracle-execute gold Day Spines under shell (no neural net).

If oracle same_outcome < Mark plan bar → compiler/shell bug, do not train.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import (
    SPINE_DIR,
    SPINE_INDEX,
    DaySpine,
    compile_spine_from_soul,
    spine_to_plan,
    write_spine,
    write_spine_index,
)
from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import (
    execute_mark_soul_day,
    run_plan,
)

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
ORACLE_CACHE = os.path.join(OUT, "MARK_ORACLE_CACHE__50d.json")
ORACLE_SCORE = os.path.join(OUT, "spine_oracle_score__latest.json")
MAX_ES = 20


def _load_oracle_cache() -> Dict[str, dict]:
    if not os.path.isfile(ORACLE_CACHE):
        return {}
    raw = json.load(open(ORACLE_CACHE, encoding="utf-8"))
    out: Dict[str, dict] = {}
    for k, v in raw.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {int(a): int(b) for a, b in vv["plan"].items()}
        out[k] = vv
    return out


def _save_oracle_cache(cache: Dict[str, dict]) -> None:
    serial = {}
    for k, v in cache.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {str(a): int(b) for a, b in vv["plan"].items()}
        serial[k] = {kk: vv[kk] for kk in vv if kk != "day"}
    with open(ORACLE_CACHE, "w", encoding="utf-8") as f:
        json.dump(serial, f)


def ensure_mark(
    oracle: Dict[str, dict],
    day_map: Dict[str, Any],
    date: str,
    t: float,
    r: float,
    *,
    max_entry_samples: int = MAX_ES,
) -> dict:
    key = f"{date}|{t}|{r}"
    if key in oracle and (
        oracle[key].get("plan") is not None
        or oracle[key].get("source") == "soul_online_fallback"
    ):
        return oracle[key]
    print(f"    oracle search {date} T/R={t}/{r}…", flush=True)
    m = execute_mark_soul_day(
        day_map[date], date, float(t), float(r), max_entry_samples=max_entry_samples
    )
    blob = {k: v for k, v in m.items() if k != "day"}
    if blob.get("plan"):
        blob["plan"] = {int(a): int(b) for a, b in blob["plan"].items()}
    oracle[key] = blob
    _save_oracle_cache(oracle)
    return blob


def execute_gold_spine(
    m1,
    spine: DaySpine,
) -> Dict[str, Any]:
    """Execute gold spine via run_plan (shell + size lock). Force-gate is in shell laws."""
    # Online fallback has no sparse soul plan — re-run soul walk (not empty HOLD plan).
    if spine.mark_source == "soul_online_fallback" or (
        spine.plan is None and spine.t1 is None
    ):
        m = execute_mark_soul_day(
            m1, spine.day, spine.target_pct, spine.risk_pct, max_entry_samples=MAX_ES
        )
        return {
            "cleared": bool(m["cleared"]),
            "breached": bool(m["breached"]),
            "pnl_pct": m["pnl_pct"],
            "n_entries": m["n_entries"],
            "source": "online_fallback_reexec",
        }
    plan = spine_to_plan(spine)
    dir_bars = [t for t, a in plan.items() if int(a) != 0]
    if not plan or not dir_bars:
        return {
            "cleared": False,
            "breached": False,
            "pnl_pct": 0.0,
            "n_entries": 0,
            "source": "empty_plan",
            "error": "no_plan",
        }
    res = run_plan(
        m1,
        spine.day,
        float(spine.target_pct),
        float(spine.risk_pct),
        risk_use_frac=float(spine.risk_use_frac),
        per_trade_cap_pct=float(spine.per_trade_cap_pct),
        plan=plan,
    )
    res.pop("day", None)
    res["source"] = "gold_spine_run_plan"
    return res


def compile_practice_spines(
    mark_rows: Sequence[dict],
    day_map: Dict[str, Any],
    oracle: Dict[str, dict],
    *,
    max_entry_samples: int = MAX_ES,
) -> Tuple[List[DaySpine], List[str]]:
    spines: List[DaySpine] = []
    paths: List[str] = []
    for i, mr in enumerate(mark_rows):
        date = str(mr["date"])
        t, r = float(mr["target_pct"]), float(mr["risk_pct"])
        if (i + 1) % 10 == 1 or i == 0:
            print(f"  compile {i+1}/{len(mark_rows)} {date}", flush=True)
        mark = ensure_mark(oracle, day_map, date, t, r, max_entry_samples=max_entry_samples)
        # decision indices from plan keys or day runner
        indices = None
        if mark.get("plan"):
            indices = sorted(int(k) for k in mark["plan"].keys())
        else:
            from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay

            day0 = GoalEquityDay(
                day_map[date],
                target_pct=t,
                risk_pct=r,
                date_str=date,
                eyes_mode="mark_doctrine",
                mark_soul=False,
            )
            indices = list(day0.runner.decision_indices())
        spine = compile_spine_from_soul(date, t, r, mark, decision_indices=indices)
        # Prefer baseline mark_award as ground truth for same_outcome compare
        if mr.get("mark_award") is not None:
            spine.cleared = bool(mr["mark_award"])
            spine.breached = False
        path = write_spine(spine)
        spines.append(spine)
        paths.append(path)
    write_spine_index(
        spines,
        paths,
        meta={
            "window": "practice_50d_seed42",
            "n": len(spines),
            "recipe": "BASELINE_50D rows T/R",
        },
    )
    return spines, paths


def score_oracle_spines(
    spines: Sequence[DaySpine],
    day_map: Dict[str, Any],
    mark_rows: Sequence[dict],
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    for i, (spine, mr) in enumerate(zip(spines, mark_rows)):
        date = spine.day
        if (i + 1) % 10 == 1 or i == 0:
            print(f"  oracle-exec {i+1}/{len(spines)} {date}", flush=True)
        m1 = day_map[date]
        res = execute_gold_spine(m1, spine)
        mark_award = bool(mr.get("mark_award", spine.cleared))
        oracle_award = bool(res.get("cleared") and not res.get("breached"))
        rows.append(
            {
                "date": date,
                "target_pct": spine.target_pct,
                "risk_pct": spine.risk_pct,
                "mark_award": mark_award,
                "oracle_award": oracle_award,
                "oracle_cleared": bool(res.get("cleared")),
                "oracle_breached": bool(res.get("breached")),
                "oracle_pnl": res.get("pnl_pct"),
                "oracle_n_entries": res.get("n_entries"),
                "same_outcome": bool(mark_award == oracle_award),
                "source": res.get("source"),
                "n_spine_events": len(spine.events),
                "mark_source": spine.mark_source,
                "side": spine.side,
                "t1": spine.t1,
            }
        )
    same = sum(1 for r in rows if r["same_outcome"])
    n_breach = sum(1 for r in rows if r["oracle_breached"])
    mark_clear = sum(1 for r in rows if r["mark_award"])
    oracle_clear = sum(1 for r in rows if r["oracle_award"])
    return {
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "n_days": len(rows),
        "same_outcome": same,
        "mark_clear": mark_clear,
        "oracle_clear": oracle_clear,
        "n_breach": n_breach,
        "pass_gate": same >= 48 and n_breach == 0,
        "spine_index": SPINE_INDEX,
        "rows": rows,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Compile spines + oracle score practice 50d")
    ap.add_argument("--max-days", type=int, default=50)
    ap.add_argument("--max-entry-samples", type=int, default=MAX_ES)
    ap.add_argument("--skip-compile", action="store_true")
    args = ap.parse_args(list(argv) if argv is not None else None)

    os.makedirs(OUT, exist_ok=True)
    os.makedirs(SPINE_DIR, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"][: args.max_days]
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[: args.max_days]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = _load_oracle_cache()
    print(f"oracle cache={len(oracle)} rows={len(mark_rows)}", flush=True)

    if args.skip_compile and os.path.isfile(SPINE_INDEX):
        from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import load_spine, load_spine_index

        idx = load_spine_index()
        spines = [load_spine(it["path"]) for it in idx["items"][: args.max_days]]
        print(f"loaded {len(spines)} spines from index", flush=True)
    else:
        print("Compile spines…", flush=True)
        spines, _ = compile_practice_spines(
            mark_rows, day_map, oracle, max_entry_samples=args.max_entry_samples
        )
        print(f"wrote {len(spines)} spines → {SPINE_DIR}", flush=True)

    print("Oracle-execute gold spines…", flush=True)
    score = score_oracle_spines(spines, day_map, mark_rows)
    with open(ORACLE_SCORE, "w", encoding="utf-8") as f:
        json.dump(score, f, indent=2)
    print(
        f"ORACLE same={score['same_outcome']}/{score['n_days']} "
        f"oracle_clear={score['oracle_clear']} mark_clear={score['mark_clear']} "
        f"breach={score['n_breach']} pass={score['pass_gate']}",
        flush=True,
    )
    print(f"wrote {ORACLE_SCORE}", flush=True)
    return 0 if score["pass_gate"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
