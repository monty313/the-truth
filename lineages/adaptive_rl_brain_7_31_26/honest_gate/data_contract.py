"""Chronological data contract: eligible days, practice/forward, no synthetic pad.

CHANGE LOG:
- 2026-07-31  honest gate — WHY: 100-day conclusion needs real days; claim has 90.
  Forward must never appear in dial search.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    load_calendar_days,
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.hashes import file_sha256
from lineages.adaptive_rl_brain_7_31_26.price_data import RAW_DIR, resolve_raw_csv
from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import load_pairs_config

_LINEAGE = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT_PATH = _LINEAGE / "checkpoints" / "honest_gate" / "data_contract.json"

# Spread/fee assumptions as used by GoalEquityDay (documented, not invented costs).
SPREAD_FEE_ASSUMPTIONS = {
    "spread_source": "per-bar spread column from MT5 export when present; else engine default path",
    "eq0": 100_000.0,
    "point_size": 0.01,
    "contract_scale": 100.0,
    "bank_after_flatten": True,
    "synthetic_days_allowed": False,
}

SESSION_CONVENTION = {
    "timezone": "as stored in MT5 M1 index (data_io.loader.read_mt5_m1)",
    "session": "full M1 calendar day slices via real_curriculum.split_calendar_days",
    "min_bars_eligible": 900,
    "symbol": "XAUUSD",
}


def assert_no_day_leak(
    search_dates: Sequence[str],
    forward_dates: Sequence[str],
) -> Dict[str, Any]:
    """Fail if any forward date appears in the search/practice set."""
    s = set(str(d) for d in search_dates)
    f = set(str(d) for d in forward_dates)
    overlap = sorted(s & f)
    if overlap:
        raise AssertionError(
            "LEAK: forward dates in search set: " + ", ".join(overlap[:20])
            + (f" ... ({len(overlap)} total)" if len(overlap) > 20 else "")
        )
    return {
        "ok": True,
        "search_n": len(s),
        "forward_n": len(f),
        "overlap_n": 0,
        "overlap": [],
    }


def build_data_contract(
    *,
    data_source: str | None = None,
    practice_n: int | None = None,
    min_bars: int = 900,
) -> Dict[str, Any]:
    cfg = load_pairs_config()
    src_name = (data_source or cfg.get("data_source", "XAUUSD_curriculum_2026.csv"))
    if "/" in str(src_name) or "\\" in str(src_name):
        src_name = Path(src_name).name
    practice_n = int(practice_n if practice_n is not None else cfg.get("practice_day_count", 50))

    csv_path = resolve_raw_csv(src_name)
    data_sha = file_sha256(csv_path)

    days = load_calendar_days(src_name, min_bars=min_bars)
    practice, forward = split_practice_forward(days, practice_n=practice_n)

    day_rows: List[Dict[str, Any]] = []
    for date_str, m1 in days:
        day_rows.append(
            {
                "date": str(date_str),
                "n_bars": int(len(m1)),
                "split": "practice" if any(str(d) == str(date_str) for d, _ in practice) else "forward",
            }
        )
    # Assign split via ordered index (more reliable than re-scan)
    practice_dates = [str(d) for d, _ in practice]
    forward_dates = [str(d) for d, _ in forward]
    pset = set(practice_dates)
    for row in day_rows:
        row["split"] = "practice" if row["date"] in pset else "forward"

    leak = assert_no_day_leak(practice_dates, forward_dates)

    n_eligible = len(days)
    hundred_day = {
        "eligible_days": n_eligible,
        "required_for_100_day_conclusion": 100,
        "status": "NOT_YET_MEASURABLE" if n_eligible < 100 else "MEASURABLE",
        "note": (
            f"{n_eligible} eligible real days < 100; do not call any result a 100-day pass. "
            "Do not pad with synthetic days."
            if n_eligible < 100
            else "≥100 eligible real days available."
        ),
    }

    contract: Dict[str, Any] = {
        "schema_version": 1,
        "track": "multi_pair_tutor",
        "symbol": SESSION_CONVENTION["symbol"],
        "data_source": str(src_name),
        "data_path": str(csv_path).replace("\\", "/"),
        "data_sha256": data_sha,
        "min_bars_eligible": min_bars,
        "session_convention": SESSION_CONVENTION,
        "spread_fee_assumptions": SPREAD_FEE_ASSUMPTIONS,
        "filter_rule": f"calendar day kept iff n_bars >= {min_bars}",
        "n_eligible_days": n_eligible,
        "practice_day_count": len(practice_dates),
        "forward_day_count": len(forward_dates),
        "practice_dates": practice_dates,
        "forward_dates": forward_dates,
        "practice_first": practice_dates[0] if practice_dates else None,
        "practice_last": practice_dates[-1] if practice_dates else None,
        "forward_first": forward_dates[0] if forward_dates else None,
        "forward_last": forward_dates[-1] if forward_dates else None,
        "leak_check": leak,
        "hundred_day_conclusion": hundred_day,
        "days": day_rows,
        "prior_claim_label": {
            "ten_pair_score_all": "IN_SAMPLE_CLAIM",
            "reason": (
                "Historical dial search could use ALL days (train_multi_pair --search-dials). "
                "Do not present ten_pair_score_all / ten_pair_score_forward as pure unseen "
                "if dials were fit with forward visible. Preserve as in-sample claim only."
            ),
        },
    }
    return contract


def write_data_contract(path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else DEFAULT_CONTRACT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    contract = build_data_contract()
    path.write_text(
        json.dumps(contract, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return contract
