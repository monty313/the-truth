"""Score rules + conclusion artifact schema (defined before training results).

CHANGE LOG:
- 2026-07-31  honest gate — WHY: clear/breach/streak only; windows reported
  separately; machine KEEP/REJECT/NOT_YET_MEASURABLE.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from lineages.adaptive_rl_brain_7_31_26.honest_gate.hashes import sha256_obj

# Official meters (diagnostics never pass/fail alone)
OFFICIAL_METERS = ("clear_count", "breach_count", "clear_streak_max", "clear_streak_end")
DIAGNOSTIC_ONLY = ("pnl", "mean_pnl", "n_entries", "looks_good")

SCORE_SCHEMA_VERSION = 1

BAR_EXPORT_SCHEMA: List[Dict[str, str]] = [
    {"name": "date", "type": "str"},
    {"name": "t", "type": "int"},
    {"name": "target_pct", "type": "float"},
    {"name": "risk_pct", "type": "float"},
    {"name": "equity_pct", "type": "float"},
    {"name": "min_eq_pct", "type": "float"},
    {"name": "danger", "type": "float"},
    {"name": "progress_to_goal", "type": "float"},
    {"name": "heat_ok", "type": "bool"},
    {"name": "action", "type": "int"},
    {"name": "side", "type": "str"},
    {"name": "higher_dir", "type": "str"},
    {"name": "lower_dir", "type": "str"},
    {"name": "pullback", "type": "bool"},
    {"name": "scale_conflict", "type": "bool"},
    {"name": "tag", "type": "str"},
    {"name": "banked", "type": "bool"},
    {"name": "breached", "type": "bool"},
    {"name": "split", "type": "str"},  # practice | forward
]


def clear_definition() -> str:
    return (
        "Clear day: final equity% (after flatten/bank costs path) >= target% "
        "AND floor (−risk%) was never touched that day."
    )


def breach_definition() -> str:
    return (
        "Breach day: equity% (incl. intrabar worst on open risk) touched −risk% "
        "at any point; day cannot be clear even if later recovers."
    )


def streaks_from_day_rows(day_rows: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    """Max and ending clear streak from ordered day_rows with cleared bool."""
    max_s = 0
    cur = 0
    for r in day_rows:
        if r.get("cleared"):
            cur += 1
            max_s = max(max_s, cur)
        else:
            cur = 0
    end_s = 0
    for r in reversed(list(day_rows)):
        if r.get("cleared"):
            end_s += 1
        else:
            break
    return {"max_clear_streak": max_s, "end_clear_streak": end_s}


def near_floor_rate(
    day_rows: Sequence[Mapping[str, Any]],
    risk_pct: float,
    *,
    frac: float = 0.85,
) -> float:
    if not day_rows:
        return 0.0
    thr = -abs(float(risk_pct)) * float(frac)
    n = sum(1 for r in day_rows if float(r.get("min_eq_pct", 0.0)) <= thr)
    return 100.0 * n / len(day_rows)


def enrich_pair_report(pair_rep: Dict[str, Any]) -> Dict[str, Any]:
    """Add streak + near_floor_rate to a score_pair_on_days-style report."""
    out = dict(pair_rep)
    rows = list(pair_rep.get("day_rows") or [])
    st = streaks_from_day_rows(rows)
    out.update(st)
    out["near_floor_rate_pct"] = near_floor_rate(
        rows, float(pair_rep.get("risk_pct", 0.0))
    )
    n = int(pair_rep.get("n_days") or len(rows) or 0)
    cleared = int(pair_rep.get("cleared") or 0)
    breached = int(pair_rep.get("breached") or 0)
    out["clear_count"] = cleared
    out["breach_count"] = breached
    out["clear_rate_pct"] = 100.0 * cleared / max(n, 1)
    out["breach_rate_pct"] = 100.0 * breached / max(n, 1)
    return out


def window_pass_rules() -> Dict[str, Any]:
    return {
        "official_meters": list(OFFICIAL_METERS),
        "diagnostic_only": list(DIAGNOSTIC_ONLY),
        "clear": clear_definition(),
        "breach": breach_definition(),
        "windows_reported_separately": ["practice", "forward", "claim_all"],
        "never_compare_raw_clears_across_window_lengths_as_equal_rates": True,
        "claim_legacy_bar": {
            "label": "IN_SAMPLE_CLAIM",
            "min_clear_days_per_pair": 30,
            "require_breach_count": 0,
            "n_pairs": 10,
            "note": "Absolute ≥30 clears depends on window length; report rates too.",
        },
        "forward_bar": {
            "any_breach": "REJECT",
            "breach_0_but_weak_clear": "diagnose regime/sensor on practice; do not fit forward",
        },
        "hundred_day": {
            "if_eligible_days_lt_100": "NOT_YET_MEASURABLE",
            "no_synthetic_pad": True,
        },
        "required_report_fields": [
            "clear_count",
            "clear_rate_pct",
            "breach_count",
            "max_clear_streak",
            "end_clear_streak",
            "near_floor_rate_pct",
            "checkpoint_sha256",
            "dials_sha256",
            "meaning_hash",
            "data_sha256",
            "practice_dates",
            "forward_dates",
            "decode",
            "seed",
            "window",
        ],
    }


def build_conclusion_artifact(
    *,
    verdict: str,
    reason: str,
    experiment_id: str,
    pins: Mapping[str, Any],
    windows: Optional[Mapping[str, Any]] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """verdict in KEEP | REJECT | NOT_YET_MEASURABLE | CONTAMINATED | GATE_PASS | GATE_FAIL."""
    allowed = {
        "KEEP",
        "REJECT",
        "NOT_YET_MEASURABLE",
        "CONTAMINATED",
        "GATE_PASS",
        "GATE_FAIL",
    }
    if verdict not in allowed:
        raise ValueError(f"verdict must be one of {sorted(allowed)}")
    art: Dict[str, Any] = {
        "schema_version": SCORE_SCHEMA_VERSION,
        "verdict": verdict,
        "reason": reason,
        "experiment_id": experiment_id,
        "score_rules": window_pass_rules(),
        "bar_export_schema": BAR_EXPORT_SCHEMA,
        "pins": dict(pins),
        "windows": dict(windows or {}),
        "extra": dict(extra or {}),
    }
    art["artifact_sha256"] = sha256_obj(
        {k: v for k, v in art.items() if k != "artifact_sha256"}
    )
    return art


def write_conclusion(path: Path | str, artifact: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path
