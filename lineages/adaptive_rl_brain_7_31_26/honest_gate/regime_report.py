"""Practice vs forward regime/sensor-trust report (tags). No shell changes.

CHANGE LOG:
- 2026-07-31  honest gate — WHY: detect when senses degrade on forward without
  fitting to forward or calling small samples “lying.”
"""
from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


DEFAULT_MIN_SAMPLES = 8


def _wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for a proportion (uncertainty band)."""
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z * z / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    lo = (centre - margin) / denom
    hi = (centre + margin) / denom
    return (max(0.0, lo), min(1.0, hi))


def aggregate_by_tag(
    day_tag_rows: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Dict[str, Dict[str, Any]]:
    """day_tag_rows: {tag, cleared, breached} per day (or per trade window).

    Small samples marked insufficient_evidence — never “lying.”
    """
    buckets: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for r in day_tag_rows:
        tag = str(r.get("tag") or "UNTAGGED")
        buckets[tag].append(r)

    out: Dict[str, Dict[str, Any]] = {}
    for tag, rows in sorted(buckets.items()):
        n = len(rows)
        n_clear = sum(1 for r in rows if r.get("cleared"))
        n_breach = sum(1 for r in rows if r.get("breached"))
        p_clear = n_clear / n if n else float("nan")
        p_breach = n_breach / n if n else float("nan")
        lo, hi = _wilson_interval(n_clear, n)
        insufficient = n < int(min_samples)
        out[tag] = {
            "tag": tag,
            "n": n,
            "n_clear": n_clear,
            "n_breach": n_breach,
            "p_clear": p_clear,
            "p_breach": p_breach,
            "wilson_p_clear_low": lo,
            "wilson_p_clear_high": hi,
            "insufficient_evidence": insufficient,
            "min_samples": int(min_samples),
            "may_call_sensor_lying": False,  # never on small n
        }
        if insufficient:
            out[tag]["status"] = "INSUFFICIENT_EVIDENCE"
        else:
            out[tag]["status"] = "OK"
    return out


def compare_practice_forward(
    practice_rows: Sequence[Mapping[str, Any]],
    forward_rows: Sequence[Mapping[str, Any]],
    *,
    min_samples: int = DEFAULT_MIN_SAMPLES,
) -> Dict[str, Any]:
    """Compare P_clear / breach by tag. May flag degradation; never authorizes shell change."""
    p_agg = aggregate_by_tag(practice_rows, min_samples=min_samples)
    f_agg = aggregate_by_tag(forward_rows, min_samples=min_samples)
    tags = sorted(set(p_agg) | set(f_agg))
    comparisons: List[Dict[str, Any]] = []
    for tag in tags:
        pr = p_agg.get(tag)
        fr = f_agg.get(tag)
        if pr is None or fr is None:
            comparisons.append(
                {
                    "tag": tag,
                    "status": "MISSING_SIDE",
                    "note": "tag only on one window",
                    "shell_change_authorized": False,
                }
            )
            continue
        if pr["insufficient_evidence"] or fr["insufficient_evidence"]:
            comparisons.append(
                {
                    "tag": tag,
                    "status": "INSUFFICIENT_EVIDENCE",
                    "practice": pr,
                    "forward": fr,
                    "delta_p_clear": None,
                    "shell_change_authorized": False,
                    "note": "Do not call sensor lying; sample below threshold.",
                }
            )
            continue
        delta = float(fr["p_clear"]) - float(pr["p_clear"])
        degraded = delta < -0.15 and fr["n"] >= min_samples
        comparisons.append(
            {
                "tag": tag,
                "status": "DEGRADED" if degraded else "STABLE_OR_UP",
                "practice": pr,
                "forward": fr,
                "delta_p_clear": delta,
                "shell_change_authorized": False,
                "note": (
                    "Sensor may have degraded on forward — attention/practice only; shell locked."
                    if degraded
                    else "No strong degradation flag."
                ),
            }
        )
    return {
        "schema_version": 1,
        "min_samples": min_samples,
        "practice_n_rows": len(practice_rows),
        "forward_n_rows": len(forward_rows),
        "comparisons": comparisons,
        "shell_change_authorized": False,
        "rule": "Report may identify degraded sensors; it must not unlock shell laws.",
    }


def write_regime_report(path: Path | str, report: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path
