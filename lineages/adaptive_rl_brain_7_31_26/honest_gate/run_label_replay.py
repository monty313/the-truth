"""Run Label Contract V1 5+5 practice/forward replay (logger only).

Usage (repo root):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/honest_gate/run_label_replay.py

Writes:
  checkpoints/honest_gate/replay_practice_5d_v1.json
  checkpoints/honest_gate/replay_forward_5d_v1.json

Does NOT train, search dials, change shell, or touch PROVEN.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import pandas as pd

_HERE = Path(__file__).resolve().parent
_LINEAGE = _HERE.parent
_ROOT = _LINEAGE.parent.parent
_CODE = _ROOT / "code"
for _p in (str(_ROOT), str(_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days, split_practice_forward
from lineages.adaptive_rl_brain_7_31_26.honest_gate.data_contract import assert_no_day_leak
from lineages.adaptive_rl_brain_7_31_26.honest_gate.hashes import sha256_obj
from lineages.adaptive_rl_brain_7_31_26.honest_gate.label_contract_v1 import (
    LABEL_CONTRACT_VERSION,
    assert_decision_schema,
    assert_eod_schema,
    decision_columns,
    eod_columns,
)
from lineages.adaptive_rl_brain_7_31_26.honest_gate.label_logger import log_equity_day, make_day
from lineages.adaptive_rl_brain_7_31_26.honest_gate.meaning_manifest import (
    build_meaning_manifest,
    meaning_hash,
)
from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import load_pairs_config

OUT_DIR = _LINEAGE / "checkpoints" / "honest_gate"
DIALS_PATH = _LINEAGE / "checkpoints" / "multi_pair_dials.json"

DEFAULT_TARGET = 1.0
DEFAULT_RISK = 2.0
DEFAULT_N = 5


def _load_dials() -> Dict[str, Any]:
    if DIALS_PATH.is_file():
        blob = json.loads(DIALS_PATH.read_text(encoding="utf-8"))
        return dict(blob.get("dials") or blob)
    return {
        "risk_use_frac": 0.35,
        "stop_atr_mult": 2.0,
        "per_trade_cap_pct": 0.25,
        "decode": "heuristic",
    }


def _days_by_date(
    days: Sequence[Tuple[str, pd.DataFrame]],
) -> Dict[str, pd.DataFrame]:
    return {str(d): g for d, g in days}


def replay_window(
    day_list: Sequence[Tuple[str, pd.DataFrame]],
    *,
    split: str,
    target: float,
    risk: float,
    dials: Dict[str, Any],
    m_hash: str,
    n_days: int = DEFAULT_N,
) -> Dict[str, Any]:
    selected = list(day_list)[: int(n_days)]
    all_decisions: List[Dict[str, Any]] = []
    all_eod: List[Dict[str, Any]] = []
    for date_str, m1 in selected:
        day = make_day(
            m1,
            date_str=str(date_str),
            target_pct=float(target),
            risk_pct=float(risk),
            dials=dials,
        )
        decisions, eod = log_equity_day(day, meaning_hash=m_hash, split=split)
        for r in decisions:
            assert_decision_schema(r)
            assert r.get("future_bar_used") is False
            assert r.get("meaning_hash") == m_hash
            assert r.get("label_contract_version") == LABEL_CONTRACT_VERSION
        assert_eod_schema(eod)
        assert eod.get("meaning_hash") == m_hash
        all_decisions.extend(decisions)
        all_eod.append(eod)

    payload = {
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "meaning_hash": m_hash,
        "split": split,
        "target_pct": float(target),
        "risk_pct": float(risk),
        "decode": "heuristic",
        "n_days": len(selected),
        "dates": [str(d) for d, _ in selected],
        "decision_columns": decision_columns(),
        "eod_columns": eod_columns(),
        "decision_rows": all_decisions,
        "eod_rows": all_eod,
        "n_decision_rows": len(all_decisions),
        "payload_hash": "",  # filled after body
    }
    body = {k: v for k, v in payload.items() if k != "payload_hash"}
    payload["payload_hash"] = sha256_obj(body)
    return payload


def validate_pair(
    practice: Dict[str, Any],
    forward: Dict[str, Any],
    *,
    practice_dates_allowed: Sequence[str],
    forward_dates_allowed: Sequence[str],
) -> Dict[str, Any]:
    """Asserts from 00_LABEL_CONTRACT_V1.md replay tests."""
    errors: List[str] = []

    # Schema / columns
    if practice["decision_columns"] != decision_columns():
        errors.append("practice decision_columns mismatch")
    if forward["decision_columns"] != decision_columns():
        errors.append("forward decision_columns mismatch")
    if set(practice["decision_columns"]) != set(forward["decision_columns"]):
        errors.append("practice vs forward decision column set mismatch")
    if set(practice["eod_columns"]) != set(forward["eod_columns"]):
        errors.append("practice vs forward eod column set mismatch")

    p_set = set(practice["dates"])
    f_set = set(forward["dates"])
    allow_p = set(practice_dates_allowed)
    allow_f = set(forward_dates_allowed)
    if not p_set.issubset(allow_p):
        errors.append(f"practice dates outside practice set: {sorted(p_set - allow_p)}")
    if not f_set.issubset(allow_f):
        errors.append(f"forward dates outside forward set: {sorted(f_set - allow_f)}")
    try:
        assert_no_day_leak(practice["dates"], forward["dates"])
    except AssertionError as e:
        errors.append(str(e))

    for split_name, blob in (("practice", practice), ("forward", forward)):
        mh = blob["meaning_hash"]
        for r in blob["decision_rows"]:
            try:
                assert_decision_schema(r)
            except AssertionError as e:
                errors.append(f"{split_name}: {e}")
            if r.get("future_bar_used") is not False:
                errors.append(f"{split_name}: future_bar_used not false at t={r.get('t')}")
            if r.get("meaning_hash") != mh:
                errors.append(f"{split_name}: meaning_hash row mismatch")
        for r in blob["eod_rows"]:
            try:
                assert_eod_schema(r)
            except AssertionError as e:
                errors.append(f"{split_name} eod: {e}")

    return {"ok": len(errors) == 0, "errors": errors}


def run(
    *,
    n_days: int = DEFAULT_N,
    target: float = DEFAULT_TARGET,
    risk: float = DEFAULT_RISK,
    write: bool = True,
) -> Dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_pairs_config()
    src = cfg.get("data_source", "XAUUSD_curriculum_2026.csv")
    # ten_pairs may store "data/raw/foo.csv"; loader wants basename only
    src = Path(str(src)).name
    practice_n = int(cfg.get("practice_day_count", 50))

    days = load_calendar_days(src, min_bars=900)
    practice_days, forward_days = split_practice_forward(days, practice_n=practice_n)

    dials = _load_dials()
    manifest = build_meaning_manifest()
    m_hash = meaning_hash(manifest)

    practice = replay_window(
        practice_days,
        split="practice",
        target=target,
        risk=risk,
        dials=dials,
        m_hash=m_hash,
        n_days=n_days,
    )
    forward = replay_window(
        forward_days,
        split="forward",
        target=target,
        risk=risk,
        dials=dials,
        m_hash=m_hash,
        n_days=n_days,
    )

    # Determinism: re-run practice first day payload hash path (full windows)
    practice2 = replay_window(
        practice_days,
        split="practice",
        target=target,
        risk=risk,
        dials=dials,
        m_hash=m_hash,
        n_days=n_days,
    )
    det_ok = practice["payload_hash"] == practice2["payload_hash"]

    verdict = validate_pair(
        practice,
        forward,
        practice_dates_allowed=[str(d) for d, _ in practice_days],
        forward_dates_allowed=[str(d) for d, _ in forward_days],
    )
    if not det_ok:
        verdict["ok"] = False
        verdict["errors"] = list(verdict.get("errors") or []) + [
            "determinism fail: practice payload_hash changed on re-run"
        ]

    practice_path = OUT_DIR / "replay_practice_5d_v1.json"
    forward_path = OUT_DIR / "replay_forward_5d_v1.json"
    if write:
        practice_path.write_text(
            json.dumps(practice, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        forward_path.write_text(
            json.dumps(forward, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )

    summary = {
        "verdict": "PASS" if verdict["ok"] else "FAIL",
        "label_contract_version": LABEL_CONTRACT_VERSION,
        "meaning_hash": m_hash,
        "target_pct": target,
        "risk_pct": risk,
        "practice_dates": practice["dates"],
        "forward_dates": forward["dates"],
        "practice_n_decision_rows": practice["n_decision_rows"],
        "forward_n_decision_rows": forward["n_decision_rows"],
        "deterministic": det_ok,
        "errors": verdict.get("errors") or [],
        "paths": {
            "practice": str(practice_path.relative_to(_ROOT)) if write else None,
            "forward": str(forward_path.relative_to(_ROOT)) if write else None,
        },
        "shell_changed": False,
        "heuristic_changed": False,
        "proven_touched": False,
    }
    summary_path = OUT_DIR / "replay_label_v1_summary.json"
    if write:
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Label Contract V1 5+5 replay")
    p.add_argument("--n-days", type=int, default=DEFAULT_N)
    p.add_argument("--target", type=float, default=DEFAULT_TARGET)
    p.add_argument("--risk", type=float, default=DEFAULT_RISK)
    p.add_argument("--no-write", action="store_true")
    args = p.parse_args(list(argv) if argv is not None else None)
    summary = run(
        n_days=int(args.n_days),
        target=float(args.target),
        risk=float(args.risk),
        write=not args.no_write,
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
