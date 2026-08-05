"""Tests for multi-pair tutor honest pre-training gate (shipped paths).

CHANGE LOG:
- 2026-07-31  gate tests — WHY: leak, shell lock, meaning hash, data split must
  drive real modules before any training cycle.
"""
from __future__ import annotations

import json
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_LINEAGE = _ROOT / "lineages" / "adaptive_rl_brain_7_31_26"
_GATE = _LINEAGE / "honest_gate"
_OUT = _LINEAGE / "checkpoints" / "honest_gate"


def test_assert_no_day_leak_detects_overlap():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.data_contract import (
        assert_no_day_leak,
    )

    assert_no_day_leak(["2026-01-20", "2026-01-21"], ["2026-03-31", "2026-04-01"])
    try:
        assert_no_day_leak(["2026-01-20", "2026-03-31"], ["2026-03-31"])
        raise AssertionError("expected leak")
    except AssertionError as e:
        assert "LEAK" in str(e)


def test_data_contract_practice_forward_no_overlap_and_lt_100_days():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.data_contract import (
        build_data_contract,
    )

    c = build_data_contract()
    p, f = set(c["practice_dates"]), set(c["forward_dates"])
    assert len(p & f) == 0
    assert len(c["practice_dates"]) == 50
    assert len(c["forward_dates"]) == 40
    assert c["n_eligible_days"] == 90
    assert c["hundred_day_conclusion"]["status"] == "NOT_YET_MEASURABLE"
    assert c["prior_claim_label"]["ten_pair_score_all"] == "IN_SAMPLE_CLAIM"


def test_meaning_manifest_stable_hash_and_gate():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.meaning_manifest import (
        assert_meaning_matches_frozen,
        build_meaning_manifest,
        meaning_hash,
        write_frozen_manifest,
    )

    m1 = build_meaning_manifest()
    m2 = build_meaning_manifest()
    assert meaning_hash(m1) == meaning_hash(m2)
    assert m1["meaning_hash"] == m2["meaning_hash"]
    path = _OUT / "meaning_manifest_test.json"
    write_frozen_manifest(path)
    ok = assert_meaning_matches_frozen(path)
    assert ok["ok"] is True
    # Tamper freeze → mismatch
    blob = json.loads(path.read_text(encoding="utf-8"))
    blob["meaning_hash"] = "0" * 64
    path.write_text(json.dumps(blob), encoding="utf-8")
    try:
        assert_meaning_matches_frozen(path)
        raise AssertionError("expected mismatch")
    except ValueError as e:
        assert "MISMATCH" in str(e)


def test_shell_locked_on_equity_day():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.shell_lock import (
        assert_shell_locked,
        verify_shell_locked,
    )

    r = verify_shell_locked()
    assert r["ok"] is True, r["errors"]
    assert_shell_locked()
    assert r["checks"].get("every_bar_marks") is True
    assert r["checks"].get("floor_scaled_sizing") is True


def test_search_dials_rejects_forward_leak():
    from lineages.adaptive_rl_brain_7_31_26.train_multi_pair import search_dials

    # Minimal fake days list (empty m1 not scored — leak fires before grid)
    fake_days = [("2026-03-31", None), ("2026-04-01", None)]
    try:
        search_dials(
            fake_days,
            [{"target_pct": 1.0, "risk_pct": 2.0, "id": 1}],
            forbidden_dates=["2026-03-31"],
            search_window="practice",
        )
        raise AssertionError("expected leak")
    except AssertionError as e:
        assert "LEAK" in str(e)


def test_score_schema_streaks_and_conclusion():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.score_schema import (
        build_conclusion_artifact,
        enrich_pair_report,
        streaks_from_day_rows,
    )

    rows = [
        {"cleared": True, "breached": False, "min_eq_pct": -0.5},
        {"cleared": True, "breached": False, "min_eq_pct": -0.2},
        {"cleared": False, "breached": False, "min_eq_pct": -1.0},
        {"cleared": True, "breached": False, "min_eq_pct": -0.1},
    ]
    st = streaks_from_day_rows(rows)
    assert st["max_clear_streak"] == 2
    assert st["end_clear_streak"] == 1
    rep = enrich_pair_report(
        {
            "target_pct": 1.0,
            "risk_pct": 2.0,
            "n_days": 4,
            "cleared": 3,
            "breached": 0,
            "day_rows": rows,
        }
    )
    assert rep["clear_count"] == 3
    assert rep["max_clear_streak"] == 2
    art = build_conclusion_artifact(
        verdict="NOT_YET_MEASURABLE",
        reason="n_days < 100",
        experiment_id="test",
        pins={"seed": 42},
    )
    assert art["verdict"] == "NOT_YET_MEASURABLE"
    assert "artifact_sha256" in art


def test_regime_insufficient_evidence_not_lying():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.regime_report import (
        compare_practice_forward,
    )

    practice = [{"tag": "rare", "cleared": True, "breached": False}] * 2
    forward = [{"tag": "rare", "cleared": False, "breached": False}] * 2
    rep = compare_practice_forward(practice, forward, min_samples=8)
    rare = [c for c in rep["comparisons"] if c["tag"] == "rare"][0]
    assert rare["status"] == "INSUFFICIENT_EVIDENCE"
    assert rare["shell_change_authorized"] is False
    assert rep["shell_change_authorized"] is False


def test_bar_export_schema_shared():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.bar_export import (
        schema_column_names,
    )

    cols = schema_column_names()
    assert "date" in cols and "tag" in cols and "split" in cols
    assert "equity_pct" in cols


def test_train_multi_pair_default_search_is_not_all_days_only():
    src = (_LINEAGE / "train_multi_pair.py").read_text(encoding="utf-8")
    assert "practice only" in src.lower() or "PRACTICE days only" in src
    assert "--search-all-days" in src
    assert "CONTAMINATED" in src
    # Must not be the old sole path without practice option
    assert "search_days = practice" in src or "search_window = \"practice\"" in src


def test_run_gate_produces_artifacts():
    """Drive the real gate entry point (no training)."""
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.run_gate import main

    rc = main([])
    assert rc == 0, "gate must PASS"
    assert (_OUT / "EXPERIMENT_CONTRACT.md").is_file()
    assert (_OUT / "PRETRAIN_GATE_REPORT.json").is_file()
    assert (_OUT / "last_score_verdict.json").is_file()
    assert (_OUT / "data_contract.json").is_file()
    assert (_OUT / "meaning_manifest.json").is_file()
    assert (_OUT / "practice_dates.json").is_file()
    assert (_OUT / "forward_dates.json").is_file()
    report = json.loads((_OUT / "PRETRAIN_GATE_REPORT.json").read_text(encoding="utf-8"))
    assert report["verdict"] == "GATE_PASS"
    assert report["checks"]["pins"]["checkpoint"]["sha256"]
    assert report["checks"]["pins"]["dials"]["sha256"]
    contract = json.loads((_OUT / "data_contract.json").read_text(encoding="utf-8"))
    assert contract["leak_check"]["overlap_n"] == 0
    verdict = json.loads((_OUT / "last_score_verdict.json").read_text(encoding="utf-8"))
    assert verdict["extra"]["training_started"] is False
    assert "100-day" in " ".join(verdict["extra"]["cannot_yet_claim"])
