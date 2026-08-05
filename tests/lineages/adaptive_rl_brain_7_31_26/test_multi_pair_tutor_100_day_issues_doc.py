"""Structural + evidence checks for the multi-pair-tutor 100-day issues catalog.

CHANGE LOG:
- 2026-07-31  doc gate — WHY: analysis deliverable must exist under references/plans/
  and every cited claim score must match frozen JSON (no invented 100/100 clears).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_DOC = _ROOT / "references" / "plans" / "MULTI_PAIR_TUTOR_100_DAY_ISSUES_AND_RULES.md"
_LINEAGE = _ROOT / "lineages" / "adaptive_rl_brain_7_31_26"
_CLAIM = _LINEAGE / "checkpoints" / "ten_pair_score_all.json"
_FORWARD = _LINEAGE / "checkpoints" / "ten_pair_score_forward.json"
_PAIRS = _LINEAGE / "ten_pairs.json"
_DIALS = _LINEAGE / "checkpoints" / "multi_pair_dials.json"


def test_deliverable_exists_under_references_plans_not_root():
    assert _DOC.is_file(), f"missing deliverable {_DOC}"
    assert "references" in _DOC.parts and "plans" in _DOC.parts
    root_dump = _ROOT / "MULTI_PAIR_TUTOR_100_DAY_ISSUES_AND_RULES.md"
    assert not root_dump.exists(), "must not dump catalog at repo root"


def test_doc_has_issues_and_mc_rules_and_goal_terms():
    text = _DOC.read_text(encoding="utf-8")
    # Issues present (ISS-01 through ISS-26)
    for n in range(1, 27):
        assert f"ISS-{n:02d}" in text, f"missing issue ISS-{n:02d}"
    # MC options pattern
    assert re.search(r"\*\*A \(Recommended\)\*\*", text)
    assert "### ISS-01" in text or "## Issue catalog" in text
    # GOAL terms
    for term in ("clear%", "breach%", "target%", "risk%", "no retrain", "streak"):
        assert term in text, f"missing GOAL term {term!r}"
    # Conclusion protocol section
    assert "100-day conclusion" in text.lower() or "Chosen rules" in text
    assert "Random inputs" in text or "random" in text.lower()
    # Must not claim already 100/100 clears as fact
    assert "not yet meet" in text.lower() or "does **not** yet meet" in text or "NOT YET" in text


def test_cited_claim_json_matches_doc_baseline_numbers():
    """Drive real frozen score artifact — not a hard-coded oracle of wishful scores."""
    assert _CLAIM.is_file()
    claim = json.loads(_CLAIM.read_text(encoding="utf-8"))
    assert claim["n_pass"] == 10
    assert claim["all_pass"] is True
    by_id = {int(p["id"]): p for p in claim["pairs"]}
    # Documented table anchors
    assert by_id[1]["cleared"] == 76 and by_id[1]["breached"] == 0
    assert by_id[10]["cleared"] == 40 and by_id[10]["breached"] == 0
    assert by_id[10]["n_days"] == 90
    assert by_id[10]["target_pct"] == 3.0 and by_id[10]["risk_pct"] == 3.5
    # Every pair breach 0
    for p in claim["pairs"]:
        assert p["breached"] == 0, p
        assert p["cleared"] >= 30


def test_cited_forward_json_is_not_all_pass():
    assert _FORWARD.is_file()
    fwd = json.loads(_FORWARD.read_text(encoding="utf-8"))
    assert fwd["n_pass"] == 5
    assert fwd["all_pass"] is False
    # Still 0 breach on all pairs (honesty fact in doc)
    for p in fwd["pairs"]:
        assert p["breached"] == 0
        assert p["n_days"] == 40
    # Hard pairs fail absolute ≥30 clear on 40d
    hard = [p for p in fwd["pairs"] if p["target_pct"] >= 2.0]
    assert any(not p["pass"] for p in hard)


def test_ten_pairs_and_dials_match_protocol_pins():
    cfg = json.loads(_PAIRS.read_text(encoding="utf-8"))
    assert len(cfg["pairs"]) == 10
    assert int(cfg["practice_day_count"]) == 50
    assert int(cfg["forward_day_count"]) == 40
    assert int(cfg["min_clear_days_per_pair"]) == 30
    assert float(cfg["require_breach_pct"]) == 0.0
    dials = json.loads(_DIALS.read_text(encoding="utf-8"))
    d = dials["dials"]
    assert d["decode"] == "heuristic"
    assert float(d["risk_use_frac"]) == 0.35
    assert float(d["stop_atr_mult"]) == 2.0
    assert float(d["per_trade_cap_pct"]) == 0.25


def test_doc_cites_key_evidence_paths():
    text = _DOC.read_text(encoding="utf-8")
    for path in (
        "ten_pair_score_all.json",
        "ten_pair_score_forward.json",
        "equity_day.py",
        "UNSEEN_CONSISTENCY_RECIPE.md",
        "GOAL_FROM_TEN_PAIR_IRAC.md",
        "ten_pairs.json",
        "multi_pair_dials.json",
    ):
        assert path in text, f"doc must cite {path}"


def test_dial_search_honesty_path_in_shipped_train_script():
    """ISS-14: practice-only default; all-day search explicit CONTAMINATED path only."""
    src = (_LINEAGE / "train_multi_pair.py").read_text(encoding="utf-8")
    assert "search_dials" in src
    assert "--search-all-days" in src
    assert "CONTAMINATED" in src
    assert "practice" in src.lower()


def test_score_pair_on_days_real_path_still_drives_equity_day():
    """Smoke: real scorer path used by multi-pair tutor still works (shipped entry)."""
    from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
    from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import score_pair_on_days

    days = load_calendar_days()
    assert len(days) == 90, "curriculum eligible day count is evidence anchor (90 not 100)"
    rep = score_pair_on_days(
        days[:2],
        target=1.0,
        risk=2.0,
        use_heuristic=True,
    )
    assert rep["n_days"] == 2
    assert rep["breached"] + rep["cleared"] <= 2
    for row in rep["day_rows"]:
        if row["cleared"]:
            assert row["breached"] is False
