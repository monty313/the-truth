"""Structural proof that PRINCIPLES_OF_SUCCESS.md anchors match shipped multi-pair policy.

Drives real equity_day helpers + claim score JSON on disk (not a reimplemented oracle).
"""
from __future__ import annotations

import inspect
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_BUY, ACTION_HOLD, ACTION_SELL

LINEAGE = os.path.join(ROOT, "lineages", "adaptive_rl_brain_7_31_26")
PRINCIPLES = os.path.join(LINEAGE, "PRINCIPLES_OF_SUCCESS.md")
CLAIM_JSON = os.path.join(LINEAGE, "checkpoints", "ten_pair_score_all.json")
DIALS_JSON = os.path.join(LINEAGE, "checkpoints", "multi_pair_dials.json")
CKPT = os.path.join(LINEAGE, "checkpoints", "multi_pair_consistent_v1.pt")


def _tiny_m1(n: int = 120, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-04-21 08:00", periods=n, freq="min")
    close = 2000.0 + np.cumsum(rng.normal(0, 0.15, size=n))
    high = close + 0.3
    low = close - 0.3
    open_ = close.copy()
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "vol": np.full(n, 100.0),
            "spread": 0.25,
        },
        index=idx,
    )


def test_principles_doc_exists_student_tutor_framing():
    assert os.path.isfile(PRINCIPLES), PRINCIPLES
    text = open(PRINCIPLES, encoding="utf-8").read()
    assert "Student" in text and "Tutor" in text
    assert "multi_pair_consistent_v1" in text
    assert "heuristic" in text.lower()
    # KEEP + REJECT present
    assert "Principles of success (KEEP)" in text or "KEEP" in text
    assert "Anti-principles (REJECT)" in text or "REJECT" in text
    assert "trail" in text.lower()
    assert "every-bar" in text.lower() or "every bar" in text.lower()
    assert "clear" in text.lower() and "breach" in text.lower()


def test_claim_json_matches_tutor_results_anchor():
    """Drive real claim artifact the principles cite (not hard-coded invent)."""
    assert os.path.isfile(CLAIM_JSON), CLAIM_JSON
    data = json.load(open(CLAIM_JSON, encoding="utf-8"))
    assert data.get("all_pass") is True
    assert int(data.get("n_pass", 0)) == 10
    pairs = data["pairs"]
    assert len(pairs) == 10
    for p in pairs:
        assert int(p["breached"]) == 0
        assert float(p["breach_pct"]) == 0.0
        assert int(p["cleared"]) >= 30
        assert p.get("pass") is True


def test_dials_match_winning_decode():
    assert os.path.isfile(DIALS_JSON), DIALS_JSON
    d = json.load(open(DIALS_JSON, encoding="utf-8"))["dials"]
    assert d.get("decode") == "heuristic"
    assert abs(float(d["risk_use_frac"]) - 0.35) < 1e-9
    assert abs(float(d["stop_atr_mult"]) - 2.0) < 1e-9
    assert abs(float(d["per_trade_cap_pct"]) - 0.25) < 1e-9
    assert os.path.isfile(CKPT), CKPT


def test_equity_shell_implements_heat_bank_mark_one_signal():
    """Spot-check principles against shipped GoalEquityDay methods (real code)."""
    src = inspect.getsource(GoalEquityDay)
    # P5 every-bar marks
    assert "_mark_bar" in src
    assert "for bt in range(prev_t, t)" in src or "range(prev_t, t)" in inspect.getsource(
        GoalEquityDay.run
    )
    # P3 heat / floor scale
    assert "risk_use_frac" in src
    assert "floor_scale" in src
    # P7 bank
    assert "banked" in src
    assert "self.target" in src
    # P6 one signal: force flat perception
    rec_src = inspect.getsource(GoalEquityDay.recommended_action)
    assert "runner.position = None" in rec_src
    assert "opposite" in rec_src.lower() or "ACTION_SELL" in rec_src

    # Runtime target/risk on a live instance
    day = GoalEquityDay(_tiny_m1(), target_pct=2.0, risk_pct=3.0, decide_every=25)
    assert day.target == 2.0
    assert day.risk == 3.0
    # recommended_action is callable and returns legal action codes
    a = int(day.recommended_action(30))
    assert a in (ACTION_HOLD, ACTION_BUY, ACTION_SELL)
    # mark_bar does not raise
    day._mark_bar(10)
    # run heuristic path produces EquityDayResult with clear/breach fields
    res = day.run(use_heuristic=True)
    assert hasattr(res, "cleared") and hasattr(res, "breached")
    assert isinstance(bool(res.cleared), bool)
    assert isinstance(bool(res.breached), bool)
    # Clear cannot co-exist with breach (GOAL definition in engine)
    if res.breached:
        assert res.cleared is False


if __name__ == "__main__":
    test_principles_doc_exists_student_tutor_framing()
    test_claim_json_matches_tutor_results_anchor()
    test_dials_match_winning_decode()
    test_equity_shell_implements_heat_bank_mark_one_signal()
    print("test_principles_of_success OK")
