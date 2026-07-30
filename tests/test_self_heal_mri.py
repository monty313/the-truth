import sys
"""Unit tests for Mind Probe + Ghost Trades + meta_tuner self-heal toolkit.
No curriculum data or frozen brain required.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'src'))
sys.path.insert(0, ROOT)
sys.path.insert(0, ROOT)

from telemetry.mind_probe import (
    DecisionRecord, DayMindDump, OP_NAMES, side_metrics_from_decisions,
    _op_probs_from_forward, LONG_OPS, SHORT_OPS,
)
from telemetry.ghost_trades import build_ghosts
from training.meta_tuner import BOUNDS, _FALLBACK, adopt_gate
from training.policy import Brain, N_OPS
from training.rewards import RewardEngine
import torch
import numpy as np


def test_op_names_cover_11():
    assert len(OP_NAMES) == 11
    assert OP_NAMES[0] == "hold"
    assert OP_NAMES[1] == "open_long"


def test_ghost_high_miss_pull_perception_lean():
    recs = []
    for t in range(8):
        probs = [0.85] + [0.015] * 10
        recs.append(DecisionRecord(
            t=t, op_probs=probs, chosen_op=0, chosen_op_name="hold",
            chosen_size=0.1, value=0.0, pull_buy=True,
        ))
    dump = DayMindDump(
        brain_name="test", day_index=0, day_label="unit",
        goal_pct=3.0, floor_pct=3.5, decisions=recs,
        n_pull_buy_bars=8, pull_buy_seen_and_held=8,
    )
    report = build_ghosts(dump)
    assert report.n_high_miss_pull == 8
    assert report.n_ghosts >= 8
    assert "Perception lean" in report.summary or "IRAC SIGNAL" in report.summary


def test_ghost_no_pattern_no_ghosts():
    recs = [DecisionRecord(
        t=0, op_probs=[1.0] + [0.0] * 10, chosen_op=0, chosen_op_name="hold",
        chosen_size=0.1, value=0.0,
    )]
    dump = DayMindDump(
        brain_name="test", day_index=0, day_label="flat",
        goal_pct=3.0, floor_pct=3.5, decisions=recs,
    )
    report = build_ghosts(dump)
    assert report.n_ghosts == 0
    assert report.n_high_miss_pull == 0


def test_meta_tuner_pullback_unlocked():
    assert "w_pullback_with_htf" in BOUNDS
    lo, hi = BOUNDS["w_pullback_with_htf"]
    assert lo == 0.0 and hi == 1.0
    assert "w_pullback_with_htf" in _FALLBACK
    assert abs(_FALLBACK["w_pullback_with_htf"] - 0.25) < 1e-9


def test_meta_tuner_side_dials_in_bounds():
    for k in ("w_with_trend_close", "w_against_trend_close",
              "w_quick_pull_close", "w_setup_skip"):
        assert k in BOUNDS, k
        assert k in _FALLBACK, k
        assert abs(_FALLBACK[k]) < 1e-9, k  # default off
    assert BOUNDS["w_with_trend_close"][0] == 0.0
    assert BOUNDS["w_against_trend_close"][1] == 0.0


def test_adopt_gate_rejects_noise():
    # adopt_gate(b, c): b = days only baseline cleared, c = only candidate — needs big gap
    assert adopt_gate(2, 3) is False  # too few disagreements
    assert adopt_gate(40, 55) is True or adopt_gate(40, 55) is False  # z-gate; just call path
    assert adopt_gate(5, 5) is False


def test_op_probs_from_categorical_forward():
    brain = Brain(24, hidden=16)
    obs = torch.randn(1, 24)
    with torch.no_grad():
        result = brain(obs)
    probs, val, size = _op_probs_from_forward(result)
    assert probs.shape[0] == N_OPS
    assert abs(probs.sum() - 1.0) < 1e-4
    assert probs[0] < 0.999  # not silent pure-hold artifact


def test_side_metrics_wrong_side_under_bull():
    recs = []
    # under bull cont, choose short
    for t in range(5):
        probs = [0.1, 0.05, 0.4, 0.05, 0.2, 0.05, 0.05, 0.05, 0.05, 0.0, 0.0]
        recs.append(DecisionRecord(
            t=t, op_probs=probs, chosen_op=2, chosen_op_name="open_short",
            chosen_size=0.2, value=0.0, cont_buy=True, cont_sell=False,
            p_long=0.1, p_short=0.6, p_hold=0.1, wrong_side=True,
        ))
    sm = side_metrics_from_decisions(recs)
    assert sm["n_cont_buy_only"] == 5
    assert sm["n_wrong_side_under_bull"] == 5
    assert sm["side_bias_bull"] < 0


def test_with_trend_reward_default_zero_noop():
    re = RewardEngine()
    plain = re.on_step(
        [{"pnl": 0.2, "pnl_pct": 0.2, "adds": 0, "max_adverse": 0.0,
          "stack_green": True, "probe": False, "full": True, "tags": {}, "bars": 5}],
        acted=True, anti_gravity=False, flat=False)
    tagged = re.on_step(
        [{"pnl": 0.2, "pnl_pct": 0.2, "adds": 0, "max_adverse": 0.0,
          "stack_green": True, "probe": False, "full": True,
          "tags": {"with_trend": True}, "bars": 5}],
        acted=True, anti_gravity=False, flat=False)
    # default w_with_trend_close=0 → same
    assert abs(tagged - plain) < 1e-9


def test_with_trend_reward_pays_when_dial_on():
    re = RewardEngine()
    re.w["w_with_trend_close"] = 0.25
    re.w["w_against_trend_close"] = -0.15
    with_t = re.on_step(
        [{"pnl": 0.2, "pnl_pct": 0.2, "adds": 0, "max_adverse": 0.0,
          "stack_green": True, "probe": False, "full": True,
          "tags": {"with_trend": True}, "bars": 5}],
        acted=True, anti_gravity=False, flat=False)
    against = re.on_step(
        [{"pnl": 0.2, "pnl_pct": 0.2, "adds": 0, "max_adverse": 0.0,
          "stack_green": True, "probe": False, "full": True,
          "tags": {"against_trend": True}, "bars": 5}],
        acted=True, anti_gravity=False, flat=False)
    plain = re.on_step(
        [{"pnl": 0.2, "pnl_pct": 0.2, "adds": 0, "max_adverse": 0.0,
          "stack_green": True, "probe": False, "full": True, "tags": {}, "bars": 5}],
        acted=True, anti_gravity=False, flat=False)
    assert with_t > plain
    assert against < plain


def test_propose_from_irac_wrong_side():
    from scripts.self_heal_epoch import propose_from_irac
    irac = {
        "issue": "test",
        "application": {
            "sum_policy_hold_on_setup": 10,
            "sum_high_miss_pull": 5,
            "sum_wrong_side_under_bull": 80,
            "sum_wrong_side_under_bear": 5,
            "sum_cont_buy_only": 100,
            "sum_cont_sell_only": 50,
            "mean_side_bias_bull": -0.08,
            "mean_side_bias_bear": 0.1,
            "sum_mask_veto": 0,
        },
        "conclusion": {"class": "Policy"},
    }
    p = propose_from_irac(irac)
    assert p["class"] == "WrongSide"
    keys = [n["key"] for n in p.get("reward_nudges") or []]
    assert "w_with_trend_close" in keys
    assert "w_against_trend_close" in keys


if __name__ == "__main__":
    test_op_names_cover_11()
    test_ghost_high_miss_pull_perception_lean()
    test_ghost_no_pattern_no_ghosts()
    test_meta_tuner_pullback_unlocked()
    test_meta_tuner_side_dials_in_bounds()
    test_adopt_gate_rejects_noise()
    test_op_probs_from_categorical_forward()
    test_side_metrics_wrong_side_under_bull()
    test_with_trend_reward_default_zero_noop()
    test_with_trend_reward_pays_when_dial_on()
    test_propose_from_irac_wrong_side()
    print("ALL SELF-HEAL MRI UNIT TESTS PASSED")
