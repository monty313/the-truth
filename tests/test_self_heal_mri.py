"""Unit tests for Mind Probe + Ghost Trades + meta_tuner pullback unlock.
No curriculum data or frozen brain required.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from telemetry.mind_probe import DecisionRecord, DayMindDump, OP_NAMES
from telemetry.ghost_trades import build_ghosts
from training.meta_tuner import BOUNDS, _FALLBACK, adopt_gate
import torch


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
    assert abs(_FALLBACK["w_pullback_with_htf"] - 0.02) < 1e-9


def test_adopt_gate_rejects_noise():
    n = 200
    champ = torch.zeros(n, dtype=torch.bool)
    champ[:100] = True
    cand = champ.clone()
    cand[0] = False
    cand[100] = True
    adopt, info = adopt_gate(champ, cand)
    assert adopt is False


if __name__ == "__main__":
    test_op_names_cover_11()
    test_ghost_high_miss_pull_perception_lean()
    test_ghost_no_pattern_no_ghosts()
    test_meta_tuner_pullback_unlocked()
    test_adopt_gate_rejects_noise()
    print("ALL SELF-HEAL MRI UNIT TESTS PASSED")
