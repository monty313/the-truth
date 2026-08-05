"""Shipped Mark-clone policy path: same obs, teacher, checkpoint contract."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from lineages.adaptive_rl_brain_7_31_26.perception.mark_sets_opportunity import (
    mark_dir_to_action,
    official_sets_table,
    scan_mark_opportunities,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction, SetConfluence, VelocityStrength
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)

_LINEAGE = Path(__file__).resolve().parents[3] / "lineages" / "adaptive_rl_brain_7_31_26"
# Prefer five-law doctrine brain; fall back to multi-set BC brain
_CKPT_CANDIDATES = (
    _LINEAGE / "checkpoints" / "mark_clone_doctrine_v1.pt",
    _LINEAGE / "checkpoints" / "mark_clone_channel1_v1.pt",
    _LINEAGE / "checkpoints" / "mark_clone_latest.pt",
)


def _conf(sid: int, d: Direction, v: VelocityStrength = VelocityStrength.MEDIUM) -> SetConfluence:
    return SetConfluence(
        set_key=f"official:{sid}",
        direction=d,
        velocity=v,
        votes=(),
        n_bull=2 if d == Direction.BULL else 0,
        n_bear=2 if d == Direction.BEAR else 0,
        n_neutral=1,
    )


def test_four_set_table_is_monty_mark_lock():
    t = official_sets_table()
    assert [x["stack"] for x in t] == [
        ["1m", "15m", "30m"],
        ["5m", "30m", "1h"],
        ["15m", "1h", "4h"],
        ["30m", "4h", "1d"],
    ]
    for row in t:
        assert row["ltf_entry"] == row["stack"][0]
        assert row["htf_confirm"] == row["stack"][1:]


def test_teacher_aligned_maps_to_buy_sell_not_against_macro():
    official = {i: _conf(i, Direction.BULL) for i in (1, 2, 3, 4)}
    entry = {i: Direction.BULL for i in (1, 2, 3, 4)}
    opp = scan_mark_opportunities(official, entry)
    assert opp.action_dir == Direction.BULL
    assert mark_dir_to_action(opp.action_dir) == ACTION_BUY


def test_channel1_obs_dim_fixed_for_new_policy():
    """Same-obs contract: new Mark clone brain must use CHANNEL1_DIM."""
    assert CHANNEL1_DIM == 32
    p = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=48)
    x = torch.zeros(CHANNEL1_DIM)
    logits = p(x)
    assert tuple(logits.shape) == (1, 3)


def test_mark_clone_checkpoint_same_obs_not_all_hold():
    """Load shipped Mark-clone weights; pure greedy not 100% HOLD on random obs."""
    ckpt = next((p for p in _CKPT_CANDIDATES if p.is_file()), None)
    if ckpt is None:
        raise AssertionError(
            "missing Mark-clone ckpt; expected one of "
            + ", ".join(str(p.name) for p in _CKPT_CANDIDATES)
        )
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert int(blob.get("obs_dim", -1)) == CHANNEL1_DIM
    assert blob.get("proven_touched") is False
    assert "state_dict" in blob
    pol = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=int(blob.get("hidden", 48)))
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    # Deterministic probes from teacher-like one-hot-ish directions in first slots
    acts = []
    rng = np.random.default_rng(0)
    with torch.no_grad():
        for _ in range(64):
            x = torch.as_tensor(rng.standard_normal(CHANNEL1_DIM).astype(np.float32))
            # inject strong bull set features (dir slots 0,3,6,9 ≈ +1)
            x[0] = 1.0
            x[3] = 1.0
            x[6] = 1.0
            x[9] = 1.0
            a = int(torch.argmax(pol(x), dim=-1).item())
            acts.append(a)
    hold_frac = sum(1 for a in acts if a == ACTION_HOLD) / len(acts)
    assert hold_frac < 0.999
    assert any(a in (ACTION_BUY, ACTION_SELL) for a in acts)


def test_train_mark_clone_script_exists():
    script = _LINEAGE / "train_mark_clone_bc.py"
    if not script.is_file():
        script = _LINEAGE / "train_mark_clone.py"
    assert script.is_file()
    text = script.read_text(encoding="utf-8")
    assert "CHANNEL1_DIM" in text
    assert "PROVEN" in text
    assert ("mark_doctrine" in text) or ("mark_all_sets" in text)
