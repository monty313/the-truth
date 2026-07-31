"""Phase 2 Slice 5: multi-TF resample + day runner + train stub."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from data_io.loader import synthetic_m1
from lineages.adaptive_rl_brain_7_31_26.data.mtf import (
    LINEAGE_TFS,
    build_mtf_pack,
    lineage_tf_to_loader,
    resample_lineage,
)
from lineages.adaptive_rl_brain_7_31_26.day_runner import (
    DayRunner,
    build_perception_at,
    setup_active,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    StructureFlags,
    TradeTag,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    Channel1Policy,
    action_to_trade_side,
)
from lineages.adaptive_rl_brain_7_31_26.train_stub import run_day_rollout, train_stub_epoch


def test_lineage_tf_mapping():
    assert lineage_tf_to_loader("1m") == "1min"
    assert lineage_tf_to_loader("5m") == "5min"
    assert lineage_tf_to_loader("1h") == "1h"
    assert lineage_tf_to_loader("1d") == "1d"


def test_build_mtf_pack_has_all_tfs():
    m1 = synthetic_m1(days=3, seed=3)
    pack = build_mtf_pack(m1)
    for tf in LINEAGE_TFS:
        assert tf in pack, tf
        assert len(pack[tf]) > 0
        for col in ("open", "high", "low", "close"):
            assert col in pack[tf].columns


def test_resample_lineage_1m_matches_m1_len_approx():
    m1 = synthetic_m1(days=1, seed=1)
    r = resample_lineage(m1, "1m")
    assert abs(len(r) - len(m1)) <= 2


def test_build_perception_obs_shape():
    m1 = synthetic_m1(days=2, seed=5)
    pack = build_mtf_pack(m1)
    ts = m1.index[min(300, len(m1) - 1)]
    perc = build_perception_at(pack, ts, trade_side=Direction.BULL)
    assert perc["obs"].shape == (CHANNEL1_DIM,)
    assert "official" in perc and "structure" in perc


def test_day_runner_step_hold():
    m1 = synthetic_m1(days=2, seed=7)
    day = m1.iloc[:500]
    runner = DayRunner(day, decide_every=30)
    idxs = runner.decision_indices()
    assert len(idxs) >= 1
    step = runner.step(idxs[0], ACTION_HOLD)
    assert step.obs.shape == (CHANNEL1_DIM,)
    assert isinstance(step.reward, float)


def test_day_runner_buy_and_hold_path():
    m1 = synthetic_m1(days=2, seed=8)
    day = m1.iloc[:500]
    runner = DayRunner(day, decide_every=40)
    idxs = runner.decision_indices()
    s0 = runner.step(idxs[0], ACTION_BUY)
    assert s0.action == ACTION_BUY
    s1 = runner.step(idxs[min(1, len(idxs) - 1)], ACTION_HOLD)
    assert s1.obs.shape == (CHANNEL1_DIM,)


def test_policy_stub_act_dim():
    p = Channel1Policy(hidden=8)
    obs = np.zeros(CHANNEL1_DIM, dtype=np.float32)
    a, logp = p.act(obs, greedy=True)
    assert a in (0, 1, 2)
    assert action_to_trade_side(0) is None
    assert action_to_trade_side(1) == Direction.BULL
    assert action_to_trade_side(2) == Direction.BEAR
    assert logp.shape == ()


def test_setup_active_with_vector():
    from lineages.adaptive_rl_brain_7_31_26.perception.types import Classification
    cl = Classification(tag=TradeTag.WITH_VECTOR, mindless=False, reasons=())
    assert setup_active(cl, StructureFlags(False, False)) is True
    cl2 = Classification(tag=TradeTag.MINDLESS, mindless=True, reasons=())
    assert setup_active(cl2, StructureFlags(False, False)) is False
    assert setup_active(None, StructureFlags(True, False)) is True


def test_run_day_rollout_smoke():
    r = run_day_rollout(max_steps=5, greedy=True, seed=2, decide_every=30)
    assert r.n_steps >= 1
    assert r.n_steps == len(r.steps)


def test_train_stub_epoch_smoke():
    out = train_stub_epoch(steps=5, seed=3)
    assert "loss" in out and "mean_reward" in out
    assert out["n"] >= 1


if __name__ == "__main__":
    test_lineage_tf_mapping()
    test_build_mtf_pack_has_all_tfs()
    test_resample_lineage_1m_matches_m1_len_approx()
    test_build_perception_obs_shape()
    test_day_runner_step_hold()
    test_day_runner_buy_and_hold_path()
    test_policy_stub_act_dim()
    test_setup_active_with_vector()
    test_run_day_rollout_smoke()
    test_train_stub_epoch_smoke()
    print("test_mtf_and_train OK")
