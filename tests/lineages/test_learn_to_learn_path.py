"""Tests for path learn-to-learn (classes + learn≠copy), real functions."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.learn_to_learn_path import (
    CLASS_TO_I,
    PATH_CLASSES,
    PathL2LPolicy,
    class_boosts_from_memory,
    classify_path_error,
    train_l2l,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.student_interface import check_learn_not_copy
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM


def test_classify_anti_thrash_and_miss_fire():
    a = classify_path_error(mark_act=ACTION_HOLD, policy_act=ACTION_SELL, t=720, t1=1000)
    assert a.path_class == "anti_thrash"
    assert a.weight >= 10
    m = classify_path_error(mark_act=ACTION_BUY, policy_act=ACTION_HOLD, t=1000, t1=1000)
    assert m.path_class == "miss_fire"
    w = classify_path_error(mark_act=ACTION_HOLD, policy_act=ACTION_HOLD, t=800, t1=1000)
    assert w.path_class == "wait_loaded"
    f = classify_path_error(mark_act=ACTION_SELL, policy_act=ACTION_SELL, t=1000, t1=1000)
    assert f.path_class == "fire_window"


def test_learn_not_copy_gate():
    # high act, low topology = copying
    g = check_learn_not_copy(act_match=0.95, topology_match=0.1, role_map_match=0.1)
    assert g["copying"] is True
    assert g["pass"] is False
    g2 = check_learn_not_copy(act_match=0.8, topology_match=0.75, role_map_match=0.7)
    assert g2["copying"] is False
    assert g2["pass"] is True


def test_class_boost_from_reject_memory():
    mem = [
        {"decision": "REJECT", "dominant_class": "anti_thrash", "class_counts": {"anti_thrash": 10}},
        {"decision": "REJECT", "dominant_class": "anti_thrash", "class_counts": {"anti_thrash": 8}},
        {"decision": "KEEP", "dominant_class": "miss_fire", "class_counts": {"miss_fire": 2}},
    ]
    b = class_boosts_from_memory(mem)
    assert b["anti_thrash"] > b["fire_window"]
    assert b["anti_thrash"] > 1.0


def test_l2l_policy_roundtrip_channel1():
    pol = PathL2LPolicy(obs_dim=MARK_FULL_DIM, hidden=128)
    # fake channel1-like state
    c1 = {
        "net.0.weight": torch.randn(128, MARK_FULL_DIM),
        "net.0.bias": torch.randn(128),
        "net.2.weight": torch.randn(3, 128),
        "net.2.bias": torch.randn(3),
    }
    pol.load_from_channel1(c1)
    st = pol.to_channel1_state()
    assert st["net.0.weight"].shape == (128, MARK_FULL_DIM)
    assert st["net.2.weight"].shape == (3, 128)
    x = np.random.randn(MARK_FULL_DIM).astype(np.float32)
    a, _ = pol.act(x, greedy=True)
    assert a in (0, 1, 2)


def test_train_l2l_improves_and_reports_gate():
    n, d = 64, MARK_FULL_DIM
    X = np.random.randn(n, d).astype(np.float32) * 0.1
    # separable-ish: class correlated with act
    y_act = np.array([0, 0, 1, 2] * (n // 4), dtype=np.int64)
    y_cls = np.array(
        [CLASS_TO_I["wait_loaded"], CLASS_TO_I["anti_thrash"], CLASS_TO_I["fire_window"], CLASS_TO_I["miss_fire"]]
        * (n // 4),
        dtype=np.int64,
    )
    pol, losses, metrics = train_l2l(
        X,
        y_act,
        y_cls,
        epochs=8,
        lr=1e-2,
        seed=0,
        kl_coef=0.0,
        class_coef=0.5,
    )
    assert len(losses) == 8
    assert "act_match" in metrics
    assert "path_class_match" in metrics
    assert "learn_not_copy_pass" in metrics
    assert 0.0 <= metrics["act_match"] <= 1.0
