"""Spine Shadow full heads — unit tests (real functions)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "code")]

from lineages.adaptive_rl_brain_7_31_26.mark_shadow_policy import (
    EVENT_I,
    PHASE_I,
    SIZE_I,
    SpineShadowNet,
    as_channel1,
    event_at_t,
    event_to_action,
    phase_at_t,
    size_at_event,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_BUY, ACTION_HOLD, ACTION_SELL


def test_phase_and_event_labels():
    assert phase_at_t(100, 500, None) == "before_first_fire"
    assert phase_at_t(600, 500, 700) == "in_trade"
    plan = {500: ACTION_BUY, 700: ACTION_BUY, 100: ACTION_HOLD}
    assert event_at_t(100, plan, 500, 700) == "wait_loaded"
    assert event_at_t(500, plan, 500, 700) == "fire"
    assert event_at_t(700, plan, 500, 700) == "add"
    assert event_to_action("fire", "BUY") == ACTION_BUY
    assert event_to_action("wait_loaded", "BUY") == ACTION_HOLD
    assert size_at_event("fire", "base") == "base"
    assert size_at_event("wait_loaded", "base") == "none"


def test_shadow_net_forward_and_export():
    net = SpineShadowNet(obs_dim=MARK_FULL_DIM, hidden=128)
    x = torch.randn(4, MARK_FULL_DIM)
    ph, ev, sz, act, gate = net(x)
    assert ph.shape == (4, len(PHASE_I))
    assert ev.shape == (4, len(EVENT_I))
    assert sz.shape == (4, len(SIZE_I))
    assert act.shape == (4, 3)
    assert gate.shape == (4, MARK_FULL_DIM)
    assert ((gate >= 0) & (gate <= 1)).all()
    pol = as_channel1(net)
    a, _ = pol.act(np.random.randn(MARK_FULL_DIM).astype(np.float32), greedy=True)
    assert a in (0, 1, 2)


def test_heads_named_for_spine_shadow_doctrine():
    # structural: all four product heads exist
    net = SpineShadowNet()
    assert hasattr(net, "phase_head")
    assert hasattr(net, "event_head")
    assert hasattr(net, "size_head")
    assert hasattr(net, "clue_gate")
    assert hasattr(net, "act_head")
