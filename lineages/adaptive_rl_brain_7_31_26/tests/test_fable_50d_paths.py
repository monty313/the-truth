"""Tests for fable 50d match loop shipped functions (real code paths)."""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIN = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_LIN))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE, _LIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import (
    better,
    gate_pass,
    not_worse,
    sample_weight,
)
from lineages.adaptive_rl_brain_7_31_26.mark_aligned_decode import (
    mark_force_gate_action,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, ACTION_SELL
from lineages.adaptive_rl_brain_7_31_26.rewards import default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc


def test_sample_weight_mwt_entry_heavier_than_award_hold():
    d = default_streak_dials()
    w_mwt = sample_weight("MARK_WOULD_TAKE", ACTION_SELL, d)
    w_aw = sample_weight("AWARD", ACTION_HOLD, d)
    assert w_mwt > w_aw


def test_not_worse_rejects_clear_drop():
    pre = {"policy_clear": 30, "same_outcome": 40, "n_breach": 0, "mark_would_take": 5}
    post = {"policy_clear": 28, "same_outcome": 42, "n_breach": 0, "mark_would_take": 3}
    assert not not_worse(post, pre, baseline_policy_clear=30)


def test_not_worse_rejects_breach():
    pre = {"policy_clear": 30, "same_outcome": 40, "n_breach": 0, "mark_would_take": 5}
    post = {"policy_clear": 31, "same_outcome": 41, "n_breach": 1, "mark_would_take": 3}
    assert not not_worse(post, pre, baseline_policy_clear=30)


def test_better_requires_strict_gain():
    pre = {"policy_clear": 30, "same_outcome": 40, "n_breach": 0, "mark_would_take": 5}
    same = {"policy_clear": 30, "same_outcome": 40, "n_breach": 0, "mark_would_take": 5}
    up = {"policy_clear": 31, "same_outcome": 40, "n_breach": 0, "mark_would_take": 5}
    assert not better(same, pre, 30)
    assert better(up, pre, 30)


def test_gate_pass_same_outcome_50():
    rows = [
        {
            "mark_award": True,
            "policy_award": True,
            "policy_breached": False,
        }
        for _ in range(50)
    ]
    s = {
        "n_days": 50,
        "same_outcome": 50,
        "n_breach": 0,
        "policy_clear": 50,
        "mark_clear": 50,
        "rows": rows,
    }
    assert gate_pass(s)


def test_gate_pass_fails_if_misses_mark_clear_day():
    rows = [{"mark_award": True, "policy_award": True, "policy_breached": False}] * 49
    rows.append({"mark_award": True, "policy_award": False, "policy_breached": False})
    s = {
        "n_days": 50,
        "same_outcome": 49,
        "n_breach": 0,
        "policy_clear": 49,
        "mark_clear": 50,
        "rows": rows,
    }
    assert not gate_pass(s)


def test_force_gate_flat_undefined_allows_mark_agreed():
    a = mark_force_gate_action(
        ACTION_SELL,
        side=None,
        equity_pct=0.0,
        risk_pct=3.0,
        force_dir=0.0,
        m_conf=0.75,
        regime="flat_undefined",
        recommended=ACTION_SELL,
    )
    assert a == ACTION_SELL


def test_train_bc_weighted_real_path():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((64, 32)).astype(np.float32)
    y = rng.integers(0, 3, size=64).astype(np.int64)
    w = np.ones(64, dtype=np.float32)
    w[:20] = 4.0
    pol, losses = train_bc(
        X, y, epochs=2, batch=16, hidden=16, seed=1, obs_dim=32, sample_weights=w, lr=1e-2
    )
    assert len(losses) == 2
    assert pol is not None


if __name__ == "__main__":
    test_sample_weight_mwt_entry_heavier_than_award_hold()
    test_not_worse_rejects_clear_drop()
    test_not_worse_rejects_breach()
    test_better_requires_strict_gain()
    test_gate_pass_same_outcome_50()
    test_gate_pass_fails_if_misses_mark_clear_day()
    test_force_gate_flat_undefined_allows_mark_agreed()
    test_train_bc_weighted_real_path()
    print("ALL_PASS")
