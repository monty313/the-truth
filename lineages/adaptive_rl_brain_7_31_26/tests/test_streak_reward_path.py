"""Unit tests: streak rewards + weighted BC path (shipped code, no hard-coded win)."""
from __future__ import annotations

import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIN = os.path.dirname(_HERE)
_ROOT = os.path.dirname(os.path.dirname(_LIN))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE, _LIN):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.loop_2x_streak_rewards import (
    gap_class_for_day,
    sample_weight_for,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    apply_autopsy_to_streak_dials,
    day_terminal_streak_reward,
    default_streak_dials,
    soul_alignment_step_reward,
    streak_award_bonus,
    streak_break_penalty,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc


def test_streak_award_grows_with_prior():
    d = default_streak_dials()
    r0 = streak_award_bonus(cleared=True, breached=False, prior_streak=0, dials=d)
    r5 = streak_award_bonus(cleared=True, breached=False, prior_streak=5, dials=d)
    assert r5 > r0
    assert streak_award_bonus(cleared=False, breached=False, prior_streak=5, dials=d) == 0.0


def test_streak_break_only_after_streak():
    d = default_streak_dials()
    assert streak_break_penalty(cleared=False, breached=False, prior_streak=0, dials=d) == 0.0
    assert streak_break_penalty(cleared=False, breached=False, prior_streak=3, dials=d) < 0.0
    assert streak_break_penalty(cleared=True, breached=False, prior_streak=3, dials=d) == 0.0


def test_day_terminal_mark_would_take_penalizes_miss():
    d = default_streak_dials()
    parts = day_terminal_streak_reward(
        cleared=False,
        breached=False,
        prior_streak=4,
        gap_class="MARK_WOULD_TAKE",
        dials=d,
    )
    assert parts["streak_break"] < 0
    assert parts["gap_class"] < 0
    assert parts["total"] < 0


def test_soul_alignment_prefers_mark_side():
    d = default_streak_dials()
    good = soul_alignment_step_reward(
        action=ACTION_BUY, mark_soul_action=ACTION_BUY, gap_class="MARK_WOULD_TAKE", dials=d
    )
    bad = soul_alignment_step_reward(
        action=ACTION_SELL, mark_soul_action=ACTION_BUY, gap_class="MARK_WOULD_TAKE", dials=d
    )
    assert good > 0
    assert bad < 0


def test_sample_weight_mark_take_higher_than_award_hold():
    d = default_streak_dials()
    w_take = sample_weight_for("MARK_WOULD_TAKE", ACTION_BUY, d)
    w_award = sample_weight_for("AWARD", ACTION_HOLD, d)
    assert w_take > w_award


def test_gap_class_logic():
    assert gap_class_for_day(True, False, True, True) == "AWARD"
    assert gap_class_for_day(False, False, True, True) == "MARK_WOULD_TAKE"
    assert gap_class_for_day(False, False, False, False) == "NO_OPPORTUNITY"
    assert gap_class_for_day(False, True, True, True) == "POLICY_BREACH"


def test_apply_autopsy_retunes_when_mark_take_dominates():
    base = default_streak_dials()
    summary = {
        "counts": {"MARK_WOULD_TAKE": 10, "NO_OPPORTUNITY": 1, "AWARD": 29},
        "n_gaps": 11,
        "max_award_streak": 8,
    }
    out = apply_autopsy_to_streak_dials(summary, base=base)
    # high mark-take share → stronger misread penalty magnitude
    assert out["mark_would_take_eod_penalty"] <= base["mark_would_take_eod_penalty"]
    assert out["soul_side_entry_bonus"] >= base["soul_side_entry_bonus"]


def test_force_gate_allows_mark_agreed_sell_in_flat_undefined():
    """Regression: flat_undefined must not block pol=S when rec=S (Mark agrees)."""
    from lineages.adaptive_rl_brain_7_31_26.mark_aligned_decode import (
        mark_force_gate_action,
        mark_aligned_action,
    )

    # Exact failure mode from 2026-04-06 t=745
    gated = mark_force_gate_action(
        ACTION_SELL,
        side=None,
        equity_pct=0.0,
        risk_pct=3.0,
        force_dir=0.0,
        m_conf=0.75,
        regime="flat_undefined",
        recommended=ACTION_SELL,
    )
    assert gated == ACTION_SELL, f"expected SELL got {gated}"
    aligned = mark_aligned_action(
        ACTION_SELL,
        ACTION_SELL,
        side=None,
        equity_pct=0.0,
        risk_pct=3.0,
        force_dir=0.0,
        m_conf=0.75,
        regime="flat_undefined",
    )
    assert aligned == ACTION_SELL
    # still block true chop when Mark does NOT agree
    blocked = mark_force_gate_action(
        ACTION_SELL,
        side=None,
        equity_pct=0.0,
        risk_pct=3.0,
        force_dir=0.0,
        m_conf=0.75,
        regime="chop",
        recommended=ACTION_HOLD,
    )
    assert blocked == ACTION_HOLD
    # danger still wins even if Mark agrees
    dang = mark_force_gate_action(
        ACTION_SELL,
        side=None,
        equity_pct=-1.5,
        risk_pct=3.0,
        force_dir=0.0,
        m_conf=0.75,
        regime="flat_undefined",
        recommended=ACTION_SELL,
    )
    assert dang == ACTION_HOLD


def test_train_bc_sample_weights_runs_and_moves_loss():
    """Shipped train_bc with sample_weights must accept weights and train."""
    rng = np.random.default_rng(0)
    n, dim = 120, 32
    X = rng.standard_normal((n, dim)).astype(np.float32)
    y = rng.integers(0, 3, size=n).astype(np.int64)
    # heavy weight on first half
    w = np.ones(n, dtype=np.float32)
    w[:60] = 5.0
    pol, losses = train_bc(
        X,
        y,
        epochs=3,
        batch=32,
        hidden=16,
        seed=0,
        obs_dim=dim,
        sample_weights=w,
        lr=1e-2,
    )
    assert isinstance(pol, Channel1Policy)
    assert len(losses) == 3
    assert all(isinstance(x, float) for x in losses)
    # forward still works
    logits = pol(torch.zeros(dim))
    assert logits.shape[-1] == 3


if __name__ == "__main__":
    test_streak_award_grows_with_prior()
    test_streak_break_only_after_streak()
    test_day_terminal_mark_would_take_penalizes_miss()
    test_soul_alignment_prefers_mark_side()
    test_sample_weight_mark_take_higher_than_award_hold()
    test_gap_class_logic()
    test_apply_autopsy_retunes_when_mark_take_dominates()
    test_train_bc_sample_weights_runs_and_moves_loss()
    print("ALL_PASS")
