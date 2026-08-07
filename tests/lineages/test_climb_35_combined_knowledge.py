"""Tests for combined climb KEEP gate + single-head BEST constraint.

Drives shipped functions in climb_35_combined_knowledge (not reimplemented).
"""
from __future__ import annotations

import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.climb_35_combined_knowledge import (  # noqa: E402
    SOURCES,
    keep_gate,
    meters,
)
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import (  # noqa: E402
    CKPT,
    load_policy,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy  # noqa: E402
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import (  # noqa: E402
    MARK_FULL_DIM,
)


def test_keep_gate_requires_same_up_and_breach0():
    pre = {"same_outcome": 35, "policy_clear": 35, "mark_would_take": 15, "n_breach": 0}
    # equal same → reject
    ok, reason = keep_gate(
        {"same_outcome": 35, "policy_clear": 35, "mark_would_take": 15, "n_breach": 0},
        pre,
    )
    assert ok is False and reason == "same_not_up"
    # breach → reject
    ok, reason = keep_gate(
        {"same_outcome": 36, "policy_clear": 36, "mark_would_take": 14, "n_breach": 1},
        pre,
    )
    assert ok is False and reason == "breach"
    # clear below floor → reject
    ok, reason = keep_gate(
        {"same_outcome": 36, "policy_clear": 20, "mark_would_take": 14, "n_breach": 0},
        pre,
        baseline_clear=27,
    )
    assert ok is False and reason == "below_baseline_clear"
    # true climb → keep
    ok, reason = keep_gate(
        {"same_outcome": 36, "policy_clear": 36, "mark_would_take": 14, "n_breach": 0},
        pre,
    )
    assert ok is True and reason == "same_up_breach0"


def test_strategy_only_decline_is_not_keep():
    """Counterexample path: 35→15 must fail keep_gate (prior REJECT)."""
    pre = {"same_outcome": 35, "policy_clear": 35, "mark_would_take": 15, "n_breach": 0}
    post = {"same_outcome": 15, "policy_clear": 15, "mark_would_take": 35, "n_breach": 0}
    ok, reason = keep_gate(post, pre)
    assert ok is False
    # may fail baseline clear first (15 < 27) or same_not_up — both reject
    assert reason in ("same_not_up", "below_baseline_clear", "clear_down")


def test_live_best_is_single_head_full_obs():
    """Climb path requires single-head BEST; multi-head full replace is banned."""
    assert os.path.isfile(CKPT), f"missing BEST ckpt {CKPT}"
    pol = load_policy(CKPT)
    assert isinstance(pol, Channel1Policy)
    assert pol.multi_head is False
    assert int(pol.obs_dim) == MARK_FULL_DIM
    # act works
    obs = torch.zeros(MARK_FULL_DIM)
    a, _ = pol.act(obs.numpy(), greedy=True)
    assert int(a) in (0, 1, 2)


def test_sources_include_mark_and_pack_protect():
    assert "mark_path_family_fire_skill" in SOURCES
    assert "kl_anchor_best" in SOURCES
    assert "award_self_protect" in SOURCES or "hold_floor_inject" in SOURCES
    assert "strategy_aux_light" in SOURCES
    # must not be strategy-only
    assert len(SOURCES) >= 3
    assert any("path_family" in s or "mark_" in s for s in SOURCES)


def test_meters_helper():
    m = meters(
        {
            "same_outcome": 35,
            "policy_clear": 35,
            "mark_would_take": 15,
            "n_breach": 0,
            "rows": [],
        }
    )
    assert m == {
        "same_outcome": 35,
        "policy_clear": 35,
        "mark_would_take": 15,
        "n_breach": 0,
    }
