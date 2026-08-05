"""Signal-agent active consensus idle — 92 agents on REAL data/raw prices.

Rule:
  ≥10 agents active AND ≥60% of those agree on one side
  (e.g. 6 of 10) → penalty for HOLD unless ≥2 trades already open.
"""
from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CODE = os.path.join(ROOT, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_BUY, ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.price_data import (
    data_banner,
    load_recent_bars,
    resolve_raw_csv,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    MAJORITY_AGREE_FRAC,
    MAJORITY_IDLE_PENALTY,
    MAJORITY_MIN_ACTIVE,
    MAJORITY_MIN_OPEN_EXEMPT,
    majority_agents_idle_penalty,
)
from lineages.adaptive_rl_brain_7_31_26.signal_majority import (
    SIGNAL_SLOTS_YAML,
    compute_panel_matrix,
    load_all_signal_agents,
    majority_at,
    majority_from_votes,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction


def _real_m1(n_bars: int = 2500):
    return load_recent_bars(n_bars)


def test_raw_price_file_exists():
    p = resolve_raw_csv()
    assert p.is_file(), p
    print("using", data_banner())


def test_yaml_has_92_filled_agents():
    assert SIGNAL_SLOTS_YAML.is_file(), SIGNAL_SLOTS_YAML
    assert len(load_all_signal_agents(only_enabled=False)) == 92


def test_active_consensus_rule_math():
    # 10 active, 6 bull = 60% → YES (60% of 10)
    votes = [1] * 6 + [-1] * 4 + [0] * 82
    s = majority_from_votes(votes)
    assert s.n_active == 10
    assert abs(s.agree_frac - 0.60) < 1e-9
    assert s.has_majority is True
    assert s.direction == Direction.BULL

    # 10 active, 5 bull / 5 bear = 50% → NO
    votes_tie = [1] * 5 + [-1] * 5 + [0] * 82
    s2 = majority_from_votes(votes_tie)
    assert s2.n_active == 10
    assert s2.has_majority is False

    # 9 active even if 100% bull → not enough active
    votes_few = [1] * 9 + [0] * 83
    s3 = majority_from_votes(votes_few)
    assert s3.n_active == 9
    assert s3.has_majority is False

    # constants locked
    assert MAJORITY_MIN_ACTIVE == 10
    assert MAJORITY_AGREE_FRAC == 0.60
    assert MAJORITY_MIN_OPEN_EXEMPT == 2


def test_idle_penalty_exempt_when_two_open():
    assert majority_agents_idle_penalty(
        has_majority=True, action_is_hold=True, n_open=0
    ) == MAJORITY_IDLE_PENALTY
    assert majority_agents_idle_penalty(
        has_majority=True, action_is_hold=True, n_open=1
    ) == MAJORITY_IDLE_PENALTY
    assert majority_agents_idle_penalty(
        has_majority=True, action_is_hold=True, n_open=2
    ) == 0.0
    assert majority_agents_idle_penalty(
        has_majority=True, action_is_hold=False, n_open=0
    ) == 0.0


def test_compute_panel_matrix_is_92_wide_on_real_prices():
    m1 = _real_m1(2000)
    mat, names = compute_panel_matrix(m1, only_enabled=False)
    assert mat.shape[1] == 92
    ever = int(np.any(mat != 0, axis=0).sum())
    print(f"real panel: bars={len(m1)} agents_ever_nonzero={ever}/92")
    assert ever >= 5


def test_day_runner_consensus_idle_and_scale_in_exempt():
    m1 = _real_m1(2000)
    runner = DayRunner(m1, decide_every=40, use_signal_majority=True)
    assert runner.agent_matrix.shape[1] == 92
    t = runner.decision_indices()[0]

    # Force 10 bull active (100% agree) on this bar — meets ≥10 & ≥60%
    row = np.zeros(92, dtype=np.float32)
    row[:10] = 1.0
    runner.agent_matrix[t] = row

    # HOLD with 0 open → penalty
    step = runner.step(t, ACTION_HOLD)
    assert step.info.get("majority_idle") is True
    assert float(step.info["majority_penalty"]) == -1.5
    assert int(step.info["majority_n_active"]) == 10
    assert step.info.get("n_open", 0) == 0

    # Open trade 1
    t2 = runner.decision_indices()[1]
    runner.agent_matrix[t2] = row
    s1 = runner.step(t2, ACTION_BUY)
    assert runner.n_open == 1
    assert s1.info.get("entry") is True

    # HOLD with 1 open → still penalty
    t3 = runner.decision_indices()[2]
    runner.agent_matrix[t3] = row
    s_hold1 = runner.step(t3, ACTION_HOLD)
    assert runner.n_open == 1
    assert s_hold1.info.get("majority_idle") is True

    # Scale-in → 2 open
    t4 = runner.decision_indices()[3]
    runner.agent_matrix[t4] = row
    s2 = runner.step(t4, ACTION_BUY)
    assert runner.n_open == 2
    assert s2.info.get("scale_in") is True

    # HOLD with 2 open → NO majority idle penalty
    t5 = runner.decision_indices()[4]
    runner.agent_matrix[t5] = row
    s_hold2 = runner.step(t5, ACTION_HOLD)
    assert runner.n_open == 2
    assert s_hold2.info.get("majority_idle") is not True


def test_real_data_active_consensus_scan():
    """How often does ≥20 active & >70% agree appear on real gold?"""
    m1 = _real_m1(4000)
    runner = DayRunner(m1, decide_every=15, use_signal_majority=True)
    idxs = runner.decision_indices()
    sample = idxs[-120:] if len(idxs) > 120 else idxs
    hits = 0
    max_active = 0
    max_agree = 0.0
    for t in sample:
        snap = majority_at(runner.agent_matrix, t)
        max_active = max(max_active, snap.n_active)
        max_agree = max(max_agree, snap.agree_frac if snap.n_active else 0.0)
        if snap.has_majority:
            hits += 1
    print(
        f"REAL active-consensus scan: bars={len(sample)} hits={hits} "
        f"max_active={max_active} max_agree_frac={max_agree:.2f} "
        f"(need active>={MAJORITY_MIN_ACTIVE}, agree>={MAJORITY_AGREE_FRAC})"
    )
    assert runner.agent_matrix.shape[1] == 92


if __name__ == "__main__":
    test_raw_price_file_exists()
    test_yaml_has_92_filled_agents()
    test_active_consensus_rule_math()
    test_idle_penalty_exempt_when_two_open()
    test_compute_panel_matrix_is_92_wide_on_real_prices()
    test_day_runner_consensus_idle_and_scale_in_exempt()
    test_real_data_active_consensus_scan()
    print("test_signal_majority OK (active consensus + real prices)")
