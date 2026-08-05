"""Phase A thrash limits on DayRunner (lineage only)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CODE = os.path.join(ROOT, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from lineages.adaptive_rl_brain_7_31_26.curriculum_data import thrust_m1_day
from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    DID_NOTHING_EOD_PENALTY,
    MAX_OPEN_UNITS,
)


def _runner() -> DayRunner:
    m1 = thrust_m1_day(n_bars=2000, direction=1, seed=3)
    return DayRunner(
        m1,
        decide_every=25,
        use_signal_majority=False,
        max_open_units=3,
        reverse_cooldown_bars=100,
    )


def test_max_scale_ins_blocked():
    r = _runner()
    idxs = r.decision_indices()[:20]
    assert len(idxs) >= 6
    # enter buy then keep buying — n_open must cap at max_open_units
    for t in idxs[:6]:
        r.step(t, ACTION_BUY)
    assert r.n_open <= MAX_OPEN_UNITS
    assert r.n_open == 3
    assert r.n_scale_blocks >= 1


def test_reverse_sets_cooldown_and_blocks_flip():
    r = _runner()
    idxs = r.decision_indices()[:30]
    t0 = idxs[0]
    r.step(t0, ACTION_BUY)
    assert r.position is not None
    # reverse to sell
    t1 = idxs[1]
    s = r.step(t1, ACTION_SELL)
    assert s.info.get("reverse") is True
    assert "flip_flop_penalty" in s.info
    assert r.cooldown_until_t > t1
    # immediate reverse back to buy should be coerced to HOLD
    t2 = idxs[2]
    assert t2 < r.cooldown_until_t
    s2 = r.step(t2, ACTION_BUY)
    assert s2.info.get("cooldown_block") is True
    assert s2.action == ACTION_HOLD
    assert r.n_cooldown_blocks >= 1


def test_hold_still_works():
    r = _runner()
    t = r.decision_indices()[0]
    s = r.step(t, ACTION_HOLD)
    assert s.action == ACTION_HOLD
    assert r.n_entries == 0


def test_eod_did_nothing_via_day_runner():
    """Shipped DayRunner.end_day applies did-nothing wall when never entered."""
    r = _runner()
    for t in r.decision_indices()[:8]:
        r.step(t, ACTION_HOLD)
    assert r.n_entries == 0
    assert abs(r.realized) < 1e-9
    eod = r.end_day()
    assert eod == DID_NOTHING_EOD_PENALTY
    # once only
    assert r.end_day() == 0.0


if __name__ == "__main__":
    test_max_scale_ins_blocked()
    test_reverse_sets_cooldown_and_blocks_flip()
    test_hold_still_works()
    test_eod_did_nothing_via_day_runner()
    print("test_thrash OK")
