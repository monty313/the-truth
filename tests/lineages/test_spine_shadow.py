"""Unit/integration tests for Spine Shadow shipped code (real functions)."""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIN = os.path.dirname(os.path.dirname(_HERE))  # the-truth root when tests/lineages
# file is tests/lineages/test_spine_shadow.py → parents: lineages, tests, the-truth
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
_LINEAGE = os.path.join(_ROOT, "lineages", "adaptive_rl_brain_7_31_26")
for _p in (_ROOT, _CODE, _LINEAGE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import (
    DaySpine,
    SpineEvent,
    classify_spine_error,
    compile_spine_from_soul,
    size_bucket_for,
    spine_to_plan,
)
from lineages.adaptive_rl_brain_7_31_26.mark_aligned_decode import mark_force_gate_action
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_BUY, ACTION_HOLD, ACTION_SELL


def _sample_mark_plan() -> dict:
    # sparse plan: HOLD everywhere, BUY at 1020, BUY add at 1170
    indices = list(range(720, 1400, 25))
    plan = {int(t): ACTION_HOLD for t in indices}
    plan[1020] = ACTION_BUY
    plan[1170] = ACTION_BUY
    return {
        "source": "soul_plan",
        "mode": "entry_plus_add",
        "cleared": True,
        "breached": False,
        "banked": True,
        "risk_use_frac": 0.35,
        "per_trade_cap_pct": 0.25,
        "plan": plan,
        "t1": 1020,
        "t2": 1170,
        "side": "BUY",
        "n_entries": 2,
        "n_adds": 0,
    }


def test_compile_spine_has_fire_and_wait():
    mark = _sample_mark_plan()
    spine = compile_spine_from_soul("2026-02-25", 1.5, 3.0, mark)
    kinds = [e.kind for e in spine.events]
    assert "fire" in kinds
    assert "wait_loaded" in kinds
    assert "add" in kinds
    assert spine.t1 == 1020
    assert spine.t2 == 1170
    assert spine.side == "BUY"
    fires = [e for e in spine.events if e.kind == "fire"]
    assert fires[0].t == 1020
    assert fires[0].size_bucket == size_bucket_for(0.35, 0.25)


def test_spine_plan_round_trip_preserves_dirs():
    mark = _sample_mark_plan()
    spine = compile_spine_from_soul("2026-02-25", 1.5, 3.0, mark)
    plan2 = spine_to_plan(spine)
    # all non-HOLD in original must match
    orig = mark["plan"]
    for t, a in orig.items():
        if int(a) != ACTION_HOLD:
            assert int(plan2[int(t)]) == int(a), f"bar {t} lost"
    # fire bars present
    assert plan2[1020] == ACTION_BUY
    assert plan2[1170] == ACTION_BUY


def test_spine_to_dict_from_dict_round_trip():
    mark = _sample_mark_plan()
    spine = compile_spine_from_soul("2026-03-01", 2.0, 3.0, mark)
    raw = spine.to_dict()
    spine2 = DaySpine.from_dict(raw)
    assert spine2.day == "2026-03-01"
    assert spine2.t1 == 1020
    assert len(spine2.events) == len(spine.events)
    p1 = spine_to_plan(spine)
    p2 = spine_to_plan(spine2)
    for t in (1020, 1170):
        assert p1[t] == p2[t] == ACTION_BUY


def test_classify_false_hold_and_false_fire():
    mark = _sample_mark_plan()
    spine = compile_spine_from_soul("2026-02-25", 1.5, 3.0, mark)
    assert (
        classify_spine_error(
            spine=spine,
            policy_fire_ts=[],
            policy_n_entries=0,
            policy_award=False,
            policy_breached=False,
        )
        == "false_hold"
    )
    assert (
        classify_spine_error(
            spine=spine,
            policy_fire_ts=[800, 900, 1000, 1100],
            policy_n_entries=5,
            policy_award=False,
            policy_breached=False,
        )
        == "false_fire"
    )
    assert (
        classify_spine_error(
            spine=spine,
            policy_fire_ts=[1200],
            policy_n_entries=1,
            policy_award=False,
            policy_breached=False,
        )
        == "late_entry"
    )


def test_force_gate_wraps_still_on_score_path():
    """Static law: force-gate still zeros against HTF (pt5)."""
    # against force: BUY when force is strong short
    a = mark_force_gate_action(
        ACTION_BUY,
        side=None,
        equity_pct=0.0,
        risk_pct=3.0,
        force_dir=-0.9,
        m_conf=0.8,
        regime="trend",
        recommended=ACTION_HOLD,
    )
    assert a == ACTION_HOLD


def test_size_bucket_nearest():
    assert size_bucket_for(0.25, 0.20) == "micro"
    assert size_bucket_for(0.50, 0.35) == "std"
    assert size_bucket_for(1.0, 0.70) == "max"


def test_compile_from_real_oracle_cache_if_present():
    cache_path = os.path.join(
        _ROOT,
        "lineages",
        "adaptive_rl_brain_7_31_26",
        "checkpoints",
        "fable_50d_match",
        "MARK_ORACLE_CACHE__50d.json",
    )
    if not os.path.isfile(cache_path):
        pytest.skip("oracle cache missing")
    raw = json.load(open(cache_path, encoding="utf-8"))
    # pick first soul_plan with plan
    item = None
    key = None
    for k, v in raw.items():
        if v.get("source") == "soul_plan" and v.get("plan"):
            item = v
            key = k
            break
    assert item is not None
    date, t, r = key.split("|")
    mark = dict(item)
    mark["plan"] = {int(a): int(b) for a, b in mark["plan"].items()}
    spine = compile_spine_from_soul(date, float(t), float(r), mark)
    plan = spine_to_plan(spine)
    # at least one directional bar
    assert any(int(a) != ACTION_HOLD for a in plan.values())
    assert any(e.kind == "fire" for e in spine.events)
