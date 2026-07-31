"""Phase 2 Slice 2: live structure + classify end-to-end."""
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

from lineages.adaptive_rl_brain_7_31_26.perception.classify import make_mindless_inputs
from lineages.adaptive_rl_brain_7_31_26.perception.pipeline import (
    assess_from_directions,
    assess_trade,
    direction_from_single_tf_ohlc,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import (
    Direction,
    TradeTag,
    VelocityStrength,
)

B, R, N = Direction.BULL, Direction.BEAR, Direction.NEUTRAL


def _ohlc_trend(n: int = 300, *, direction: int = 1) -> pd.DataFrame:
    close = 100.0 if direction > 0 else 500.0
    closes = []
    for i in range(n):
        thrust = (i % 10) < 7
        close += direction * (1.5 if thrust else -0.6)
        closes.append(close)
    c = np.asarray(closes, dtype=float)
    return pd.DataFrame({
        "open": c - direction * 0.2,
        "high": c + 1.0,
        "low": c - 1.0,
        "close": c,
    })


def test_with_vector_from_directions():
    a = assess_from_directions(
        B, B, B,
        make_mindless_inputs(B),  # wall not armed (with higher)
        higher_velocity=VelocityStrength.STRONG,
        lower_velocity=VelocityStrength.STRONG,
    )
    assert a.classification.tag == TradeTag.WITH_VECTOR
    assert a.structure.pullback is False
    assert a.structure.scale_conflict is False


def test_qualified_macro_pullback_from_directions():
    # higher bull, lower bear, trade with higher → MACRO
    a = assess_from_directions(
        B, B, R,
        make_mindless_inputs(B),
        higher_velocity=VelocityStrength.STRONG,
        lower_velocity=VelocityStrength.MEDIUM,
    )
    assert a.structure.pullback is True
    assert a.classification.tag == TradeTag.QUALIFIED_MACRO
    assert a.classification.mindless is False


def test_qualified_micro_wall_pass_from_directions():
    # trade against higher, wall all true → MICRO
    m = make_mindless_inputs(R, turned=True, velocity_confirms=True, higher_weakening=True)
    a = assess_from_directions(
        R, B, R, m,
        higher_velocity=VelocityStrength.STRONG,
        lower_velocity=VelocityStrength.STRONG,
    )
    assert a.classification.tag == TradeTag.QUALIFIED_MICRO
    assert a.classification.mindless is False


def test_mindless_wall_from_directions():
    m = make_mindless_inputs(R, turned=False, velocity_confirms=False, higher_weakening=False)
    a = assess_from_directions(
        R, B, R, m,
        higher_velocity=VelocityStrength.STRONG,
        lower_velocity=VelocityStrength.WEAK,
    )
    assert a.classification.tag == TradeTag.MINDLESS
    assert a.classification.mindless is True


def test_live_ohlc_pipeline_produces_classification():
    conf = _ohlc_trend(300, direction=1)
    entry = _ohlc_trend(300, direction=1)
    a = assess_trade(B, conf, conf.copy(), entry, set_key="official:1")
    assert a.classification.tag in TradeTag
    assert isinstance(a.higher.direction, Direction)
    assert isinstance(a.lower_direction, Direction)
    assert isinstance(a.structure.pullback, bool)


def test_live_pullback_shape_when_entry_opposes_conf():
    conf = _ohlc_trend(300, direction=1)
    entry = _ohlc_trend(300, direction=-1)
    # trade with higher bull
    a = assess_trade(
        B, conf, conf.copy(), entry,
        set_key="official:2",
        mindless=make_mindless_inputs(B),
    )
    # If higher clear bull and lower clear bear → pullback + MACRO when trade B
    if a.higher.direction == B and a.lower_direction == R:
        assert a.structure.pullback is True
        assert a.classification.tag == TradeTag.QUALIFIED_MACRO


def test_entry_direction_helper_runs():
    up = _ohlc_trend(200, direction=1)
    d = direction_from_single_tf_ohlc(up)
    assert d in (B, R, N)


if __name__ == "__main__":
    test_with_vector_from_directions()
    test_qualified_macro_pullback_from_directions()
    test_qualified_micro_wall_pass_from_directions()
    test_mindless_wall_from_directions()
    test_live_ohlc_pipeline_produces_classification()
    test_live_pullback_shape_when_entry_opposes_conf()
    test_entry_direction_helper_runs()
    print("test_pipeline OK")
