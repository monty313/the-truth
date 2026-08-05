"""Tests for Label Contract V1 pure labels + replay schema asserts.

CHANGE LOG:
- 2026-07-31  Principle 9 — WHY: freeze explanation schema before attention work.

Run as script (same as other lineage tests; avoids tests/lineages package shadow):
  python tests/lineages/adaptive_rl_brain_7_31_26/test_label_contract_v1.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
CODE = os.path.join(ROOT, "code")
if CODE not in sys.path:
    sys.path.insert(0, CODE)

# If pytest already loaded the *test* package as lineages.adaptive_*, drop it.
_shadow = "lineages.adaptive_rl_brain_7_31_26"
if _shadow in sys.modules:
    mod = sys.modules[_shadow]
    f = getattr(mod, "__file__", "") or ""
    if f.replace("\\", "/").find("/tests/lineages/") >= 0:
        for k in list(sys.modules):
            if k == "lineages" or k.startswith("lineages."):
                # only drop the shadowed adaptive package branch
                if k == _shadow or k.startswith(_shadow + "."):
                    del sys.modules[k]

_ROOT = Path(ROOT)
_LINEAGE = _ROOT / "lineages" / "adaptive_rl_brain_7_31_26"
_OUT = _LINEAGE / "checkpoints" / "honest_gate"


def test_pure_label_functions():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.label_contract_v1 import (
        POINT_SIZE,
        alignment_label,
        cci_state_label,
        channel_slope_label,
        day_activity_class_label,
        market_condition_label,
        momentum_velocity_change_label,
        rsi_state_label,
    )

    assert alignment_label("bullish", "bullish") == "aligned"
    assert alignment_label("bullish", "bearish") == "conflicting"
    assert alignment_label("bullish", "neutral") == "neutral"
    assert alignment_label("unknown", "bullish") == "unknown"

    assert channel_slope_label(1.0, 1.0) == "flat"
    assert channel_slope_label(1.0 + POINT_SIZE, 1.0) == "rising"
    assert channel_slope_label(1.0 - POINT_SIZE, 1.0) == "falling"

    assert cci_state_label(120.0, 110.0, None) == "extended_high"
    assert cci_state_label(-120.0, -110.0, None) == "extended_low"
    assert cci_state_label(10.0, 5.0, 8.0) == "strengthening"
    assert cci_state_label(10.0, 5.0, 12.0) == "weakening"

    assert rsi_state_label(80.0, 75.0, None) == "extended_high"
    assert rsi_state_label(20.0, 25.0, None) == "extended_low"

    assert momentum_velocity_change_label("strong", "weak") == "strengthening"
    assert momentum_velocity_change_label("weak", "strong") == "weakening"
    assert momentum_velocity_change_label("medium", "medium") == "flat"

    assert (
        market_condition_label(
            alignment="aligned",
            htf_trend_dir="bullish",
            htf_trend_strength="strong",
            channel_position="above",
            channel_slope="rising",
        )
        == "trend"
    )
    assert (
        day_activity_class_label(
            cleared=False, breached=True, n_entries=0, reversal_count=0
        )
        == "breached"
    )
    assert (
        day_activity_class_label(
            cleared=True, breached=False, n_entries=2, reversal_count=0
        )
        == "cleared"
    )
    assert (
        day_activity_class_label(
            cleared=False, breached=False, n_entries=2, reversal_count=1
        )
        == "miss_low_activity"
    )
    assert (
        day_activity_class_label(
            cleared=False, breached=False, n_entries=8, reversal_count=3
        )
        == "miss_high_activity"
    )


def test_schema_field_lists_stable():
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.label_contract_v1 import (
        DECISION_FIELD_NAMES,
        EOD_FIELD_NAMES,
        LABEL_CONTRACT_VERSION,
        decision_columns,
        eod_columns,
    )

    assert LABEL_CONTRACT_VERSION == "label_contract_v1"
    assert "htf_trend_dir" in DECISION_FIELD_NAMES
    assert "day_activity_class" in EOD_FIELD_NAMES
    assert "future_bar_used" in DECISION_FIELD_NAMES
    assert "thin_liquidity_flag" not in DECISION_FIELD_NAMES  # NOT AVAILABLE
    assert decision_columns() == list(DECISION_FIELD_NAMES)
    assert eod_columns() == list(EOD_FIELD_NAMES)


def test_replay_smoke_1d():
    """1+1 day smoke: schema, determinism, no leak, no PROVEN."""
    from lineages.adaptive_rl_brain_7_31_26.honest_gate.run_label_replay import run

    summary = run(n_days=1, target=1.0, risk=2.0, write=False)
    assert summary["verdict"] == "PASS", summary.get("errors")
    assert summary["deterministic"] is True
    assert summary["proven_touched"] is False
    assert summary["shell_changed"] is False


def test_replay_artifacts_if_present():
    practice_path = _OUT / "replay_practice_5d_v1.json"
    forward_path = _OUT / "replay_forward_5d_v1.json"
    if not (practice_path.is_file() and forward_path.is_file()):
        return
    p = json.loads(practice_path.read_text(encoding="utf-8"))
    f = json.loads(forward_path.read_text(encoding="utf-8"))
    assert p["decision_columns"] == f["decision_columns"]
    assert p["label_contract_version"] == "label_contract_v1"
    assert all(r.get("future_bar_used") is False for r in p["decision_rows"])
    assert all(r.get("future_bar_used") is False for r in f["decision_rows"])
    assert set(p["dates"]).isdisjoint(set(f["dates"]))


if __name__ == "__main__":
    test_pure_label_functions()
    print("test_pure_label_functions OK")
    test_schema_field_lists_stable()
    print("test_schema_field_lists_stable OK")
    test_replay_smoke_1d()
    print("test_replay_smoke_1d OK")
    test_replay_artifacts_if_present()
    print("test_replay_artifacts_if_present OK")
    print("ALL test_label_contract_v1 OK")
