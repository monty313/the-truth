"""Tests for multi-pair consistency helpers (shipped path).

CHANGE LOG:
- 2026-07-31  ten-pair tests — WHY: pair list load + clear counting must drive
  real equity_day / score helpers, not reimplemented oracles.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[3]
_LINEAGE = _ROOT / "lineages" / "adaptive_rl_brain_7_31_26"
_PAIRS = _LINEAGE / "ten_pairs.json"


class _Skip(Exception):
    """Local skip when optional artifacts are missing (no pytest required)."""


def test_ten_pairs_file_has_exactly_10_distinct_pairs():
    assert _PAIRS.is_file(), f"missing {_PAIRS}"
    cfg = json.loads(_PAIRS.read_text(encoding="utf-8"))
    pairs = cfg["pairs"]
    assert len(pairs) == 10
    seen = set()
    for p in pairs:
        key = (float(p["target_pct"]), float(p["risk_pct"]))
        assert key not in seen
        seen.add(key)
        assert p["target_pct"] > 0
        assert p["risk_pct"] > 0


def test_load_pairs_config_matches_file():
    from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import load_pairs_config

    cfg = load_pairs_config(str(_PAIRS))
    assert len(cfg["pairs"]) == 10
    assert int(cfg["seed"]) == 42
    assert int(cfg["min_clear_days_per_pair"]) == 30


def test_score_pair_on_days_real_path_smoke():
    """Drive real GoalEquityDay scorer on a few real days (not a fake counter)."""
    from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
    from lineages.adaptive_rl_brain_7_31_26.score_ten_pairs import score_pair_on_days

    days = load_calendar_days()
    assert len(days) >= 5
    sample = days[:3]
    rep = score_pair_on_days(
        sample,
        target=1.0,
        risk=2.5,
        use_heuristic=True,
        decide_every=25,
    )
    assert rep["n_days"] == 3
    assert 0 <= rep["cleared"] <= 3
    assert 0 <= rep["breached"] <= 3
    assert len(rep["day_rows"]) == 3
    for row in rep["day_rows"]:
        assert "date" in row and "pnl_pct" in row
        assert "cleared" in row and "breached" in row
        # clear implies not breach on that day
        if row["cleared"]:
            assert row["breached"] is False


def test_split_practice_forward_chronological():
    from lineages.adaptive_rl_brain_7_31_26.equity_day import (
        load_calendar_days,
        split_practice_forward,
    )

    days = load_calendar_days()
    practice, forward = split_practice_forward(days, practice_n=50)
    assert len(practice) == 50
    assert len(forward) == len(days) - 50
    assert practice[0][0] < practice[-1][0]
    assert practice[-1][0] < forward[0][0]


def test_checkpoint_dials_sidecar_if_present():
    ckpt = _LINEAGE / "checkpoints" / "multi_pair_consistent_v1.pt"
    if not ckpt.is_file():
        raise _Skip("checkpoint not built yet")
    import torch

    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    assert "state_dict" in blob
    assert "dials" in blob
    assert blob["dials"].get("decode") == "heuristic"
    assert blob.get("proven_touched") is False


if __name__ == "__main__":
    import traceback

    failed = 0
    skipped = 0
    for name, fn in list(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            print("RUN", name, flush=True)
            fn()
            print("  OK", flush=True)
        except _Skip as e:
            skipped += 1
            print("  SKIP", e, flush=True)
        except Exception as e:
            failed += 1
            print("  FAIL", e, flush=True)
            traceback.print_exc()
    print("SUMMARY failed=%d skipped=%d" % (failed, skipped), flush=True)
    raise SystemExit(failed)
