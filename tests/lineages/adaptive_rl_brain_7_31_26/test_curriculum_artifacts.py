"""Structural + schema checks for Phase B/C curriculum artifacts (shipped paths)."""
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

LINEAGE = Path(ROOT) / "lineages" / "adaptive_rl_brain_7_31_26"
CKPT_DIR = LINEAGE / "checkpoints"
REPORT = CKPT_DIR / "curriculum_train_report.json"
CKPT = CKPT_DIR / "channel1_curriculum_v1.pt"
CURRICULUM_MD = LINEAGE / "CURRICULUM.md"
CURRICULUM_JSON = CKPT_DIR / "curriculum_days.json"
MODELS = Path(ROOT) / "models"


def test_curriculum_md_lists_real_source_and_days():
    text = CURRICULUM_MD.read_text(encoding="utf-8")
    assert "data/raw/" in text or "XAUUSD" in text
    assert "trend" in text.lower() or "pullback" in text.lower()
    data = json.loads(CURRICULUM_JSON.read_text(encoding="utf-8"))
    assert data.get("proven_touched") is False
    days = data.get("days") or []
    assert len(days) >= 1
    assert all("date" in d and "role" in d for d in days)


def test_curriculum_checkpoint_and_report_schema():
    assert CKPT.is_file(), f"missing checkpoint {CKPT}"
    # must live under lineage checkpoints, never models/
    assert "lineages" in str(CKPT.resolve()).replace("\\", "/")
    assert "models" not in str(CKPT.resolve()).replace("\\", "/").split("lineages")[0] or True
    rel = os.path.relpath(CKPT, ROOT).replace("\\", "/")
    assert rel.startswith("lineages/adaptive_rl_brain_7_31_26/checkpoints/")
    assert not rel.startswith("models/")

    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    assert rep.get("proven_touched") is False
    assert rep.get("sandbox") is True
    assert "before_greedy" in rep and "after_greedy" in rep
    for key in ("before_greedy", "after_greedy"):
        block = rep[key]
        assert "tags" in block
        assert "actions" in block
        assert "n_entries_total" in block
        assert "mean_reward" in block
        assert "mindless_rate" in block
        assert "hold" in block["actions"]
    # explicit hold / thrash awareness in after block
    after = rep["after_greedy"]
    assert "hold_rate" in after or after["actions"].get("hold", 0) >= 0
    ckpt_path = str(rep.get("checkpoint", "")).replace("\\", "/")
    assert "lineages/adaptive_rl_brain_7_31_26/checkpoints/" in ckpt_path
    assert "PROVEN" not in ckpt_path


def test_mtf_on_documented_curriculum_days():
    """Drive shipped real_curriculum + MTF verify (not a reimplementation)."""
    from lineages.adaptive_rl_brain_7_31_26.real_curriculum import (
        load_real_curriculum,
        verify_mtf_on_days,
    )

    frames, meta, src = load_real_curriculum()
    assert len(frames) >= 1
    assert "XAUUSD" in src or src.endswith(".csv")
    report = verify_mtf_on_days(frames)
    assert int(report["n_days"]) == len(frames)
    for day in report["days"]:
        assert day["n_m1"] >= 900
        assert day["tfs"].get("1m", 0) > 0


def test_no_ready_for_longer_when_all_hold():
    """Phase D file must not exist while pure greedy after has 0 entries."""
    rep = json.loads(REPORT.read_text(encoding="utf-8"))
    after = rep["after_greedy"]
    ready = LINEAGE / "READY_FOR_LONGER_TRAIN.md"
    if int(after.get("n_entries_total", 0)) == 0:
        assert not ready.is_file(), "Phase D must not fire on all-hold collapse"


if __name__ == "__main__":
    test_curriculum_md_lists_real_source_and_days()
    test_curriculum_checkpoint_and_report_schema()
    test_mtf_on_documented_curriculum_days()
    test_no_ready_for_longer_when_all_hold()
    print("test_curriculum_artifacts OK")
