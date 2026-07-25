"""Per-signal accuracy vs trigger (LTF) timeframe — SIGON.

CHANGE LOG:
- 2026-07-25  created — WHY: each signal agent reports hit rate on 5/10/20 LTF bars.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import numpy as np

from core.configs import path as rpath, load as load_cfg


def _horizons():
    try:
        sch = load_cfg("signal_accuracy_schema") or {}
        return list(sch.get("default_horizons_bars") or [5, 10, 20])
    except Exception:
        return [5, 10, 20]


def write_placeholder_accuracy(n_slots: int = 500) -> str:
    """Write schema-valid stubs so HUD/CMO can read folder before full scoreboard runs."""
    out_dir = rpath("artifacts", "signal_accuracy")
    os.makedirs(out_dir, exist_ok=True)
    horizons = _horizons()
    index = []
    for slot in range(n_slots):
        row = {
            "slot": slot,
            "name": "sig_%03d" % slot,
            "trigger_tf": "ltf",
            "horizons": {str(h): {"n": 0, "hit_pct": None} for h in horizons},
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "note": "stub — run scripts/score_signal_accuracy.py for live hits",
        }
        path = os.path.join(out_dir, "sig_%03d.json" % slot)
        # only write a few filled slots to avoid 500 tiny files on every train tick
        if slot < 5 or slot in (80, 81, 82, 83):
            with open(path, "w", encoding="utf-8") as f:
                json.dump(row, f, indent=2)
        index.append({"slot": slot, "name": row["name"]})
    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"slots": index, "horizons": horizons, "updated_at": datetime.now(timezone.utc).isoformat()}, f, indent=2)
    return out_dir


def update_from_scoreboard(rows: list[dict]) -> str:
    """rows: [{slot, name, trigger_tf, horizons: {5: {n, hit_pct}, ...}}]"""
    out_dir = rpath("artifacts", "signal_accuracy")
    os.makedirs(out_dir, exist_ok=True)
    for row in rows:
        slot = int(row["slot"])
        path = os.path.join(out_dir, "sig_%03d.json" % slot)
        row = dict(row)
        row["updated_at"] = datetime.now(timezone.utc).isoformat()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
    return out_dir
