"""Signal-agent forward accuracy (trigger TF, horizons 5/10/20).

CHANGE LOG:
- 2026-07-25  index.json + priority stubs 0-4/80-83 + update_from_scoreboard — WHY: SIGON board.
- 2026-07-25  created — WHY: SIGON signal_accuracy for slots bus.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.configs import path as rpath, load as load_cfg


def accuracy_dir() -> str:
    try:
        schema = load_cfg("signal_accuracy_schema") or {}
        rel = schema.get("output_dir", "artifacts/signal_accuracy")
        if rel.startswith("artifacts"):
            d = rpath(*rel.split("/"))
        else:
            d = rpath("artifacts", "signal_accuracy")
    except Exception:
        d = rpath("artifacts", "signal_accuracy")
    os.makedirs(d, exist_ok=True)
    return d


def _schema() -> dict:
    try:
        return load_cfg("signal_accuracy_schema") or {}
    except Exception:
        return {
            "default_horizons_bars": [5, 10, 20],
            "priority_slots": [0, 1, 2, 3, 4, 80, 81, 82, 83],
        }


def write_placeholder_accuracy(n_slots: int = 500, path: str | None = None) -> str:
    """Write index.json + per-slot stubs (priority 0–4 and 80–83)."""
    sch = _schema()
    horizons = list(sch.get("default_horizons_bars") or [5, 10, 20])
    priority = list(sch.get("priority_slots") or [0, 1, 2, 3, 4, 80, 81, 82, 83])
    d = accuracy_dir()
    slots_meta = {}
    for i in priority:
        stub = {
            "slot": i,
            "status": "stub",
            "score_on": sch.get("score_on", "trigger_tf_only"),
            "horizons": horizons,
            "hit_5": None,
            "hit_10": None,
            "hit_20": None,
            "n_fires": 0,
            "note": "Run scripts/score_signal_accuracy.py for real rates.",
        }
        if i in (80, 81, 82, 83):
            stub["highlight"] = {
                80: "agree_seA_r2A ~75%@10 (PART4)",
                81: "agree_seB_r2B_epB ~70-72%",
                82: "agree_2of_top4 ~76/71%",
                83: "agree_seA_r2A_atr ~78-81%",
            }.get(i)
        slots_meta[str(i)] = stub
        with open(os.path.join(d, f"slot_{i:03d}.json"), "w", encoding="utf-8") as f:
            json.dump(stub, f, indent=2)

    index = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "placeholder",
        "n_slots": n_slots,
        "horizons": horizons,
        "score_on": sch.get("score_on", "trigger_tf_only"),
        "priority_slots": priority,
        "slots": slots_meta,
        "note": "Accuracy is relative to each signal's TRIGGER (LTF) timeframe at 5/10/20 bars.",
    }
    out = path or os.path.join(d, sch.get("index_file", "index.json"))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    # keep legacy name for older HUD
    with open(os.path.join(d, "placeholder.json"), "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return out


def score_slots_on_close(
    close: np.ndarray,
    signals: dict[int, np.ndarray],
    horizons: list[int] | None = None,
    min_fires: int = 10,
) -> dict[str, Any]:
    """When slot fires +1/-1, does forward close move agree? Bars = trigger TF grid."""
    sch = _schema()
    horizons = horizons or list(sch.get("default_horizons_bars") or [5, 10, 20])
    close = np.asarray(close, dtype=np.float64)
    T = len(close)
    rows = []
    for idx, sig in sorted(signals.items()):
        s = np.asarray(sig, dtype=np.float32).reshape(-1)
        if s.shape[0] != T:
            s = np.resize(s, T)
        fires = np.where(s != 0)[0]
        n = int(len(fires))
        rec: dict[str, Any] = {
            "slot": int(idx),
            "n_fires": n,
            "score_on": "trigger_tf_only",
        }
        for h in horizons:
            hits = []
            for i in fires:
                j = i + h
                if j >= T or not np.isfinite(close[i]) or close[i] == 0:
                    continue
                r = (close[j] - close[i]) / close[i]
                hits.append(1.0 if s[i] * r > 0 else 0.0)
            key = f"hit_{h}"
            rec[key] = round(float(np.mean(hits)), 4) if hits and len(hits) >= min_fires else None
            rec[f"n_{h}"] = len(hits)
        rows.append(rec)
    return {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "scored",
        "horizons": horizons,
        "score_on": "trigger_tf_only",
        "min_fires": min_fires,
        "rows": rows,
    }


def update_from_scoreboard(rows: list[dict], name: str = "index.json") -> str:
    """Merge live/scoreboard hit rows into index + per-slot files (for later live hits)."""
    d = accuracy_dir()
    sch = _schema()
    horizons = list(sch.get("default_horizons_bars") or [5, 10, 20])
    index_path = os.path.join(d, name)
    if os.path.isfile(index_path):
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"slots": {}, "horizons": horizons}
    slots = index.setdefault("slots", {})
    for r in rows:
        sid = str(int(r["slot"]))
        entry = slots.get(sid, {"slot": int(r["slot"])})
        entry.update(r)
        entry["status"] = "scored"
        entry["updated_at"] = datetime.now(timezone.utc).isoformat()
        slots[sid] = entry
        with open(os.path.join(d, f"slot_{int(r['slot']):03d}.json"), "w", encoding="utf-8") as f:
            json.dump(entry, f, indent=2)
    index["updated_at"] = datetime.now(timezone.utc).isoformat()
    index["status"] = "live"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2)
    return index_path


def write_accuracy_report(report: dict, name: str = "latest.json") -> str:
    out = os.path.join(accuracy_dir(), name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    if report.get("rows"):
        update_from_scoreboard(report["rows"])
    return out
