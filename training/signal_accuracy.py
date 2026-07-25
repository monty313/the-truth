"""Signal-agent forward accuracy (trigger TF, horizons 5/10/20).

CHANGE LOG:
- 2026-07-25  created — WHY: SIGON signal_accuracy board for slots 80–83 / 500 bus.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

import numpy as np

from core.configs import path as rpath


def accuracy_dir() -> str:
    d = rpath("artifacts", "signal_accuracy")
    os.makedirs(d, exist_ok=True)
    return d


def write_placeholder_accuracy(n_slots: int = 500, path: str | None = None) -> str:
    """Stub scoreboard so HUD / gpu_train always have a file before full score runs."""
    out = path or os.path.join(accuracy_dir(), "placeholder.json")
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "placeholder",
        "n_slots": n_slots,
        "horizons": [5, 10, 20],
        "note": "Run scripts/score_signal_accuracy.py for real 5/10/20 hit rates.",
        "slots": {
            str(i): {"fires": 0, "hit_5": None, "hit_10": None, "hit_20": None}
            for i in range(min(n_slots, 20))  # compact placeholder
        },
        "highlight": {
            "80": "agree_seA_r2A ~75%@10 (PART4)",
            "81": "agree_seB_r2B_epB ~70-72%",
            "82": "agree_2of_top4 ~76/71%",
            "83": "agree_seA_r2A_atr ~78-81%",
        },
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def score_slots_on_close(
    close: np.ndarray,
    signals: dict[int, np.ndarray],
    horizons: list[int] | None = None,
    min_fires: int = 10,
) -> dict[str, Any]:
    """When slot fires +1/-1, does forward close move agree?

    Parameters
    ----------
    close : (T,) prices
    signals : slot_index -> (T,) series in {-1,0,+1}
    horizons : bar horizons (default 5, 10, 20) on the scoring grid (usually M1)
    """
    horizons = horizons or [5, 10, 20]
    close = np.asarray(close, dtype=np.float64)
    T = len(close)
    rows = []
    for idx, sig in sorted(signals.items()):
        s = np.asarray(sig, dtype=np.float32).reshape(-1)
        if s.shape[0] != T:
            s = np.resize(s, T)
        fires = np.where(s != 0)[0]
        n = int(len(fires))
        rec: dict[str, Any] = {"slot": int(idx), "n_fires": n}
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
        "min_fires": min_fires,
        "rows": rows,
    }


def write_accuracy_report(report: dict, name: str = "latest.json") -> str:
    out = os.path.join(accuracy_dir(), name)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return out
