"""Live day board JSON for SIGON training HUD.

Writes artifacts/llm_curriculum/day_board.json with per-instance emoji rows
so Iron Man HUD / Colab can refresh without parsing logs.

CHANGE LOG:
- 2026-07-25  created — WHY: SIGON live emoji board (gpu_train write_day_board).
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timezone
from typing import Any

from core.configs import path as rpath

# ⚪ pending / warm  🟢 cleared (hit target, no breach)  🔴 breach  🟡 hit miss (not clear)  🔵 in-progress-ish
EMOJI = {
    "pending": "⚪",
    "clear": "🟢",
    "breach": "🔴",
    "miss": "🟡",
    "active": "🔵",
}


def _board_path() -> str:
    return rpath("artifacts", "llm_curriculum", "day_board.json")


def classify_row(pnl: float, target: float, risk: float, breached: bool) -> str:
    if breached:
        return "breach"
    if pnl >= target:
        return "clear"
    if pnl > 0:
        return "miss"
    return "pending"


def write_day_board(
    day_pnl,
    targets,
    risks,
    breached=None,
    symbols=None,
    clear_rate: float | None = None,
    breach_rate: float | None = None,
    row: int = 0,
    obs_dim: int | None = None,
    extra: dict | None = None,
    path: str | None = None,
) -> str:
    """Write live board. Arrays are 1-d length N (batch slice)."""
    import numpy as np

    pnl = np.asarray(day_pnl, dtype=np.float64).reshape(-1)
    tg = np.asarray(targets, dtype=np.float64).reshape(-1)
    rk = np.asarray(risks, dtype=np.float64).reshape(-1)
    n = int(pnl.shape[0])
    if breached is None:
        br = np.zeros(n, dtype=bool)
    else:
        br = np.asarray(breached, dtype=bool).reshape(-1)
        if br.shape[0] != n:
            br = np.resize(br, n)

    rows = []
    n_clear = n_breach = n_miss = 0
    for i in range(n):
        status = classify_row(float(pnl[i]), float(tg[i]), float(rk[i]), bool(br[i]))
        if status == "clear":
            n_clear += 1
        elif status == "breach":
            n_breach += 1
        elif status == "miss":
            n_miss += 1
        sym = None
        if symbols is not None and i < len(symbols):
            sym = symbols[i]
        rows.append({
            "i": i,
            "emoji": EMOJI.get(status, "⚪"),
            "status": status,
            "pnl": round(float(pnl[i]), 3),
            "target": round(float(tg[i]), 3),
            "risk": round(float(rk[i]), 3),
            "breached": bool(br[i]),
            "symbol": sym,
        })

    payload: dict[str, Any] = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "unix": time.time(),
        "n": n,
        "counts": {
            "clear": n_clear,
            "breach": n_breach,
            "miss": n_miss,
            "pending": n - n_clear - n_breach - n_miss,
        },
        "clear_rate": clear_rate,
        "breach_rate": breach_rate,
        "row": int(row),
        "obs_dim": obs_dim,
        "legend": EMOJI,
        "rows": rows,
    }
    if extra:
        payload["extra"] = extra

    out = path or _board_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    tmp = out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, out)
    return out
