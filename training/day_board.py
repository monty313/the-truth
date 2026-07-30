"""Live day board JSON for Iron Man HUD — SIGON.

Writes artifacts/llm_curriculum/day_board.json
  emoji: clear 🟢 | near 🟡 | miss ⚪ | breach 🔴

CHANGE LOG:
- 2026-07-25  created — WHY: Monty wants live green/emoji target results while training.
"""
from __future__ import annotations
import json
import os
from datetime import datetime, timezone

import numpy as np

from core.configs import path as rpath


def classify_day(pnl: float, goal: float, breached: bool) -> str:
    if breached:
        return "breach"
    if pnl >= goal:
        return "clear"
    if pnl > 0:
        return "near"
    return "miss"


EMOJI = {"clear": "🟢", "near": "🟡", "miss": "⚪", "breach": "🔴"}


def write_day_board(
    pnls,
    goals,
    floors,
    breached=None,
    dates=None,
    symbols=None,
    clear_rate=None,
    breach_rate=None,
    row=None,
    obs_dim=None,
    extra=None,
) -> str:
    pnls = np.asarray(pnls, dtype=float).reshape(-1)
    goals = np.asarray(goals, dtype=float).reshape(-1)
    n = len(pnls)
    if breached is None:
        breached = np.zeros(n, dtype=bool)
    else:
        breached = np.asarray(breached, dtype=bool).reshape(-1)
    days = []
    for i in range(n):
        st = classify_day(float(pnls[i]), float(goals[i]), bool(breached[i]))
        days.append({
            "i": i,
            "status": st,
            "emoji": EMOJI[st],
            "pnl": round(float(pnls[i]), 4),
            "goal": round(float(goals[i]), 4),
            "floor": round(float(floors[i]), 4) if floors is not None else None,
            "date": (str(dates[i]) if dates is not None and i < len(dates) else None),
            "symbol": (symbols[i] if symbols is not None and i < len(symbols) else None),
        })
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "clear_rate": clear_rate,
        "breach_rate": breach_rate,
        "row": row,
        "obs_dim": obs_dim,
        "counts": {
            "clear": int(sum(1 for d in days if d["status"] == "clear")),
            "near": int(sum(1 for d in days if d["status"] == "near")),
            "miss": int(sum(1 for d in days if d["status"] == "miss")),
            "breach": int(sum(1 for d in days if d["status"] == "breach")),
            "n": n,
        },
        "days": days[:256],  # HUD-friendly cap
    }
    if extra:
        payload["extra"] = extra
    out_dir = rpath("artifacts", "llm_curriculum")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "day_board.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    # also champion snapshot for HUD header
    champ = {
        "clear_rate": clear_rate,
        "breach_rate": breach_rate,
        "row": row,
        "obs_dim": obs_dim,
        "updated_at": payload["updated_at"],
    }
    cp = rpath("models", "best_sigon_record.json")
    os.makedirs(os.path.dirname(cp), exist_ok=True)
    if clear_rate is not None:
        # only overwrite record if better clear and breach ok — caller decides; here write live view
        live = rpath("artifacts", "llm_curriculum", "champion_live.json")
        with open(live, "w", encoding="utf-8") as f:
            json.dump(champ, f, indent=2)
    return path
