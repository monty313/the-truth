"""Encode up to 500 strategy slots into observation columns obs::sig_000..sig_499.

Values: +1 buy, -1 sell, 0 empty/flat.

CHANGE LOG:
- 2026-07-25  created — WHY: Monty 500-slot advisor obs; unfilled slots stay 0.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

N_SLOTS = 500
ROOT = Path(__file__).resolve().parents[1]
SLOTS_CFG = ROOT / "configs" / "signal_slots.yaml"


def signal_column_names() -> list[str]:
    return [f"obs::sig_{i:03d}" for i in range(N_SLOTS)]


def load_filled_slots(path: Path | None = None) -> dict[int, dict[str, Any]]:
    path = path or SLOTS_CFG
    if not path.is_file():
        return {}
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    filled = raw.get("filled") or {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in filled.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < N_SLOTS and isinstance(v, dict) and v.get("enabled", True):
            out[idx] = v
    return out


def _series_pull(F: pd.DataFrame, set_name: str = "set1") -> pd.Series:
    buy = F.get(f"{set_name}::pull_buy")
    sell = F.get(f"{set_name}::pull_sell")
    if buy is None or sell is None:
        return pd.Series(0.0, index=F.index, dtype=np.float32)
    out = np.where(buy.to_numpy() > 0, 1.0, np.where(sell.to_numpy() > 0, -1.0, 0.0))
    return pd.Series(out, index=F.index, dtype=np.float32)


def _series_cont(F: pd.DataFrame, set_name: str = "set1") -> pd.Series:
    buy = F.get(f"{set_name}::cont_buy")
    sell = F.get(f"{set_name}::cont_sell")
    if buy is None or sell is None:
        return pd.Series(0.0, index=F.index, dtype=np.float32)
    out = np.where(buy.to_numpy() > 0, 1.0, np.where(sell.to_numpy() > 0, -1.0, 0.0))
    return pd.Series(out, index=F.index, dtype=np.float32)


def _series_zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


KIND_HANDLERS = {
    "pull_set1": lambda F: _series_pull(F, "set1"),
    "pull_set2": lambda F: _series_pull(F, "set2"),
    "pull_set3": lambda F: _series_pull(F, "set3"),
    "cont_set1": lambda F: _series_cont(F, "set1"),
    "cont_set2": lambda F: _series_cont(F, "set2"),
    "cont_set3": lambda F: _series_cont(F, "set3"),
    "zero": _series_zero,
}


def compute_slot(F: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    kind = (spec.get("kind") or "zero").strip()
    handler = KIND_HANDLERS.get(kind, _series_zero)
    return handler(F).astype(np.float32)


def append_signal_obs(F: pd.DataFrame, new: dict | None = None) -> dict:
    """Fill new[obs::sig_XXX] for all 500 slots."""
    if new is None:
        new = {}
    filled = load_filled_slots()
    n = len(F)
    zeros = np.zeros(n, dtype=np.float32)
    for i in range(N_SLOTS):
        col = f"obs::sig_{i:03d}"
        if i in filled:
            new[col] = compute_slot(F, filled[i]).to_numpy(dtype=np.float32)
        else:
            new[col] = zeros
    return new
