"""Multi-TF resample for adaptive_rl_brain_7_31_26.

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 5 — WHY: M1 → Official/Sub TF bars; reuses
  data_io.loader.resample (no look-ahead). Parallel lineage only.

Lineage TF labels: 1m, 5m, 15m, 30m, 1h, 4h, 1d
Loader keys:       1min, 5min, 15min, 30min, 1h, 4h, 1d
"""
from __future__ import annotations

from typing import Dict, Iterable, Mapping, Tuple

import pandas as pd

from data_io.loader import resample as _loader_resample

# Canonical set of TFs this lineage needs.
LINEAGE_TFS: Tuple[str, ...] = ("1m", "5m", "15m", "30m", "1h", "4h", "1d")

# Map lineage short names → data_io.loader TF_RULE keys.
_LINEAGE_TO_LOADER: Dict[str, str] = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1d",
    # also accept loader names passthrough
    "1min": "1min",
    "5min": "5min",
    "15min": "15min",
    "30min": "30min",
}


def lineage_tf_to_loader(tf: str) -> str:
    key = str(tf).strip().lower()
    if key not in _LINEAGE_TO_LOADER:
        raise KeyError(f"unknown lineage TF {tf!r}; expected one of {LINEAGE_TFS}")
    return _LINEAGE_TO_LOADER[key]


def resample_lineage(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """Resample M1 OHLC to a lineage TF via data_io.loader.resample."""
    loader_tf = lineage_tf_to_loader(tf)
    out = _loader_resample(m1, loader_tf)
    # Ensure required columns for live_indicators
    for col in ("open", "high", "low", "close"):
        if col not in out.columns:
            raise ValueError(f"resample_lineage({tf}): missing {col}")
    return out


def build_mtf_pack(
    m1: pd.DataFrame,
    tfs: Iterable[str] | None = None,
) -> Dict[str, pd.DataFrame]:
    """Build {lineage_tf: OHLCV frame} for all requested TFs (default full set)."""
    want = tuple(tfs) if tfs is not None else LINEAGE_TFS
    pack: Dict[str, pd.DataFrame] = {}
    for tf in want:
        # normalize key to lineage short form when possible
        short = str(tf).strip().lower()
        if short.endswith("min") and short not in ("1min",):
            # 5min → 5m etc.
            if short in ("5min", "15min", "30min"):
                short = short.replace("min", "m")
        elif short == "1min":
            short = "1m"
        pack[short if short in LINEAGE_TFS else tf] = resample_lineage(m1, tf)
    return pack


def bar_asof(frame: pd.DataFrame, ts: pd.Timestamp) -> int:
    """Index of last bar with index <= ts; -1 if none."""
    if frame is None or len(frame) == 0:
        return -1
    idx = frame.index
    # pad if naive
    pos = idx.searchsorted(ts, side="right") - 1
    if pos < 0:
        return -1
    return int(pos)
