"""Signal-agent majority panel for adaptive_rl_brain_7_31_26.

CHANGE LOG:
- 2026-07-31  majority idle penalty — WHY: if more than half of the signal
  agents agree on one side and the policy does nothing (flat HOLD), apply a
  clear per-step penalty. Lineage only; never touches PROVEN.
- 2026-07-31  full YAML panel — WHY: use ALL filled agents from
  configs/signal_slots.yaml (92 slots), not a short lightweight subset.
  Builds features via features.engine so MO/camillion/native kinds can vote.
- 2026-07-31  active consensus rule — WHY: ≥20 agents active AND >70% of those
  agree on one side → idle penalty unless ≥2 trades already open.
- 2026-07-31  soften consensus — WHY: ≥10 active AND ≥60% of those agree
  (e.g. 6 of 10). Fires more often on real data/raw prices.

Vote values: +1 buy / -1 sell / 0 flat (same as signals/encode).

Consensus (default):
  n_active >= 10
  and max(n_bull, n_bear) / n_active >= 0.60
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    MAJORITY_AGREE_FRAC,
    MAJORITY_MIN_ACTIVE,
    MAJORITY_MIN_OPEN_EXEMPT,
)

# Repo root = parents[2] from this file (lineages/<name>/signal_majority.py)
_LINEAGE_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _LINEAGE_DIR.parents[1]
# Official slot book (NOT code/configs — that path does not exist)
SIGNAL_SLOTS_YAML = _REPO_ROOT / "configs" / "signal_slots.yaml"


def resolve_slots_path(path: Path | str | None = None) -> Path:
    """Path to signal_slots.yaml (repo configs/)."""
    if path is None:
        return SIGNAL_SLOTS_YAML
    return Path(path)


def load_all_signal_agents(
    path: Path | str | None = None,
    *,
    only_enabled: bool = False,
) -> Dict[int, dict[str, Any]]:
    """Load every filled signal agent from signal_slots.yaml.

    Default only_enabled=False → all 92 filled slots (disabled ones still count
    in the panel; they vote 0). Pass only_enabled=True for the 91 enabled only.
    """
    from signals.encode import load_filled_slots

    cfg = resolve_slots_path(path)
    if not cfg.is_file():
        raise FileNotFoundError(
            f"signal_slots.yaml not found at {cfg}. "
            "Expected repo configs/signal_slots.yaml"
        )
    return load_filled_slots(cfg, only_enabled=only_enabled)


def build_signal_feature_frame(m1: pd.DataFrame) -> pd.DataFrame:
    """Build the feature frame signal agents need (sets, TFs, obs helpers)."""
    from features.engine import build_features

    F = m1.sort_index().copy()
    if "vol" not in F.columns and "volume" in F.columns:
        F["vol"] = F["volume"]
    if "vol" not in F.columns:
        F["vol"] = 1.0
    if "spread" not in F.columns:
        F["spread"] = 0.0
    return build_features(F)


@dataclass(frozen=True)
class MajoritySnapshot:
    """Consensus read at one bar across the full agent panel."""

    n_agents: int
    n_bull: int
    n_bear: int
    n_flat: int
    has_majority: bool
    direction: Direction  # BULL / BEAR / NEUTRAL
    frac_bull: float  # of full panel
    frac_bear: float
    n_active: int = 0  # non-zero votes
    agree_frac: float = 0.0  # max(bull,bear) / active (0 if none active)
    min_active: int = MAJORITY_MIN_ACTIVE
    agree_threshold: float = MAJORITY_AGREE_FRAC

    @property
    def majority_side(self) -> Optional[Direction]:
        if not self.has_majority:
            return None
        return self.direction


def majority_from_votes(
    votes: Sequence[float] | np.ndarray,
    *,
    min_active: int = MAJORITY_MIN_ACTIVE,
    agree_frac: float = MAJORITY_AGREE_FRAC,
    # legacy kwargs ignored (kept so old callers don't crash)
    threshold: float | None = None,
) -> MajoritySnapshot:
    """Active-agent consensus from one row of votes (+1 / -1 / 0).

    Fires when:
      - n_active >= min_active (default 10)
      - max(n_bull, n_bear) / n_active >= agree_frac (default 0.60)
        e.g. 6 of 10 active on the same side
    Full panel size is still reported; zeros do not count as active.
    """
    v = np.asarray(votes, dtype=float).reshape(-1)
    n = int(v.size)
    min_a = int(min_active)
    thr = float(agree_frac)
    if n <= 0:
        return MajoritySnapshot(
            n_agents=0,
            n_bull=0,
            n_bear=0,
            n_flat=0,
            has_majority=False,
            direction=Direction.NEUTRAL,
            frac_bull=0.0,
            frac_bear=0.0,
            n_active=0,
            agree_frac=0.0,
            min_active=min_a,
            agree_threshold=thr,
        )
    n_bull = int(np.sum(v > 0))
    n_bear = int(np.sum(v < 0))
    n_flat = n - n_bull - n_bear
    n_active = n_bull + n_bear
    if n_active > 0:
        top = max(n_bull, n_bear)
        af = float(top) / float(n_active)
    else:
        af = 0.0

    has = False
    direction = Direction.NEUTRAL
    # >= so "60% of 10" = 6/10 counts
    if n_active >= min_a and af + 1e-12 >= thr:
        if n_bull > n_bear:
            direction = Direction.BULL
            has = True
        elif n_bear > n_bull:
            direction = Direction.BEAR
            has = True
        # exact tie among sides → no consensus

    return MajoritySnapshot(
        n_agents=n,
        n_bull=n_bull,
        n_bear=n_bear,
        n_flat=n_flat,
        has_majority=has,
        direction=direction,
        frac_bull=float(n_bull) / float(n),
        frac_bear=float(n_bear) / float(n),
        n_active=n_active,
        agree_frac=af,
        min_active=min_a,
        agree_threshold=thr,
    )


def compute_panel_matrix(
    m1: pd.DataFrame,
    *,
    slots_path: Path | str | None = None,
    only_enabled: bool = False,
    feature_frame: pd.DataFrame | None = None,
) -> Tuple[np.ndarray, List[str]]:
    """Run ALL filled signal agents from YAML on M1.

    Returns
    -------
    matrix : (n_bars, n_agents) float32 in {-1, 0, +1}
    names  : list like \"000:mo_bread_and_butter_pull_set1\"
    """
    from signals.encode import compute_slot

    filled = load_all_signal_agents(slots_path, only_enabled=only_enabled)
    if not filled:
        return np.zeros((len(m1), 0), dtype=np.float32), []

    F = feature_frame if feature_frame is not None else build_signal_feature_frame(m1)
    # Align to m1 index length
    if len(F) != len(m1):
        F = F.reindex(m1.index).ffill().bfill()

    names: List[str] = []
    cols: List[np.ndarray] = []
    for idx in sorted(filled.keys()):
        spec = filled[idx]
        label = f"{int(idx):03d}:{spec.get('name') or spec.get('kind') or idx}"
        try:
            s = compute_slot(F, spec)
            arr = np.asarray(s.reindex(m1.index).fillna(0.0), dtype=np.float32)
            if arr.shape[0] != len(m1):
                arr = np.zeros(len(m1), dtype=np.float32)
        except Exception:
            arr = np.zeros(len(m1), dtype=np.float32)
        cols.append(np.sign(arr).astype(np.float32))
        names.append(label)

    mat = np.stack(cols, axis=1) if cols else np.zeros((len(m1), 0), dtype=np.float32)
    return mat, names


def majority_at(
    matrix: np.ndarray,
    t: int,
    *,
    min_active: int = MAJORITY_MIN_ACTIVE,
    agree_frac: float = MAJORITY_AGREE_FRAC,
    threshold: float | None = None,  # legacy ignored
) -> MajoritySnapshot:
    """Consensus snapshot at bar index t."""
    if matrix is None or matrix.size == 0:
        return majority_from_votes([], min_active=min_active, agree_frac=agree_frac)
    t = int(t)
    if t < 0 or t >= matrix.shape[0]:
        return majority_from_votes(
            np.zeros(matrix.shape[1], dtype=np.float32),
            min_active=min_active,
            agree_frac=agree_frac,
        )
    return majority_from_votes(
        matrix[t], min_active=min_active, agree_frac=agree_frac
    )


def majority_idle_penalty(
    snap: MajoritySnapshot,
    *,
    action_is_hold: bool,
    n_open: int = 0,
    min_open_exempt: int = MAJORITY_MIN_OPEN_EXEMPT,
    penalty: float,
    is_flat: bool | None = None,  # legacy
) -> float:
    """Per-step penalty when active consensus and bot is idle (<2 open)."""
    if not action_is_hold:
        return 0.0
    if not snap.has_majority:
        return 0.0
    if int(n_open) >= int(min_open_exempt):
        return 0.0
    return -abs(float(penalty))


def panel_summary(matrix: np.ndarray, names: Sequence[str]) -> Dict[str, Any]:
    """Small diagnostics for logs / reports."""
    n_agents = int(matrix.shape[1]) if matrix is not None and matrix.ndim == 2 else 0
    if n_agents == 0:
        return {"n_agents": 0, "agents_ever_nonzero": 0, "names_head": []}
    ever = int(np.any(matrix != 0, axis=0).sum())
    return {
        "n_agents": n_agents,
        "agents_ever_nonzero": ever,
        "names_head": list(names[:8]),
        "slots_yaml": str(SIGNAL_SLOTS_YAML),
    }
