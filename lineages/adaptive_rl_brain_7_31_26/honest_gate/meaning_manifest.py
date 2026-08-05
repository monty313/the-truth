"""Deterministic meaning-version (eyes) manifest + hash gate.

CHANGE LOG:
- 2026-07-31  honest gate — WHY: silent indicator/TF/tag edits must not look
  like “market got hard.” Any eye change forces new practice/forward eval.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

from lineages.adaptive_rl_brain_7_31_26.data.mtf import LINEAGE_TFS
from lineages.adaptive_rl_brain_7_31_26.honest_gate.hashes import (
    file_sha256,
    sha256_obj,
)
from lineages.adaptive_rl_brain_7_31_26.perception.live_indicators import (
    CCI_FAST,
    CCI_SLOW,
    CHANNEL_N,
    CHANNEL_SHIFT,
    GROUP_KEYS,
    REF_SMA_N,
    REF_SMA_SHIFT,
    RSI_FAST,
    RSI_SLOW,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import (
    CHANNEL1_DIM,
    FEATURES_PER_SET,
    N_OFFICIAL,
    N_SUB,
)
from lineages.adaptive_rl_brain_7_31_26.perception.sets import (
    OFFICIAL_SETS,
    SUB_SETS,
)

_LINEAGE = Path(__file__).resolve().parents[1]
_REPO = _LINEAGE.parents[1]
FROZEN_MANIFEST_PATH = _LINEAGE / "checkpoints" / "honest_gate" / "meaning_manifest.json"

# Source files that define “eyes” — any content change must retag meaning_version.
MEANING_SOURCE_FILES = (
    "lineages/adaptive_rl_brain_7_31_26/perception/live_indicators.py",
    "lineages/adaptive_rl_brain_7_31_26/perception/structure.py",
    "lineages/adaptive_rl_brain_7_31_26/perception/confluence.py",
    "lineages/adaptive_rl_brain_7_31_26/perception/classify.py",
    "lineages/adaptive_rl_brain_7_31_26/perception/observation.py",
    "lineages/adaptive_rl_brain_7_31_26/perception/sets.py",
    "lineages/adaptive_rl_brain_7_31_26/data/mtf.py",
    "lineages/adaptive_rl_brain_7_31_26/day_runner.py",
    "lineages/adaptive_rl_brain_7_31_26/equity_day.py",
)


def _channel1_layout() -> list[dict[str, Any]]:
    slots: list[dict[str, Any]] = []
    # Official 1..4 × (direction, velocity, confluence_score)
    for i, s in enumerate(OFFICIAL_SETS):
        base = i * FEATURES_PER_SET
        slots.append({"index": base, "name": f"official_{s.set_id}_direction"})
        slots.append({"index": base + 1, "name": f"official_{s.set_id}_velocity"})
        slots.append({"index": base + 2, "name": f"official_{s.set_id}_confluence"})
    for j, s in enumerate(SUB_SETS):
        base = N_OFFICIAL * FEATURES_PER_SET + j * FEATURES_PER_SET
        slots.append({"index": base, "name": f"sub_{s.sub_id}_direction"})
        slots.append({"index": base + 1, "name": f"sub_{s.sub_id}_velocity"})
        slots.append({"index": base + 2, "name": f"sub_{s.sub_id}_confluence"})
    slots.append({"index": 27, "name": "pullback"})
    slots.append({"index": 28, "name": "scale_conflict"})
    slots.append({"index": 29, "name": "progress_to_goal"})
    slots.append({"index": 30, "name": "danger"})
    slots.append({"index": 31, "name": "session_phase"})
    return slots


def build_meaning_manifest() -> Dict[str, Any]:
    """Build live meaning manifest from code constants + source file hashes."""
    source_hashes: Dict[str, str] = {}
    for rel in MEANING_SOURCE_FILES:
        p = _REPO / rel
        source_hashes[rel] = file_sha256(p) if p.is_file() else "MISSING"

    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "track": "multi_pair_tutor",
        "not_tracks": ["PROVEN_champion", "channel1_rl_sandbox"],
        "indicators": {
            "cci_fast": CCI_FAST,
            "cci_slow": CCI_SLOW,
            "rsi_fast": RSI_FAST,
            "rsi_slow": RSI_SLOW,
            "ref_sma_n": REF_SMA_N,
            "ref_sma_shift": REF_SMA_SHIFT,
            "channel_n": CHANNEL_N,
            "channel_shift": CHANNEL_SHIFT,
            "group_keys": list(GROUP_KEYS),
            "entry_tf_votes": False,
            "dual_confirm_required": True,
        },
        "timeframe_stack": list(LINEAGE_TFS),
        "official_sets": [
            {
                "id": s.set_id,
                "name": s.name,
                "entry_tf": s.entry_tf,
                "confirmation_tfs": list(s.confirmation_tfs),
            }
            for s in OFFICIAL_SETS
        ],
        "sub_sets": [
            {
                "id": s.sub_id,
                "entry_tf": s.entry_tf,
                "confirmation_tf": s.confirmation_tf,
            }
            for s in SUB_SETS
        ],
        "structure_rules": {
            "pullback": "higher clear AND lower clear AND lower == opposite(higher)",
            "scale_conflict": "major clear AND minor clear AND major != minor",
            "heuristic_direction": "prefer higher TF; if NEUTRAL fall back to lower/entry",
            "flat_and_in_trade_same_eyes": True,
            "reverse_only_on_opposite": True,
        },
        "channel1": {
            "dim": CHANNEL1_DIM,
            "n_official": N_OFFICIAL,
            "n_sub": N_SUB,
            "features_per_set": FEATURES_PER_SET,
            "layout": _channel1_layout(),
            "normalization": {
                "direction": "int enum BULL=+1 BEAR=-1 NEUTRAL=0 as float",
                "velocity": "NONE=0 WEAK=1/3 MEDIUM=2/3 STRONG=1",
                "confluence_score": "(n_bull - n_bear) / max(n_groups, 1) in [-1,1]",
                "progress_to_goal": "clip(equity_pct / target_pct, -1, 1)",
                "danger": "clip((-equity_pct)/risk_pct, 0, 1) when equity < 0 else 0",
                "session_phase": "t / (n_bars-1) in [0,1]",
            },
        },
        "equity_day_decision_stride_default": 25,
        "source_file_hashes": source_hashes,
    }
    # Hash excludes a nested self-hash field; computed after body is fixed.
    body_hash = sha256_obj(manifest)
    manifest["meaning_hash"] = body_hash
    manifest["meaning_version"] = f"mp_tutor_meaning_v1_{body_hash[:12]}"
    return manifest


def meaning_hash(manifest: Optional[Dict[str, Any]] = None) -> str:
    m = manifest if manifest is not None else build_meaning_manifest()
    # Recompute from body without the stamp fields for stability if present.
    body = {k: v for k, v in m.items() if k not in ("meaning_hash", "meaning_version")}
    return sha256_obj(body)


def write_frozen_manifest(path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else FROZEN_MANIFEST_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_meaning_manifest()
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_frozen_manifest(path: Path | str | None = None) -> Dict[str, Any]:
    path = Path(path) if path is not None else FROZEN_MANIFEST_PATH
    if not path.is_file():
        raise FileNotFoundError(f"frozen meaning manifest missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def assert_meaning_matches_frozen(
    path: Path | str | None = None,
) -> Dict[str, Any]:
    """Refuse if live eyes differ from frozen manifest hash.

    Returns dict with ok, live_hash, frozen_hash, meaning_version.
    Raises ValueError on mismatch.
    """
    path = Path(path) if path is not None else FROZEN_MANIFEST_PATH
    live = build_meaning_manifest()
    live_h = meaning_hash(live)
    if not path.is_file():
        raise FileNotFoundError(
            f"no frozen meaning manifest at {path}; run write_frozen_manifest first"
        )
    frozen = load_frozen_manifest(path)
    frozen_h = frozen.get("meaning_hash") or meaning_hash(frozen)
    if live_h != frozen_h:
        raise ValueError(
            "MEANING HASH MISMATCH — eyes changed. "
            f"live={live_h} frozen={frozen_h}. "
            "Do not train/score/promote until practice+forward re-eval under new version."
        )
    return {
        "ok": True,
        "live_hash": live_h,
        "frozen_hash": frozen_h,
        "meaning_version": live.get("meaning_version"),
        "path": str(path).replace("\\", "/"),
    }
