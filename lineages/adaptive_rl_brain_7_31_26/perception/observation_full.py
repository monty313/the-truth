"""Mark FULL observation — Channel1 + doctrine + signal agents + self context.

CHANGE LOG:
- 2026-08-05  created — WHY: Fable full-clone order — meta/policy = Mark must
  see context clues and pattern panels (92 agents), not only 32 set floats.
  Lineage-local. Does NOT change PROVEN frame_dim. Never warm-start PROVEN.

Layout (MARK_FULL_DIM = 168):
  [0:32]     CHANNEL1 base (sets + structure + progress/danger/session)
  [32:48]    Doctrine context (force, regime, play, m_conf, m_regime, opp)
  [48:60]    Signal majority summary (panel vote stats)
  [60:152]   All 92 signal agent votes (-1/0/+1), padded/truncated to 92
  [152:168]  Self-state / goal context (side, heat room, hardness, adds, …)

Teacher (Mark soul) still owns labels. Policy learns attention over this board.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from lineages.adaptive_rl_brain_7_31_26.perception.observation import (
    CHANNEL1_DIM,
    build_channel1_obs,
)
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction

# Fixed blocks
N_SIGNAL_SLOTS = 92
DOCTRINE_DIM = 16
MAJORITY_DIM = 12
SELF_DIM = 16
MARK_FULL_DIM = CHANNEL1_DIM + DOCTRINE_DIM + MAJORITY_DIM + N_SIGNAL_SLOTS + SELF_DIM
# 32 + 16 + 12 + 92 + 16 = 168

assert MARK_FULL_DIM == 168


def _dir01(d: Any) -> float:
    """Map Direction / int / str to [-1, 0, 1]."""
    if d is None:
        return 0.0
    if isinstance(d, (int, float)):
        v = float(d)
        if v > 0.5:
            return 1.0
        if v < -0.5:
            return -1.0
        return 0.0
    name = str(getattr(d, "value", d) or "").lower()
    if "bull" in name or name in ("1", "buy", "long"):
        return 1.0
    if "bear" in name or name in ("-1", "sell", "short"):
        return -1.0
    return 0.0


def pack_doctrine(dec: Any) -> np.ndarray:
    """16 floats: force / regime / play / confidence."""
    out = np.zeros(DOCTRINE_DIM, dtype=np.float32)
    if dec is None:
        return out
    out[0] = _dir01(getattr(dec, "force_dir", None))
    out[1] = float(getattr(dec, "n_force_bull", 0) or 0) / 4.0
    out[2] = float(getattr(dec, "n_force_bear", 0) or 0) / 4.0
    out[3] = float(getattr(dec, "n_aligned", 0) or 0) / 4.0
    out[4] = float(getattr(dec, "n_breather", 0) or 0) / 4.0
    out[5] = float(np.clip(getattr(dec, "m_conf", 0.0) or 0.0, 0.0, 1.5)) / 1.5
    out[6] = float(np.clip(getattr(dec, "m_regime", 0.0) or 0.0, 0.0, 1.0))
    play = str(getattr(getattr(dec, "play", None), "value", getattr(dec, "play", "")) or "").lower()
    out[7] = 1.0 if play == "launch" else 0.0
    out[8] = 1.0 if play in ("breather", "breath") else 0.0
    out[9] = 1.0 if play == "aligned" else 0.0
    regime = str(getattr(getattr(dec, "regime", None), "value", getattr(dec, "regime", "")) or "").lower()
    out[10] = 1.0 if "bull" in regime else 0.0
    out[11] = 1.0 if "bear" in regime else 0.0
    out[12] = 1.0 if "chop" in regime else 0.0
    out[13] = 1.0 if "flat" in regime else 0.0
    out[14] = float(int(getattr(dec, "action", 0) or 0) == 1)  # teacher raw buy
    out[15] = float(int(getattr(dec, "action", 0) or 0) == 2)  # teacher raw sell
    return out


def pack_majority(
    *,
    frac_bull: float = 0.0,
    frac_bear: float = 0.0,
    agree_frac: float = 0.0,
    n_active: float = 0.0,
    n_agents: float = 0.0,
    has_majority: bool = False,
    maj_dir: float = 0.0,
    mean_vote: float = 0.0,
    std_vote: float = 0.0,
    n_bull: float = 0.0,
    n_bear: float = 0.0,
    n_flat: float = 0.0,
) -> np.ndarray:
    out = np.zeros(MAJORITY_DIM, dtype=np.float32)
    out[0] = float(np.clip(frac_bull, 0.0, 1.0))
    out[1] = float(np.clip(frac_bear, 0.0, 1.0))
    out[2] = float(np.clip(agree_frac, 0.0, 1.0))
    out[3] = float(np.clip(n_active / max(n_agents, 1.0), 0.0, 1.0))
    out[4] = 1.0 if has_majority else 0.0
    out[5] = float(maj_dir)
    out[6] = float(np.clip(mean_vote, -1.0, 1.0))
    out[7] = float(np.clip(std_vote, 0.0, 1.0))
    n = max(n_agents, 1.0)
    out[8] = float(n_bull) / n
    out[9] = float(n_bear) / n
    out[10] = float(n_flat) / n
    out[11] = float(np.clip(n_agents / float(N_SIGNAL_SLOTS), 0.0, 2.0))
    return out


def pack_agent_votes(votes: Optional[Sequence[float] | np.ndarray]) -> np.ndarray:
    """Fixed 92-slot vote board (-1/0/+1)."""
    out = np.zeros(N_SIGNAL_SLOTS, dtype=np.float32)
    if votes is None:
        return out
    v = np.asarray(votes, dtype=np.float32).reshape(-1)
    n = min(int(v.size), N_SIGNAL_SLOTS)
    if n > 0:
        out[:n] = np.clip(v[:n], -1.0, 1.0)
    return out


def pack_self_state(
    *,
    side: float = 0.0,
    n_open_units: float = 0.0,
    n_entries: float = 0.0,
    n_adds: float = 0.0,
    progress: float = 0.0,
    danger: float = 0.0,
    target_pct: float = 2.0,
    risk_pct: float = 3.0,
    equity_pct: float = 0.0,
    room_to_floor: float = 0.0,
    remaining_to_target: float = 0.0,
    mark_soul: float = 1.0,
    soul_flips: float = 0.0,
    session_phase: float = 0.0,
    banked: float = 0.0,
    in_trade: float = 0.0,
) -> np.ndarray:
    out = np.zeros(SELF_DIM, dtype=np.float32)
    out[0] = float(np.clip(side, -1.0, 1.0))
    out[1] = float(np.clip(n_open_units / 8.0, 0.0, 1.0))
    out[2] = float(np.clip(n_entries / 12.0, 0.0, 1.0))
    out[3] = float(np.clip(n_adds / 4.0, 0.0, 1.0))
    out[4] = float(np.clip(progress, -1.0, 1.0))
    out[5] = float(np.clip(danger, 0.0, 1.0))
    out[6] = float(np.clip(target_pct / 5.0, 0.0, 1.0))
    out[7] = float(np.clip(risk_pct / 5.0, 0.0, 1.0))
    hardness = target_pct / max(risk_pct, 1e-6)
    out[8] = float(np.clip(hardness, 0.0, 2.0) / 2.0)
    out[9] = float(np.clip(equity_pct / 5.0, -1.0, 1.0))
    out[10] = float(np.clip(room_to_floor / max(risk_pct, 1e-6), 0.0, 2.0) / 2.0)
    out[11] = float(np.clip(remaining_to_target / max(target_pct, 1e-6), -1.0, 2.0) / 2.0)
    out[12] = float(mark_soul)
    out[13] = float(np.clip(soul_flips / 4.0, 0.0, 1.0))
    out[14] = float(np.clip(session_phase, 0.0, 1.0))
    out[15] = 1.0 if banked or in_trade > 0.5 and banked else float(in_trade)
    out[15] = float(in_trade)  # clean: in_trade only; banked is rare mid-decision
    return out


def build_mark_full_obs(
    channel1: np.ndarray,
    *,
    doctrine_vec: Optional[np.ndarray] = None,
    majority_vec: Optional[np.ndarray] = None,
    agent_votes: Optional[Sequence[float] | np.ndarray] = None,
    self_vec: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Concatenate fixed blocks → MARK_FULL_DIM."""
    c1 = np.asarray(channel1, dtype=np.float32).reshape(-1)
    if c1.size != CHANNEL1_DIM:
        # pad / truncate defensively
        tmp = np.zeros(CHANNEL1_DIM, dtype=np.float32)
        n = min(CHANNEL1_DIM, int(c1.size))
        tmp[:n] = c1[:n]
        c1 = tmp
    d = doctrine_vec if doctrine_vec is not None else np.zeros(DOCTRINE_DIM, np.float32)
    m = majority_vec if majority_vec is not None else np.zeros(MAJORITY_DIM, np.float32)
    a = pack_agent_votes(agent_votes)
    s = self_vec if self_vec is not None else np.zeros(SELF_DIM, np.float32)
    d = np.asarray(d, dtype=np.float32).reshape(-1)[:DOCTRINE_DIM]
    m = np.asarray(m, dtype=np.float32).reshape(-1)[:MAJORITY_DIM]
    s = np.asarray(s, dtype=np.float32).reshape(-1)[:SELF_DIM]
    if d.size < DOCTRINE_DIM:
        d = np.pad(d, (0, DOCTRINE_DIM - d.size))
    if m.size < MAJORITY_DIM:
        m = np.pad(m, (0, MAJORITY_DIM - m.size))
    if s.size < SELF_DIM:
        s = np.pad(s, (0, SELF_DIM - s.size))
    return np.concatenate([c1, d, m, a, s], axis=0).astype(np.float32)


def mark_full_layout() -> Dict[str, object]:
    return {
        "dim": MARK_FULL_DIM,
        "blocks": {
            "channel1": "0:32",
            "doctrine": "32:48",
            "majority": "48:60",
            "signal_agents_92": "60:152",
            "self_state": "152:168",
        },
        "channel1_layout": "see observation.channel1_layout()",
        "note": "Mark full clone eyes — not PROVEN 1820/6820",
        "proven_touched": False,
    }
