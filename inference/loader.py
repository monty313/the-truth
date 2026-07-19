"""Frozen-champion loader — the only door from checkpoint to live brain.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (audit R6: torch.load appeared NOWHERE — nothing
       could ever run a trained brain).
WHAT:  load_brain(name) -> (Brain in eval mode, metadata). Reads
       artifacts/checkpoints/<name>.pt saved by training/ppo.PPO.save
       (model weights + obs_dim + reward state + seed).
WHEN:  2026-07-19 (audit round 2).
WHERE: scripts/run_live.py (dry-run/live), evaluation/champion.py.
WHY:   Frozen means frozen (Monty's ruling) — this loads, never trains.
INTERCONNECTED WITH: training/policy.Brain, training/ppo.save,
       artifacts/checkpoints/.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created  — WHY: nothing loaded a checkpoint; frozen champion had no door to run (audit R6).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import os

import torch

from core.configs import path as rpath
from training.policy import Brain


def load_brain(name: str = "champion_candidate"):
    """Returns (brain, meta) or (None, {}) if no checkpoint exists."""
    p = rpath("artifacts", "checkpoints", f"{name}.pt")
    if not os.path.exists(p):
        return None, {}
    d = torch.load(p, map_location="cpu", weights_only=False)
    brain = Brain(int(d["obs_dim"]))
    try:
        brain.load_state_dict(d["model"])
    except RuntimeError:
        # architecture drift (e.g. v1 sigmoid head checkpoints) — refuse quietly
        return None, {"error": "checkpoint_architecture_mismatch", "path": p}
    brain.eval()
    meta = {k: v for k, v in d.items() if k != "model"}
    meta["path"] = p
    return brain, meta
