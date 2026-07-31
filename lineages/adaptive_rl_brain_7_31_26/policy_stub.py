"""Minimal Channel-1 policy for adaptive_rl_brain_7_31_26 (not PROVEN).

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 5 — WHY: lineage-local tiny policy over CHANNEL1_DIM.
  Never loads or writes PROVEN 1820 checkpoints.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction

# Discrete actions for the stub day loop
# 0=hold/flat, 1=buy (bull), 2=sell (bear)
N_ACTIONS = 3
ACTION_HOLD, ACTION_BUY, ACTION_SELL = 0, 1, 2


def action_to_trade_side(action: int) -> Direction | None:
    """Map action → trade side; HOLD → None (flat)."""
    a = int(action)
    if a == ACTION_BUY:
        return Direction.BULL
    if a == ACTION_SELL:
        return Direction.BEAR
    return None


class Channel1Policy(nn.Module):
    """Tiny MLP: CHANNEL1_DIM → N_ACTIONS. Lineage-local only."""

    def __init__(self, obs_dim: int = CHANNEL1_DIM, hidden: int = 32):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.net = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, N_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return action logits (B, N_ACTIONS)."""
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        return self.net(obs)

    @torch.no_grad()
    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        *,
        greedy: bool = False,
        generator: torch.Generator | None = None,
    ) -> Tuple[int, torch.Tensor]:
        """Sample or greedy action; returns (action, log_prob)."""
        if isinstance(obs, np.ndarray):
            t = torch.as_tensor(obs, dtype=torch.float32)
        else:
            t = obs.float()
        logits = self.forward(t).squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            action = int(torch.argmax(logits).item())
        else:
            if generator is not None:
                # sample via Gumbel-max with generator for determinism in tests
                g = -torch.log(-torch.log(torch.rand(N_ACTIONS, generator=generator) + 1e-8) + 1e-8)
                action = int(torch.argmax(logits + g).item())
            else:
                action = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(action))
        return action, logp
