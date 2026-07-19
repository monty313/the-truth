"""The brain — compact goal-conditioned recurrent actor-critic.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0007: meta = context-based, staged).
WHAT:  GRU over the frame-stacked observation -> three heads:
       op logits (11), size in (0,1] (sigmoid), value. The recurrent
       state + goal/floor inputs ARE the meta layer: the same frozen
       weights adapt in-context to regime and to any X Monty types.
WHEN:  2026-07-19 overnight build.
WHERE: training/ppo.py (learning) and inference/ + bridge (frozen).
WHY:   Compact by law — 2-core laptop (ADR-0007); observability over
       cleverness; a small honest brain beats a large unexplainable one.
INTERCONNECTED WITH: training/env.TradingEnv (obs layout), configs/
       training.yaml (hidden size), evaluation/champion.py (checkpoints).
----------------------------------------------------------------------
"""
from __future__ import annotations
import torch
import torch.nn as nn

N_OPS = 11


class Brain(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.op_head = nn.Linear(hidden, N_OPS)
        self.size_head = nn.Linear(hidden, 1)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor, h: torch.Tensor | None = None):
        """obs: (B, obs_dim) one step, or (B, T, obs_dim) sequence."""
        seq = obs.dim() == 3
        x = self.inp(obs if seq else obs.unsqueeze(1))
        y, h2 = self.gru(x, h)
        if not seq:
            y = y.squeeze(1)
        return (self.op_head(y), torch.sigmoid(self.size_head(y)).clamp(0.05, 1.0),
                self.value_head(y).squeeze(-1), h2)

    @torch.no_grad()
    def act(self, obs, h=None, greedy: bool = False):
        logits, size, value, h2 = self.forward(obs, h)
        dist = torch.distributions.Categorical(logits=logits)
        op = torch.argmax(logits, -1) if greedy else dist.sample()
        return (int(op.item()), float(size.squeeze().item()),
                float(dist.log_prob(op).item()), float(value.item()), h2)
