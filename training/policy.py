"""The brain v2 — recurrent actor-critic with a LEARNABLE size dial.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0007; audit T1 fix).
WHAT:  GRU torso -> three heads: op logits (11), size as a BETA
       DISTRIBUTION over (0,1] of the cap (v2 — the audit proved the v1
       sigmoid head had exactly zero reward gradient: the "smooth dial"
       Monty ruled could 'learn freely inside the cap' was frozen).
       Joint log-prob (op + size) drives the PPO ratio; joint entropy
       drives exploration. Value head for GAE.
WHEN:  2026-07-19 (v2, post-audit).
WHERE: training/ppo.py (learning); inference/loader.py (frozen).
WHY:   Sizing by conviction is a ruled behavior (smooth dial, multi-TF
       agreement); it must be trainable or the ruling is decoration.
INTERCONNECTED WITH: training/env.TradingEnv (obs layout), configs/
       training.yaml (hidden), evaluation/champion.py, tests/
       test_training_fixes.py (nonzero size-gradient proof).
----------------------------------------------------------------------
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

N_OPS = 11


class Brain(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 128):
        super().__init__()
        self.inp = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.Tanh())
        self.gru = nn.GRU(hidden, hidden, batch_first=True)
        self.op_head = nn.Linear(hidden, N_OPS)
        self.size_head = nn.Linear(hidden, 2)      # Beta alpha, beta params
        self.value_head = nn.Linear(hidden, 1)

    # ---------- distribution plumbing ----------
    def _dists(self, y: torch.Tensor):
        logits = self.op_head(y)
        ab = F.softplus(self.size_head(y)) + 1.0   # alpha,beta >= 1 (unimodal)
        op_dist = torch.distributions.Categorical(logits=logits)
        size_dist = torch.distributions.Beta(ab[..., 0], ab[..., 1])
        return op_dist, size_dist

    def forward(self, obs: torch.Tensor, h: torch.Tensor | None = None):
        """obs: (B, D) one step or (B, T, D) sequence ->
        (op_dist, size_dist, value, h)."""
        seq = obs.dim() == 3
        x = self.inp(obs if seq else obs.unsqueeze(1))
        y, h2 = self.gru(x, h)
        if not seq:
            y = y.squeeze(1)
        op_dist, size_dist = self._dists(y)
        return op_dist, size_dist, self.value_head(y).squeeze(-1), h2

    @staticmethod
    def _clamp_size(z: torch.Tensor) -> torch.Tensor:
        return z.clamp(0.05, 1.0)

    def joint_logprob(self, op_dist, size_dist, ops: torch.Tensor,
                      sizes: torch.Tensor) -> torch.Tensor:
        """log p(op) + log p(size) — size grad flows for every action; ops
        that ignore size (hold/close) still train the dial gently toward
        the sizes it proposed."""
        return op_dist.log_prob(ops) + size_dist.log_prob(
            sizes.clamp(1e-4, 1 - 1e-4))

    @torch.no_grad()
    def act(self, obs, h=None, greedy: bool = False):
        op_dist, size_dist, value, h2 = self.forward(obs, h)
        if greedy:
            op = torch.argmax(op_dist.logits, -1)
            size = self._clamp_size(size_dist.mean)
        else:
            op = op_dist.sample()
            size = self._clamp_size(size_dist.sample())
        lp = self.joint_logprob(op_dist, size_dist, op, size)
        return (int(op.item()), float(size.squeeze().item()),
                float(lp.item()), float(value.item()), h2)
