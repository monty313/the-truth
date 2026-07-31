"""Minimal rollout / training stub (lineage-local, not PROVEN).

CHANGE LOG:
- 2026-07-31  Phase 2 Slice 5 — WHY: prove Channel1Policy can act in DayRunner
  and receive lineage rewards. No checkpoint writes to models/PROVEN.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch

from data_io.loader import synthetic_m1
from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner, DayStepResult
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import DEFAULT_DIALS


@dataclass
class RolloutResult:
    steps: List[DayStepResult] = field(default_factory=list)
    total_reward: float = 0.0
    n_steps: int = 0
    loss: Optional[float] = None


def _first_day_frame(m1, min_bars: int = 400):
    days = m1.groupby(m1.index.date)
    first_key = next(iter(days.groups))
    day = days.get_group(first_key)
    if len(day) < min_bars:
        return m1.iloc[: min(len(m1), max(min_bars, 600))]
    return day


def run_day_rollout(
    m1=None,
    *,
    policy: Channel1Policy | None = None,
    decide_every: int = 15,
    greedy: bool = True,
    dials: dict | None = None,
    max_steps: int = 40,
    seed: int = 0,
) -> RolloutResult:
    """One synthetic (or provided) day: policy acts, runner rewards."""
    if m1 is None:
        m1 = synthetic_m1(days=2, seed=seed)
    day = _first_day_frame(m1)

    pol = policy or Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=32)
    pol.eval()
    runner = DayRunner(day, decide_every=decide_every, dials=dials or DEFAULT_DIALS)
    idxs = runner.decision_indices()[:max_steps]
    out = RolloutResult()
    for t in idxs:
        obs = runner.observe(t)
        action, _ = pol.act(obs, greedy=greedy)
        step = runner.step(t, action=int(action))
        out.steps.append(step)
        out.total_reward += float(step.reward)
        out.n_steps += 1
    return out


def train_stub_epoch(
    *,
    steps: int = 20,
    lr: float = 1e-3,
    seed: int = 1,
) -> Dict[str, float]:
    """One tiny REINFORCE-ish epoch on synthetic day (smoke trainable path)."""
    torch.manual_seed(seed)
    m1 = synthetic_m1(days=2, seed=seed)
    day = _first_day_frame(m1, min_bars=500)
    policy = Channel1Policy(hidden=16)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    runner = DayRunner(day, decide_every=20, dials=DEFAULT_DIALS)
    idxs = runner.decision_indices()[:steps]
    logps: List[torch.Tensor] = []
    rewards: List[float] = []
    for t in idxs:
        obs = runner.observe(t)
        action, logp = policy.act(obs, greedy=False)
        # need grad through logp — recompute with grad
        logits = policy.forward(torch.as_tensor(obs, dtype=torch.float32))
        dist = torch.distributions.Categorical(logits=logits.squeeze(0))
        logp_g = dist.log_prob(torch.tensor(int(action)))
        step = runner.step(t, action=int(action))
        logps.append(logp_g)
        rewards.append(float(step.reward))
    if not logps:
        return {"loss": 0.0, "mean_reward": 0.0, "n": 0.0}
    r = torch.tensor(rewards, dtype=torch.float32)
    r = r - r.mean()
    loss = -(torch.stack(logps) * r.detach()).mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    return {
        "loss": float(loss.item()),
        "mean_reward": float(np.mean(rewards)),
        "n": float(len(rewards)),
    }


if __name__ == "__main__":
    r = run_day_rollout(max_steps=10, greedy=True)
    print("rollout", r.n_steps, r.total_reward)
    tr = train_stub_epoch(steps=8)
    print("train_stub", tr)
