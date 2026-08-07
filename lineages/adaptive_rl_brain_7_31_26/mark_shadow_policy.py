"""Spine Shadow policy — phase + event + size + clue_gate (meta).

Doctrine: 01_SYSTEM/Fable 5 Alternate — Spine Shadow.md §2.2

  phase  → before_first_fire | in_trade | breath_reload | done_bank | killed
  event  → wait_loaded | fire | add | hold_on_spine | bank | kill
  size   → micro|base|std|lag_add|heavy|max|none  (when fire/add)
  clue   → soft gate scalar [0,1] over observation (learn who to trust)

Act for shell still derived: fire/add → side from spine; else HOLD.
mark_aligned_decode / mark_align_policy wraps at GoalEquityDay (unchanged law).

Learn-to-learn: multi-head loss + learn≠copy (act high, phase/event low → REJECT).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import SIZE_BUCKETS
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
    N_ACTIONS,
)

PHASES = [
    "before_first_fire",
    "in_trade",
    "breath_reload",
    "done_bank",
    "killed",
]
EVENTS = [
    "wait_loaded",
    "fire",
    "add",
    "hold_on_spine",
    "bank",
    "kill",
]
SIZES = ["none"] + [name for name, _, _ in SIZE_BUCKETS]

PHASE_I = {p: i for i, p in enumerate(PHASES)}
EVENT_I = {e: i for i, e in enumerate(EVENTS)}
SIZE_I = {s: i for i, s in enumerate(SIZES)}
N_PHASE, N_EVENT, N_SIZE = len(PHASES), len(EVENTS), len(SIZES)


def phase_at_t(t: int, t1: Optional[int], t2: Optional[int], banked: bool = False) -> str:
    if banked:
        return "done_bank"
    if t1 is None:
        return "before_first_fire"
    if t < int(t1):
        return "before_first_fire"
    if t2 is not None and int(t1) <= t < int(t2):
        return "in_trade"
    if t2 is not None and t == int(t2):
        return "breath_reload"  # add moment / reload
    return "in_trade"


def event_at_t(t: int, plan: Dict[int, int], t1: Optional[int], t2: Optional[int]) -> str:
    a = int(plan.get(int(t), ACTION_HOLD))
    if t1 is not None and int(t) == int(t1) and a != ACTION_HOLD:
        return "fire"
    if t2 is not None and int(t) == int(t2) and a != ACTION_HOLD:
        return "add"
    if a != ACTION_HOLD:
        return "fire" if t1 is None or int(t) <= int(t1) else "add"
    if t1 is not None and int(t) < int(t1):
        return "wait_loaded"
    return "hold_on_spine"


def size_at_event(event: str, bucket: str) -> str:
    if event in ("fire", "add"):
        return bucket if bucket in SIZE_I else "base"
    return "none"


def event_to_action(event: str, side: Optional[str]) -> int:
    if event in ("fire", "add"):
        if side in ("BUY", "+1"):
            return ACTION_BUY
        if side in ("SELL", "-1"):
            return ACTION_SELL
        return ACTION_HOLD
    return ACTION_HOLD


class SpineShadowNet(nn.Module):
    """Shared trunk + phase/event/size heads + clue gate (meta)."""

    def __init__(self, obs_dim: int = MARK_FULL_DIM, hidden: int = 128):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        # Clue gate: soft mask over dims (learn-to-learn attention)
        self.clue_gate = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, self.obs_dim),
            nn.Sigmoid(),
        )
        self.trunk = nn.Sequential(
            nn.Linear(self.obs_dim, hidden),
            nn.Tanh(),
        )
        self.phase_head = nn.Linear(hidden, N_PHASE)
        self.event_head = nn.Linear(hidden, N_EVENT)
        self.size_head = nn.Linear(hidden, N_SIZE)
        # Legacy act head for decode + Channel1 export
        self.act_head = nn.Linear(hidden, N_ACTIONS)

    def forward(
        self, obs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        gate = self.clue_gate(obs)
        h = self.trunk(obs * gate)
        return (
            self.phase_head(h),
            self.event_head(h),
            self.size_head(h),
            self.act_head(h),
            gate,
        )

    @torch.no_grad()
    def act(self, obs: np.ndarray | torch.Tensor, *, greedy: bool = True) -> Tuple[int, torch.Tensor]:
        if isinstance(obs, np.ndarray):
            t = torch.as_tensor(obs, dtype=torch.float32)
        else:
            t = obs.float()
        _, _, _, act_log, _ = self.forward(t)
        logits = act_log.squeeze(0)
        action = int(torch.argmax(logits).item()) if greedy else int(
            torch.distributions.Categorical(logits=logits).sample().item()
        )
        logp = F.log_softmax(logits, dim=-1)[action]
        return action, logp

    def load_from_channel1(self, state: Dict[str, Any]) -> None:
        try:
            self.trunk[0].weight.data.copy_(state["net.0.weight"])
            self.trunk[0].bias.data.copy_(state["net.0.bias"])
            self.act_head.weight.data.copy_(state["net.2.weight"])
            self.act_head.bias.data.copy_(state["net.2.bias"])
            print("  SpineShadow warm: trunk+act from Channel1", flush=True)
        except Exception as e:
            print(f"  SpineShadow warm skip: {e}", flush=True)

    def to_channel1_state(self) -> Dict[str, torch.Tensor]:
        return {
            "net.0.weight": self.trunk[0].weight.detach().clone(),
            "net.0.bias": self.trunk[0].bias.detach().clone(),
            "net.2.weight": self.act_head.weight.detach().clone(),
            "net.2.bias": self.act_head.bias.detach().clone(),
        }


def as_channel1(net: SpineShadowNet) -> Channel1Policy:
    pol = Channel1Policy(obs_dim=net.obs_dim, hidden=net.hidden)
    pol.load_state_dict(net.to_channel1_state())
    pol.eval()
    return pol
