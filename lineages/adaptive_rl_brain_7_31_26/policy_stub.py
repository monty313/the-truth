"""Channel-1 policy for adaptive_rl_brain_7_31_26 (not PROVEN).

CHANGE LOG:
- 2026-08-06  Multi-head physics brain — WHY: teach LTF pullback/continuation
  via aux topology+wait heads over frozen 168-dim obs (find links, no cheat
  features). Shared trunk Linear(168,128)->ReLU->Linear(128,128)->ReLU;
  action 3 / topology 4 / wait 3. Legacy single-head (Tanh, net.*) still
  loads for 35/50 mark_clone_full_obs_v1.pt.
- 2026-07-31  Phase 2 Slice 5 — WHY: lineage-local tiny policy over CHANNEL1_DIM.
  Never loads or writes PROVEN 1820 checkpoints.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.types import Direction

# Discrete actions for the stub day loop
# 0=hold/flat, 1=buy (bull), 2=sell (bear)
N_ACTIONS = 3
ACTION_HOLD, ACTION_BUY, ACTION_SELL = 0, 1, 2

# Topology head (GROK_PROMPT_TEACH_PULLBACKS)
N_TOPOLOGY = 4
TOPO_PULLBACK = 0       # Load / slingshot_load
TOPO_CONTINUATION = 1   # Release / slingshot_release
TOPO_LAUNCH = 2
TOPO_CHOP = 3
TOPOLOGY_NAMES = ("pullback_load", "continuation_release", "launch", "chop")

# Wait head
N_WAIT = 3
WAIT_LOADED = 0
WAIT_NO_TRADE = 1
WAIT_HEAT = 2
WAIT_NAMES = ("loaded", "no_trade", "heat")


def greedy_action_ban_hold_if_directional(
    logits: np.ndarray | "torch.Tensor",
    recommended: int,
) -> int:
    """Pure-greedy decode experiment: ban HOLD when structure is directional.

    If recommended is BUY or SELL → argmax among {BUY, SELL} only.
    If recommended is HOLD → argmax among all three (HOLD still allowed).
    Stochastic / training paths must not call this.
    """
    if hasattr(logits, "detach"):
        arr = logits.detach().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(logits, dtype=np.float64).reshape(-1)
    rec = int(recommended)
    if rec in (ACTION_BUY, ACTION_SELL):
        return (
            ACTION_BUY
            if float(arr[ACTION_BUY]) >= float(arr[ACTION_SELL])
            else ACTION_SELL
        )
    return int(np.argmax(arr[:N_ACTIONS]))


def action_to_trade_side(action: int) -> Direction | None:
    """Map action → trade side; HOLD → None (flat)."""
    a = int(action)
    if a == ACTION_BUY:
        return Direction.BULL
    if a == ACTION_SELL:
        return Direction.BEAR
    return None


class Channel1Policy(nn.Module):
    """Policy over Channel1 / full-obs board.

    multi_head=True (physics teach):
      trunk: Linear(obs,h)->ReLU->Linear(h,h)->ReLU
      heads: action(3), topology(4), wait(3)

    multi_head=False (legacy 35/50 embryo):
      net: Linear(obs,h)->Tanh->Linear(h,3)
    """

    def __init__(
        self,
        obs_dim: int = CHANNEL1_DIM,
        hidden: int = 32,
        *,
        multi_head: bool = False,
    ):
        super().__init__()
        self.obs_dim = int(obs_dim)
        self.hidden = int(hidden)
        self.multi_head = bool(multi_head)
        h = self.hidden
        if self.multi_head:
            self.trunk = nn.Sequential(
                nn.Linear(self.obs_dim, h),
                nn.ReLU(),
                nn.Dropout(0.15),
                nn.Linear(h, h),
                nn.ReLU(),
                nn.Dropout(0.10),
            )
            self.action_head = nn.Linear(h, N_ACTIONS)
            self.topology_head = nn.Linear(h, N_TOPOLOGY)
            self.wait_head = nn.Linear(h, N_WAIT)
            self.net = None  # type: ignore
        else:
            self.net = nn.Sequential(
                nn.Linear(self.obs_dim, h),
                nn.Tanh(),
                nn.Linear(h, N_ACTIONS),
            )
            self.trunk = None  # type: ignore
            self.action_head = None  # type: ignore
            self.topology_head = None  # type: ignore
            self.wait_head = None  # type: ignore

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if self.multi_head:
            assert self.trunk is not None
            return self.trunk(obs)
        assert self.net is not None
        # legacy: hidden is first layer + tanh
        return self.net[1](self.net[0](obs))  # type: ignore[index]

    def forward(
        self,
        obs: torch.Tensor,
        *,
        return_all: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Default: action logits (B, 3) — keeps day loops / train_bc working.

        return_all=True (multi_head only): dict action/topology/wait_subtype.
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if not self.multi_head:
            assert self.net is not None
            logits = self.net(obs)
            if return_all:
                b = logits.shape[0]
                z_t = torch.zeros(b, N_TOPOLOGY, device=logits.device, dtype=logits.dtype)
                z_w = torch.zeros(b, N_WAIT, device=logits.device, dtype=logits.dtype)
                return {
                    "action": logits,
                    "topology": z_t,
                    "wait_subtype": z_w,
                }
            return logits
        assert self.trunk is not None
        h = self.trunk(obs)
        act = self.action_head(h)
        if not return_all:
            return act
        return {
            "action": act,
            "topology": self.topology_head(h),
            "wait_subtype": self.wait_head(h),
        }

    def forward_heads(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Always return multi-task logits dict."""
        out = self.forward(obs, return_all=True)
        assert isinstance(out, dict)
        return out

    def trunk_parameters(self):
        if self.multi_head:
            assert self.trunk is not None
            return self.trunk.parameters()
        assert self.net is not None
        return self.net[0].parameters()  # type: ignore[index]

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
        logits = self.forward(t)
        assert isinstance(logits, torch.Tensor)
        logits = logits.squeeze(0)
        dist = torch.distributions.Categorical(logits=logits)
        if greedy:
            action = int(torch.argmax(logits).item())
        else:
            if generator is not None:
                g = -torch.log(-torch.log(torch.rand(N_ACTIONS, generator=generator) + 1e-8) + 1e-8)
                action = int(torch.argmax(logits + g).item())
            else:
                action = int(dist.sample().item())
        logp = dist.log_prob(torch.tensor(action))
        return action, logp

    def load_state_dict_flexible(self, state_dict: Dict[str, Any], strict: bool = True) -> Dict[str, Any]:
        """Load multi-head or legacy weights; optional partial trunk/action transfer."""
        keys = list(state_dict.keys())
        is_legacy = any(k.startswith("net.") for k in keys)
        is_mh = any(k.startswith("trunk.") or k.startswith("action_head.") for k in keys)
        if self.multi_head and is_legacy and not is_mh:
            # Warm-start: map net.0 -> trunk.0, net.2 -> action_head
            mapped: Dict[str, torch.Tensor] = {}
            for k, v in state_dict.items():
                if k == "net.0.weight":
                    mapped["trunk.0.weight"] = v
                elif k == "net.0.bias":
                    mapped["trunk.0.bias"] = v
                elif k == "net.2.weight":
                    mapped["action_head.weight"] = v
                elif k == "net.2.bias":
                    mapped["action_head.bias"] = v
            inc = self.load_state_dict(mapped, strict=False)
            return {
                "mapped_from_legacy": True,
                "missing_keys": list(inc.missing_keys),
                "unexpected_keys": list(inc.unexpected_keys),
            }
        inc = self.load_state_dict(state_dict, strict=strict)
        return {
            "mapped_from_legacy": False,
            "missing_keys": list(inc.missing_keys),
            "unexpected_keys": list(inc.unexpected_keys),
        }


def policy_from_checkpoint(
    blob: Dict[str, Any] | str,
    *,
    prefer_multi_head: Optional[bool] = None,
) -> Channel1Policy:
    """Build Channel1Policy from a saved blob path or dict.

    prefer_multi_head: force architecture; None = detect from state keys / blob flag.
    """
    if isinstance(blob, str):
        blob = torch.load(blob, map_location="cpu", weights_only=False)
    assert isinstance(blob, dict)
    sd = blob.get("state_dict") or blob
    if not isinstance(sd, dict):
        raise ValueError("checkpoint missing state_dict")
    keys = list(sd.keys())
    multi = bool(blob.get("multi_head", False))
    if prefer_multi_head is not None:
        multi = bool(prefer_multi_head)
    elif any(k.startswith("trunk.") or k.startswith("action_head.") for k in keys):
        multi = True
    elif any(k.startswith("net.") for k in keys):
        multi = False
    obs_dim = int(blob.get("obs_dim", CHANNEL1_DIM))
    hidden = int(blob.get("hidden", 128 if obs_dim >= 100 else 32))
    pol = Channel1Policy(obs_dim=obs_dim, hidden=hidden, multi_head=multi)
    if multi and any(k.startswith("net.") for k in keys) and not any(
        k.startswith("trunk.") for k in keys
    ):
        pol.load_state_dict_flexible(sd, strict=False)
    else:
        pol.load_state_dict(sd, strict=False)
    pol.eval()
    return pol
