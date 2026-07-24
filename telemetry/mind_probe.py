"""Mind Probe — the MRI Scanner for conversational diagnosis.

5W+I -----------------------------------------------------------------
WHO:   Fable 5 for Monty (Project Instructions Diagnostic LLM layer).
WHAT:  Runs a frozen brain over one or more days and records, for every
       decision step: market/gravity pattern flags already present in the
       observation (cont/pull/rev per set), the policy's action probability
       distribution, chosen op/size, and a minimal self-state snapshot.
       Produces a day-level mind dump that the Diagnostic LLM can load and
       reason over (Perception vs Policy vs Generalization).
WHEN:  2026-07-24 Phase 1 of autonomous self-heal plan.
WHERE: Called by scripts/mind_probe_day.py; consumed by future IRAC dialog.
WHY:   The RL model must learn to see the chart and recognize patterns that
       correlate with consistent daily clears. Without a mind dump we cannot
       tell whether a missed clear was blindness (Perception) or fear/bad
       incentives (Policy). Constraint: read-only — no weight or obs change.
INTERCONNECTED WITH: training/policy.Brain, inference/loader, training/fastsim
       or backtesting/simulator, features/engine (cont/pull/rev columns),
       doctrine/STANDING_LAWS.md (bread-and-butter, IRAC), configs/rewards.yaml.
----------------------------------------------------------------------

CHANGE LOG (newest first):
- 2026-07-24  created — WHY: Phase 1 MRI Scanner; enable conversational diagnosis of chart-pattern recognition without touching core weights or obs space.
# NEXT EDITOR: append dated WHY; keep this line.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch

from training.policy import Brain, N_OPS

OP_NAMES = {
    0: "hold",
    1: "open_long",
    2: "open_short",
    3: "add_long",
    4: "add_short",
    5: "close_half_long",
    6: "close_long",
    7: "close_half_short",
    8: "close_short",
    9: "probe_long",
    10: "probe_short",
}


@dataclass
class DecisionRecord:
    t: int
    op_probs: list[float]
    chosen_op: int
    chosen_op_name: str
    chosen_size: float
    value: float
    cont_buy: bool = False
    cont_sell: bool = False
    pull_buy: bool = False
    pull_sell: bool = False
    rev_buy: bool = False
    rev_sell: bool = False
    mask_buy_blocked: bool = False
    mask_sell_blocked: bool = False
    dist_to_goal: float = 0.0
    dist_to_floor: float = 0.0
    open_risk: float = 0.0
    position_sign: float = 0.0
    trades_used: float = 0.0


@dataclass
class DayMindDump:
    brain_name: str
    day_index: int
    day_label: str
    goal_pct: float
    floor_pct: float
    day_pnl: float = 0.0
    goal_hit: bool = False
    breached: bool = False
    n_decisions: int = 0
    n_pull_buy_bars: int = 0
    n_pull_sell_bars: int = 0
    n_cont_buy_bars: int = 0
    n_cont_sell_bars: int = 0
    n_rev_buy_bars: int = 0
    n_rev_sell_bars: int = 0
    pull_buy_seen_and_acted: int = 0
    pull_buy_seen_and_held: int = 0
    pull_sell_seen_and_acted: int = 0
    pull_sell_seen_and_held: int = 0
    mean_op_entropy: float = 0.0
    decisions: list[DecisionRecord] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def write(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - logits.max()
    e = np.exp(x)
    return e / e.sum()


def _extract_pattern_flags(row: np.ndarray, col_index: dict[str, int]) -> dict[str, bool]:
    def flag(name: str) -> bool:
        i = col_index.get(name)
        if i is None:
            return False
        return bool(row[i] > 0.5)

    return {
        "cont_buy": any(flag(f"set{k}::cont_buy") for k in (1, 2, 3, 4)),
        "cont_sell": any(flag(f"set{k}::cont_sell") for k in (1, 2, 3, 4)),
        "pull_buy": any(flag(f"set{k}::pull_buy") for k in (1, 2, 3, 4)),
        "pull_sell": any(flag(f"set{k}::pull_sell") for k in (1, 2, 3, 4)),
        "rev_buy": any(flag(f"set{k}::rev_buy") for k in (1, 2, 3, 4)),
        "rev_sell": any(flag(f"set{k}::rev_sell") for k in (1, 2, 3, 4)),
        "mask_buy_blocked": flag("mask_buy_blocked"),
        "mask_sell_blocked": flag("mask_sell_blocked"),
    }


def _self_from_obs(obs: torch.Tensor, frame: int = 10, self_dim: int = 12) -> dict[str, float]:
    flat = obs.detach().cpu().numpy().reshape(-1)
    last_self = flat[-self_dim:]
    return {
        "dist_to_goal": float(last_self[2]),
        "dist_to_floor": float(last_self[3]),
        "open_risk": float(last_self[7]),
        "position_sign": float(last_self[8]),
        "trades_used": float(last_self[10]),
    }


@torch.no_grad()
def probe_day(
    brain: Brain,
    day_obs: np.ndarray,
    day_phys: np.ndarray,
    cols: list[str],
    goal_pct: float,
    floor_pct: float,
    brain_name: str = "unknown",
    day_index: int = 0,
    day_label: str = "",
    decide_every: int = 5,
    device: str = "cpu",
) -> DayMindDump:
    """Run frozen brain greedily over one day; record mind for Perception diagnosis."""
    brain = brain.to(device).eval()
    L, C = day_obs.shape
    col_index = {name: i for i, name in enumerate(cols)}
    frame = 10
    self_dim = 12

    self_hist = np.zeros((L, self_dim), dtype=np.float32)
    self_hist[:, 0] = goal_pct / 5.0
    self_hist[:, 1] = floor_pct / 6.0

    dump = DayMindDump(
        brain_name=brain_name,
        day_index=day_index,
        day_label=day_label or str(day_index),
        goal_pct=goal_pct,
        floor_pct=floor_pct,
    )

    h = None
    entropies: list[float] = []
    t = frame - 1

    while t < L - 1:
        pos = np.clip(np.arange(t - frame + 1, t + 1), 0, L - 1)
        mk = day_obs[pos]
        sf = self_hist[pos]
        obs_np = np.concatenate([mk, sf], axis=1).reshape(-1).astype(np.float32)
        obs = torch.from_numpy(obs_np).unsqueeze(0).to(device)

        op_dist, size_dist, value, h = brain.forward(obs, h)
        logits = op_dist.logits[0].detach().cpu().numpy()
        probs = _softmax(logits)
        chosen_op = int(torch.argmax(op_dist.logits, -1).item())
        chosen_size = float(size_dist.mean.clamp(0.05, 1.0).item())
        val = float(value.item())

        flags = _extract_pattern_flags(day_obs[t], col_index)
        self_snap = _self_from_obs(obs)

        rec = DecisionRecord(
            t=int(t),
            op_probs=[float(p) for p in probs],
            chosen_op=chosen_op,
            chosen_op_name=OP_NAMES.get(chosen_op, str(chosen_op)),
            chosen_size=chosen_size,
            value=val,
            cont_buy=flags["cont_buy"],
            cont_sell=flags["cont_sell"],
            pull_buy=flags["pull_buy"],
            pull_sell=flags["pull_sell"],
            rev_buy=flags["rev_buy"],
            rev_sell=flags["rev_sell"],
            mask_buy_blocked=flags["mask_buy_blocked"],
            mask_sell_blocked=flags["mask_sell_blocked"],
            dist_to_goal=self_snap["dist_to_goal"],
            dist_to_floor=self_snap["dist_to_floor"],
            open_risk=self_snap["open_risk"],
            position_sign=self_snap["position_sign"],
            trades_used=self_snap["trades_used"],
        )
        dump.decisions.append(rec)

        if flags["pull_buy"]:
            dump.n_pull_buy_bars += 1
            if chosen_op in (1, 3, 9):
                dump.pull_buy_seen_and_acted += 1
            elif chosen_op == 0:
                dump.pull_buy_seen_and_held += 1
        if flags["pull_sell"]:
            dump.n_pull_sell_bars += 1
            if chosen_op in (2, 4, 10):
                dump.pull_sell_seen_and_acted += 1
            elif chosen_op == 0:
                dump.pull_sell_seen_and_held += 1
        if flags["cont_buy"]:
            dump.n_cont_buy_bars += 1
        if flags["cont_sell"]:
            dump.n_cont_sell_bars += 1
        if flags["rev_buy"]:
            dump.n_rev_buy_bars += 1
        if flags["rev_sell"]:
            dump.n_rev_sell_bars += 1

        p = probs + 1e-12
        entropies.append(float(-(p * np.log(p)).sum()))
        t += max(1, decide_every)

    dump.n_decisions = len(dump.decisions)
    dump.mean_op_entropy = float(np.mean(entropies)) if entropies else 0.0
    dump.summary = _summarize(dump)
    return dump


def _summarize(d: DayMindDump) -> str:
    lines = [
        f"Brain {d.brain_name} on day {d.day_label} (goal {d.goal_pct}% / floor {d.floor_pct}%).",
        f"Decisions: {d.n_decisions}. Mean op entropy: {d.mean_op_entropy:.3f}.",
    ]
    if d.n_pull_buy_bars or d.n_pull_sell_bars:
        lines.append(
            f"Bread-and-butter (pull) bars: buy={d.n_pull_buy_bars} "
            f"(acted {d.pull_buy_seen_and_acted}, held {d.pull_buy_seen_and_held}); "
            f"sell={d.n_pull_sell_bars} "
            f"(acted {d.pull_sell_seen_and_acted}, held {d.pull_sell_seen_and_held})."
        )
        total_pull = d.n_pull_buy_bars + d.n_pull_sell_bars
        acted = d.pull_buy_seen_and_acted + d.pull_sell_seen_and_acted
        held = d.pull_buy_seen_and_held + d.pull_sell_seen_and_held
        if total_pull > 5 and held > 2 * max(1, acted):
            lines.append(
                "SIGNAL: pull pattern was frequently present but policy mostly held — "
                "candidate Perception or Policy issue for IRAC."
            )
    else:
        lines.append("No pull (bread-and-butter) flags present on this day in the observation.")
    if d.n_rev_buy_bars or d.n_rev_sell_bars:
        lines.append(
            f"Reversal flags present: rev_buy={d.n_rev_buy_bars}, rev_sell={d.n_rev_sell_bars}."
        )
    return " ".join(lines)


def load_and_probe(
    brain_name: str,
    day_obs: np.ndarray,
    cols: list[str],
    goal_pct: float = 3.0,
    floor_pct: float = 3.5,
    day_index: int = 0,
    day_label: str = "",
    day_phys: np.ndarray | None = None,
    decide_every: int = 5,
) -> DayMindDump:
    from inference.loader import load_brain

    brain, meta = load_brain(brain_name)
    if brain is None:
        raise FileNotFoundError(
            f"Could not load brain '{brain_name}': {meta.get('error', 'missing')}"
        )
    if day_phys is None:
        day_phys = np.zeros((day_obs.shape[0], 7), dtype=np.float32)
    return probe_day(
        brain=brain,
        day_obs=day_obs,
        day_phys=day_phys,
        cols=cols,
        goal_pct=goal_pct,
        floor_pct=floor_pct,
        brain_name=brain_name,
        day_index=day_index,
        day_label=day_label,
        decide_every=decide_every,
    )
