"""Mind Probe — the MRI Scanner for conversational diagnosis.

CHANGE LOG (newest first):
- 2026-07-24  regime language on every DecisionRecord — WHY: document HTF regime + LTF setup + skip_reason for trend-without-entry audits.
- 2026-07-24  created — WHY: Phase 1 MRI Scanner.
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
from telemetry.regime_language import document_decision, summarize_day_skips

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
    htf_regime: str = "unknown"
    ltf_setup: str = "none"
    setup_side: str = "none"
    skip_reason: str = "flat_ok"
    matched_setup: bool = False
    why: str = ""


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
    n_policy_hold_on_setup: int = 0
    n_mask_veto: int = 0
    n_no_ltf_setup: int = 0
    skip_counts: dict = field(default_factory=dict)

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
    from training.fastsim import FastSim, SELF_DIM

    col_index = {c: i for i, c in enumerate(cols)}
    L = int(day_obs.shape[0])
    dump = DayMindDump(
        brain_name=brain_name,
        day_index=day_index,
        day_label=day_label or str(day_index),
        goal_pct=goal_pct,
        floor_pct=floor_pct,
    )
    brain = brain.to(device)
    brain.eval()
    # Minimal step loop: frame-stack last 10 rows as obs window when available
    frame = 10
    self_dim = SELF_DIM if hasattr(__import__('training.fastsim', fromlist=['SELF_DIM']), 'SELF_DIM') else 12
    t = frame
    entropies = []
    while t < L:
        window = day_obs[max(0, t - frame + 1): t + 1]
        if window.shape[0] < frame:
            pad = np.repeat(window[:1], frame - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        # append zeros for self-state if not in matrix
        flat = window.reshape(-1).astype(np.float32)
        # policy expects obs_dim = frame * (n_cols + self) often; use brain path via logits
        try:
            obs = torch.as_tensor(flat, device=device).unsqueeze(0)
            # If dim mismatch, pad/truncate to brain input
            need = None
            try:
                # probe via get_action if available
                out = brain(obs) if False else None
            except Exception:
                out = None
            # Use inference-style: expand to expected dim if needed
            from training.policy import Brain as _B
            # fallback: construct full obs with self zeros
            n_feat = day_obs.shape[1]
            full = np.zeros((frame, n_feat + 12), dtype=np.float32)
            full[:, :n_feat] = window
            obs = torch.as_tensor(full.reshape(1, -1), device=device)
            logits, val, size_p = None, 0.0, 0.0
            if hasattr(brain, "forward"):
                result = brain(obs)
                if isinstance(result, (tuple, list)):
                    logits = result[0]
                    val = float(result[1].reshape(-1)[0].item()) if len(result) > 1 else 0.0
                else:
                    logits = result
            if logits is None:
                t += max(1, decide_every)
                continue
            probs = _softmax(logits.detach().cpu().numpy().reshape(-1)[:N_OPS])
            chosen_op = int(probs.argmax())
            chosen_size = 0.0
        except Exception:
            # last-resort: uniform hold
            probs = np.zeros(N_OPS, dtype=np.float64); probs[0] = 1.0
            chosen_op = 0
            chosen_size = 0.0
            val = 0.0

        row = day_obs[t]
        flags = _extract_pattern_flags(row, col_index)
        self_snap = {
            "dist_to_goal": 0.0,
            "dist_to_floor": 0.0,
            "open_risk": 0.0,
            "position_sign": 0.0,
            "trades_used": 0.0,
        }
        op_name = OP_NAMES.get(chosen_op, str(chosen_op))
        state = document_decision(
            cont_buy=flags["cont_buy"],
            cont_sell=flags["cont_sell"],
            pull_buy=flags["pull_buy"],
            pull_sell=flags["pull_sell"],
            rev_buy=flags["rev_buy"],
            rev_sell=flags["rev_sell"],
            mask_buy_blocked=flags["mask_buy_blocked"],
            mask_sell_blocked=flags["mask_sell_blocked"],
            chosen_op_name=op_name,
        )
        rec = DecisionRecord(
            t=int(t),
            op_probs=[float(p) for p in probs],
            chosen_op=chosen_op,
            chosen_op_name=op_name,
            chosen_size=float(chosen_size),
            value=float(val),
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
            htf_regime=state["htf_regime"],
            ltf_setup=state["ltf_setup"],
            setup_side=state["setup_side"],
            skip_reason=state["skip_reason"],
            matched_setup=state["matched_setup"],
            why=state["why"],
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
        p = np.clip(probs, 1e-8, 1.0)
        entropies.append(float(-(p * np.log(p)).sum()))
        t += max(1, decide_every)

    dump.n_decisions = len(dump.decisions)
    dump.mean_op_entropy = float(np.mean(entropies)) if entropies else 0.0
    docs = [
        {"skip_reason": r.skip_reason, "ltf_setup": r.ltf_setup, "htf_regime": r.htf_regime}
        for r in dump.decisions
    ]
    agg = summarize_day_skips(docs)
    dump.n_policy_hold_on_setup = int(agg["n_policy_hold_on_setup"])
    dump.n_mask_veto = int(agg["n_mask_veto"])
    dump.n_no_ltf_setup = int(agg["n_no_ltf_setup"])
    dump.skip_counts = dict(agg["skip_counts"])
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
    else:
        lines.append("No pull (bread-and-butter) flags present on this day in the observation.")
    if d.n_rev_buy_bars or d.n_rev_sell_bars:
        lines.append(
            f"Reversal flags present: rev_buy={d.n_rev_buy_bars}, rev_sell={d.n_rev_sell_bars}."
        )
    if getattr(d, "skip_counts", None):
        lines.append(
            f"Regime skips: policy_hold_on_setup={d.n_policy_hold_on_setup}, "
            f"mask_veto={d.n_mask_veto}, no_ltf_setup={d.n_no_ltf_setup}, "
            f"counts={d.skip_counts}."
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
