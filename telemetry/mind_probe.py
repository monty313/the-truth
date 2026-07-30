"""Mind Probe — the MRI Scanner for conversational diagnosis.

CHANGE LOG (newest first):
- 2026-07-30  Categorical.probs + obs_dim assert; side-bias / wrong_side metrics — WHY: silent except→hold lied to IRAC; meta needs honest wrong_side class.
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

LONG_OPS = (1, 3, 9)
SHORT_OPS = (2, 4, 10)


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
    p_long: float = 0.0
    p_short: float = 0.0
    p_hold: float = 0.0
    wrong_side: bool = False


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
    # Side-bias toolkit (meta / IRAC)
    n_cont_buy_only: int = 0
    n_cont_sell_only: int = 0
    n_wrong_side_under_bull: int = 0
    n_wrong_side_under_bear: int = 0
    n_aligned_under_bull: int = 0
    n_aligned_under_bear: int = 0
    mean_p_long_under_bull: float = 0.0
    mean_p_short_under_bull: float = 0.0
    mean_p_long_under_bear: float = 0.0
    mean_p_short_under_bear: float = 0.0
    side_bias_bull: float = 0.0   # P(long)-P(short) under cont_buy only
    side_bias_bear: float = 0.0   # P(short)-P(long) under cont_sell only
    forward_ok: int = 0
    forward_fail: int = 0

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


def _brain_obs_dim(brain: Brain) -> int:
    return int(brain.inp[0].in_features)


def _op_probs_from_forward(result) -> tuple[np.ndarray, float, float]:
    """Brain.forward -> (op_dist, size_dist, value, h). Extract probs safely."""
    if isinstance(result, (tuple, list)):
        op_part = result[0]
        val = 0.0
        size_mean = 0.0
        if len(result) > 2 and torch.is_tensor(result[2]):
            val = float(result[2].reshape(-1)[0].item())
        if len(result) > 1 and hasattr(result[1], "mean"):
            try:
                size_mean = float(result[1].mean.reshape(-1)[0].item())
            except Exception:
                size_mean = 0.0
        if hasattr(op_part, "probs"):
            probs = op_part.probs.detach().cpu().numpy().reshape(-1)
        elif hasattr(op_part, "logits"):
            probs = _softmax(op_part.logits.detach().cpu().numpy().reshape(-1))
        elif torch.is_tensor(op_part):
            probs = _softmax(op_part.detach().cpu().numpy().reshape(-1))
        else:
            raise TypeError(f"unknown op head type: {type(op_part)}")
        return probs[:N_OPS].astype(np.float64), val, size_mean
    if torch.is_tensor(result):
        return _softmax(result.detach().cpu().numpy().reshape(-1)[:N_OPS]), 0.0, 0.0
    raise TypeError(f"unknown forward result: {type(result)}")


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


def side_metrics_from_decisions(decisions: list[DecisionRecord]) -> dict[str, float]:
    """Aggregate side-bias / wrong_side for IRAC + meta proposals."""
    bull_pl, bull_ps = [], []
    bear_pl, bear_ps = [], []
    n_cb = n_cs = 0
    wrong_bull = wrong_bear = 0
    align_bull = align_bear = 0
    for r in decisions:
        cb, cs = r.cont_buy, r.cont_sell
        p_long = r.p_long if r.p_long or r.op_probs else (
            float(sum(r.op_probs[i] for i in LONG_OPS if i < len(r.op_probs)))
            if r.op_probs else 0.0)
        p_short = r.p_short if r.p_short or r.op_probs else (
            float(sum(r.op_probs[i] for i in SHORT_OPS if i < len(r.op_probs)))
            if r.op_probs else 0.0)
        if cb and not cs:
            n_cb += 1
            bull_pl.append(p_long)
            bull_ps.append(p_short)
            if r.chosen_op in SHORT_OPS:
                wrong_bull += 1
            elif r.chosen_op in LONG_OPS:
                align_bull += 1
        elif cs and not cb:
            n_cs += 1
            bear_pl.append(p_long)
            bear_ps.append(p_short)
            if r.chosen_op in LONG_OPS:
                wrong_bear += 1
            elif r.chosen_op in SHORT_OPS:
                align_bear += 1
    mpl_b = float(np.mean(bull_pl)) if bull_pl else 0.0
    mps_b = float(np.mean(bull_ps)) if bull_ps else 0.0
    mpl_s = float(np.mean(bear_pl)) if bear_pl else 0.0
    mps_s = float(np.mean(bear_ps)) if bear_ps else 0.0
    return {
        "n_cont_buy_only": n_cb,
        "n_cont_sell_only": n_cs,
        "n_wrong_side_under_bull": wrong_bull,
        "n_wrong_side_under_bear": wrong_bear,
        "n_aligned_under_bull": align_bull,
        "n_aligned_under_bear": align_bear,
        "mean_p_long_under_bull": mpl_b,
        "mean_p_short_under_bull": mps_b,
        "mean_p_long_under_bear": mpl_s,
        "mean_p_short_under_bear": mps_s,
        "side_bias_bull": mpl_b - mps_b,
        "side_bias_bear": mps_s - mpl_s,
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
    from training.fastsim import SELF_DIM

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
    frame = 10
    self_dim = int(SELF_DIM) if SELF_DIM else 12
    need = _brain_obs_dim(brain)
    n_feat = int(day_obs.shape[1])
    # Strip signal slots if cache is SIGON-wide but brain is PROVEN 1820
    expected_market = need // frame - self_dim
    feat_slice = None
    if n_feat != expected_market and expected_market > 0:
        # Prefer first expected_market non-sig columns if cols known
        non_sig = [i for i, c in enumerate(cols) if not str(c).startswith("obs::sig_")]
        if len(non_sig) == expected_market:
            feat_slice = non_sig
            n_feat = expected_market
            col_index = {cols[i]: j for j, i in enumerate(non_sig)}
        elif n_feat > expected_market:
            feat_slice = list(range(expected_market))
            n_feat = expected_market
            col_index = {cols[i]: i for i in feat_slice if i < len(cols)}

    t = frame
    entropies = []
    while t < L:
        raw = day_obs[max(0, t - frame + 1): t + 1]
        if feat_slice is not None:
            window = raw[:, feat_slice]
        else:
            window = raw
        if window.shape[0] < frame:
            pad = np.repeat(window[:1], frame - window.shape[0], axis=0)
            window = np.concatenate([pad, window], axis=0)
        full = np.zeros((frame, n_feat + self_dim), dtype=np.float32)
        full[:, :n_feat] = window
        full[:, n_feat + 0] = float(goal_pct)
        full[:, n_feat + 1] = float(floor_pct)
        full[:, n_feat + 2] = 1.0
        full[:, n_feat + 3] = 1.0
        obs = torch.as_tensor(full.reshape(1, -1), device=device)
        if obs.shape[-1] != need:
            raise RuntimeError(
                f"Mind Probe obs_dim mismatch: built {obs.shape[-1]} vs brain {need} "
                f"(day_obs cols={day_obs.shape[1]}, market_used={n_feat}). "
                f"Use PROVEN cache (signal slots off) or matching brain."
            )
        try:
            result = brain(obs)
            probs, val, size_mean = _op_probs_from_forward(result)
            if probs.shape[0] < N_OPS:
                pad = np.zeros(N_OPS, dtype=np.float64)
                pad[: probs.shape[0]] = probs
                probs = pad
            chosen_op = int(probs.argmax())
            chosen_size = float(size_mean)
            dump.forward_ok += 1
        except Exception as e:
            # Do not silently invent pure-hold for dim errors (already raised).
            # Only soft-fail unexpected numeric issues — still mark fail.
            dump.forward_fail += 1
            probs = np.ones(N_OPS, dtype=np.float64) / N_OPS
            chosen_op = 0
            chosen_size = 0.0
            val = 0.0
            if dump.forward_fail <= 2:
                dump.summary += f" [forward_warn: {type(e).__name__}]"

        row_src = day_obs[t]
        if feat_slice is not None:
            row = row_src[feat_slice]
        else:
            row = row_src
        flags = _extract_pattern_flags(row, col_index)
        p_long = float(probs[list(LONG_OPS)].sum())
        p_short = float(probs[list(SHORT_OPS)].sum())
        p_hold = float(probs[0])
        wrong = False
        if flags["cont_buy"] and not flags["cont_sell"] and chosen_op in SHORT_OPS:
            wrong = True
        if flags["cont_sell"] and not flags["cont_buy"] and chosen_op in LONG_OPS:
            wrong = True
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
            op_probs=[float(p) for p in probs[:N_OPS]],
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
            htf_regime=state["htf_regime"],
            ltf_setup=state["ltf_setup"],
            setup_side=state["setup_side"],
            skip_reason=state["skip_reason"],
            matched_setup=state["matched_setup"],
            why=state["why"],
            p_long=p_long,
            p_short=p_short,
            p_hold=p_hold,
            wrong_side=wrong,
        )
        dump.decisions.append(rec)
        if flags["pull_buy"]:
            dump.n_pull_buy_bars += 1
            if chosen_op in LONG_OPS:
                dump.pull_buy_seen_and_acted += 1
            elif chosen_op == 0:
                dump.pull_buy_seen_and_held += 1
        if flags["pull_sell"]:
            dump.n_pull_sell_bars += 1
            if chosen_op in SHORT_OPS:
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
        p = np.clip(probs[:N_OPS], 1e-8, 1.0)
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
    sm = side_metrics_from_decisions(dump.decisions)
    for k, v in sm.items():
        setattr(dump, k, v if not isinstance(v, float) else float(v))
    dump.summary = _summarize(dump)
    return dump


def _summarize(d: DayMindDump) -> str:
    lines = [
        f"Brain {d.brain_name} on day {d.day_label} (goal {d.goal_pct}% / floor {d.floor_pct}%).",
        f"Decisions: {d.n_decisions}. Mean op entropy: {d.mean_op_entropy:.3f}.",
        f"Forward ok/fail: {d.forward_ok}/{d.forward_fail}.",
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
    lines.append(
        f"Side-bias: bull={d.side_bias_bull:+.3f} (wrong={d.n_wrong_side_under_bull}/"
        f"{d.n_cont_buy_only}) bear={d.side_bias_bear:+.3f} "
        f"(wrong={d.n_wrong_side_under_bear}/{d.n_cont_sell_only})."
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
