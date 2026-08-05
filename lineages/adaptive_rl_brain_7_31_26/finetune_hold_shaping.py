"""Short retrain: bake directional HOLD shaping into logits (lineage only).

Loads channel1_curriculum_v1.pt, fine-tunes on real curriculum with new
rewards (directional HOLD penalty + structure-match entry bonus). No hard
ban in the train sampler. Pure greedy eval without decode ban.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/finetune_hold_shaping.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.real_curriculum import load_real_curriculum
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    DID_NOTHING_EOD_PENALTY,
    DIRECTIONAL_HOLD_PENALTY,
    FLIP_FLOP_PENALTY,
    MAX_OPEN_UNITS,
    REVERSE_COOLDOWN_BARS,
    STRUCTURE_MATCH_ENTRY_BONUS,
    make_dials,
)
from lineages.adaptive_rl_brain_7_31_26.train_curriculum import (
    evaluate_policy,
    print_stats,
    save_ckpt,
    train_policy,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
SRC_CKPT = os.path.join(CKPT_DIR, "channel1_curriculum_v1.pt")
OUT_CKPT = os.path.join(CKPT_DIR, "channel1_curriculum_v2_hold_shape.pt")
OUT_LATEST = os.path.join(CKPT_DIR, "channel1_sandbox_latest.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "hold_shape_finetune_report.json")

STEPS_PER_DAY = 40
DECIDE_EVERY = 25
# Short retrain — reward shaping + CE only on flat+directional (no CE→HOLD)
REAL_EPOCHS = 16
REAL_BC_EPOCHS = 6
LR = 1e-3
SEED = 17
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
EVAL_DATES = {"2026-04-21", "2026-05-06"}


def mean_logits_on_days(policy: Channel1Policy, days: List) -> Dict[str, float]:
    dials = make_dials(
        w_with_vector=1.0,
        w_qualified_macro=1.0,
        w_qualified_micro=0.4,
        w_inactivity=0.85,
    ).as_dict()
    all_logits: List[np.ndarray] = []
    for day in days:
        runner = DayRunner(
            day,
            decide_every=DECIDE_EVERY,
            dials=dials,
            use_signal_majority=False,
            max_open_units=MAX_OPEN_UNITS,
            reverse_cooldown_bars=REVERSE_COOLDOWN_BARS,
            flip_flop_penalty_val=FLIP_FLOP_PENALTY,
        )
        idxs = runner.decision_indices()[:STEPS_PER_DAY]
        for t in idxs:
            obs = runner.observe(t)
            with torch.no_grad():
                logits = (
                    policy.forward(torch.as_tensor(obs, dtype=torch.float32))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            all_logits.append(logits)
            # advance state with pure greedy so path matches eval
            action = int(np.argmax(logits))
            runner.step(t, action=action)
    if not all_logits:
        return {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
    m = np.mean(np.stack(all_logits), axis=0)
    return {"HOLD": float(m[0]), "BUY": float(m[1]), "SELL": float(m[2])}


def pure_greedy_detail(policy: Channel1Policy, days: List, metas: List) -> Dict[str, Any]:
    """Raw argmax only — no ban_hold_if_directional, no anti_hold_greedy."""
    dials = make_dials(
        w_with_vector=1.0,
        w_qualified_macro=1.0,
        w_qualified_micro=0.4,
        w_inactivity=0.85,
    ).as_dict()
    chosen_c: Counter = Counter()
    tags: Counter = Counter()
    entries_by_day: Dict[str, int] = {}
    match_entries = 0
    n_entries = 0
    n_dir_flat = 0
    match_dir_flat = 0
    n_steps = 0

    for day, meta in zip(days, metas):
        date = str(meta.date)
        runner = DayRunner(
            day,
            decide_every=DECIDE_EVERY,
            dials=dials,
            use_signal_majority=False,
            max_open_units=MAX_OPEN_UNITS,
            reverse_cooldown_bars=REVERSE_COOLDOWN_BARS,
            flip_flop_penalty_val=FLIP_FLOP_PENALTY,
        )
        idxs = runner.decision_indices()[:STEPS_PER_DAY]
        for t in idxs:
            n_steps += 1
            was_flat = runner.position is None
            rec = int(runner.recommended_action(t)) if was_flat else ACTION_HOLD
            obs = runner.observe(t)
            with torch.no_grad():
                logits = (
                    policy.forward(torch.as_tensor(obs, dtype=torch.float32))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            action = int(np.argmax(logits))  # pure raw greedy
            step = runner.step(t, action=action)
            tag = step.tag.value if hasattr(step.tag, "value") else str(step.tag)
            tags[tag] += 1
            chosen_c[NAMES[action]] += 1
            if was_flat and rec in (ACTION_BUY, ACTION_SELL):
                n_dir_flat += 1
                if action == rec:
                    match_dir_flat += 1
            if step.info.get("entry") and not step.info.get("scale_in"):
                n_entries += 1
                if action == rec:
                    match_entries += 1
            elif step.info.get("entry"):
                n_entries += 1
                # scale-in: rec is HOLD while in pos — skip match
        entries_by_day[date] = int(runner.n_entries)

    hold_rate = chosen_c.get("HOLD", 0) / max(n_steps, 1)
    return {
        "n_steps": n_steps,
        "chosen_counts": dict(chosen_c),
        "hold_rate": hold_rate,
        "tags": dict(tags),
        "entries_by_day": entries_by_day,
        "n_entries_total": int(sum(entries_by_day.values())),
        "entries_every_day": all(v > 0 for v in entries_by_day.values()),
        "match_structure_on_dir_flat": {
            "n": n_dir_flat,
            "match": match_dir_flat,
            "rate": match_dir_flat / max(n_dir_flat, 1),
        },
        "mindless_rate": tags.get("MINDLESS", 0) / max(n_steps, 1),
    }


def main() -> None:
    print("=" * 60, flush=True)
    print("HOLD-SHAPING FINETUNE — lineage adaptive_rl_brain_7_31_26", flush=True)
    print("PROVEN: not touched", flush=True)
    print(
        f"DIRECTIONAL_HOLD_PENALTY={DIRECTIONAL_HOLD_PENALTY} "
        f"STRUCTURE_MATCH_ENTRY_BONUS={STRUCTURE_MATCH_ENTRY_BONUS}",
        flush=True,
    )
    print(f"src_ckpt={SRC_CKPT}", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if not os.path.isfile(SRC_CKPT):
        raise FileNotFoundError(SRC_CKPT)

    payload = torch.load(SRC_CKPT, map_location="cpu", weights_only=False)
    hidden = int(payload.get("hidden", 48))
    policy = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=hidden)
    policy.load_state_dict(payload["model"])
    print(f"loaded hidden={hidden}", flush=True)

    frames, meta, src = load_real_curriculum(n_trend=4, n_mix=2)
    if len(frames) >= 4:
        train_days = frames[:-2]
        eval_days = frames[-2:]
        train_meta = meta[:-2]
        eval_meta = meta[-2:]
    else:
        train_days = frames
        eval_days = frames
        train_meta = meta
        eval_meta = meta

    # Prefer named eval dates if present
    named = [(f, m) for f, m in zip(frames, meta) if str(m.date) in EVAL_DATES]
    if len(named) == 2:
        eval_days = [f for f, _ in named]
        eval_meta = [m for _, m in named]

    print(
        f"source={src} train={len(train_days)} eval={len(eval_days)} "
        f"epochs={REAL_EPOCHS}",
        flush=True,
    )

    print("\n[1/3] BEFORE pure greedy (raw argmax)...", flush=True)
    before_detail = pure_greedy_detail(policy, eval_days, eval_meta)
    before_stats = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="BEFORE_pure_greedy",
        ban_hold_if_directional=False,
        anti_hold_greedy=False,
    )
    print_stats(before_stats)
    print("before_detail", before_detail, flush=True)
    before_logits = mean_logits_on_days(policy, eval_days)
    print("before_mean_logits", before_logits, flush=True)

    print("\n[2/3] FINETUNE (rewards only; no HOLD ban in sampler)...", flush=True)
    curve = train_policy(
        policy,
        train_days,
        epochs=REAL_EPOCHS,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        lr=LR,
        bc_epochs=REAL_BC_EPOCHS,
        phase_name="hold_shape",
    )

    print("\n[3/3] AFTER pure greedy (raw argmax, NO decode ban)...", flush=True)
    after_detail = pure_greedy_detail(policy, eval_days, eval_meta)
    after_stats = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_pure_greedy",
        ban_hold_if_directional=False,
        anti_hold_greedy=False,
    )
    print_stats(after_stats)
    print("after_detail", after_detail, flush=True)
    after_logits = mean_logits_on_days(policy, eval_days)
    print("after_mean_logits", after_logits, flush=True)

    # Optional: still report ban decode for comparison
    after_ban = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_ban_decode",
        ban_hold_if_directional=True,
    )
    print_stats(after_ban)

    meta_out = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source": src,
        "src_ckpt": os.path.abspath(SRC_CKPT),
        "out_ckpt": os.path.abspath(OUT_CKPT),
        "real_epochs": REAL_EPOCHS,
        "real_bc_epochs": REAL_BC_EPOCHS,
        "directional_hold_penalty": DIRECTIONAL_HOLD_PENALTY,
        "structure_match_entry_bonus": STRUCTURE_MATCH_ENTRY_BONUS,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
        "train_dates": [m.date for m in train_meta],
        "eval_dates": [m.date for m in eval_meta],
        "before_pure_greedy": before_detail,
        "after_pure_greedy": after_detail,
        "before_mean_logits": before_logits,
        "after_mean_logits": after_logits,
        "before_hold_rate": before_detail["hold_rate"],
        "after_hold_rate": after_detail["hold_rate"],
        "after_entries_by_day": after_detail["entries_by_day"],
        "after_actions": after_stats.actions,
        "train_epoch_means": curve,
        "success_raw_not_100_hold": after_detail["hold_rate"] < 1.0,
        "success_entries": after_detail["n_entries_total"] > 0,
        "proven_touched": False,
        "sandbox": True,
    }

    path = save_ckpt(
        policy,
        {
            **meta_out,
            "note": "hold_shape_v2 finetune; directional reward shaping",
            "phase": "hold_shape_v2",
        },
        OUT_CKPT,
    )
    # also refresh latest pointer inside lineage
    save_ckpt(
        policy,
        {
            **meta_out,
            "note": "hold_shape_v2 latest pointer",
            "phase": "hold_shape_v2",
        },
        OUT_LATEST,
    )

    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                **meta_out,
                "checkpoint": path,
                "report": os.path.abspath(REPORT_PATH),
                "after_stats": {
                    "label": after_stats.label,
                    "n_steps": after_stats.n_steps,
                    "actions": after_stats.actions,
                    "tags": after_stats.tags,
                    "hold_rate": after_stats.hold_rate,
                    "mindless_rate": after_stats.mindless_rate,
                    "mean_reward": after_stats.mean_reward,
                    "n_entries_total": after_stats.n_entries_total,
                    "days_with_entry": after_stats.days_with_entry,
                    "days_did_nothing": after_stats.days_did_nothing,
                },
            },
            f,
            indent=2,
        )

    print("\n=== SUMMARY (pure greedy raw argmax) ===", flush=True)
    print(
        f"hold_rate before={100 * before_detail['hold_rate']:.1f}% "
        f"after={100 * after_detail['hold_rate']:.1f}%",
        flush=True,
    )
    print(f"entries_by_day after={after_detail['entries_by_day']}", flush=True)
    print(
        f"match_dir_flat after="
        f"{after_detail['match_structure_on_dir_flat']}",
        flush=True,
    )
    print(f"mindless after={100 * after_detail['mindless_rate']:.1f}%", flush=True)
    print(f"mean_logits before={before_logits}", flush=True)
    print(f"mean_logits after={after_logits}", flush=True)
    print(f"ckpt={path}", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print("PROVEN untouched.", flush=True)


if __name__ == "__main__":
    main()
