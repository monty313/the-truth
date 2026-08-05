"""Short finetune: second-best logit regret on flat HOLD (lineage only).

Loads channel1_curriculum_v2_hold_shape.pt, fine-tunes with missed-opportunity
penalty when 2nd-best entry would have been profitable after fees.
Pure greedy raw argmax eval (no decode ban). No PROVEN writes.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/finetune_second_best_regret.py
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
    SECOND_BEST_FEE_PX,
    SECOND_BEST_HORIZON_BARS,
    SECOND_BEST_REGRET_PENALTY,
    STRUCTURE_MATCH_ENTRY_BONUS,
    make_dials,
    second_best_entry_from_logits,
)
from lineages.adaptive_rl_brain_7_31_26.train_curriculum import (
    evaluate_policy,
    print_stats,
    save_ckpt,
    train_policy,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
SRC_CKPT = os.path.join(CKPT_DIR, "channel1_curriculum_v2_hold_shape.pt")
OUT_CKPT = os.path.join(CKPT_DIR, "channel1_curriculum_v3_second_best.pt")
OUT_LATEST = os.path.join(CKPT_DIR, "channel1_sandbox_latest.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "second_best_regret_report.json")

STEPS_PER_DAY = 40
DECIDE_EVERY = 25
REAL_EPOCHS = 14
REAL_BC_EPOCHS = 4
LR = 8e-4
SEED = 23
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
EVAL_DATES = {"2026-04-21", "2026-05-06"}


def _dials() -> dict:
    return make_dials(
        w_with_vector=1.0,
        w_qualified_macro=1.0,
        w_qualified_micro=0.4,
        w_inactivity=0.85,
    ).as_dict()


def mean_logits_on_days(policy: Channel1Policy, days: List) -> Dict[str, float]:
    all_logits: List[np.ndarray] = []
    for day in days:
        runner = DayRunner(
            day,
            decide_every=DECIDE_EVERY,
            dials=_dials(),
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
            action = int(np.argmax(logits))
            runner.step(t, action=action, logits=logits)
    if not all_logits:
        return {"HOLD": 0.0, "BUY": 0.0, "SELL": 0.0}
    m = np.mean(np.stack(all_logits), axis=0)
    return {"HOLD": float(m[0]), "BUY": float(m[1]), "SELL": float(m[2])}


def pure_greedy_detail(policy: Channel1Policy, days: List, metas: List) -> Dict[str, Any]:
    """Raw argmax only — no ban / anti-hold decode. Tracks 2nd-best CF stats."""
    chosen_c: Counter = Counter()
    tags: Counter = Counter()
    entries_by_day: Dict[str, int] = {}
    n_dir_flat = 0
    match_dir_flat = 0
    n_steps = 0
    n_hold_flat = 0
    n_sb_valid = 0
    n_sb_profitable = 0
    n_regret_applied = 0
    sb_side: Counter = Counter()

    for day, meta in zip(days, metas):
        date = str(meta.date)
        runner = DayRunner(
            day,
            decide_every=DECIDE_EVERY,
            dials=_dials(),
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
            action = int(np.argmax(logits))
            step = runner.step(t, action=action, logits=logits)
            tag = step.tag.value if hasattr(step.tag, "value") else str(step.tag)
            tags[tag] += 1
            chosen_c[NAMES[action]] += 1

            if was_flat and rec in (ACTION_BUY, ACTION_SELL):
                n_dir_flat += 1
                if action == rec:
                    match_dir_flat += 1

            if was_flat and action == ACTION_HOLD:
                n_hold_flat += 1
                sb = second_best_entry_from_logits(logits)
                if sb is not None:
                    n_sb_valid += 1
                    sb_side[NAMES[sb]] += 1
                    if step.info.get("second_best_profitable"):
                        n_sb_profitable += 1
                    if step.info.get("second_best_regret") is not None:
                        n_regret_applied += 1


        entries_by_day[date] = int(runner.n_entries)

    hold_rate = chosen_c.get("HOLD", 0) / max(n_steps, 1)
    buy_n = chosen_c.get("BUY", 0)
    sell_n = chosen_c.get("SELL", 0)
    return {
        "n_steps": n_steps,
        "chosen_counts": dict(chosen_c),
        "hold_rate": hold_rate,
        "tags": dict(tags),
        "entries_by_day": entries_by_day,
        "n_entries_total": int(sum(entries_by_day.values())),
        "entries_every_day": all(v > 0 for v in entries_by_day.values())
        if entries_by_day
        else False,
        "match_structure_on_dir_flat": {
            "n": n_dir_flat,
            "match": match_dir_flat,
            "rate": match_dir_flat / max(n_dir_flat, 1),
        },
        "mindless_rate": tags.get("MINDLESS", 0) / max(n_steps, 1),
        "second_best_on_hold": {
            "n_hold_flat": n_hold_flat,
            "n_second_best_valid": n_sb_valid,
            "n_second_best_profitable": n_sb_profitable,
            "profitable_rate_when_hold": n_sb_profitable / max(n_sb_valid, 1),
            "n_regret_applied": n_regret_applied,
            "second_best_side_counts": dict(sb_side),
        },
        "buy_vs_sell": {
            "buy": buy_n,
            "sell": sell_n,
            "sell_heavy": sell_n > buy_n,
            "buy_bias_improved": buy_n > 0,
        },
    }


def main() -> None:
    print("=" * 60, flush=True)
    print("SECOND-BEST REGRET FINETUNE — adaptive_rl_brain_7_31_26", flush=True)
    print("PROVEN: not touched", flush=True)
    print(
        f"REGRET={SECOND_BEST_REGRET_PENALTY} horizon={SECOND_BEST_HORIZON_BARS} "
        f"fee_px={SECOND_BEST_FEE_PX}",
        flush=True,
    )
    print(
        f"keep DIRECTIONAL_HOLD={DIRECTIONAL_HOLD_PENALTY} "
        f"STRUCTURE_MATCH={STRUCTURE_MATCH_ENTRY_BONUS}",
        flush=True,
    )
    print(f"src={SRC_CKPT}", flush=True)
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

    print("\n[2/3] FINETUNE (second-best regret; no HOLD ban in sampler)...", flush=True)
    curve = train_policy(
        policy,
        train_days,
        epochs=REAL_EPOCHS,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        lr=LR,
        bc_epochs=REAL_BC_EPOCHS,
        phase_name="second_best",
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

    meta_out = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source": src,
        "src_ckpt": os.path.abspath(SRC_CKPT),
        "out_ckpt": os.path.abspath(OUT_CKPT),
        "real_epochs": REAL_EPOCHS,
        "directional_hold_penalty": DIRECTIONAL_HOLD_PENALTY,
        "structure_match_entry_bonus": STRUCTURE_MATCH_ENTRY_BONUS,
        "second_best_regret_penalty": SECOND_BEST_REGRET_PENALTY,
        "second_best_horizon_bars": SECOND_BEST_HORIZON_BARS,
        "second_best_fee_px": SECOND_BEST_FEE_PX,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
        "train_dates": [m.date for m in train_meta],
        "eval_dates": [m.date for m in eval_meta],
        "before_pure_greedy": before_detail,
        "after_pure_greedy": after_detail,
        "before_mean_logits": before_logits,
        "after_mean_logits": after_logits,
        "train_epoch_means": curve,
        "proven_touched": False,
        "sandbox": True,
    }

    path = save_ckpt(
        policy,
        {
            **meta_out,
            "note": "second_best_regret v3; CF missed-opportunity on flat HOLD",
            "phase": "second_best_v3",
        },
        OUT_CKPT,
    )
    save_ckpt(
        policy,
        {
            **meta_out,
            "note": "second_best_v3 latest pointer",
            "phase": "second_best_v3",
        },
        OUT_LATEST,
    )

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                **meta_out,
                "checkpoint": path,
                "report": os.path.abspath(REPORT_PATH),
                "after_stats": {
                    "label": after_stats.label,
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
    print(f"mean_logits before={before_logits}", flush=True)
    print(f"mean_logits after={after_logits}", flush=True)
    print(
        f"2nd-best profitable when HOLD: "
        f"{after_detail['second_best_on_hold']}",
        flush=True,
    )
    print(f"mindless after={100 * after_detail['mindless_rate']:.1f}%", flush=True)
    print(f"buy_vs_sell after={after_detail['buy_vs_sell']}", flush=True)
    print(f"ckpt={path}", flush=True)
    print(f"report={REPORT_PATH}", flush=True)
    print("PROVEN untouched.", flush=True)


if __name__ == "__main__":
    main()
