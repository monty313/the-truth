"""Probe: pure greedy with optional directional HOLD ban (no retrain).

Default (this experiment): if structure rec is BUY/SELL, argmax among BUY/SELL
only; HOLD allowed only when structure rec is HOLD.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/probe_hold_logits.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter
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
    greedy_action_ban_hold_if_directional,
)
from lineages.adaptive_rl_brain_7_31_26.real_curriculum import load_real_curriculum
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    FLIP_FLOP_PENALTY,
    MAX_OPEN_UNITS,
    REVERSE_COOLDOWN_BARS,
    make_dials,
)
from lineages.adaptive_rl_brain_7_31_26.train_curriculum import evaluate_policy

CKPT = os.path.join(_HERE, "checkpoints", "channel1_curriculum_v1.pt")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
EVAL_DATES = {"2026-04-21", "2026-05-06"}
STEPS_PER_DAY = 40
DECIDE_EVERY = 25
# Experiment flag: ban HOLD when structure is directional (pure greedy only)
BAN_HOLD_IF_DIRECTIONAL = True


def main() -> None:
    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    state = payload["model"]
    hidden = int(payload.get("hidden", state["net.0.weight"].shape[0]))
    print(f"ckpt={CKPT}", flush=True)
    print(f"hidden={hidden} obs_dim={CHANNEL1_DIM}", flush=True)
    print(f"BAN_HOLD_IF_DIRECTIONAL={BAN_HOLD_IF_DIRECTIONAL}", flush=True)

    policy = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=hidden)
    policy.load_state_dict(state)
    policy.eval()

    frames, meta, src = load_real_curriculum(n_trend=4, n_mix=2)
    print(f"source={src} n_days={len(frames)}", flush=True)

    eval_pairs = [
        (f, m) for f, m in zip(frames, meta) if str(m.date) in EVAL_DATES
    ]
    if not eval_pairs:
        eval_pairs = list(zip(frames[-2:], meta[-2:]))
        print("fallback: last 2 curriculum days", flush=True)

    dials = make_dials(
        w_with_vector=1.0,
        w_qualified_macro=1.0,
        w_qualified_micro=0.4,
        w_inactivity=0.85,
    ).as_dict()

    rows: List[Dict[str, Any]] = []
    second_best_raw: Counter = Counter()
    margins_raw: List[float] = []
    rec_vs_chosen: Counter = Counter()
    match_rec = 0
    n_dir_rec = 0
    tags: Counter = Counter()
    entries_by_day: Dict[str, int] = {}
    chosen_counts: Counter = Counter()

    for frame, m in eval_pairs:
        date = str(m.date)
        runner = DayRunner(
            frame,
            decide_every=DECIDE_EVERY,
            dials=dials,
            use_signal_majority=False,
            max_open_units=MAX_OPEN_UNITS,
            reverse_cooldown_bars=REVERSE_COOLDOWN_BARS,
            flip_flop_penalty_val=FLIP_FLOP_PENALTY,
        )
        idxs = runner.decision_indices()[:STEPS_PER_DAY]
        print(f"day {date} decisions={len(idxs)}", flush=True)
        day_entries_before = runner.n_entries
        for t in idxs:
            obs = runner.observe(t)
            rec = int(runner.recommended_action(t))
            with torch.no_grad():
                logits = (
                    policy.forward(torch.as_tensor(obs, dtype=torch.float32))
                    .squeeze(0)
                    .cpu()
                    .numpy()
                )
            probs = torch.softmax(torch.tensor(logits), dim=-1).numpy()
            order = list(np.argsort(-logits))
            raw_chosen = int(order[0])
            raw_second = int(order[1])
            margin_raw = float(logits[raw_chosen] - logits[raw_second])

            if BAN_HOLD_IF_DIRECTIONAL and runner.position is None:
                chosen = greedy_action_ban_hold_if_directional(logits, rec)
            else:
                chosen = raw_chosen
            # In-trade pure greedy still uses policy (manage not part of this ban)
            if runner.position is not None:
                chosen = raw_chosen

            step = runner.step(t, action=int(chosen))
            tag = step.tag.value if hasattr(step.tag, "value") else str(step.tag)
            tags[tag] += 1
            chosen_counts[NAMES[int(chosen)]] += 1

            row = {
                "date": date,
                "t": int(t),
                "raw_chosen": NAMES[raw_chosen],
                "raw_second": NAMES[raw_second],
                "chosen": NAMES[int(chosen)],
                "logits": {NAMES[i]: float(logits[i]) for i in range(3)},
                "probs": {NAMES[i]: float(probs[i]) for i in range(3)},
                "margin_raw_1_2": margin_raw,
                "rec": NAMES[rec],
                "tag": tag,
                "match_rec": int(chosen) == rec,
            }
            rows.append(row)
            rec_vs_chosen[(NAMES[rec], NAMES[int(chosen)])] += 1
            if rec != ACTION_HOLD:
                n_dir_rec += 1
                if int(chosen) == rec:
                    match_rec += 1
            if raw_chosen == ACTION_HOLD:
                second_best_raw[NAMES[raw_second]] += 1
                margins_raw.append(margin_raw)

        entries_by_day[date] = int(runner.n_entries) - int(day_entries_before)
        # day_entries_before was 0 at start; n_entries is cumulative for the day
        entries_by_day[date] = int(runner.n_entries)

    print("\n=== PURE GREEDY PROBE (directional HOLD ban) ===", flush=True)
    print(f"n_decisions={len(rows)}", flush=True)
    print("chosen_counts", dict(chosen_counts), flush=True)
    raw_c = Counter(r["raw_chosen"] for r in rows)
    print("raw_argmax_counts (before ban)", dict(raw_c), flush=True)
    print("second_best_WHEN_RAW_HOLD", dict(second_best_raw), flush=True)
    if margins_raw:
        print(
            "raw margin(hold-2nd): "
            f"mean={float(np.mean(margins_raw)):.4f} "
            f"median={float(np.median(margins_raw)):.4f}",
            flush=True,
        )
    print(
        "rec_vs_chosen",
        {f"{a}->{b}": c for (a, b), c in rec_vs_chosen.most_common()},
        flush=True,
    )
    print(
        f"chosen_matches_structure_rec (directional only): "
        f"{match_rec}/{n_dir_rec} = {100.0 * match_rec / max(n_dir_rec, 1):.1f}%",
        flush=True,
    )
    print("tag_distribution", dict(tags), flush=True)
    print("entries_by_day", entries_by_day, flush=True)
    print(
        f"entries_every_eval_day: {all(v > 0 for v in entries_by_day.values())} "
        f"(days={list(entries_by_day.keys())})",
        flush=True,
    )
    hold_rate = chosen_counts.get("HOLD", 0) / max(len(rows), 1)
    print(f"hold_rate={100 * hold_rate:.1f}%", flush=True)

    print("\n=== EXAMPLE steps (first 12) ===", flush=True)
    for r in rows[:12]:
        print(
            f"  {r['date']} t={r['t']} tag={r['tag']} rec={r['rec']} | "
            f"raw={r['raw_chosen']}→exec={r['chosen']} | "
            f"logits H={r['logits']['HOLD']:+.3f} "
            f"B={r['logits']['BUY']:+.3f} "
            f"S={r['logits']['SELL']:+.3f} | "
            f"match_rec={r['match_rec']}",
            flush=True,
        )

    # Also curriculum evaluate_policy path with same ban
    print("\n=== curriculum evaluate_policy (ban_hold_if_directional=True) ===", flush=True)
    eval_frames = [f for f, _ in eval_pairs]
    stats = evaluate_policy(
        policy,
        eval_frames,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_greedy_ban_hold_directional",
        ban_hold_if_directional=True,
    )
    print(
        f"label={stats.label} n_steps={stats.n_steps} "
        f"entries={stats.n_entries_total} hold_rate={100 * stats.hold_rate:.1f}% "
        f"days_with_entry={stats.days_with_entry} "
        f"days_did_nothing={stats.days_did_nothing}",
        flush=True,
    )
    print("actions", stats.actions, flush=True)
    print("tags", stats.tags, flush=True)
    print(
        f"mindless_rate={100 * stats.mindless_rate:.1f}% "
        f"good_tag_rate={100 * stats.good_tag_rate:.1f}% "
        f"mean_reward={stats.mean_reward:+.4f}",
        flush=True,
    )
    print(f"eod_penalties={stats.eod_penalties}", flush=True)
    print("proven_touched=False (probe only; no ckpt write)", flush=True)


if __name__ == "__main__":
    main()
