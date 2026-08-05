"""Phase C — first serious Channel1 train on real curriculum days.

CHANGE LOG:
- 2026-07-31  curriculum train — WHY: take sandbox entries → real-data train
  with thrash limits. Checkpoints ONLY under this lineage. Never PROVEN.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/train_curriculum.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.curriculum_data import curriculum_days
from lineages.adaptive_rl_brain_7_31_26.day_runner import DayRunner
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
    greedy_action_ban_hold_if_directional,
)
from lineages.adaptive_rl_brain_7_31_26.real_curriculum import (
    load_real_curriculum,
    verify_mtf_on_days,
    write_curriculum_docs,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    DID_NOTHING_EOD_PENALTY,
    FLIP_FLOP_PENALTY,
    MAX_OPEN_UNITS,
    REVERSE_COOLDOWN_BARS,
    make_dials,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "channel1_curriculum_v1.pt")
CKPT_LATEST = os.path.join(CKPT_DIR, "channel1_sandbox_latest.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "curriculum_train_report.json")

# Serious-but-bounded first run
STEPS_PER_DAY = 40
DECIDE_EVERY = 25
# Warm on synthetic thrust (anti-hold green path), then real fine-tune
WARM_EPOCHS = 10
WARM_BC_EPOCHS = 5
REAL_EPOCHS = 14
REAL_BC_EPOCHS = 4
LR = 1e-3
HIDDEN = 48
SEED = 11
ENTROPY_COEF = 0.03
IMITATION_COEF = 1.1
GUIDE_PROB = 0.65
# Majority panel is slow on multi-day real train; anti-hold + thrash are primary.
USE_SIGNAL_MAJORITY = False

PRACTICE_DIALS = make_dials(
    w_with_vector=1.0,
    w_qualified_macro=1.0,
    w_qualified_micro=0.4,
    w_inactivity=0.85,
).as_dict()


@dataclass
class PhaseStats:
    label: str
    n_steps: int = 0
    total_reward: float = 0.0
    mean_reward: float = 0.0
    tags: Dict[str, int] = field(default_factory=dict)
    actions: Dict[str, int] = field(default_factory=dict)
    mindless_rate: float = 0.0
    good_tag_rate: float = 0.0
    reward_curve: List[float] = field(default_factory=list)
    eod_pnls: List[float] = field(default_factory=list)
    eod_penalties: List[float] = field(default_factory=list)
    n_entries_total: int = 0
    days_with_entry: int = 0
    days_did_nothing: int = 0
    n_reverses: int = 0
    n_scale_blocks: int = 0
    n_cooldown_blocks: int = 0
    hold_rate: float = 0.0


def _stats_from_steps(
    label: str,
    steps,
    day_means: List[float],
    eod_pnls: List[float],
    eod_penalties: List[float],
    n_entries_list: List[int],
    thrash: Dict[str, int],
) -> PhaseStats:
    tags = Counter(s.tag.value for s in steps)
    acts = Counter(s.action for s in steps)
    rews = [float(s.reward) for s in steps]
    n = len(steps)
    mindless = tags.get("MINDLESS", 0)
    good = (
        tags.get("WITH_VECTOR", 0)
        + tags.get("QUALIFIED_MACRO", 0)
        + tags.get("QUALIFIED_MICRO", 0)
    )
    n_entries_total = int(sum(n_entries_list))
    days_with = sum(1 for e in n_entries_list if e > 0)
    days_nothing = sum(
        1 for p, e in zip(eod_pnls, n_entries_list) if e == 0 and abs(p) < 1e-6
    )
    hold = int(acts.get(ACTION_HOLD, 0))
    return PhaseStats(
        label=label,
        n_steps=n,
        total_reward=float(sum(rews)),
        mean_reward=float(np.mean(rews)) if n else 0.0,
        tags=dict(tags),
        actions={
            "hold": hold,
            "buy": int(acts.get(ACTION_BUY, 0)),
            "sell": int(acts.get(ACTION_SELL, 0)),
        },
        mindless_rate=float(mindless) / float(n) if n else 0.0,
        good_tag_rate=float(good) / float(n) if n else 0.0,
        reward_curve=list(day_means),
        eod_pnls=list(eod_pnls),
        eod_penalties=list(eod_penalties),
        n_entries_total=n_entries_total,
        days_with_entry=int(days_with),
        days_did_nothing=int(days_nothing),
        n_reverses=int(thrash.get("n_reverses", 0)),
        n_scale_blocks=int(thrash.get("n_scale_blocks", 0)),
        n_cooldown_blocks=int(thrash.get("n_cooldown_blocks", 0)),
        hold_rate=float(hold) / float(n) if n else 0.0,
    )


def _run_day(
    policy: Channel1Policy,
    day,
    *,
    steps_per_day: int,
    decide_every: int,
    greedy: bool,
    train_mode: bool,
    guide_prob: float = 0.0,
    anti_hold_greedy: bool = False,
    ban_hold_if_directional: bool = False,
) -> Tuple[List[Any], float, float, int, List, List[float], List, List, Dict[str, int]]:
    runner = DayRunner(
        day,
        decide_every=decide_every,
        dials=PRACTICE_DIALS,
        use_signal_majority=USE_SIGNAL_MAJORITY,
        max_open_units=MAX_OPEN_UNITS,
        reverse_cooldown_bars=REVERSE_COOLDOWN_BARS,
        flip_flop_penalty_val=FLIP_FLOP_PENALTY,
    )
    idxs = runner.decision_indices()[:steps_per_day]
    steps = []
    logps: List[torch.Tensor] = []
    ents: List[torch.Tensor] = []
    ce_losses: List[torch.Tensor] = []
    rews: List[float] = []
    for t in idxs:
        obs = runner.observe(t)
        rec = int(runner.recommended_action(t))
        logits_np: Optional[np.ndarray] = None
        if train_mode:
            logits = policy.forward(torch.as_tensor(obs, dtype=torch.float32)).squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            if runner.position is not None:
                if float(np.random.random()) < max(guide_prob, 0.75):
                    action = ACTION_HOLD
                else:
                    action = int(dist.sample().item())
            elif (
                rec != ACTION_HOLD
                and guide_prob > 0.0
                and float(np.random.random()) < float(guide_prob)
            ):
                action = rec
            else:
                action = int(dist.sample().item())
            logp = dist.log_prob(torch.tensor(action))
            logps.append(logp)
            ents.append(dist.entropy())
            # CE only when flat + structure directional.
            # Never CE→HOLD: obs has no position bit, so in-trade HOLD labels
            # were poisoning flat bars into all-hold under argmax.
            if runner.position is None and rec != ACTION_HOLD:
                ce_losses.append(
                    torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([rec], dtype=torch.long),
                    )
                )
            logits_np = logits.detach().cpu().numpy()
        else:
            with torch.no_grad():
                logits_t = policy.forward(
                    torch.as_tensor(obs, dtype=torch.float32)
                ).squeeze(0)
                logits_np = logits_t.cpu().numpy()
            if greedy and ban_hold_if_directional and runner.position is None:
                # Experiment: pure greedy with HOLD banned when structure is
                # BUY/SELL — argmax among entry sides only. No retrain.
                action = greedy_action_ban_hold_if_directional(logits_t, rec)
            else:
                action, _ = policy.act(obs, greedy=greedy)
                action = int(action)
            # Greedy manage: while in a trade, hold if higher TF still agrees
            # (allows exit/reverse only when higher flipped — cuts thrash)
            if greedy and runner.position is not None:
                perc = runner.perceive(t)
                higher = perc["higher"]
                pos = runner.position
                still_agrees = (
                    higher.name == "NEUTRAL"
                    or (pos.name == "BULL" and higher.name == "BULL")
                    or (pos.name == "BEAR" and higher.name == "BEAR")
                )
                if still_agrees:
                    action = ACTION_HOLD
            # Anti-hold greedy: if flat + policy holds but structure recommends
            # a side, take the recommendation (breaks pure all-hold collapse)
            if anti_hold_greedy and runner.position is None and action == ACTION_HOLD:
                if rec != ACTION_HOLD:
                    action = rec
        step = runner.step(t, action=action, logits=logits_np)
        steps.append(step)
        rews.append(float(step.reward))

    eod = float(runner.end_day())
    if steps and eod != 0.0:
        last = steps[-1]
        last.reward = float(last.reward) + eod
        last.info = dict(last.info)
        last.info["eod_penalty"] = eod
        rews[-1] = float(last.reward)

    thrash = {
        "n_reverses": int(runner.n_reverses),
        "n_scale_blocks": int(runner.n_scale_blocks),
        "n_cooldown_blocks": int(runner.n_cooldown_blocks),
    }
    return (
        steps,
        float(runner.realized),
        eod,
        int(runner.n_entries),
        logps,
        rews,
        ents,
        ce_losses,
        thrash,
    )


@torch.no_grad()
def evaluate_policy(
    policy: Channel1Policy,
    days: List,
    *,
    steps_per_day: int,
    decide_every: int,
    greedy: bool = True,
    label: str = "eval",
    anti_hold_greedy: bool = False,
    ban_hold_if_directional: bool = False,
) -> PhaseStats:
    policy.eval()
    all_steps = []
    day_means: List[float] = []
    eod_pnls: List[float] = []
    eod_penalties: List[float] = []
    n_entries_list: List[int] = []
    thrash_tot = {"n_reverses": 0, "n_scale_blocks": 0, "n_cooldown_blocks": 0}
    for day in days:
        steps, realized, eod, n_ent, _, rews, _, _, thr = _run_day(
            policy,
            day,
            steps_per_day=steps_per_day,
            decide_every=decide_every,
            greedy=greedy,
            train_mode=False,
            anti_hold_greedy=anti_hold_greedy,
            ban_hold_if_directional=ban_hold_if_directional,
        )
        all_steps.extend(steps)
        day_means.append(float(np.mean(rews)) if rews else 0.0)
        eod_pnls.append(realized)
        eod_penalties.append(eod)
        n_entries_list.append(n_ent)
        for k in thrash_tot:
            thrash_tot[k] += thr[k]
    return _stats_from_steps(
        label, all_steps, day_means, eod_pnls, eod_penalties, n_entries_list, thrash_tot
    )


def train_policy(
    policy: Channel1Policy,
    days: List,
    *,
    epochs: int,
    steps_per_day: int,
    decide_every: int,
    lr: float,
    bc_epochs: int,
    phase_name: str = "train",
) -> List[float]:
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    epoch_means: List[float] = []
    policy.train()
    for ep in range(epochs):
        ep_rewards: List[float] = []
        ep_logps: List[torch.Tensor] = []
        ep_ents: List[torch.Tensor] = []
        ep_ces: List[torch.Tensor] = []
        bc_only = ep < int(bc_epochs)
        if bc_only:
            gprob = 0.92
        else:
            frac = (ep - bc_epochs) / max(epochs - bc_epochs, 1)
            gprob = float(GUIDE_PROB) * max(0.45, 1.0 - frac)
        for day in days:
            _steps, _realized, _eod, _n_ent, logps, rews, ents, ces, _thr = _run_day(
                policy,
                day,
                steps_per_day=steps_per_day,
                decide_every=decide_every,
                greedy=False,
                train_mode=True,
                guide_prob=gprob,
            )
            ep_logps.extend(logps)
            ep_rewards.extend(rews)
            ep_ents.extend(ents)
            ep_ces.extend(ces)
        if not ep_logps and not ep_ces:
            epoch_means.append(0.0)
            continue
        ent = torch.stack(ep_ents).mean() if ep_ents else torch.tensor(0.0)
        ce = torch.stack(ep_ces).mean() if ep_ces else torch.tensor(0.0)
        if bc_only:
            loss = ce if ep_ces else torch.tensor(0.0, requires_grad=True)
            mode = "BC"
        else:
            r = torch.tensor(ep_rewards, dtype=torch.float32)
            r = r - r.mean()
            reinforce = -(torch.stack(ep_logps) * r.detach()).mean()
            loss = reinforce - float(ENTROPY_COEF) * ent + float(IMITATION_COEF) * ce
            mode = "RF"
        opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        epoch_means.append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)
        print(
            f"  [{phase_name}] epoch {ep + 1:02d}/{epochs} [{mode}]  "
            f"mean_reward={epoch_means[-1]:+.4f}  loss={float(loss.item()):.4f}  "
            f"guide={gprob:.2f}",
            flush=True,
        )
    return epoch_means


def save_ckpt(policy: Channel1Policy, meta: dict, path: str) -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    abs_path = os.path.abspath(path)
    norm = abs_path.replace("\\", "/")
    if "/models/" in norm and "lineages" not in norm:
        raise RuntimeError("refusing to write outside lineage")
    payload = {
        "model": policy.state_dict(),
        "obs_dim": CHANNEL1_DIM,
        "hidden": HIDDEN,
        "meta": meta,
        "lineage": "adaptive_rl_brain_7_31_26",
        "note": "CURRICULUM v1 sandbox — not PROVEN",
        "anti_hold": True,
        "thrash_control": {
            "max_open_units": MAX_OPEN_UNITS,
            "reverse_cooldown_bars": REVERSE_COOLDOWN_BARS,
            "flip_flop_penalty": FLIP_FLOP_PENALTY,
        },
        "dials": PRACTICE_DIALS,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
    }
    torch.save(payload, path)
    return path


def print_stats(s: PhaseStats) -> None:
    print(f"\n--- {s.label} ---", flush=True)
    print(f"steps: {s.n_steps}", flush=True)
    print(f"mean_reward: {s.mean_reward:+.4f}   total: {s.total_reward:+.4f}", flush=True)
    print(f"mindless_rate: {100 * s.mindless_rate:.1f}%", flush=True)
    print(f"hold_rate: {100 * s.hold_rate:.1f}%", flush=True)
    print(f"tags: {s.tags}", flush=True)
    print(f"actions: {s.actions}", flush=True)
    print(
        f"entries_total={s.n_entries_total}  "
        f"days_with_entry={s.days_with_entry}  "
        f"days_did_nothing={s.days_did_nothing}",
        flush=True,
    )
    print(
        f"thrash: reverses={s.n_reverses} scale_blocks={s.n_scale_blocks} "
        f"cooldown_blocks={s.n_cooldown_blocks}",
        flush=True,
    )
    if s.eod_pnls:
        print("eod_pnls: " + ", ".join(f"{x:+.4f}" for x in s.eod_pnls), flush=True)
    if s.eod_penalties:
        print(
            "eod_penalties: " + ", ".join(f"{x:+.1f}" for x in s.eod_penalties),
            flush=True,
        )


def main() -> None:
    print("=" * 60, flush=True)
    print("PHASE C — curriculum train (real XAUUSD days)", flush=True)
    print("lineage: adaptive_rl_brain_7_31_26", flush=True)
    print(f"ckpt out: {CKPT_PATH}", flush=True)
    print("PROVEN: not touched", flush=True)
    print(
        f"thrash: max_open={MAX_OPEN_UNITS} cooldown_bars={REVERSE_COOLDOWN_BARS} "
        f"flip={FLIP_FLOP_PENALTY}",
        flush=True,
    )
    print(f"majority={USE_SIGNAL_MAJORITY} (off for train speed)", flush=True)
    print("=" * 60, flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    frames, meta, src = load_real_curriculum(n_trend=4, n_mix=2)
    write_curriculum_docs(meta, src)
    mtf = verify_mtf_on_days(frames)
    print(f"source={src} n_days={len(frames)} mtf_ok={mtf['n_days']}", flush=True)
    for m in meta:
        print(
            f"  day {m.date} role={m.role} bars={m.n_bars} "
            f"strength={m.trend_strength:.3f}",
            flush=True,
        )

    # Hold-out: last 2 days eval, rest train (or 50/50 if small)
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

    print(
        f"train_days={len(train_days)} eval_days={len(eval_days)} "
        f"steps/day={STEPS_PER_DAY} warm_epochs={WARM_EPOCHS} "
        f"real_epochs={REAL_EPOCHS}",
        flush=True,
    )

    policy = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=HIDDEN)

    print("\n[1/5] BEFORE greedy (untrained, real eval)...", flush=True)
    before = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="BEFORE_greedy",
    )
    print_stats(before)

    print("\n[2/5] BEFORE stochastic...", flush=True)
    before_stoch = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=False,
        label="BEFORE_stoch",
    )
    print_stats(before_stoch)

    # Synthetic warm-start: teach anti-hold + correct-side before real noise
    warm_days = curriculum_days(4, seed=SEED + 3, pattern="alternate")
    print("\n[3/5] WARM-START on synthetic thrust...", flush=True)
    warm_curve = train_policy(
        policy,
        warm_days,
        epochs=WARM_EPOCHS,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        lr=LR,
        bc_epochs=WARM_BC_EPOCHS,
        phase_name="warm",
    )
    mid = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_WARM_greedy",
    )
    print_stats(mid)

    print("\n[4/5] FINE-TUNE on real curriculum days...", flush=True)
    train_curve = train_policy(
        policy,
        train_days,
        epochs=REAL_EPOCHS,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        lr=LR * 0.7,
        bc_epochs=REAL_BC_EPOCHS,
        phase_name="real",
    )
    train_curve = list(warm_curve) + list(train_curve)

    print("\n[5/5] AFTER greedy + stoch (real eval)...", flush=True)
    after = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_greedy",
    )
    print_stats(after)
    after_stoch = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=False,
        label="AFTER_stoch",
    )
    print_stats(after_stoch)

    after_guided = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_antihold_greedy",
        anti_hold_greedy=True,
    )
    print_stats(after_guided)

    # Synthetic eval (sanity: thrash + entries on thrust)
    syn_eval = curriculum_days(2, seed=SEED + 21, pattern="alternate")
    after_syn = evaluate_policy(
        policy,
        syn_eval,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="AFTER_syn_greedy",
        anti_hold_greedy=True,
    )
    print_stats(after_syn)

    meta_out = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source": src,
        "warm_epochs": WARM_EPOCHS,
        "real_epochs": REAL_EPOCHS,
        "steps_per_day": STEPS_PER_DAY,
        "decide_every": DECIDE_EVERY,
        "train_dates": [m.date for m in train_meta],
        "eval_dates": [m.date for m in eval_meta],
        "anti_hold": True,
        "thrash_control": True,
        "use_signal_majority": USE_SIGNAL_MAJORITY,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
        "dials": PRACTICE_DIALS,
        "before_mean_reward": before.mean_reward,
        "after_mean_reward": after.mean_reward,
        "before_mindless_rate": before.mindless_rate,
        "after_mindless_rate": after.mindless_rate,
        "after_actions": after.actions,
        "after_hold_rate": after.hold_rate,
        "after_entries_total": after.n_entries_total,
        "after_days_did_nothing": after.days_did_nothing,
    }
    path = save_ckpt(policy, meta_out, CKPT_PATH)
    # also refresh sandbox latest pointer (lineage only)
    save_ckpt(policy, meta_out, CKPT_LATEST)
    print(f"\nSaved: {path}", flush=True)
    print(f"Also:  {CKPT_LATEST}", flush=True)

    print("\n" + "=" * 60, flush=True)
    print("COMPARE (greedy)", flush=True)
    print("=" * 60, flush=True)
    for k in sorted(set(before.tags) | set(after.tags)):
        print(
            f"  tag {k:18s} before={before.tags.get(k, 0):4d} "
            f"after={after.tags.get(k, 0):4d}",
            flush=True,
        )
    for k in ("hold", "buy", "sell"):
        print(
            f"  act {k:6s} before={before.actions.get(k, 0):4d} "
            f"after={after.actions.get(k, 0):4d}",
            flush=True,
        )
    print(
        f"  entries before={before.n_entries_total} after={after.n_entries_total}",
        flush=True,
    )
    print(
        f"  mean_reward before={before.mean_reward:+.4f} after={after.mean_reward:+.4f}",
        flush=True,
    )
    print(
        f"  mindless before={100*before.mindless_rate:.1f}% "
        f"after={100*after.mindless_rate:.1f}%",
        flush=True,
    )
    print(
        f"  hold_rate after={100*after.hold_rate:.1f}% "
        f"days_did_nothing after={after.days_did_nothing}",
        flush=True,
    )

    # health gates for Phase D
    # Prefer anti-hold greedy on real (pure greedy often freezes OOD)
    primary = after_guided if after_guided.n_entries_total > 0 else after_stoch
    healthy = (
        primary.n_entries_total > 0
        and primary.days_did_nothing == 0
        and primary.mindless_rate < 0.55
        and primary.hold_rate < 0.95
        and after_syn.n_entries_total > 0
    )
    partial = (
        after_stoch.n_entries_total > 0
        or after_guided.n_entries_total > 0
        or after_syn.n_entries_total > 0
    )
    print(f"\nhealth_gate (Phase D eligible): {healthy}", flush=True)
    print(f"partial_progress: {partial}", flush=True)
    print(
        f"primary_metric=antihold_greedy entries={after_guided.n_entries_total} "
        f"hold_rate={100*after_guided.hold_rate:.1f}% "
        f"mindless={100*after_guided.mindless_rate:.1f}%",
        flush=True,
    )

    report = {
        "lineage": "adaptive_rl_brain_7_31_26",
        "phase": "C",
        "checkpoint": os.path.abspath(path),
        "report": os.path.abspath(REPORT_PATH),
        "source": src,
        "train_dates": [m.date for m in train_meta],
        "eval_dates": [m.date for m in eval_meta],
        "anti_hold": True,
        "thrash_control": {
            "max_open_units": MAX_OPEN_UNITS,
            "reverse_cooldown_bars": REVERSE_COOLDOWN_BARS,
            "flip_flop_penalty": FLIP_FLOP_PENALTY,
        },
        "dials": PRACTICE_DIALS,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
        "before_greedy": asdict(before),
        "before_stoch": asdict(before_stoch),
        "after_warm_greedy": asdict(mid),
        "after_greedy": asdict(after),
        "after_stoch": asdict(after_stoch),
        "after_antihold_greedy": asdict(after_guided),
        "after_syn_greedy": asdict(after_syn),
        "train_epoch_means": train_curve,
        "meta": meta_out,
        "health_gate": healthy,
        "partial_progress": partial,
        "sandbox": True,
        "proven_touched": False,
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport: {REPORT_PATH}", flush=True)
    print("Done. No prove_it. No PROVEN.", flush=True)


if __name__ == "__main__":
    main()
