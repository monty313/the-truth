"""Sandbox practice for adaptive_rl_brain_7_31_26 only (anti-hold focus).

CHANGE LOG:
- 2026-07-31  multi-day train + before/after tag report — WHY: measure MINDLESS
  vs WITH_VECTOR/QUALIFIED under longer practice. Checkpoints ONLY under this
  lineage folder. Never touches models/PROVEN.
- 2026-07-31  anti-hold collapse practice — WHY: EOD did-nothing + stronger
  inactivity + correct-side entry; report action / EOD PnL / tags. Lineage only.

Usage (from repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/practice_long.py
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

# Repo root on path when run as script
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
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    DEFAULT_DIALS,
    DID_NOTHING_EOD_PENALTY,
    FLIP_FLOP_PENALTY,
    MAX_OPEN_UNITS,
    REVERSE_COOLDOWN_BARS,
    make_dials,
)

# --- lineage-only checkpoint home (NEVER models/) ---
CKPT_DIR = os.path.join(_HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "channel1_sandbox_latest.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "practice_long_report.json")

# Short sandbox — anti-hold levers on (EOD / inactivity / correct-side / MINDLESS)
N_EVAL_DAYS = 3
N_TRAIN_DAYS = 4
STEPS_PER_DAY = 35
DECIDE_EVERY = 25
TRAIN_EPOCHS = 14
BC_EPOCHS = 6  # pure imitation first so greedy can leave HOLD
LR = 1e-3
HIDDEN = 48
SEED = 7
ENTROPY_COEF = 0.02
IMITATION_COEF = 0.8  # strong match-higher push
GUIDE_PROB = 0.7  # train-time: usually take recommended side
# Majority panel (92 agents) is heavy; anti-hold report focuses on EOD/inactivity.
# Keep False for this short sandbox speed; signal consensus still in day_runner.
USE_SIGNAL_MAJORITY = False

# Active dials: strong inactivity (anti-hold)
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
    good_tag_rate: float = 0.0  # WITH_VECTOR + QUALIFIED_*
    reward_curve: List[float] = field(default_factory=list)  # per-day means
    eod_pnls: List[float] = field(default_factory=list)
    eod_penalties: List[float] = field(default_factory=list)
    n_entries_total: int = 0
    days_with_entry: int = 0
    days_did_nothing: int = 0


def _stats_from_steps(
    label: str,
    steps,
    day_means: List[float],
    eod_pnls: List[float],
    eod_penalties: List[float],
    n_entries_list: List[int],
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
    return PhaseStats(
        label=label,
        n_steps=n,
        total_reward=float(sum(rews)),
        mean_reward=float(np.mean(rews)) if n else 0.0,
        tags=dict(tags),
        actions={
            "hold": int(acts.get(ACTION_HOLD, 0)),
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
    )


def _run_day(
    policy: Channel1Policy,
    day,
    *,
    steps_per_day: int,
    decide_every: int,
    greedy: bool,
    train_mode: bool,
    opt: Optional[torch.optim.Optimizer] = None,
    guide_prob: float = 0.0,
) -> Tuple[
    List[Any],
    float,
    float,
    int,
    List[torch.Tensor],
    List[float],
    List[torch.Tensor],
    List[torch.Tensor],
]:
    """One day: steps + EOD penalty on last reward. Returns train tensors if train_mode."""
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
        if train_mode:
            logits = policy.forward(torch.as_tensor(obs, dtype=torch.float32)).squeeze(0)
            dist = torch.distributions.Categorical(logits=logits)
            # Guided explore: match higher when flat; while in a trade, prefer
            # HOLD so we do not thrash open/close every bar.
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
            # Imitation aux only when there is a real directional recommendation
            # (do not train the policy to HOLD via CE when market is flat)
            if rec != ACTION_HOLD:
                ce_losses.append(
                    torch.nn.functional.cross_entropy(
                        logits.unsqueeze(0),
                        torch.tensor([rec], dtype=torch.long),
                    )
                )
        else:
            action, _ = policy.act(obs, greedy=greedy)
            action = int(action)
        step = runner.step(t, action=action)
        steps.append(step)
        rews.append(float(step.reward))

    eod = float(runner.end_day())
    if steps and eod != 0.0:
        # attach once to last step so REINFORCE / metrics see it
        last = steps[-1]
        last.reward = float(last.reward) + eod
        last.info = dict(last.info)
        last.info["eod_penalty"] = eod
        rews[-1] = float(last.reward)

    return (
        steps,
        float(runner.realized),
        eod,
        int(runner.n_entries),
        logps,
        rews,
        ents,
        ce_losses,
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
) -> PhaseStats:
    policy.eval()
    all_steps = []
    day_means: List[float] = []
    eod_pnls: List[float] = []
    eod_penalties: List[float] = []
    n_entries_list: List[int] = []
    for day in days:
        steps, realized, eod, n_ent, _, rews, _, _ = _run_day(
            policy,
            day,
            steps_per_day=steps_per_day,
            decide_every=decide_every,
            greedy=greedy,
            train_mode=False,
        )
        all_steps.extend(steps)
        day_means.append(float(np.mean(rews)) if rews else 0.0)
        eod_pnls.append(realized)
        eod_penalties.append(eod)
        n_entries_list.append(n_ent)
    return _stats_from_steps(
        label, all_steps, day_means, eod_pnls, eod_penalties, n_entries_list
    )


def train_policy(
    policy: Channel1Policy,
    days: List,
    *,
    epochs: int,
    steps_per_day: int,
    decide_every: int,
    lr: float,
) -> List[float]:
    """BC warmup then REINFORCE + imitation aux across days."""
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    epoch_means: List[float] = []
    policy.train()
    for ep in range(epochs):
        ep_rewards: List[float] = []
        ep_logps: List[torch.Tensor] = []
        ep_ents: List[torch.Tensor] = []
        ep_ces: List[torch.Tensor] = []
        bc_only = ep < int(BC_EPOCHS)
        # high guide during BC; decay later so policy owns more
        if bc_only:
            gprob = 0.9
        else:
            frac = (ep - BC_EPOCHS) / max(epochs - BC_EPOCHS, 1)
            gprob = float(GUIDE_PROB) * max(0.4, 1.0 - frac)
        for day in days:
            steps, _realized, _eod, _n_ent, logps, rews, ents, ces = _run_day(
                policy,
                day,
                steps_per_day=steps_per_day,
                decide_every=decide_every,
                greedy=False,
                train_mode=True,
                opt=opt,
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
            # pure behavioral cloning toward match-higher actions
            loss = ce if ep_ces else torch.tensor(0.0, requires_grad=True)
            mode = "BC"
        else:
            r = torch.tensor(ep_rewards, dtype=torch.float32)
            r = r - r.mean()
            reinforce = -(torch.stack(ep_logps) * r.detach()).mean()
            loss = (
                reinforce
                - float(ENTROPY_COEF) * ent
                + float(IMITATION_COEF) * ce
            )
            mode = "RF"
        opt.zero_grad()
        if loss.requires_grad:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        epoch_means.append(float(np.mean(ep_rewards)) if ep_rewards else 0.0)
        print(
            f"  epoch {ep + 1:02d}/{epochs} [{mode}]  "
            f"mean_reward={epoch_means[-1]:+.4f}  loss={float(loss.item()):.4f}  "
            f"entropy={float(ent.item()):.3f}  ce={float(ce.item()):.3f}  "
            f"guide={gprob:.2f}",
            flush=True,
        )
    return epoch_means


def save_ckpt(policy: Channel1Policy, meta: dict) -> str:
    os.makedirs(CKPT_DIR, exist_ok=True)
    abs_path = os.path.abspath(CKPT_PATH)
    norm = abs_path.replace("\\", "/")
    if "/models/" in norm and "lineages" not in norm:
        raise RuntimeError("refusing to write outside lineage")
    payload = {
        "model": policy.state_dict(),
        "obs_dim": CHANNEL1_DIM,
        "hidden": HIDDEN,
        "meta": meta,
        "lineage": "adaptive_rl_brain_7_31_26",
        "note": "SANDBOX only — not PROVEN",
        "anti_hold": True,
        "dials": PRACTICE_DIALS,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
    }
    torch.save(payload, CKPT_PATH)
    return CKPT_PATH


def print_stats(s: PhaseStats) -> None:
    print(f"\n--- {s.label} ---", flush=True)
    print(f"steps: {s.n_steps}", flush=True)
    print(f"mean_reward: {s.mean_reward:+.4f}   total: {s.total_reward:+.4f}", flush=True)
    print(f"mindless_rate: {100 * s.mindless_rate:.1f}%", flush=True)
    print(
        f"good_tag_rate (WITH_VECTOR+QUALIFIED*): {100 * s.good_tag_rate:.1f}%",
        flush=True,
    )
    print(f"tags: {s.tags}", flush=True)
    print(f"actions: {s.actions}", flush=True)
    print(
        f"entries_total={s.n_entries_total}  "
        f"days_with_entry={s.days_with_entry}  "
        f"days_did_nothing={s.days_did_nothing}",
        flush=True,
    )
    if s.eod_pnls:
        print(
            "eod_pnls: " + ", ".join(f"{x:+.4f}" for x in s.eod_pnls),
            flush=True,
        )
    if s.eod_penalties:
        print(
            "eod_penalties: " + ", ".join(f"{x:+.1f}" for x in s.eod_penalties),
            flush=True,
        )
    if s.reward_curve:
        print(
            "per-day mean rewards: "
            + ", ".join(f"{x:+.3f}" for x in s.reward_curve),
            flush=True,
        )


def main() -> None:
    print("=" * 60, flush=True)
    print("NEW BRAIN sandbox practice — anti-hold collapse", flush=True)
    print("lineage: adaptive_rl_brain_7_31_26", flush=True)
    print("checkpoints: lineages/.../checkpoints/ ONLY", flush=True)
    print("PROVEN: not touched", flush=True)
    print(f"DID_NOTHING_EOD_PENALTY={DID_NOTHING_EOD_PENALTY}", flush=True)
    print(f"dials={PRACTICE_DIALS}", flush=True)
    print(f"use_signal_majority={USE_SIGNAL_MAJORITY}", flush=True)
    print(
        "anti-hold: EOD did-nothing, setup-hold floor, flat tax, "
        "correct-side bonus, MINDLESS wall",
        flush=True,
    )
    print("=" * 60, flush=True)

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # Thrust curriculum — directional days so confluence is not always NEUTRAL
    eval_days = curriculum_days(N_EVAL_DAYS, seed=SEED, pattern="alternate")
    train_days = curriculum_days(N_TRAIN_DAYS, seed=SEED + 99, pattern="alternate")
    print(
        f"eval_days={len(eval_days)} train_days={len(train_days)} "
        f"(thrust curriculum)",
        flush=True,
    )
    print(
        f"steps/day={STEPS_PER_DAY} epochs={TRAIN_EPOCHS} decide_every={DECIDE_EVERY}",
        flush=True,
    )

    policy = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=HIDDEN)

    # ----- BEFORE (fresh policy, greedy) -----
    print("\n[1/4] BEFORE greedy (untrained)...", flush=True)
    before = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=True,
        label="BEFORE_greedy",
    )
    print_stats(before)

    print("\n[2/4] BEFORE stochastic (untrained, sample)...", flush=True)
    before_stoch = evaluate_policy(
        policy,
        eval_days,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        greedy=False,
        label="BEFORE_stoch",
    )
    print_stats(before_stoch)

    # ----- TRAIN -----
    print("\n[3/4] TRAIN (multi-day REINFORCE)...", flush=True)
    train_curve = train_policy(
        policy,
        train_days,
        epochs=TRAIN_EPOCHS,
        steps_per_day=STEPS_PER_DAY,
        decide_every=DECIDE_EVERY,
        lr=LR,
    )

    # ----- AFTER -----
    print("\n[4/4] AFTER greedy + stochastic...", flush=True)
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

    # ----- SAVE lineage-only ckpt -----
    meta = {
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "train_epochs": TRAIN_EPOCHS,
        "steps_per_day": STEPS_PER_DAY,
        "n_train_days": len(train_days),
        "n_eval_days": len(eval_days),
        "anti_hold": True,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
        "dials": PRACTICE_DIALS,
        "before_mindless_rate": before.mindless_rate,
        "after_mindless_rate": after.mindless_rate,
        "before_mean_reward": before.mean_reward,
        "after_mean_reward": after.mean_reward,
        "after_actions": after.actions,
        "after_entries_total": after.n_entries_total,
    }
    path = save_ckpt(policy, meta)
    print(f"\nSaved checkpoint (lineage only): {path}", flush=True)

    # ----- COMPARE -----
    print("\n" + "=" * 60, flush=True)
    print("COMPARE (greedy eval — what policy prefers)", flush=True)
    print("=" * 60, flush=True)

    def tag_get(st: PhaseStats, k: str) -> int:
        return int(st.tags.get(k, 0))

    print("\n1) Tag counts BEFORE vs AFTER (greedy):", flush=True)
    all_tags = sorted(set(before.tags) | set(after.tags))
    for k in all_tags:
        b, a = tag_get(before, k), tag_get(after, k)
        print(f"   {k:18s}  before={b:4d}  after={a:4d}  delta={a - b:+d}", flush=True)

    print("\n2) Action counts BEFORE vs AFTER (greedy):", flush=True)
    for k in ("hold", "buy", "sell"):
        b, a = before.actions.get(k, 0), after.actions.get(k, 0)
        print(f"   {k:6s}  before={b:4d}  after={a:4d}  delta={a - b:+d}", flush=True)
    print(
        f"   real_entries: before={before.n_entries_total}  "
        f"after={after.n_entries_total}",
        flush=True,
    )
    print(
        f"   days_with_entry: before={before.days_with_entry}  "
        f"after={after.days_with_entry}",
        flush=True,
    )

    print("\n   Action counts stochastic AFTER (sampling):", flush=True)
    print(f"   {after_stoch.actions}  entries={after_stoch.n_entries_total}", flush=True)

    print("\n3) End-of-day PnL distribution (greedy):", flush=True)
    print(f"   BEFORE eod_pnls: {before.eod_pnls}", flush=True)
    print(f"   AFTER  eod_pnls: {after.eod_pnls}", flush=True)
    print(f"   BEFORE eod_penalties: {before.eod_penalties}", flush=True)
    print(f"   AFTER  eod_penalties: {after.eod_penalties}", flush=True)
    print(
        f"   days_did_nothing: before={before.days_did_nothing}  "
        f"after={after.days_did_nothing}",
        flush=True,
    )

    print("\n4) Average reward trend:", flush=True)
    print(f"   BEFORE mean_reward: {before.mean_reward:+.4f}", flush=True)
    print(f"   AFTER  mean_reward: {after.mean_reward:+.4f}", flush=True)
    print(f"   delta: {after.mean_reward - before.mean_reward:+.4f}", flush=True)
    print("   train epoch means:", flush=True)
    print("   " + ", ".join(f"{x:+.3f}" for x in train_curve), flush=True)

    print("\n5) MINDLESS rate (greedy):", flush=True)
    print(f"   BEFORE: {100 * before.mindless_rate:.1f}%", flush=True)
    print(f"   AFTER:  {100 * after.mindless_rate:.1f}%", flush=True)
    improved = after.mindless_rate < before.mindless_rate - 1e-9
    print(f"   improved: {'YES' if improved else 'NO / N/A (tag probe path)'}", flush=True)

    real_entries = after.n_entries_total > 0 or after_stoch.n_entries_total > 0
    any_trade_actions = (
        after.actions.get("buy", 0) + after.actions.get("sell", 0) > 0
        or after_stoch.actions.get("buy", 0) + after_stoch.actions.get("sell", 0) > 0
    )
    print("\n6) Real entries appeared?", flush=True)
    print(f"   greedy entries: {after.n_entries_total}", flush=True)
    print(f"   stoch  entries: {after_stoch.n_entries_total}", flush=True)
    print(f"   any trade actions: {'YES' if any_trade_actions else 'NO'}", flush=True)
    print(f"   real entries: {'YES' if real_entries else 'NO'}", flush=True)

    report = {
        "lineage": "adaptive_rl_brain_7_31_26",
        "checkpoint": path,
        "anti_hold": True,
        "dials": PRACTICE_DIALS,
        "did_nothing_eod_penalty": DID_NOTHING_EOD_PENALTY,
        "before_greedy": asdict(before),
        "before_stoch": asdict(before_stoch),
        "after_greedy": asdict(after),
        "after_stoch": asdict(after_stoch),
        "train_epoch_means": train_curve,
        "meta": meta,
        "sandbox": True,
        "proven_touched": False,
    }
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport JSON: {REPORT_PATH}", flush=True)
    print("\nDone. Still sandbox — no prove_it, no PROVEN.", flush=True)


if __name__ == "__main__":
    main()
