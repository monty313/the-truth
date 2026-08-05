"""Train a NEW same-obs Channel1 policy as MARK HERE clone (Fable translator).

CHANGE LOG:
- 2026-08-04  created — WHY: BC pure greedy to Mark multi-set teacher so the
  bot moves like Mark (ENTJ pt5: HTF permission, LTF trigger, all 4 sets).
  Same CHANNEL1_DIM obs. Checkpoints ONLY under this lineage. Never PROVEN.

Pipeline (10-role single process):
  1 eyes/teacher  2 label dump  3 train  4 day-walk audit  5 thrash metrics
  6 A/B score  7 doctrine  8 docs  9 breach guard  10 next-diff

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/train_mark_clone.py
  python lineages/adaptive_rl_brain_7_31_26/train_mark_clone.py --epochs 20 --practice-n 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT_CKPT = os.path.join(CKPT_DIR, "mark_clone_channel1_v1.pt")
OUT_LATEST = os.path.join(CKPT_DIR, "mark_clone_channel1_latest.pt")
LABELS_PATH = os.path.join(CKPT_DIR, "mark_teacher_labels_practice.json")
REPORT_PATH = os.path.join(CKPT_DIR, "mark_clone_train_report.json")
MATCH_PATH = os.path.join(CKPT_DIR, "mark_clone_policy_match.json")

NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}
SEED = 42
HIDDEN = 48


def collect_teacher_labels(
    days: Sequence[Tuple[str, Any]],
    *,
    target: float = 2.0,
    risk: float = 3.0,
    max_days: int = 0,
) -> Tuple[List[np.ndarray], List[int], Dict[str, Any]]:
    """Emit (obs, Mark-teacher action) pairs on practice days. Same Channel1 obs."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    per_day: List[dict] = []
    action_ctr: Counter = Counter()
    n_days = 0
    for date_str, m1 in days:
        if max_days and n_days >= max_days:
            break
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            mark_clone=False,
            eyes_mode="mark_all_sets",
        )
        indices = day.runner.decision_indices()
        day_acts: List[int] = []
        n_dir = 0
        for t in indices:
            obs = day.observe(t)
            # Teacher: Mark multi-set eyes (flat perception for direction)
            act = int(day.recommended_action(t))
            xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            ys.append(act)
            day_acts.append(act)
            action_ctr[NAMES[act]] += 1
            if act in (ACTION_BUY, ACTION_SELL):
                n_dir += 1
            # Advance shell lightly so progress/danger evolve (teacher still structure)
            day.step_action(t, act)
            if day.banked or day.dead:
                break
        n_days += 1
        per_day.append(
            {
                "date": str(date_str),
                "n_decisions": len(day_acts),
                "n_directional": n_dir,
                "n_entries": day.n_entries,
                "actions": dict(Counter(NAMES[a] for a in day_acts)),
                "banked": bool(day.banked),
                "breached": bool(day.breached),
                "pnl_pct": float(
                    100.0 * (day.balance - day.eq0) / day.eq0
                ),
            }
        )
    meta = {
        "n_days": n_days,
        "n_samples": len(ys),
        "action_counts": dict(action_ctr),
        "frac_hold": float(action_ctr.get("HOLD", 0) / max(len(ys), 1)),
        "frac_directional": float(
            (action_ctr.get("BUY", 0) + action_ctr.get("SELL", 0)) / max(len(ys), 1)
        ),
        "target_pct": target,
        "risk_pct": risk,
        "eyes_mode": "mark_all_sets",
        "obs_dim": CHANNEL1_DIM,
        "per_day": per_day,
        "all_hold": len(ys) > 0 and action_ctr.get("HOLD", 0) == len(ys),
    }
    return xs, ys, meta


def train_bc(
    xs: List[np.ndarray],
    ys: List[int],
    *,
    epochs: int = 24,
    lr: float = 1e-3,
    batch: int = 64,
    seed: int = SEED,
    hidden: int = HIDDEN,
) -> Tuple[Channel1Policy, Dict[str, Any]]:
    """Behavioral clone: CE to Mark teacher. Same obs dim."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    policy = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=hidden)
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    X = torch.as_tensor(np.stack(xs, axis=0), dtype=torch.float32)
    Y = torch.as_tensor(np.asarray(ys, dtype=np.int64))
    n = len(Y)
    # class weights: upweight BUY/SELL so we don't collapse to HOLD
    counts = torch.bincount(Y, minlength=3).float().clamp(min=1.0)
    w = (counts.sum() / (3.0 * counts)).clamp(0.5, 4.0)
    history: List[float] = []
    for ep in range(epochs):
        perm = torch.randperm(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            logits = policy(X[idx])
            loss = F.cross_entropy(logits, Y[idx], weight=w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        history.append(ep_loss / max(nb, 1))
    return policy, {"loss_curve": history, "class_weights": w.tolist(), "epochs": epochs}


@torch.no_grad()
def eval_match(
    policy: Channel1Policy,
    xs: List[np.ndarray],
    ys: List[int],
) -> Dict[str, Any]:
    policy.eval()
    pred_ctr: Counter = Counter()
    match = 0
    n = len(ys)
    for x, y in zip(xs, ys):
        logits = policy(torch.as_tensor(x, dtype=torch.float32))
        a = int(torch.argmax(logits, dim=-1).item())
        pred_ctr[NAMES[a]] += 1
        if a == int(y):
            match += 1
    hold_rate = float(pred_ctr.get("HOLD", 0) / max(n, 1))
    return {
        "n": n,
        "match_rate": float(match / max(n, 1)),
        "pred_actions": dict(pred_ctr),
        "hold_rate": hold_rate,
        "not_all_hold": hold_rate < 0.999,
        "entries_signal": int(pred_ctr.get("BUY", 0) + pred_ctr.get("SELL", 0)),
    }


def run_policy_day(
    policy: Channel1Policy,
    m1,
    date_str: str,
    target: float,
    risk: float,
) -> Dict[str, Any]:
    """Pure greedy policy through equity shell (Mark eyes only for teacher compare)."""
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=str(date_str),
        eyes_mode="mark_all_sets",
    )
    teacher_match = 0
    n_dec = 0
    acts: List[int] = []

    def policy_fn(obs, d):
        nonlocal teacher_match, n_dec
        t_idx = d.runner.decision_indices()[min(n_dec, len(d.runner.decision_indices()) - 1)]
        # teacher at same bar
        teach = int(d.recommended_action(t_idx)) if hasattr(d, "recommended_action") else 0
        with torch.no_grad():
            logits = policy(torch.as_tensor(obs, dtype=torch.float32))
            a = int(torch.argmax(logits, dim=-1).item())
        if a == teach:
            teacher_match += 1
        n_dec += 1
        acts.append(a)
        return a

    # Use run with policy_fn — but teacher compare needs bar alignment
    # Simpler: manual loop
    indices = day.runner.decision_indices()
    prev_t = 0
    for t in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev_t, t):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = t + 1
        if day.dead or day.banked:
            break
        obs = day.observe(t)
        teach = int(day.recommended_action(t))
        with torch.no_grad():
            a = int(
                torch.argmax(
                    policy(torch.as_tensor(obs, dtype=torch.float32)), dim=-1
                ).item()
            )
        if a == teach:
            teacher_match += 1
        n_dec += 1
        acts.append(a)
        day.step_action(t, a)
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    price = float(day._close[t_last])
    sp = float(day._spread_px[t_last])
    day._flatten(price, sp)
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    day.min_eq_pct = min(day.min_eq_pct, pnl)
    if pnl <= -day.risk + 1e-12:
        day.breached = True
    cleared = (pnl >= day.target - 1e-12) and (not day.breached)
    if day.banked and not day.breached:
        cleared = True
    return {
        "date": str(date_str),
        "pnl_pct": round(float(pnl), 4),
        "min_eq_pct": round(float(day.min_eq_pct), 4),
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "banked": bool(day.banked),
        "n_entries": int(day.n_entries),
        "n_decisions": n_dec,
        "teacher_match_rate": float(teacher_match / max(n_dec, 1)),
        "actions": dict(Counter(NAMES[a] for a in acts)),
        "hold_rate": float(sum(1 for a in acts if a == ACTION_HOLD) / max(len(acts), 1)),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train Mark-clone Channel1 policy")
    ap.add_argument("--epochs", type=int, default=24)
    ap.add_argument("--practice-n", type=int, default=30)
    ap.add_argument("--max-label-days", type=int, default=15)
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--risk", type=float, default=3.0)
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    args = ap.parse_args(argv)

    os.makedirs(CKPT_DIR, exist_ok=True)
    print("=== MARK CLONE TRAIN (same obs, new brain) ===", flush=True)
    print(f"obs_dim={CHANNEL1_DIM}  epochs={args.epochs}", flush=True)

    all_days = load_calendar_days(args.data, min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=args.practice_n)
    print(f"practice days available={len(practice)}  labeling max={args.max_label_days}", flush=True)

    # --- Role 2: label dump ---
    xs, ys, label_meta = collect_teacher_labels(
        practice,
        target=args.target,
        risk=args.risk,
        max_days=args.max_label_days,
    )
    with open(LABELS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {k: v for k, v in label_meta.items() if k != "per_day"},
                "per_day": label_meta["per_day"],
                "saved_at": datetime.now(timezone.utc).isoformat(),
            },
            f,
            indent=2,
        )
    print(
        f"labels: n={label_meta['n_samples']}  "
        f"actions={label_meta['action_counts']}  "
        f"frac_dir={label_meta['frac_directional']:.3f}  "
        f"all_hold={label_meta['all_hold']}",
        flush=True,
    )
    if label_meta["all_hold"] or label_meta["n_samples"] < 10:
        print("FAIL: teacher labels empty or all HOLD", flush=True)
        return 2

    # --- Role 3: train ---
    policy, train_meta = train_bc(xs, ys, epochs=args.epochs)
    match = eval_match(policy, xs, ys)
    print(
        f"train done: match_rate={match['match_rate']:.3f}  "
        f"pred={match['pred_actions']}  not_all_hold={match['not_all_hold']}",
        flush=True,
    )

    blob = {
        "state_dict": policy.state_dict(),
        "hidden": HIDDEN,
        "obs_dim": CHANNEL1_DIM,
        "tag": "mark_clone_channel1_v1",
        "eyes_mode": "mark_all_sets",
        "teacher": "mark_multi_set_htf_permission",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "dials": {"decode": "pure_greedy", "shell": "multi_pair_locked"},
        "train": train_meta,
        "match": match,
        "label_meta": {k: v for k, v in label_meta.items() if k != "per_day"},
        "proven_touched": False,
    }
    torch.save(blob, OUT_CKPT)
    torch.save(blob, OUT_LATEST)
    print(f"saved {OUT_CKPT}", flush=True)

    # --- Role 4–5: day walks policy vs thrash/bank ---
    day_map = {str(d): m1 for d, m1 in all_days}
    walk_dates = [
        ("2026-04-02", 3.0, 3.5),
        ("2026-04-01", 1.0, 2.0),
    ]
    walks = []
    for dt, tgt, rsk in walk_dates:
        if dt not in day_map:
            continue
        w = run_policy_day(policy, day_map[dt], dt, tgt, rsk)
        walks.append(w)
        print(
            f"policy day {dt} t={tgt}/{rsk}: entries={w['n_entries']} "
            f"hold={w['hold_rate']:.2f} match={w['teacher_match_rate']:.2f} "
            f"pnl={w['pnl_pct']} cleared={w['cleared']} breach={w['breached']}",
            flush=True,
        )

    # --- Role 6: small practice A/B window ---
    ab_days = practice[:8]
    base_clear = 0
    mark_clear = 0
    pol_clear = 0
    base_ent = []
    pol_ent = []
    breaches = 0
    for date_str, m1 in ab_days:
        b = GoalEquityDay(
            m1, target_pct=3.0, risk_pct=3.5, date_str=str(date_str), eyes_mode="legacy_set2"
        ).run(use_heuristic=True)
        m = GoalEquityDay(
            m1, target_pct=3.0, risk_pct=3.5, date_str=str(date_str), eyes_mode="mark_all_sets"
        ).run(use_heuristic=True)
        p = run_policy_day(policy, m1, str(date_str), 3.0, 3.5)
        base_clear += int(b.cleared)
        mark_clear += int(m.cleared)
        pol_clear += int(p["cleared"])
        base_ent.append(b.n_entries)
        pol_ent.append(p["n_entries"])
        breaches += int(b.breached) + int(m.breached) + int(p["breached"])

    ab = {
        "n_days": len(ab_days),
        "pair": [3.0, 3.5],
        "baseline_clear": base_clear,
        "mark_eyes_clear": mark_clear,
        "policy_clear": pol_clear,
        "baseline_mean_entries": float(np.mean(base_ent)) if base_ent else 0.0,
        "policy_mean_entries": float(np.mean(pol_ent)) if pol_ent else 0.0,
        "total_breaches": breaches,
        "breach_ok": breaches == 0,
    }
    print(f"A/B practice8: {ab}", flush=True)

    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "ckpt": OUT_CKPT,
        "match": match,
        "label_meta": {k: v for k, v in label_meta.items() if k != "per_day"},
        "walks": walks,
        "ab_practice8": ab,
        "proven_touched": False,
        "stop_rule": {
            "breach_must_be_0": True,
            "not_all_hold": match["not_all_hold"],
            "teacher_match_min_hint": 0.4,
            "next_if_fail": "raise epochs / more practice labels / reinforce BC weight on BUY/SELL",
        },
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(MATCH_PATH, "w", encoding="utf-8") as f:
        json.dump({"match": match, "walks": walks, "ab": ab}, f, indent=2)
    print(f"report {REPORT_PATH}", flush=True)

    ok = (
        match["not_all_hold"]
        and match["match_rate"] >= 0.35
        and not label_meta["all_hold"]
        and ab["breach_ok"]
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
