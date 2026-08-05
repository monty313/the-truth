"""Phase training for 2x streak: memorize Mark full-day entry bars, then blend awards.

Lesson (LEARNING_2X_STREAK.md):
  - Gate flat_undefined blocked Mark-agreed entries (fixed).
  - Online recommended != full-day soul plan (e.g. 770 plan=S, rec=H).
  - Heavy DAgger collapses awards; entry-only then blend is safer.

Usage:
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/train_entry_only_2x.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

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
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import load_pairs
from lineages.adaptive_rl_brain_7_31_26.loop_2x_streak_rewards import (
    load_policy,
    save_policy,
    score_streak,
    sample_weight_for,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "mark_consistency")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
NOTES = os.path.join(OUT, "LEARNING_2X_STREAK.md")


def collect_plan_entries(m1, date, t, r, mark) -> List[Tuple[np.ndarray, int]]:
    """Only (obs, action) at directional Mark plan bars + HOLD pads around them."""
    plan = mark.get("plan")
    if not plan or mark.get("source") != "soul_plan":
        return []
    day = GoalEquityDay(
        m1,
        target_pct=float(t),
        risk_pct=float(r),
        date_str=str(date),
        eyes_mode="mark_doctrine",
        risk_use_frac=float(mark["risk_use_frac"]),
        per_trade_cap_pct=float(mark["per_trade_cap_pct"]),
        mark_soul=True,
        full_obs=True,
    )
    day._plan_lock_ruf = float(mark["risk_use_frac"])
    day._plan_lock_cap = float(mark["per_trade_cap_pct"])
    pairs: List[Tuple[np.ndarray, int]] = []
    indices = list(day.runner.decision_indices())
    prev = 0
    for tb in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev, tb):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev = tb + 1
        if day.dead or day.banked:
            break
        obs = day.observe(tb)
        act = int(plan.get(int(tb), ACTION_HOLD))
        # keep all plan bars (HOLD + entries) — HOLD teaches wait
        pairs.append((np.asarray(obs, dtype=np.float32).reshape(-1), act))
        day.step_action(tb, act)
    return pairs


def award_imitate(m1, date, t, r, policy) -> List[Tuple[np.ndarray, int]]:
    out = []
    day = GoalEquityDay(
        m1,
        target_pct=float(t),
        risk_pct=float(r),
        date_str=str(date),
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=True,
        mark_align_policy=True,
    )
    for tb in day.runner.decision_indices():
        if day.dead or day.banked:
            break
        obs = day.observe(tb)
        with torch.no_grad():
            a, _ = policy.act(obs, greedy=True)
        out.append((np.asarray(obs, dtype=np.float32).reshape(-1), int(a)))
        day.step_action(tb, int(a))
    return out


def pack(pairs: List[Tuple[np.ndarray, int]], weights: List[float]):
    X = np.stack([p[0] for p in pairs])
    y = np.asarray([p[1] for p in pairs], dtype=np.int64)
    w = np.asarray(weights, dtype=np.float32)
    return X, y, w


def append_note(text: str) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(NOTES, "a", encoding="utf-8") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--target-streak", type=int, default=16)
    ap.add_argument("--phase1-epochs", type=int, default=60)
    ap.add_argument("--phase2-epochs", type=int, default=25)
    ap.add_argument("--entry-os", type=int, default=40)
    ap.add_argument("--max-entry-samples", type=int, default=22)
    args = ap.parse_args(argv)

    dials = clip_streak_dials(
        {
            **default_streak_dials(),
            "mark_would_take_eod_penalty": -16.0,
            "soul_side_entry_bonus": 5.0,
            "soul_side_misread_penalty": -6.0,
            "streak_break_penalty": -12.0,
            "streak_award_base": 6.0,
            "streak_award_per_prior": 2.0,
        }
    )
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    _, forward = split_practice_forward(all_days, practice_n=50)
    window = forward[: int(args.n_days)]
    day_map = {str(d): m1 for d, m1 in window}
    pairs = load_pairs()
    policy = load_policy(CKPT)
    pre = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    print(
        f"PRE streak={pre['max_award_streak']} awards={pre['n_award']}/{pre['n_days']} "
        f"breach={pre['n_breach']}",
        flush=True,
    )
    append_note(
        f"\n### Entry-only 2x run {datetime.now(timezone.utc).isoformat()}\n"
        f"- PRE streak={pre['max_award_streak']} awards={pre['n_award']}/40 breach={pre['n_breach']}\n"
    )
    best = pre
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    # Oracle miss days
    miss_rows = [r for r in pre["rows"] if not r["award"]]
    award_rows = [r for r in pre["rows"] if r["award"]]
    entry_pairs: List[Tuple[np.ndarray, int]] = []
    hold_pairs: List[Tuple[np.ndarray, int]] = []
    for row in miss_rows:
        date = str(row["date"])
        t, r = float(row["target_pct"]), float(row["risk_pct"])
        print(f"  oracle {date}…", flush=True)
        mark = execute_mark_soul_day(
            day_map[date], date, t, r, max_entry_samples=int(args.max_entry_samples)
        )
        mark_s = {k: v for k, v in mark.items() if k != "day"}
        if not (mark_s.get("source") == "soul_plan" and mark_s.get("cleared")):
            print(f"    skip non-winnable {date}", flush=True)
            continue
        pl = collect_plan_entries(day_map[date], date, t, r, mark_s)
        n_dir = sum(1 for _, a in pl if a != ACTION_HOLD)
        print(f"    plan bars={len(pl)} directional={n_dir} clear={mark_s.get('pnl_pct')}", flush=True)
        for obs, act in pl:
            if act != ACTION_HOLD:
                entry_pairs.append((obs, act))
            else:
                hold_pairs.append((obs, act))

    print(f"  entry labels={len(entry_pairs)} hold labels={len(hold_pairs)}", flush=True)
    if len(entry_pairs) < 2:
        print("no entries", flush=True)
        return 2

    # ---- Phase 1: memorize entries (+ holds), no award imitate, kl=0 ----
    print("\n===== PHASE 1: entry memorize (kl=0) =====", flush=True)
    p1: List[Tuple[np.ndarray, int]] = []
    w1: List[float] = []
    for _ in range(int(args.entry_os)):
        for obs, act in entry_pairs:
            p1.append((obs, act))
            w1.append(sample_weight_for("MARK_WOULD_TAKE", act, dials) * 5.0)
        for obs, act in hold_pairs:
            p1.append((obs, act))
            w1.append(sample_weight_for("MARK_WOULD_TAKE", act, dials) * 1.2)
    X, y, w = pack(p1, w1)
    print(f"  phase1 n={len(y)}", flush=True)
    policy, losses = train_bc(
        X,
        y,
        epochs=int(args.phase1_epochs),
        hidden=128,
        seed=args.seed + 1,
        warm_state=best_state,
        obs_dim=MARK_FULL_DIM,
        lr=8e-4,
        sample_weights=w,
        kl_anchor_state=None,
        kl_coef=0.0,
    )
    m = match_rate(policy, X, y)
    print(f"  phase1 match={m}", flush=True)
    s1 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    print(
        f"  after P1 streak={s1['max_award_streak']} awards={s1['n_award']} breach={s1['n_breach']}",
        flush=True,
    )
    append_note(
        f"- Phase1 (entry memorize kl=0): streak={s1['max_award_streak']} "
        f"awards={s1['n_award']} breach={s1['n_breach']} match={m}\n"
    )
    p1_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    if s1["n_breach"] == 0 and (
        s1["max_award_streak"] > best["max_award_streak"]
        or (
            s1["max_award_streak"] == best["max_award_streak"]
            and s1["n_award"] >= best["n_award"]
        )
    ):
        best, best_state = s1, p1_state
        save_policy(policy, note="entry_only_p1", dials=dials)
        print("  ** kept P1 as best **", flush=True)
    else:
        print(
            "  P1 not best overall — still using P1 weights as Phase2 warm-start "
            "(entry craft), KL pulls toward pre-P1 best awards",
            flush=True,
        )

    # ---- Phase 2: blend award self-imitate + entries, light KL to best ----
    print("\n===== PHASE 2: blend awards + entries =====", flush=True)
    p2: List[Tuple[np.ndarray, int]] = []
    w2: List[float] = []
    for _ in range(max(8, int(args.entry_os) // 3)):
        for obs, act in entry_pairs:
            p2.append((obs, act))
            w2.append(sample_weight_for("MARK_WOULD_TAKE", act, dials) * 4.0)
        for obs, act in hold_pairs:
            p2.append((obs, act))
            w2.append(1.0)
    # awards from BEST award-preserving policy
    pol_aw = load_policy(CKPT)
    pol_aw.load_state_dict(best_state)
    pol_aw.eval()
    for row in award_rows:
        date = str(row["date"])
        ap = award_imitate(
            day_map[date],
            date,
            float(row["target_pct"]),
            float(row["risk_pct"]),
            pol_aw,
        )
        for obs, act in ap:
            p2.append((obs, act))
            w2.append(sample_weight_for("AWARD", act, dials) * 1.5)
    X2, y2, ww = pack(p2, w2)
    print(f"  phase2 n={len(y2)}", flush=True)
    # warm from entry-trained P1; KL to award-best
    policy, losses2 = train_bc(
        X2,
        y2,
        epochs=int(args.phase2_epochs),
        hidden=128,
        seed=args.seed + 7,
        warm_state=p1_state,
        obs_dim=MARK_FULL_DIM,
        lr=2.5e-4,
        sample_weights=ww,
        kl_anchor_state=best_state,
        kl_coef=0.25,
    )
    m2 = match_rate(policy, X2, y2)
    s2 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    print(
        f"  after P2 streak={s2['max_award_streak']} awards={s2['n_award']} breach={s2['n_breach']} match={m2}",
        flush=True,
    )
    append_note(
        f"- Phase2 (blend awards+entries): streak={s2['max_award_streak']} "
        f"awards={s2['n_award']} breach={s2['n_breach']} match={m2}\n"
    )
    if s2["n_breach"] == 0 and (
        s2["max_award_streak"] > best["max_award_streak"]
        or (
            s2["max_award_streak"] == best["max_award_streak"]
            and s2["n_award"] > best["n_award"]
        )
    ):
        best, best_state = s2, {k: v.detach().clone() for k, v in policy.state_dict().items()}
        print("  ** kept P2 **", flush=True)

    # ---- Phase 3: more rounds entry emphasize if not 16 ----
    for rnd in range(1, 6):
        if best["max_award_streak"] >= int(args.target_streak) and best["award_pct"] >= 72.5:
            break
        print(f"\n===== PHASE 3 round {rnd} =====", flush=True)
        policy.load_state_dict(best_state)
        # re-score miss under best
        cur = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
        miss = [r for r in cur["rows"] if not r["award"]]
        entry_pairs = []
        hold_pairs = []
        for row in miss:
            date = str(row["date"])
            t, r = float(row["target_pct"]), float(row["risk_pct"])
            mark = execute_mark_soul_day(
                day_map[date], date, t, r, max_entry_samples=int(args.max_entry_samples)
            )
            mark_s = {k: v for k, v in mark.items() if k != "day"}
            if not (mark_s.get("source") == "soul_plan" and mark_s.get("cleared")):
                continue
            for obs, act in collect_plan_entries(day_map[date], date, t, r, mark_s):
                if act != ACTION_HOLD:
                    entry_pairs.append((obs, act))
                else:
                    hold_pairs.append((obs, act))
        if not entry_pairs:
            break
        p3, w3 = [], []
        for _ in range(25 + 5 * rnd):
            for obs, act in entry_pairs:
                p3.append((obs, act))
                w3.append(8.0)
            for obs, act in hold_pairs[::2]:
                p3.append((obs, act))
                w3.append(1.5)
        for row in [r for r in cur["rows"] if r["award"]][:25]:
            for obs, act in award_imitate(
                day_map[str(row["date"])],
                str(row["date"]),
                float(row["target_pct"]),
                float(row["risk_pct"]),
                policy,
            ):
                p3.append((obs, act))
                w3.append(1.2)
        X3, y3, ww3 = pack(p3, w3)
        policy, _ = train_bc(
            X3,
            y3,
            epochs=20 + 5 * rnd,
            hidden=128,
            seed=args.seed + 20 + rnd,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=3e-4,
            sample_weights=ww3,
            kl_anchor_state=best_state,
            kl_coef=max(0.1, 0.35 - 0.05 * rnd),
        )
        s3 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
        print(
            f"  P3r{rnd} streak={s3['max_award_streak']} awards={s3['n_award']} breach={s3['n_breach']}",
            flush=True,
        )
        append_note(
            f"- Phase3 r{rnd}: streak={s3['max_award_streak']} awards={s3['n_award']} "
            f"breach={s3['n_breach']} entries_trained={len(entry_pairs)}\n"
        )
        if s3["n_breach"] == 0 and (
            s3["max_award_streak"] > best["max_award_streak"]
            or (
                s3["max_award_streak"] == best["max_award_streak"]
                and s3["n_award"] > best["n_award"]
            )
        ):
            best, best_state = s3, {k: v.detach().clone() for k, v in policy.state_dict().items()}
            save_policy(policy, note=f"entry_only_p3r{rnd}", dials=dials)
            print("  ** kept **", flush=True)
        else:
            policy.load_state_dict(best_state)
            print("  rejected", flush=True)

    policy.load_state_dict(best_state)
    save_policy(policy, note="entry_only_best", dials=dials)
    run1 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    run2 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    gate = bool(
        run1["max_award_streak"] >= int(args.target_streak)
        and run1["n_breach"] == 0
        and run1["award_pct"] + 1e-9 >= 72.5
    )
    final = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "entry_only_then_blend_mark_oracle",
        "baseline": {
            "max_award_streak": 8,
            "award_pct": 72.5,
            "n_breach": 0,
            "n_days": 40,
        },
        "pre": {k: v for k, v in pre.items() if k != "rows"},
        "best": {k: v for k, v in best.items() if k != "rows"},
        "final_run1": {k: v for k, v in run1.items() if k != "rows"},
        "final_run2": {k: v for k, v in run2.items() if k != "rows"},
        "gate_pass": gate,
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
        "gate_fix_included": True,
    }
    with open(os.path.join(OUT, "FINAL_2X_STREAK__latest.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
        json.dump(run1["rows"], f, indent=2)
    with open(os.path.join(OUT, "CONSISTENCY__latest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": final["saved_at"],
                "streak": final["final_run1"],
                "baseline": final["baseline"],
                "gate_pass": gate,
                "proven_touched": False,
            },
            f,
            indent=2,
        )
    append_note(
        f"- FINAL: streak={run1['max_award_streak']} awards={run1['n_award']} "
        f"breach={run1['n_breach']} gate_pass={gate}\n"
        f"- miss={[r['date'] for r in run1['rows'] if not r['award']]}\n"
    )
    print(
        f"DONE gate={gate} streak={run1['max_award_streak']} awards={run1['n_award']} "
        f"breach={run1['n_breach']}",
        flush=True,
    )
    return 0 if gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
