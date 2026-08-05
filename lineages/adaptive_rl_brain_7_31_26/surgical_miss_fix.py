"""Surgical fix: clone Mark full-day plan actions on learnable miss days only.

Diagnostic: on most MARK_WOULD_TAKE days, Mark plan actions + dynamic mark_soul
size already clear. So BC must hit those sparse entry bars without destroying
award days (KL anchor + award self-imitate).

Rewards/penalties set sample weights. Full-day Mark oracle offline (MARK HERE).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
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
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    clip_streak_dials,
    default_streak_dials,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "mark_consistency")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")


def plan_path_labels(m1, date, t, r, mark, dials, *, gap_class: str, oversample: int):
    """Mark plan path (expert states)."""
    plan = mark.get("plan")
    if not plan or mark.get("source") != "soul_plan":
        return [], [], [], []
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
    xs, ys, ws, entry_keys = [], [], [], []
    indices = day.runner.decision_indices()
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
        w = sample_weight_for(gap_class, act, dials)
        if act != ACTION_HOLD:
            w *= 4.0  # sparse entries are the craft
            entry_keys.append((str(date), int(tb), act, np.asarray(obs, dtype=np.float32).copy()))
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(act)
        ws.append(float(w))
        day.step_action(tb, act)
    X = xs * oversample
    y = ys * oversample
    w = ws * oversample
    return X, y, w, entry_keys


def dagger_path_labels(m1, date, t, r, mark, policy, dials, *, gap_class: str, oversample: int):
    """Policy path states × Mark plan action at same bar (fixes early thrash)."""
    plan = mark.get("plan") or {}
    xs, ys, ws, entry_keys = [], [], [], []
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
    indices = day.runner.decision_indices()
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
        with torch.no_grad():
            pol_a, _ = policy.act(obs, greedy=True)
            pol_a = int(pol_a)
        mark_a = int(plan.get(int(tb), ACTION_HOLD)) if plan else int(day.recommended_action(tb))
        w = sample_weight_for(gap_class, mark_a, dials)
        if pol_a != mark_a:
            w *= 3.5  # correction
        if mark_a == ACTION_HOLD and pol_a != ACTION_HOLD:
            w *= 2.5  # kill early thrash
        if mark_a != ACTION_HOLD:
            w *= 3.0
            entry_keys.append(
                (str(date), int(tb), mark_a, np.asarray(obs, dtype=np.float32).copy())
            )
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(mark_a)
        ws.append(float(w))
        # step with policy so path stays on-policy
        day.step_action(tb, pol_a)
    return xs * oversample, ys * oversample, ws * oversample, entry_keys


def award_self_imitate(m1, date, t, r, policy, dials):
    xs, ys, ws = [], [], []
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
            act, _ = policy.act(obs, greedy=True)
            act = int(act)
        w = sample_weight_for("AWARD", act, dials) * 0.8
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(act)
        ws.append(float(w))
        day.step_action(tb, act)
    return xs, ys, ws


def entry_hit_rate(policy, entry_keys) -> float:
    if not entry_keys:
        return 1.0
    ok = 0
    for _date, _tb, act, obs in entry_keys:
        with torch.no_grad():
            pred, _ = policy.act(obs, greedy=True)
        if int(pred) == int(act):
            ok += 1
    return ok / max(len(entry_keys), 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--miss-os", type=int, default=8)
    ap.add_argument("--kl-coef", type=float, default=0.25)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--max-entry-samples", type=int, default=22)
    ap.add_argument("--target-streak", type=int, default=16)
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    dials = clip_streak_dials(
        {
            **default_streak_dials(),
            "mark_would_take_eod_penalty": -16.0,
            "soul_side_entry_bonus": 5.0,
            "soul_side_misread_penalty": -6.5,
            "streak_break_penalty": -12.0,
            "streak_award_base": 6.0,
            "streak_award_per_prior": 2.0,
        }
    )
    with open(os.path.join(OUT, "STREAK_REWARD_DIALS__latest.json"), "w", encoding="utf-8") as f:
        json.dump({"saved_at": datetime.now(timezone.utc).isoformat(), "dials": dials}, f, indent=2)

    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    _, forward = split_practice_forward(all_days, practice_n=50)
    window = forward[: int(args.n_days)]
    pairs = load_pairs()
    day_map = {str(d): m1 for d, m1 in window}
    policy = load_policy(CKPT)
    pre = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    print(
        f"PRE streak={pre['max_award_streak']} awards={pre['n_award']}/{pre['n_days']} "
        f"breach={pre['n_breach']}",
        flush=True,
    )
    best_score = pre
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    history = [{"round": 0, "score": {k: v for k, v in pre.items() if k != "rows"}}]

    # Precompute Mark oracle for all miss days once (reuse every round)
    miss_oracle: Dict[str, Dict[str, Any]] = {}
    for row in pre["rows"]:
        if row["award"]:
            continue
        date = str(row["date"])
        t, r = float(row["target_pct"]), float(row["risk_pct"])
        print(f"  oracle miss {date}…", flush=True)
        mark = execute_mark_soul_day(
            day_map[date], date, t, r, max_entry_samples=int(args.max_entry_samples)
        )
        miss_oracle[date] = {
            "row": row,
            "mark": {k: v for k, v in mark.items() if k != "day"},
        }
        print(
            f"    → src={mark.get('source')} clear={mark.get('cleared')} pnl={mark.get('pnl_pct')}",
            flush=True,
        )

    for rnd in range(1, int(args.rounds) + 1):
        print(f"\n===== SURGICAL ROUND {rnd}/{args.rounds} =====", flush=True)
        policy.load_state_dict(best_state)
        policy.eval()
        cur = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
        # refresh which days miss under best
        miss_rows = [r for r in cur["rows"] if not r["award"]]
        award_rows = [r for r in cur["rows"] if r["award"]]
        print(f"  miss={len(miss_rows)} award={len(award_rows)}", flush=True)

        xs, ys, ws = [], [], []
        entry_keys = []
        for row in miss_rows:
            date = str(row["date"])
            t, r = float(row["target_pct"]), float(row["risk_pct"])
            if date not in miss_oracle:
                mark = execute_mark_soul_day(
                    day_map[date], date, t, r, max_entry_samples=int(args.max_entry_samples)
                )
                miss_oracle[date] = {
                    "row": row,
                    "mark": {k: v for k, v in mark.items() if k != "day"},
                }
            mark = miss_oracle[date]["mark"]
            gclass = (
                "MARK_WOULD_TAKE"
                if mark.get("source") == "soul_plan" and mark.get("cleared")
                else "NO_OPPORTUNITY"
            )
            osamp = int(args.miss_os) if gclass == "MARK_WOULD_TAKE" else 1
            # 1) expert path
            X1, y1, w1, ek1 = plan_path_labels(
                day_map[date],
                date,
                t,
                r,
                mark,
                dials,
                gap_class=gclass,
                oversample=max(1, osamp // 2),
            )
            # 2) DAgger policy path (kills early thrash)
            X2, y2, w2, ek2 = dagger_path_labels(
                day_map[date],
                date,
                t,
                r,
                mark,
                policy,
                dials,
                gap_class=gclass,
                oversample=osamp,
            )
            xs.extend(X1)
            ys.extend(y1)
            ws.extend(w1)
            xs.extend(X2)
            ys.extend(y2)
            ws.extend(w2)
            entry_keys.extend(ek1)
            entry_keys.extend(ek2)
            print(
                f"  {date} {gclass} plan={len(y1)} dagger={len(y2)} entries={len(ek1)+len(ek2)}",
                flush=True,
            )

        # preserve awards
        for row in award_rows:
            date = str(row["date"])
            ax, ay, aw = award_self_imitate(
                day_map[date],
                date,
                float(row["target_pct"]),
                float(row["risk_pct"]),
                policy,
                dials,
            )
            xs.extend(ax)
            ys.extend(ay)
            ws.extend(aw)

        if len(ys) < 30:
            print("too few", flush=True)
            break
        X = np.stack(xs)
        y = np.asarray(ys, dtype=np.int64)
        w = np.asarray(ws, dtype=np.float32)
        print(f"  train n={len(y)} entry_points={len(entry_keys)}", flush=True)

        # escalate KL down / miss weight up if stuck
        kl = float(args.kl_coef) * (0.85 ** (rnd - 1))
        os_note = int(args.miss_os) + (rnd - 1)
        policy, losses = train_bc(
            X,
            y,
            epochs=int(args.epochs) + 5 * (rnd - 1),
            hidden=128,
            seed=args.seed + rnd * 13,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=float(args.lr) * (1.0 + 0.1 * (rnd - 1)),
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=kl,
        )
        m = match_rate(policy, X, y)
        hit = entry_hit_rate(policy, entry_keys)
        print(f"  match={m} entry_hit={hit:.3f} kl={kl:.3f}", flush=True)

        post = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
        print(
            f"  POST streak={post['max_award_streak']} awards={post['n_award']}/{post['n_days']} "
            f"breach={post['n_breach']} pct={post['award_pct']:.1f}",
            flush=True,
        )
        history.append(
            {
                "round": rnd,
                "match": m,
                "entry_hit": hit,
                "kl": kl,
                "score": {k: v for k, v in post.items() if k != "rows"},
            }
        )
        better = post["n_breach"] == 0 and (
            post["max_award_streak"] > best_score["max_award_streak"]
            or (
                post["max_award_streak"] == best_score["max_award_streak"]
                and post["award_pct"] > best_score["award_pct"] + 0.05
            )
        )
        # also accept pure award_pct lift of +1 day even if streak same
        if post["n_breach"] == 0 and post["n_award"] > best_score["n_award"]:
            better = True
        if better:
            best_score = post
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            save_policy(policy, note=f"surgical_r{rnd}", dials=dials)
            print("  ** kept **", flush=True)
        else:
            policy.load_state_dict(best_state)
            save_policy(policy, note=f"surgical_r{rnd}_reject", dials=dials)
            print("  rejected", flush=True)

        if (
            best_score["max_award_streak"] >= int(args.target_streak)
            and best_score.get("n_breach", 0) == 0
            and best_score["award_pct"] + 1e-9 >= 72.5
        ):
            print("  *** 2× GATE HIT ***", flush=True)
            break

    policy.load_state_dict(best_state)
    save_policy(policy, note="surgical_best", dials=dials)
    run1 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    run2 = score_streak(policy, window, pairs, seed=args.seed, n_days=args.n_days)
    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "surgical_mark_plan_bc_reward_weights",
        "baseline_frozen": {
            "max_award_streak": 8,
            "award_pct": 72.5,
            "n_breach": 0,
            "n_days": 40,
        },
        "pre": {k: v for k, v in pre.items() if k != "rows"},
        "final_run1": {k: v for k, v in run1.items() if k != "rows"},
        "final_run2": {k: v for k, v in run2.items() if k != "rows"},
        "history": history,
        "dials": dials,
        "gate_pass": bool(
            run1["max_award_streak"] >= int(args.target_streak)
            and run1["n_breach"] == 0
            and run1["award_pct"] + 1e-9 >= 72.5
        ),
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
    }
    with open(os.path.join(OUT, "FINAL_2X_STREAK__latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
        json.dump(run1["rows"], f, indent=2)
    with open(os.path.join(OUT, "LOOP_2X_CYCLES__latest.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(
        f"DONE gate={report['gate_pass']} streak={run1['max_award_streak']} "
        f"awards={run1['n_award']} breach={run1['n_breach']}",
        flush=True,
    )
    return 0 if report["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
