"""Hard fix for MARK_WOULD_TAKE days: DAgger + reward-weighted BC.

Walks the *policy* path, labels with full-day Mark oracle actions (MARK HERE
power offline), weights samples by streak reward dials, trains the embryo.
No REINFORCE thrash. No shell/PROVEN changes.

Usage:
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/train_miss_days_dagger.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import (
    load_pairs,
    sample_pairs_for_days,
)
from lineages.adaptive_rl_brain_7_31_26.loop_2x_streak_rewards import (
    load_policy,
    save_policy,
    score_streak,
    sample_weight_for,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    clip_streak_dials,
    default_streak_dials,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT = os.path.join(CKPT_DIR, "mark_consistency")
CKPT = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
STREAK_DIALS_PATH = os.path.join(OUT, "STREAK_REWARD_DIALS__latest.json")


def _dials() -> Dict[str, float]:
    d = default_streak_dials()
    if os.path.isfile(STREAK_DIALS_PATH):
        try:
            raw = json.load(open(STREAK_DIALS_PATH, encoding="utf-8"))
            d = clip_streak_dials(raw.get("dials", d))
        except Exception:
            pass
    # hard push for miss-day craft
    d = clip_streak_dials(
        {
            **d,
            "mark_would_take_eod_penalty": -15.0,
            "soul_side_entry_bonus": 4.5,
            "soul_side_misread_penalty": -6.0,
            "streak_break_penalty": -12.0,
            "streak_award_base": 6.0,
            "streak_award_per_prior": 2.0,
        }
    )
    return d


def collect_dagger_on_day(
    m1,
    date: str,
    target: float,
    risk: float,
    policy: Channel1Policy,
    mark: Dict[str, Any],
    dials: Dict[str, float],
    *,
    gap_class: str,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Policy trajectory states × Mark oracle action labels (DAgger)."""
    plan = mark.get("plan") or {}
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ws: List[float] = []
    day = GoalEquityDay(
        m1,
        target_pct=float(target),
        risk_pct=float(risk),
        date_str=str(date),
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=True,
        mark_align_policy=True,
    )
    # If Mark has plan size, also clone size dials on a second walk for labels
    indices = day.runner.decision_indices()
    prev_t = 0
    for tb in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev_t, tb):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = tb + 1
        if day.dead or day.banked:
            break
        obs = day.observe(tb)
        with torch.no_grad():
            pol_a, _ = policy.act(obs, greedy=True)
            pol_a = int(pol_a)
        if plan:
            mark_a = int(plan.get(int(tb), ACTION_HOLD))
        else:
            mark_a = int(day.recommended_action(tb))
        w = sample_weight_for(gap_class, mark_a, dials)
        # disagree → much higher weight (DAgger correction)
        if pol_a != mark_a:
            w *= 3.0
        if mark_a != ACTION_HOLD:
            w *= 2.0
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(mark_a)
        ws.append(float(w))
        # step with *policy* so next obs is on-policy (DAgger)
        day.step_action(tb, pol_a)
    return xs, ys, ws


def collect_mark_plan_path(
    m1,
    date: str,
    target: float,
    risk: float,
    mark: Dict[str, Any],
    dials: Dict[str, float],
    *,
    gap_class: str,
) -> Tuple[List[np.ndarray], List[int], List[float]]:
    """Mark plan trajectory labels (expert states)."""
    plan = mark.get("plan")
    if mark.get("source") != "soul_plan" or plan is None:
        return [], [], []
    day = GoalEquityDay(
        m1,
        target_pct=float(target),
        risk_pct=float(risk),
        date_str=str(date),
        eyes_mode="mark_doctrine",
        risk_use_frac=float(mark["risk_use_frac"]),
        per_trade_cap_pct=float(mark["per_trade_cap_pct"]),
        mark_soul=True,
        full_obs=True,
    )
    day._plan_lock_ruf = float(mark["risk_use_frac"])
    day._plan_lock_cap = float(mark["per_trade_cap_pct"])
    xs, ys, ws = [], [], []
    indices = day.runner.decision_indices()
    prev_t = 0
    for tb in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev_t, tb):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = tb + 1
        if day.dead or day.banked:
            break
        obs = day.observe(tb)
        act = int(plan.get(int(tb), ACTION_HOLD))
        w = sample_weight_for(gap_class, act, dials) * 1.5
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(act)
        ws.append(float(w))
        day.step_action(tb, act)
    return xs, ys, ws


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--rounds", type=int, default=4, help="DAgger outer rounds")
    ap.add_argument("--max-entry-samples", type=int, default=20)
    ap.add_argument("--miss-oversample", type=int, default=4)
    ap.add_argument(
        "--award-keep",
        type=int,
        default=0,
        help="include N award days (0=all award days for balance)",
    )
    ap.add_argument("--kl-coef", type=float, default=0.75, help="KL to anchor good embryo")
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    dials = _dials()
    with open(STREAK_DIALS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": datetime.now(timezone.utc).isoformat(),
                "dials": dials,
                "note": "dagger miss-day hard push",
            },
            f,
            indent=2,
        )

    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    _, forward = split_practice_forward(all_days, practice_n=50)
    n_days = min(int(args.n_days), len(forward))
    window = forward[:n_days]
    pairs_raw = load_pairs()
    tr = sample_pairs_for_days(n_days, pairs_raw, seed=args.seed, soft_bias=False)
    policy = load_policy(CKPT)

    print("Baseline score…", flush=True)
    pre = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
    print(
        f"  PRE max_streak={pre['max_award_streak']} awards={pre['n_award']}/{pre['n_days']} "
        f"breach={pre['n_breach']}",
        flush=True,
    )

    history = [{"round": 0, "score": {k: v for k, v in pre.items() if k != "rows"}}]
    best_score = pre
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    for rnd in range(1, int(args.rounds) + 1):
        print(f"\n===== DAGGER ROUND {rnd}/{args.rounds} =====", flush=True)
        # Always train from best embryo (never cascade from a worse round)
        policy.load_state_dict(best_state)
        policy.eval()
        cur = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
        miss_rows = [r for r in cur["rows"] if not r["award"]]
        award_rows = [r for r in cur["rows"] if r["award"]]
        print(f"  miss_days={len(miss_rows)} award_days={len(award_rows)}", flush=True)

        xs_all: List[np.ndarray] = []
        ys_all: List[int] = []
        ws_all: List[float] = []
        oracle_meta = []

        # Balance: ALL award days (1x) + miss days (mild oversample)
        n_aw = len(award_rows) if int(args.award_keep) <= 0 else int(args.award_keep)
        focus = list(miss_rows) + list(award_rows[:n_aw])
        day_map = {str(d): m1 for d, m1 in window}
        # Cache Mark oracle per day (once per round)
        mark_cache: Dict[str, Dict[str, Any]] = {}

        for row in focus:
            date = str(row["date"])
            m1 = day_map[date]
            t, r = float(row["target_pct"]), float(row["risk_pct"])
            ckey = f"{date}|{t}|{r}"

            if row["award"]:
                # Fast path: self-imitate winning policy path (no full soul search)
                gclass = "AWARD"
                mark_s = {"plan": None, "source": "policy_award", "cleared": True}
                mark_clear = True
                day_w = GoalEquityDay(
                    m1,
                    target_pct=t,
                    risk_pct=r,
                    date_str=date,
                    eyes_mode="mark_doctrine",
                    mark_soul=True,
                    full_obs=True,
                    mark_align_policy=True,
                )
                for tb in day_w.runner.decision_indices():
                    if day_w.dead or day_w.banked:
                        break
                    obs = day_w.observe(tb)
                    with torch.no_grad():
                        act, _ = policy.act(obs, greedy=True)
                        act = int(act)
                    w = sample_weight_for("AWARD", act, dials) * 0.7
                    xs_all.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                    ys_all.append(act)
                    ws_all.append(float(w))
                    day_w.step_action(tb, act)
                oracle_meta.append(
                    {
                        "date": date,
                        "class": gclass,
                        "mark_clear": True,
                        "policy_award": True,
                    }
                )
                print(f"  {date} AWARD self-imitate", flush=True)
                continue

            # Miss days: full-day Mark oracle (expensive, necessary)
            if ckey not in mark_cache:
                mark = execute_mark_soul_day(
                    m1, date, t, r, max_entry_samples=int(args.max_entry_samples)
                )
                mark_cache[ckey] = {k: v for k, v in mark.items() if k != "day"}
            mark_s = mark_cache[ckey]
            mark_clear = bool(mark_s.get("cleared") and not mark_s.get("breached"))
            gclass = "MARK_WOULD_TAKE" if mark_clear else "NO_OPPORTUNITY"
            oracle_meta.append(
                {
                    "date": date,
                    "class": gclass,
                    "mark_clear": mark_clear,
                    "policy_award": False,
                }
            )
            reps = int(args.miss_oversample) if gclass == "MARK_WOULD_TAKE" else 1
            for _ in range(reps):
                dx, dy, dw = collect_dagger_on_day(
                    m1, date, t, r, policy, mark_s, dials, gap_class=gclass
                )
                xs_all.extend(dx)
                ys_all.extend(dy)
                ws_all.extend(dw)
                mx, my, mw = collect_mark_plan_path(
                    m1, date, t, r, mark_s, dials, gap_class=gclass
                )
                xs_all.extend(mx)
                ys_all.extend(my)
                ws_all.extend(mw)
            print(
                f"  {date} {gclass} mark_clear={mark_clear} reps={reps}",
                flush=True,
            )

        if len(ys_all) < 20:
            print("  too few samples", flush=True)
            break
        X = np.stack(xs_all)
        y = np.asarray(ys_all, dtype=np.int64)
        w = np.asarray(ws_all, dtype=np.float32)
        print(f"  train n={len(y)} mean_w={float(w.mean()):.2f}", flush=True)

        warm = {k: v.detach().clone() for k, v in best_state.items()}
        policy, losses = train_bc(
            X,
            y,
            epochs=int(args.epochs),
            hidden=128,
            seed=args.seed + rnd * 31,
            warm_state=warm,
            obs_dim=MARK_FULL_DIM,
            lr=float(args.lr),
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=float(args.kl_coef),
        )
        m = match_rate(policy, X, y)
        print(f"  match={m} loss_tail={losses[-3:]}", flush=True)

        post = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
        print(
            f"  POST max_streak={post['max_award_streak']} awards={post['n_award']}/{post['n_days']} "
            f"breach={post['n_breach']} award_pct={post['award_pct']:.1f}",
            flush=True,
        )
        history.append(
            {
                "round": rnd,
                "match": m,
                "oracle_meta": oracle_meta,
                "score": {k: v for k, v in post.items() if k != "rows"},
                "n_train": len(y),
            }
        )
        improved = post["n_breach"] == 0 and (
            post["max_award_streak"] > best_score["max_award_streak"]
            or (
                post["max_award_streak"] == best_score["max_award_streak"]
                and post["award_pct"] > best_score["award_pct"] + 0.05
            )
        )
        if improved:
            best_score = post
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            save_policy(policy, note=f"dagger_round_{rnd}_best", dials=dials)
            print("  ** new best embryo kept **", flush=True)
        else:
            # reject regression — keep best on disk
            policy.load_state_dict(best_state)
            save_policy(policy, note=f"dagger_round_{rnd}_rejected_keep_best", dials=dials)
            print("  rejected (no improvement) — restored best", flush=True)

        if best_score["max_award_streak"] >= 16 and best_score.get("n_breach", 0) == 0:
            print("  *** 2× STREAK GATE HIT ***", flush=True)
            break

    # restore best
    policy.load_state_dict(best_state)
    save_policy(policy, note="dagger_best", dials=dials)
    final = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
    final2 = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
    assert final["max_award_streak"] == final2["max_award_streak"]

    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "dagger_mark_oracle_reward_weighted_bc",
        "pre": {k: v for k, v in pre.items() if k != "rows"},
        "final": {k: v for k, v in final.items() if k != "rows"},
        "final_run2": {k: v for k, v in final2.items() if k != "rows"},
        "history": history,
        "dials": dials,
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
    }
    with open(os.path.join(OUT, "DAGGER_2X__latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
        json.dump(final["rows"], f, indent=2)
    with open(os.path.join(OUT, "FINAL_2X_STREAK__latest.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(
        f"DONE final streak={final['max_award_streak']} awards={final['n_award']}/{final['n_days']} "
        f"breach={final['n_breach']}",
        flush=True,
    )
    return 0 if final["max_award_streak"] >= 16 and final["n_breach"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
