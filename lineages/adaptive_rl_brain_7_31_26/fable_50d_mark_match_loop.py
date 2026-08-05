"""Fable 50-day loop: miss class → Mark labels → rewards → BC/DAgger → keep/reject.

Frozen recipe:
  - first 50 loadable calendar days (chronological, min_bars=900)
  - seed=42 for random T/R from ten_pairs.json (soft_bias=False)
  - full-obs Mark-aligned pure-greedy policy vs Mark full-day soul plans
  - keep only if policy_clear and same_outcome do not fall; n_breach stays 0

Usage (repo root):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/fable_50d_mark_match_loop.py
  python lineages/adaptive_rl_brain_7_31_26/fable_50d_mark_match_loop.py --max-cycles 8
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

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import load_pairs, sample_pairs_for_days
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_HOLD,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    apply_autopsy_to_streak_dials,
    clip_streak_dials,
    default_streak_dials,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT = os.path.join(CKPT_DIR, "fable_50d_match")
CKPT = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
LOOP_LOG = os.path.join(OUT, "LOOP_CYCLES_50D__latest.json")
FINAL = os.path.join(OUT, "FINAL_50D_MATCH__latest.json")
STREAK_DIALS = os.path.join(CKPT_DIR, "mark_consistency", "STREAK_REWARD_DIALS__latest.json")

# Frozen recipe constants (do not change mid-goal without new baseline tag)
N_DAYS = 50
SEED = 42
WINDOW = "first_50_calendar_loadable"


def load_policy(path: str = CKPT) -> Channel1Policy:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    pol = Channel1Policy(
        obs_dim=int(blob.get("obs_dim", MARK_FULL_DIM)),
        hidden=int(blob.get("hidden", 128)),
    )
    pol.load_state_dict(blob["state_dict"])
    pol.eval()
    return pol


def save_policy(policy: Channel1Policy, *, note: str, dials: Dict[str, float]) -> None:
    blob = {
        "tag": "mark_clone_full_obs_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state_dict": policy.state_dict(),
        "hidden": 128,
        "obs_dim": MARK_FULL_DIM,
        "eyes_mode": "mark_doctrine",
        "teacher": "fable_50d_mark_match",
        "full_obs": True,
        "mark_align_policy": True,
        "streak_reward_dials": dials,
        "train_note": note,
        "proven_touched": False,
        "shell_touched": False,
    }
    torch.save(blob, CKPT)
    torch.save(blob, os.path.join(CKPT_DIR, "mark_clone_latest.pt"))


def sample_weight(gap_class: str, action: int, dials: Dict[str, float]) -> float:
    """Reward-derived sample importance. Directional Mark craft must dominate HOLD."""
    d = clip_streak_dials(dials)
    g = gap_class.upper()
    act = int(action)
    if g == "MARK_WOULD_TAKE":
        # Sparse BUY/SELL is the craft; HOLD is anti-thrash but must not drown dirs
        if act != ACTION_HOLD:
            w = abs(float(d["mark_would_take_eod_penalty"])) + float(
                d["soul_side_entry_bonus"]
            )
            w *= 8.0
        else:
            w = 1.0 + 0.25 * abs(float(d["soul_side_misread_penalty"]))
        return float(w)
    if g == "NO_OPPORTUNITY":
        return float(d["no_opp_hold_bonus"]) + (1.5 if act == ACTION_HOLD else 0.3)
    if g == "AWARD":
        # Preserve good days lightly; prefer keeping directional entries if any
        if act != ACTION_HOLD:
            return 2.0 + 0.3 * float(d["streak_award_base"])
        return 0.6
    return 1.0


def score_50d(
    policy: Channel1Policy,
    days: Sequence[Tuple[str, Any]],
    pairs_raw: List[dict],
    *,
    seed: int = SEED,
    n_days: int = N_DAYS,
    max_entry_samples: int = 20,
    classify_miss: bool = True,
    mark_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Mark full-day plan vs policy pure-greedy Mark-aligned on frozen window."""
    window = list(days[:n_days])
    tr = sample_pairs_for_days(len(window), pairs_raw, seed=seed, soft_bias=False)
    if mark_cache is None:
        mark_cache = {}
    rows: List[Dict[str, Any]] = []
    for i, ((date, m1), (t, r)) in enumerate(zip(window, tr)):
        ckey = f"{date}|{t}|{r}"
        if ckey not in mark_cache:
            if (i + 1) % 5 == 1 or i == 0:
                print(f"    mark oracle {i+1}/{len(window)} {date}…", flush=True)
            mark = execute_mark_soul_day(
                m1,
                str(date),
                float(t),
                float(r),
                max_entry_samples=int(max_entry_samples),
            )
            mark_cache[ckey] = {k: v for k, v in mark.items() if k != "day"}
        mark = mark_cache[ckey]
        # plan keys may be str if loaded from JSON
        if mark.get("plan") is not None and mark["plan"] and isinstance(
            next(iter(mark["plan"].keys())), str
        ):
            mark["plan"] = {int(k): int(v) for k, v in mark["plan"].items()}
        mark_award = bool(mark.get("cleared") and not mark.get("breached"))
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
        res = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        pol_award = bool(res.cleared and not res.breached)
        plan_winnable = bool(
            mark.get("source") == "soul_plan" and mark.get("cleared") and not mark.get("breached")
        )
        if pol_award:
            mclass = "AWARD"
        elif res.breached:
            mclass = "POLICY_BREACH"
        elif plan_winnable and not pol_award:
            mclass = "MARK_WOULD_TAKE"
        elif not plan_winnable and not mark_award:
            mclass = "NO_OPPORTUNITY"
        elif mark_award and not pol_award:
            mclass = "MARK_WOULD_TAKE"
        else:
            mclass = "BOTH_MISS"

        thrash = int(res.n_entries) > int(mark.get("n_entries") or 0) + 1
        rows.append(
            {
                "date": str(date),
                "target_pct": float(t),
                "risk_pct": float(r),
                "mark_cleared": bool(mark.get("cleared")),
                "mark_breached": bool(mark.get("breached")),
                "mark_award": mark_award,
                "mark_pnl": mark.get("pnl_pct"),
                "mark_n_entries": mark.get("n_entries"),
                "mark_source": mark.get("source"),
                "policy_cleared": bool(res.cleared),
                "policy_breached": bool(res.breached),
                "policy_award": pol_award,
                "policy_pnl": round(float(res.pnl_pct), 4),
                "policy_n_entries": int(res.n_entries),
                "same_outcome": bool(mark_award == pol_award),
                "miss_class": mclass if classify_miss else "",
                "policy_thrash_vs_mark": thrash,
                "mark_plan": mark.get("plan") if mark.get("source") == "soul_plan" else None,
                "mark_ruf": mark.get("risk_use_frac"),
                "mark_cap": mark.get("per_trade_cap_pct"),
            }
        )
        # drop heavy plan from row for JSON size in aggregates path — keep for train separately
    # strip plans from scored rows for compact JSON (train re-runs oracle)
    compact = []
    for r in rows:
        c = {k: v for k, v in r.items() if k != "mark_plan"}
        compact.append(c)

    mark_clear = sum(1 for r in compact if r["mark_award"])
    policy_clear = sum(1 for r in compact if r["policy_award"])
    same = sum(1 for r in compact if r["same_outcome"])
    n_breach = sum(1 for r in compact if r["policy_breached"] or r["mark_breached"])
    counts: Dict[str, int] = {}
    for r in compact:
        counts[r["mark_class"] if "mark_class" in r else r["miss_class"]] = (
            counts.get(r["miss_class"], 0) + 1
        )
    # fix counts key
    counts = {}
    for r in compact:
        counts[r["miss_class"]] = counts.get(r["miss_class"], 0) + 1
    n_mwt = counts.get("MARK_WOULD_TAKE", 0)
    n_no = counts.get("NO_OPPORTUNITY", 0)
    return {
        "n_days": len(compact),
        "mark_clear": mark_clear,
        "policy_clear": policy_clear,
        "same_outcome": same,
        "n_breach": n_breach,
        "miss_class_counts": counts,
        "mark_would_take": n_mwt,
        "no_opportunity": n_no,
        "seed": seed,
        "window": WINDOW,
        "decode": "policy_full_obs_mark_align_pure_greedy",
        "soft_bias": False,
        "rows": compact,
        # full rows with plans only used internally during train via re-oracle
    }


def collect_train_set(
    days: Sequence[Tuple[str, Any]],
    score: Dict[str, Any],
    policy: Channel1Policy,
    dials: Dict[str, float],
    *,
    max_entry_samples: int = 20,
    miss_os: int = 5,
    oracle_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Mark plan path + DAgger policy path, reward-weighted."""
    day_map = {str(d): m1 for d, m1 in days}
    if oracle_cache is None:
        oracle_cache = {}
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ws: List[float] = []
    meta = {"n_mwt": 0, "n_award": 0, "n_no": 0, "n_samples": 0}

    for row in score["rows"]:
        date = str(row["date"])
        m1 = day_map[date]
        t, r = float(row["target_pct"]), float(row["risk_pct"])
        gclass = str(row["miss_class"])
        ckey = f"{date}|{t}|{r}"
        if ckey not in oracle_cache:
            mark = execute_mark_soul_day(
                m1, date, t, r, max_entry_samples=int(max_entry_samples)
            )
            oracle_cache[ckey] = {k: v for k, v in mark.items() if k != "day"}
        mark_s = oracle_cache[ckey]
        plan = mark_s.get("plan") if mark_s.get("source") == "soul_plan" else None

        if gclass == "MARK_WOULD_TAKE":
            meta["n_mwt"] += 1
            reps = int(miss_os)
        elif gclass == "NO_OPPORTUNITY":
            meta["n_no"] += 1
            reps = 1
        elif gclass == "AWARD":
            meta["n_award"] += 1
            reps = 1
        else:
            reps = 1

        for _ in range(reps):
            # Expert plan path — on MWT keep all bars but weight dirs high
            if plan is not None and mark_s.get("source") == "soul_plan":
                day = GoalEquityDay(
                    m1,
                    target_pct=t,
                    risk_pct=r,
                    date_str=date,
                    eyes_mode="mark_doctrine",
                    risk_use_frac=float(mark_s["risk_use_frac"]),
                    per_trade_cap_pct=float(mark_s["per_trade_cap_pct"]),
                    mark_soul=True,
                    full_obs=True,
                )
                day._plan_lock_ruf = float(mark_s["risk_use_frac"])
                day._plan_lock_cap = float(mark_s["per_trade_cap_pct"])
                prev = 0
                for tb in day.runner.decision_indices():
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
                    w = sample_weight(gclass, act, dials)
                    # Oversample directional bars; keep some HOLD for anti-breach
                    if act != ACTION_HOLD and gclass == "MARK_WOULD_TAKE":
                        copies = 5
                    elif act == ACTION_HOLD and gclass == "MARK_WOULD_TAKE":
                        copies = 1  # wait craft
                    else:
                        copies = 1
                    for _c in range(copies):
                        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                        ys.append(act)
                        ws.append(float(w))
                    day.step_action(tb, act)

            # DAgger: policy path, Mark action at bar (plan or online)
            if gclass in ("MARK_WOULD_TAKE", "NO_OPPORTUNITY"):
                dayp = GoalEquityDay(
                    m1,
                    target_pct=t,
                    risk_pct=r,
                    date_str=date,
                    eyes_mode="mark_doctrine",
                    mark_soul=True,
                    full_obs=True,
                    mark_align_policy=True,
                )
                prev = 0
                for tb in dayp.runner.decision_indices():
                    if dayp.dead or dayp.banked:
                        break
                    for bt in range(prev, tb):
                        if dayp.dead or dayp.banked:
                            break
                        dayp._mark_bar(bt)
                    prev = tb + 1
                    if dayp.dead or dayp.banked:
                        break
                    obs = dayp.observe(tb)
                    with torch.no_grad():
                        pol_a, _ = policy.act(obs, greedy=True)
                        pol_a = int(pol_a)
                    if plan is not None:
                        mark_a = int(plan.get(int(tb), ACTION_HOLD))
                    else:
                        mark_a = int(dayp.recommended_action(tb))
                    w = sample_weight(gclass, mark_a, dials)
                    if pol_a != mark_a:
                        w *= 2.0
                    # Only keep disagree bars + all directional Mark bars (cut HOLD spam)
                    keep = (mark_a != ACTION_HOLD) or (pol_a != mark_a)
                    if keep:
                        copies = 4 if mark_a != ACTION_HOLD else 1
                        for _c in range(copies):
                            xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                            ys.append(mark_a)
                            ws.append(float(w))
                    dayp.step_action(tb, pol_a)

            # Award days: self-imitate subsample (preserve good days, less HOLD flood)
            if gclass == "AWARD":
                daya = GoalEquityDay(
                    m1,
                    target_pct=t,
                    risk_pct=r,
                    date_str=date,
                    eyes_mode="mark_doctrine",
                    mark_soul=True,
                    full_obs=True,
                    mark_align_policy=True,
                )
                step_i = 0
                for tb in daya.runner.decision_indices():
                    if daya.dead or daya.banked:
                        break
                    obs = daya.observe(tb)
                    with torch.no_grad():
                        a, _ = policy.act(obs, greedy=True)
                        a = int(a)
                    w = sample_weight("AWARD", a, dials)
                    # keep every 2nd HOLD, all directional
                    if a != ACTION_HOLD or (step_i % 2 == 0):
                        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                        ys.append(a)
                        ws.append(float(w))
                    daya.step_action(tb, a)
                    step_i += 1

    meta["n_samples"] = len(ys)
    if not xs:
        return (
            np.zeros((0, MARK_FULL_DIM), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            meta,
        )
    return (
        np.stack(xs),
        np.asarray(ys, dtype=np.int64),
        np.asarray(ws, dtype=np.float32),
        meta,
    )


def not_worse(post: Dict[str, Any], pre: Dict[str, Any], baseline_policy_clear: int) -> bool:
    """Reject if worse: clear/same fell, breach, or below baseline clear."""
    if int(post["n_breach"]) != 0:
        return False
    if int(post["policy_clear"]) < int(baseline_policy_clear):
        return False
    if int(post["policy_clear"]) < int(pre["policy_clear"]):
        return False
    if int(post["same_outcome"]) < int(pre["same_outcome"]):
        return False
    return True


def better(post: Dict[str, Any], pre: Dict[str, Any], baseline_policy_clear: int) -> bool:
    """Strict improvement under not_worse (update best embryo only when better)."""
    if not not_worse(post, pre, baseline_policy_clear):
        return False
    return (
        int(post["same_outcome"]) > int(pre["same_outcome"])
        or int(post["policy_clear"]) > int(pre["policy_clear"])
        or int(post.get("mark_would_take", 99)) < int(pre.get("mark_would_take", 99))
    )


def gate_pass(score: Dict[str, Any]) -> bool:
    """Policy days match Mark days on frozen window."""
    if int(score["n_breach"]) != 0:
        return False
    if int(score["same_outcome"]) == int(score["n_days"]):
        return True
    # policy clears every day Mark clears, no extra breaches
    # (same_outcome may fail if policy clears a Mark miss — still OK if mark_clear==policy on mark days)
    rows = score["rows"]
    for r in rows:
        if r["mark_award"] and not r["policy_award"]:
            return False
        if r["policy_breached"]:
            return False
    return int(score["policy_clear"]) >= int(score["mark_clear"])


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cycles", type=int, default=10)
    ap.add_argument("--n-days", type=int, default=N_DAYS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--epochs", type=int, default=28)
    ap.add_argument("--miss-os", type=int, default=4)
    ap.add_argument("--kl-coef", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-entry-samples", type=int, default=18)
    ap.add_argument("--baseline-only", action="store_true")
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    print("Loading calendar days…", flush=True)
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    n_days = min(int(args.n_days), len(all_days))
    window = all_days[:n_days]
    pairs_raw = load_pairs()
    print(
        f"window={WINDOW} n={n_days} first={window[0][0]} last={window[-1][0]} seed={args.seed}",
        flush=True,
    )

    dials = default_streak_dials()
    if os.path.isfile(STREAK_DIALS):
        try:
            dials = clip_streak_dials(
                json.load(open(STREAK_DIALS, encoding="utf-8")).get("dials", dials)
            )
        except Exception:
            pass

    policy = load_policy(CKPT)
    mark_cache: Dict[str, Dict[str, Any]] = {}
    print("Scoring baseline 50d (Mark vs policy)…", flush=True)
    baseline_score = score_50d(
        policy,
        window,
        pairs_raw,
        seed=args.seed,
        n_days=n_days,
        max_entry_samples=args.max_entry_samples,
        mark_cache=mark_cache,
    )
    baseline = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "recipe": {
            "window": WINDOW,
            "n_days": n_days,
            "seed": args.seed,
            "soft_bias": False,
            "decode": "policy_full_obs_mark_align_pure_greedy",
            "first_date": str(window[0][0]),
            "last_date": str(window[-1][0]),
            "ckpt": "mark_clone_full_obs_v1.pt",
        },
        "mark_clear": baseline_score["mark_clear"],
        "policy_clear": baseline_score["policy_clear"],
        "same_outcome": baseline_score["same_outcome"],
        "n_breach": baseline_score["n_breach"],
        "n_days": baseline_score["n_days"],
        "miss_class_counts": baseline_score["miss_class_counts"],
        "mark_would_take": baseline_score["mark_would_take"],
        "no_opportunity": baseline_score["no_opportunity"],
        "rows": baseline_score["rows"],
        "proven_touched": False,
        "shell_touched": False,
    }
    with open(BASELINE, "w", encoding="utf-8") as f:
        json.dump(baseline, f, indent=2)
    print(
        f"BASELINE mark={baseline['mark_clear']} policy={baseline['policy_clear']} "
        f"same={baseline['same_outcome']}/50 breach={baseline['n_breach']} "
        f"mwt={baseline['mark_would_take']} no_opp={baseline['no_opportunity']}",
        flush=True,
    )
    if args.baseline_only:
        return 0
    if int(baseline["n_breach"]) != 0:
        print("baseline breach != 0 — abort", flush=True)
        return 2
    if int(baseline["mark_clear"]) == 0:
        print("mark_clear=0 — abort", flush=True)
        return 2

    base_policy_clear = int(baseline["policy_clear"])
    best_score = baseline_score
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    cycles: List[Dict[str, Any]] = []

    if gate_pass(baseline_score):
        print("Already matches Mark on 50d — no train needed", flush=True)
    else:
        for cyc in range(1, int(args.max_cycles) + 1):
            print(f"\n===== FABLE CYCLE {cyc}/{args.max_cycles} =====", flush=True)
            policy.load_state_dict(best_state)
            policy.eval()
            pre = score_50d(
                policy,
                window,
                pairs_raw,
                seed=args.seed,
                n_days=n_days,
                max_entry_samples=args.max_entry_samples,
                mark_cache=mark_cache,
            )
            print(
                f"  PRE same={pre['same_outcome']} policy={pre['policy_clear']} "
                f"mark={pre['mark_clear']} mwt={pre['mark_would_take']} breach={pre['n_breach']}",
                flush=True,
            )
            # retune dials from miss mix
            dials = apply_autopsy_to_streak_dials(
                {
                    "counts": pre["miss_class_counts"],
                    "n_gaps": pre["mark_would_take"] + pre["no_opportunity"],
                    "max_award_streak": 0,
                },
                base=dials,
            )
            # strengthen thrash/misread dials each cycle if stuck
            if cyc > 1 and pre["same_outcome"] <= best_score["same_outcome"]:
                dials = clip_streak_dials(
                    {
                        **dials,
                        "mark_would_take_eod_penalty": dials["mark_would_take_eod_penalty"] - 1.0,
                        "soul_side_entry_bonus": min(dials["soul_side_entry_bonus"] + 0.3, 5.0),
                        "soul_side_misread_penalty": max(
                            dials["soul_side_misread_penalty"] - 0.3, -8.0
                        ),
                    }
                )
            # After a breach rejection, slightly favor HOLD again next train
            if cyc > 1 and cycles and cycles[-1].get("post", {}).get("n_breach", 0) > 0:
                dials = clip_streak_dials(
                    {
                        **dials,
                        "soul_side_entry_bonus": max(dials["soul_side_entry_bonus"] - 0.5, 1.0),
                    }
                )
                print("  post-breach: ease entry bonus (more HOLD room)", flush=True)
            print(f"  dials={dials}", flush=True)

            print("  Collect Mark full-day + DAgger labels (reward-weighted)…", flush=True)
            X, y, w, meta = collect_train_set(
                window,
                pre,
                policy,
                dials,
                max_entry_samples=args.max_entry_samples,
                miss_os=int(args.miss_os) + min(cyc - 1, 3),
                oracle_cache=mark_cache,
            )
            print(f"  labels n={len(y)} meta={meta}", flush=True)
            if len(y) < 40:
                print("  too few labels", flush=True)
                break

            kl = float(args.kl_coef) * (0.92 ** (cyc - 1))
            policy, losses = train_bc(
                X,
                y,
                epochs=int(args.epochs) + 3 * (cyc - 1),
                hidden=128,
                seed=args.seed + cyc * 11,
                warm_state=best_state,
                obs_dim=MARK_FULL_DIM,
                lr=float(args.lr),
                sample_weights=w,
                kl_anchor_state=best_state,
                kl_coef=kl,
            )
            m = match_rate(policy, X, y)
            print(f"  BC match={m} kl={kl:.3f}", flush=True)

            post = score_50d(
                policy,
                window,
                pairs_raw,
                seed=args.seed,
                n_days=n_days,
                max_entry_samples=args.max_entry_samples,
                mark_cache=mark_cache,
            )
            print(
                f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
                f"mark={post['mark_clear']} mwt={post['mark_would_take']} breach={post['n_breach']}",
                flush=True,
            )
            keep = better(post, pre, base_policy_clear)
            # also keep if matches gate vs baseline
            if not keep and gate_pass(post) and int(post["policy_clear"]) >= base_policy_clear:
                keep = True
            cycle_rec = {
                "cycle": cyc,
                "pre": {k: v for k, v in pre.items() if k != "rows"},
                "post": {k: v for k, v in post.items() if k != "rows"},
                "keep": keep,
                "dials": dials,
                "label_meta": meta,
                "bc_match": m,
                "kl_coef": kl,
                "reward_note": (
                    "sample_weights from streak/gap dials; thrash→stronger HOLD; "
                    "MARK_WOULD_TAKE oversample; award self-imitate preserve"
                ),
            }
            cycles.append(cycle_rec)
            if keep:
                best_score = post
                best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
                save_policy(policy, note=f"fable_50d_cycle_{cyc}_keep", dials=dials)
                print("  KEEP", flush=True)
            else:
                policy.load_state_dict(best_state)
                save_policy(policy, note=f"fable_50d_cycle_{cyc}_reject", dials=dials)
                print("  REJECT (restored best)", flush=True)

            if gate_pass(best_score) and int(best_score["policy_clear"]) >= base_policy_clear:
                print("  *** MATCH GATE HIT ***", flush=True)
                break

    # Dual final score
    print("\n===== FINAL DUAL SCORE =====", flush=True)
    policy.load_state_dict(best_state)
    save_policy(policy, note="fable_50d_best", dials=dials)
    run1 = score_50d(
        policy,
        window,
        pairs_raw,
        seed=args.seed,
        n_days=n_days,
        max_entry_samples=args.max_entry_samples,
        mark_cache=mark_cache,
    )
    run2 = score_50d(
        policy,
        window,
        pairs_raw,
        seed=args.seed,
        n_days=n_days,
        max_entry_samples=args.max_entry_samples,
        mark_cache=mark_cache,
    )
    assert run1["same_outcome"] == run2["same_outcome"]
    assert run1["n_breach"] == run2["n_breach"]

    passed = gate_pass(run1) and int(run1["policy_clear"]) >= base_policy_clear
    final = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "baseline_policy_clear": base_policy_clear,
        "final_run1": {k: v for k, v in run1.items() if k != "rows"},
        "final_run2": {k: v for k, v in run2.items() if k != "rows"},
        "final_rows": run1["rows"],
        "cycles": cycles,
        "gate_pass": passed,
        "dials": dials,
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
        "ckpt": CKPT,
    }
    with open(FINAL, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(LOOP_LOG, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": final["saved_at"],
                "baseline_agg": {k: v for k, v in baseline.items() if k != "rows"},
                "cycles": cycles,
                "final": final["final_run1"],
                "gate_pass": passed,
            },
            f,
            indent=2,
        )
    md = [
        "# Fable 50-day Mark match",
        "",
        f"**When:** {final['saved_at']}",
        f"**Window:** {WINDOW} ({baseline['recipe']['first_date']} → {baseline['recipe']['last_date']})",
        f"**Seed:** {args.seed}",
        "",
        "## Baseline",
        f"- mark_clear **{baseline['mark_clear']}** · policy_clear **{baseline['policy_clear']}** · "
        f"same_outcome **{baseline['same_outcome']}/50** · breach **{baseline['n_breach']}**",
        f"- miss classes: `{json.dumps(baseline['miss_class_counts'])}`",
        "",
        "## Final",
        f"- mark_clear **{run1['mark_clear']}** · policy_clear **{run1['policy_clear']}** · "
        f"same_outcome **{run1['same_outcome']}/50** · breach **{run1['n_breach']}**",
        f"- mark_would_take **{run1['mark_would_take']}** (baseline {baseline['mark_would_take']})",
        f"- **gate_pass:** {passed}",
        "",
        f"Cycles: {len(cycles)} · log: `LOOP_CYCLES_50D__latest.json`",
        "",
        "PROVEN untouched · shell untouched · rewards/labels only",
        "",
    ]
    with open(os.path.join(OUT, "FINAL_50D_MATCH__latest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(
        f"DONE gate_pass={passed} same={run1['same_outcome']}/50 "
        f"policy={run1['policy_clear']} mark={run1['mark_clear']} breach={run1['n_breach']}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
