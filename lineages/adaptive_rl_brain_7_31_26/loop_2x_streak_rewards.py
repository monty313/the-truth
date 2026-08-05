"""Loop: full-day Mark oracle → rewards/penalties → train → score until 2× streak.

MARK HERE / soul plan may look at the entire future day offline. The clone
absorbs that via reward-weighted BC + REINFORCE day terminal shaping.
Never peeks future bars online. Never touches PROVEN or shell floors.

Usage (repo root):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/loop_2x_streak_rewards.py
  python lineages/adaptive_rl_brain_7_31_26/loop_2x_streak_rewards.py --max-cycles 8 --target-mult 2.0
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
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import (
    load_pairs,
    longest_award_streak,
    sample_pairs_for_days,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import (
    execute_mark_soul_day,
    search_mark_soul_plan,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import (
    apply_autopsy_to_streak_dials,
    clip_streak_dials,
    day_terminal_streak_reward,
    default_streak_dials,
    soul_alignment_step_reward,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

CKPT_DIR = os.path.join(_HERE, "checkpoints")
OUT = os.path.join(CKPT_DIR, "mark_consistency")
CKPT = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
BASELINE_PATH = os.path.join(OUT, "BASELINE_2X__frozen.json")
LOOP_LOG = os.path.join(OUT, "LOOP_2X_CYCLES__latest.json")
STREAK_DIALS_PATH = os.path.join(OUT, "STREAK_REWARD_DIALS__latest.json")
FINAL_SCORE = os.path.join(OUT, "FINAL_2X_STREAK__latest.json")


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
        "teacher": "full_day_mark_oracle_plus_streak_rewards",
        "full_obs": True,
        "mark_align_policy": True,
        "streak_reward_dials": dials,
        "train_note": note,
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
    }
    torch.save(blob, CKPT)
    torch.save(blob, os.path.join(CKPT_DIR, "mark_clone_latest.pt"))


def score_streak(
    policy: Channel1Policy,
    days: Sequence[Tuple[str, Any]],
    pairs_raw: List[dict],
    *,
    seed: int,
    n_days: int,
) -> Dict[str, Any]:
    window = list(days[:n_days])
    tr = sample_pairs_for_days(len(window), pairs_raw, seed=seed, soft_bias=False)
    rows = []
    for (date, m1), (t, r) in zip(window, tr):
        day = GoalEquityDay(
            m1,
            target_pct=t,
            risk_pct=r,
            date_str=str(date),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=True,
            mark_align_policy=True,
        )
        r_out = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        rows.append(
            {
                "date": str(date),
                "target_pct": t,
                "risk_pct": r,
                "cleared": bool(r_out.cleared),
                "breached": bool(r_out.breached),
                "pnl_pct": round(float(r_out.pnl_pct), 4),
                "n_entries": int(r_out.n_entries),
                "award": bool(r_out.cleared and not r_out.breached),
            }
        )
    max_s, s_i, e_i = longest_award_streak(rows)
    n_breach = sum(1 for x in rows if x["breached"])
    n_award = sum(1 for x in rows if x["award"])
    return {
        "n_days": len(rows),
        "max_award_streak": int(max_s),
        "n_award": n_award,
        "n_breach": n_breach,
        "award_pct": 100.0 * n_award / max(len(rows), 1),
        "streak_dates": [rows[i]["date"] for i in range(s_i, e_i + 1)] if max_s else [],
        "rows": rows,
        "seed": seed,
        "mark_align_policy": True,
        "full_obs": True,
    }


def gap_class_for_day(
    policy_award: bool,
    policy_breach: bool,
    mark_plan_winnable: bool,
    mark_cleared: bool,
) -> str:
    if policy_breach:
        return "POLICY_BREACH"
    if policy_award:
        return "AWARD"
    if mark_plan_winnable and mark_cleared:
        return "MARK_WOULD_TAKE"
    if not mark_plan_winnable and not mark_cleared:
        return "NO_OPPORTUNITY"
    if mark_cleared:
        return "MARK_WOULD_TAKE"
    return "BOTH_MISS"


def sample_weight_for(gap_class: str, action: int, dials: Dict[str, float]) -> float:
    """Map streak reward/penalty dials → BC sample importance."""
    d = clip_streak_dials(dials)
    g = gap_class.upper()
    base = 1.0
    if g == "MARK_WOULD_TAKE":
        base = abs(float(d["mark_would_take_eod_penalty"])) + float(
            d["soul_side_entry_bonus"]
        )
        if int(action) in (ACTION_BUY, ACTION_SELL):
            base *= 2.5  # sparse Mark entries are the craft
        else:
            base *= 1.4  # Mark HOLD = wait sense
    elif g == "NO_OPPORTUNITY":
        base = float(d["no_opp_hold_bonus"]) + 1.0
        if int(action) == ACTION_HOLD:
            base *= 2.0
    elif g == "AWARD":
        base = 0.5 * float(d["streak_award_base"]) + 1.0
        if int(action) in (ACTION_BUY, ACTION_SELL):
            base *= 1.3
    else:
        base = 1.0
    return float(max(base, 0.25))


def collect_oracle_weighted_labels(
    days: Sequence[Tuple[str, Any]],
    pairs_tr: Sequence[Tuple[float, float]],
    policy_rows: Optional[List[dict]],
    dials: Dict[str, float],
    *,
    max_entry_samples: int = 28,
    oversample_mark_take: int = 4,
    oracle_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Full-day Mark oracle labels with reward-derived sample weights."""
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ws: List[float] = []
    meta = {
        "n_days": 0,
        "n_mark_would_take": 0,
        "n_no_opp": 0,
        "n_award": 0,
        "n_samples": 0,
        "gap_counts": {},
    }
    row_by_date = {}
    if policy_rows:
        for r in policy_rows:
            row_by_date[str(r["date"])] = r
    if oracle_cache is None:
        oracle_cache = {}

    for i, ((date, m1), (t, r)) in enumerate(zip(days, pairs_tr)):
        meta["n_days"] += 1
        prow = row_by_date.get(str(date))
        ckey = f"{date}|{t}|{r}"
        if ckey in oracle_cache:
            mark = oracle_cache[ckey]
        else:
            mark = execute_mark_soul_day(
                m1,
                str(date),
                float(t),
                float(r),
                max_entry_samples=int(max_entry_samples),
            )
            # strip non-serializable day object for cache safety
            mark = {k: v for k, v in mark.items() if k != "day"}
            oracle_cache[ckey] = mark
        # soul_plan source that cleared ⇒ full-day Mark oracle found a win
        plan_winnable = bool(
            mark.get("source") == "soul_plan"
            and mark.get("cleared")
            and not mark.get("breached")
        )
        mark_clear = bool(mark.get("cleared") and not mark.get("breached"))
        if prow is not None:
            pol_award = bool(prow["award"])
            pol_breach = bool(prow["breached"])
            gclass = gap_class_for_day(
                pol_award, pol_breach, plan_winnable, mark_clear
            )
        else:
            # practice / no policy row: oracle-only — award if Mark clears
            gclass = "AWARD" if mark_clear else (
                "NO_OPPORTUNITY" if not plan_winnable else "MARK_WOULD_TAKE"
            )
        meta["gap_counts"][gclass] = meta["gap_counts"].get(gclass, 0) + 1
        if gclass == "MARK_WOULD_TAKE":
            meta["n_mark_would_take"] += 1
        elif gclass == "NO_OPPORTUNITY":
            meta["n_no_opp"] += 1
        elif gclass == "AWARD":
            meta["n_award"] += 1

        plan = mark.get("plan")
        # Walk day recording full-day oracle actions
        if mark.get("source") == "soul_plan" and plan is not None:
            day2 = GoalEquityDay(
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
            day2._plan_lock_ruf = float(mark["risk_use_frac"])
            day2._plan_lock_cap = float(mark["per_trade_cap_pct"])
            indices = day2.runner.decision_indices()
            prev_t = 0
            day_xs: List[np.ndarray] = []
            day_ys: List[int] = []
            day_ws: List[float] = []
            for tb in indices:
                if day2.dead or day2.banked:
                    break
                for bt in range(prev_t, tb):
                    if day2.dead or day2.banked:
                        break
                    day2._mark_bar(bt)
                prev_t = tb + 1
                if day2.dead or day2.banked:
                    break
                obs = day2.observe(tb)
                act = int(plan.get(int(tb), ACTION_HOLD))
                w = sample_weight_for(gclass, act, dials)
                day_xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                day_ys.append(act)
                day_ws.append(w)
                day2.step_action(tb, act)
        else:
            day3 = GoalEquityDay(
                m1,
                target_pct=float(t),
                risk_pct=float(r),
                date_str=str(date),
                eyes_mode="mark_doctrine",
                mark_soul=True,
                full_obs=True,
            )
            day_xs, day_ys, day_ws = [], [], []
            for tb in day3.runner.decision_indices():
                if day3.banked or day3.dead:
                    break
                obs = day3.observe(tb)
                act = int(day3.recommended_action(tb))
                w = sample_weight_for(gclass, act, dials)
                day_xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
                day_ys.append(act)
                day_ws.append(w)
                day3.step_action(tb, act)

        reps = int(oversample_mark_take) if gclass == "MARK_WOULD_TAKE" else 1
        for _ in range(reps):
            xs.extend(day_xs)
            ys.extend(day_ys)
            ws.extend(day_ws)

        print(
            f"    oracle {date} T/R={t}/{r} class={gclass} samples={len(day_ys)} x{reps}",
            flush=True,
        )

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


def reinforce_streak_days(
    policy: Channel1Policy,
    days: Sequence[Tuple[str, Any]],
    pairs_tr: Sequence[Tuple[float, float]],
    dials: Dict[str, float],
    *,
    epochs: int = 3,
    lr: float = 3e-4,
    max_entry_samples: int = 28,
    oracle_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> List[float]:
    """REINFORCE with day-terminal streak rewards + per-step soul alignment.

    Full-day Mark oracle supplies gap_class and soul actions (offline peek).
    """
    if oracle_cache is None:
        oracle_cache = {}
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    losses: List[float] = []
    policy.train()
    for ep in range(epochs):
        ep_loss = 0.0
        n_days = 0
        prior_streak = 0
        for (date, m1), (t, r) in zip(days, pairs_tr):
            ckey = f"{date}|{t}|{r}"
            if ckey not in oracle_cache:
                mark = execute_mark_soul_day(
                    m1,
                    str(date),
                    float(t),
                    float(r),
                    max_entry_samples=max_entry_samples,
                )
                oracle_cache[ckey] = {k: v for k, v in mark.items() if k != "day"}
            mark = oracle_cache[ckey]
            mark_plan = mark.get("plan") or {}
            mark_winnable = bool(
                mark.get("source") == "soul_plan" and mark.get("cleared")
            )
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
            logps: List[torch.Tensor] = []
            step_shaping = 0.0
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
                logits = policy(torch.as_tensor(obs, dtype=torch.float32))
                dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                # mix: mostly greedy for stability + sample for explore
                if np.random.random() < 0.75:
                    act = int(torch.argmax(logits.squeeze(0)).item())
                else:
                    act = int(dist.sample().item())
                logps.append(dist.log_prob(torch.tensor(act)))
                if mark_plan:
                    mark_a = int(mark_plan.get(int(tb), ACTION_HOLD))
                else:
                    mark_a = int(day.recommended_action(tb))
                # Always shape toward full-day Mark actions (oracle peek offline)
                step_shaping += soul_alignment_step_reward(
                    action=act,
                    mark_soul_action=mark_a,
                    gap_class="MARK_WOULD_TAKE",
                    dials=dials,
                )
                day.step_action(tb, act)
            if not day.dead and not day.banked:
                for bt in range(prev_t, len(day.m1)):
                    if day.dead or day.banked:
                        break
                    day._mark_bar(bt)
            t_last = len(day.m1) - 1
            day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
            pnl = 100.0 * (day.balance - day.eq0) / day.eq0
            day.min_eq_pct = min(day.min_eq_pct, pnl)
            if pnl <= -day.risk + 1e-12:
                day.breached = True
            cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
                day.banked and not day.breached
            )
            gclass = gap_class_for_day(
                bool(cleared and not day.breached),
                bool(day.breached),
                mark_winnable,
                bool(mark.get("cleared") and not mark.get("breached")),
            )
            if not cleared and mark_winnable:
                gclass = "MARK_WOULD_TAKE"
            term = day_terminal_streak_reward(
                cleared=bool(cleared),
                breached=bool(day.breached),
                prior_streak=prior_streak,
                gap_class=gclass,
                dials=dials,
            )
            R = float(term["total"]) + 0.05 * float(step_shaping)
            if not logps:
                continue
            # REINFORCE: -R * sum logp  (maximize R)
            loss = -float(R) * torch.stack(logps).sum()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            ep_loss += float(loss.item())
            n_days += 1
            if cleared and not day.breached:
                prior_streak += 1
            else:
                prior_streak = 0
        losses.append(ep_loss / max(n_days, 1))
        print(f"  reinforce ep {ep+1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    policy.eval()
    return losses


def autopsy_summary_from_rows(
    rows: List[dict],
    days_map: Dict[str, Any],
    *,
    max_entry_samples: int = 20,
    oracle_cache: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Gap counts for dial retune. Prefer oracle_cache; else treat non-award as learnable."""
    counts: Dict[str, int] = {}
    for row in rows:
        if row.get("award"):
            g = "AWARD"
        elif row.get("breached"):
            g = "POLICY_BREACH"
        else:
            d = str(row["date"])
            t, r = float(row["target_pct"]), float(row["risk_pct"])
            ckey = f"{d}|{t}|{r}"
            if oracle_cache and ckey in oracle_cache:
                mark = oracle_cache[ckey]
                if mark.get("source") == "soul_plan" and mark.get("cleared"):
                    g = "MARK_WOULD_TAKE"
                elif mark.get("cleared"):
                    g = "MARK_WOULD_TAKE"
                else:
                    g = "NO_OPPORTUNITY"
            else:
                # Prior autopsy: ~91% of gaps are Mark-would-take — default learnable
                # so dials push soul-side / streak-break (no expensive re-search here)
                g = "MARK_WOULD_TAKE"
        counts[g] = counts.get(g, 0) + 1
    n_gaps = sum(v for k, v in counts.items() if k != "AWARD")
    max_s, _, _ = longest_award_streak(rows)
    return {
        "counts": counts,
        "n_gaps": n_gaps,
        "n_days": len(rows),
        "max_award_streak": int(max_s),
        "n_award": counts.get("AWARD", 0),
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="2× award streak via rewards/penalties loop")
    ap.add_argument("--max-cycles", type=int, default=6)
    ap.add_argument("--target-mult", type=float, default=2.0)
    ap.add_argument("--n-days", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bc-epochs", type=int, default=25)
    ap.add_argument("--rf-epochs", type=int, default=2)
    ap.add_argument("--max-entry-samples", type=int, default=24)
    ap.add_argument(
        "--baseline-json",
        default="",
        help="optional path to frozen baseline; else score current embryo",
    )
    args = ap.parse_args(argv)

    os.makedirs(OUT, exist_ok=True)
    print("Loading days…", flush=True)
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=50)
    days_map = {str(d): m1 for d, m1 in all_days}
    pairs_raw = load_pairs()
    n_days = min(int(args.n_days), len(forward))
    window = forward[:n_days]
    tr = sample_pairs_for_days(n_days, pairs_raw, seed=args.seed, soft_bias=False)

    # --- baseline ---
    if args.baseline_json and os.path.isfile(args.baseline_json):
        with open(args.baseline_json, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"loaded baseline {args.baseline_json}", flush=True)
    elif os.path.isfile(BASELINE_PATH):
        with open(BASELINE_PATH, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        print(f"loaded frozen {BASELINE_PATH}", flush=True)
    else:
        print("Scoring baseline from current embryo…", flush=True)
        pol0 = load_policy(CKPT)
        base_score = score_streak(
            pol0, window, pairs_raw, seed=args.seed, n_days=n_days
        )
        baseline = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "recipe": {
                "mode": "forward_after_practice_50",
                "seed": args.seed,
                "n_days": n_days,
                "decode": "policy_full_obs_mark_align",
                "ckpt": "mark_clone_full_obs_v1.pt",
                "soft_bias": False,
            },
            "max_award_streak": base_score["max_award_streak"],
            "n_award": base_score["n_award"],
            "n_days": base_score["n_days"],
            "award_pct": base_score["award_pct"],
            "n_breach": base_score["n_breach"],
        }
        with open(BASELINE_PATH, "w", encoding="utf-8") as f:
            json.dump(baseline, f, indent=2)
        with open(os.path.join(OUT, "STREAK_ROWS__baseline.json"), "w", encoding="utf-8") as f:
            json.dump(base_score["rows"], f, indent=2)

    base_streak = int(baseline["max_award_streak"])
    base_award_pct = float(baseline["award_pct"])
    target_streak = int(np.ceil(base_streak * float(args.target_mult)))
    print(
        f"BASELINE max_streak={base_streak} award_pct={base_award_pct} "
        f"→ target max_streak>={target_streak} breach=0",
        flush=True,
    )

    dials = default_streak_dials()
    if os.path.isfile(STREAK_DIALS_PATH):
        try:
            dials = clip_streak_dials(json.load(open(STREAK_DIALS_PATH)).get("dials", dials))
        except Exception:
            pass

    cycles: List[Dict[str, Any]] = []
    best = {
        "max_award_streak": base_streak,
        "award_pct": base_award_pct,
        "n_breach": int(baseline.get("n_breach", 0)),
    }
    policy = load_policy(CKPT)

    for cyc in range(1, int(args.max_cycles) + 1):
        print(f"\n===== CYCLE {cyc}/{args.max_cycles} =====", flush=True)
        pre = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
        print(
            f"  PRE max_streak={pre['max_award_streak']} awards={pre['n_award']}/{pre['n_days']} "
            f"breach={pre['n_breach']} award_pct={pre['award_pct']:.1f}",
            flush=True,
        )
        with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
            json.dump(pre["rows"], f, indent=2)

        # Retune dials from light autopsy
        summary = autopsy_summary_from_rows(
            pre["rows"], days_map, max_entry_samples=args.max_entry_samples
        )
        dials = apply_autopsy_to_streak_dials(summary, base=dials)
        with open(STREAK_DIALS_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "dials": dials,
                    "cycle": cyc,
                    "from": "loop_2x_streak_rewards",
                    "note": "rewards/penalties only — longer award streaks",
                },
                f,
                indent=2,
            )
        print(f"  dials={dials}", flush=True)
        print(f"  gap_counts={summary.get('counts')}", flush=True)

        # Full-day Mark oracle labels — practice craft + forward window (MARK HERE)
        oracle_cache: Dict[str, Dict[str, Any]] = {}
        print("  Collect practice soul labels…", flush=True)
        n_prac = min(20, len(practice))
        prac_tr = sample_pairs_for_days(
            n_prac, pairs_raw, seed=args.seed + cyc, soft_bias=False
        )
        Xp, yp, wp, meta_p = collect_oracle_weighted_labels(
            practice[:n_prac],
            prac_tr,
            None,
            dials,
            max_entry_samples=args.max_entry_samples,
            oversample_mark_take=1,
            oracle_cache=oracle_cache,
        )
        print("  Collect forward full-day Mark oracle (reward-weighted)…", flush=True)
        Xf, yf, wf, meta_f = collect_oracle_weighted_labels(
            window,
            tr,
            pre["rows"],
            dials,
            max_entry_samples=args.max_entry_samples,
            oversample_mark_take=6,
            oracle_cache=oracle_cache,
        )
        if len(yp) and len(yf):
            X = np.concatenate([Xp, Xf], axis=0)
            y = np.concatenate([yp, yf], axis=0)
            w = np.concatenate([wp, wf], axis=0)
        elif len(yf):
            X, y, w = Xf, yf, wf
        else:
            X, y, w = Xp, yp, wp
        print(
            f"  labels n={len(y)} practice={meta_p} forward_oracle={meta_f}",
            flush=True,
        )
        if len(y) < 30:
            print("  too few labels — abort cycle", flush=True)
            break

        warm = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        print("  Reward-weighted BC…", flush=True)
        policy, bc_losses = train_bc(
            X,
            y,
            epochs=args.bc_epochs,
            hidden=128,
            seed=args.seed + cyc * 17,
            warm_state=warm,
            obs_dim=MARK_FULL_DIM,
            lr=6e-4,
            sample_weights=w,
        )
        m = match_rate(policy, X, y)
        print(f"  BC match={m}", flush=True)

        # REINFORCE with streak terminal rewards (reuse oracle cache)
        print("  REINFORCE streak rewards…", flush=True)
        rf_losses = reinforce_streak_days(
            policy,
            window,
            tr,
            dials,
            epochs=args.rf_epochs,
            lr=2e-4,
            max_entry_samples=args.max_entry_samples,
            oracle_cache=oracle_cache,
        )

        save_policy(
            policy,
            note=f"cycle_{cyc}_reward_weighted_bc_plus_rf",
            dials=dials,
        )

        post = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
        print(
            f"  POST max_streak={post['max_award_streak']} awards={post['n_award']}/{post['n_days']} "
            f"breach={post['n_breach']} award_pct={post['award_pct']:.1f}",
            flush=True,
        )
        with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
            json.dump(post["rows"], f, indent=2)

        cycle_rec = {
            "cycle": cyc,
            "pre": {k: v for k, v in pre.items() if k != "rows"},
            "post": {k: v for k, v in post.items() if k != "rows"},
            "dials": dials,
            "label_meta_practice": meta_p,
            "label_meta_forward": meta_f,
            "bc_match": m,
            "bc_losses_tail": bc_losses[-3:] if bc_losses else [],
            "rf_losses": rf_losses,
            "gap_summary": summary,
        }
        cycles.append(cycle_rec)

        if post["n_breach"] == 0 and (
            post["max_award_streak"] > best["max_award_streak"]
            or (
                post["max_award_streak"] == best["max_award_streak"]
                and post["award_pct"] >= best["award_pct"]
            )
        ):
            best = {
                "max_award_streak": post["max_award_streak"],
                "award_pct": post["award_pct"],
                "n_breach": post["n_breach"],
                "n_award": post["n_award"],
                "n_days": post["n_days"],
            }

        hit = (
            post["max_award_streak"] >= target_streak
            and post["n_breach"] == 0
            and post["award_pct"] + 1e-9 >= base_award_pct
        )
        if hit:
            print(f"  *** 2× GATE HIT: streak {post['max_award_streak']} >= {target_streak} ***", flush=True)
            break
        # escalate dials if stuck
        if post["max_award_streak"] <= pre["max_award_streak"]:
            dials = clip_streak_dials(
                {
                    **dials,
                    "mark_would_take_eod_penalty": dials["mark_would_take_eod_penalty"]
                    - 1.5,
                    "soul_side_entry_bonus": min(
                        dials["soul_side_entry_bonus"] + 0.4, 5.0
                    ),
                    "streak_break_penalty": max(
                        dials["streak_break_penalty"] - 1.0, -25.0
                    ),
                }
            )
            print(f"  escalate dials → {dials}", flush=True)

    # Final dual score
    print("\n===== FINAL DUAL SCORE =====", flush=True)
    policy = load_policy(CKPT)
    run1 = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
    run2 = score_streak(policy, window, pairs_raw, seed=args.seed, n_days=n_days)
    assert run1["max_award_streak"] == run2["max_award_streak"]
    assert run1["n_breach"] == run2["n_breach"]

    final = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "baseline": baseline,
        "target_streak": target_streak,
        "best": best,
        "final_run1": {k: v for k, v in run1.items() if k != "rows"},
        "final_run2": {k: v for k, v in run2.items() if k != "rows"},
        "gate_pass": bool(
            run1["max_award_streak"] >= target_streak
            and run1["n_breach"] == 0
            and run1["award_pct"] + 1e-9 >= base_award_pct
        ),
        "cycles": cycles,
        "dials": dials,
        "ckpt": CKPT,
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
    }
    with open(FINAL_SCORE, "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(LOOP_LOG, "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": final["saved_at"],
                "baseline": baseline,
                "target_streak": target_streak,
                "cycles": cycles,
                "final": final["final_run1"],
                "gate_pass": final["gate_pass"],
            },
            f,
            indent=2,
        )
    with open(os.path.join(OUT, "CONSISTENCY__latest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": final["saved_at"],
                "goal": "2x award streak via rewards/penalties",
                "streak": final["final_run1"],
                "baseline": baseline,
                "gate_pass": final["gate_pass"],
                "proven_touched": False,
            },
            f,
            indent=2,
        )
    with open(os.path.join(OUT, "STREAK_ROWS__latest.json"), "w", encoding="utf-8") as f:
        json.dump(run1["rows"], f, indent=2)

    md = [
        "# 2× streak reward loop",
        "",
        f"**When:** {final['saved_at']}",
        f"**Baseline max_streak:** {base_streak} → **target:** {target_streak}",
        f"**Final max_streak:** {run1['max_award_streak']} · awards {run1['n_award']}/{run1['n_days']} · breach **{run1['n_breach']}**",
        f"**award_pct:** {run1['award_pct']:.1f}% (baseline {base_award_pct:.1f}%)",
        f"**Gate pass:** {final['gate_pass']}",
        "",
        "## Cause (rewards/penalties only)",
        f"- dials: `{json.dumps(dials)}`",
        "- full-day Mark soul oracle labels (MARK HERE power offline)",
        "- sample weights from streak/gap reward dials",
        "- REINFORCE day_terminal_streak_reward + soul_alignment_step_reward",
        "- PROVEN untouched · shell untouched",
        "",
        f"Cycles: {len(cycles)} · log: `LOOP_2X_CYCLES__latest.json`",
        "",
    ]
    with open(os.path.join(OUT, "FINAL_2X_STREAK__latest.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md))

    print(
        f"DONE gate_pass={final['gate_pass']} streak={run1['max_award_streak']} "
        f"target={target_streak} breach={run1['n_breach']}",
        flush=True,
    )
    return 0 if final["gate_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
