"""Fast sprints from best embryo: entry-focused BC on remaining MARK_WOULD_TAKE only.

Uses frozen 50d recipe (seed=42, first 50 days). Loads mark outcomes from
BASELINE/FINAL rows when possible to avoid re-searching Mark every cycle.
"""
from __future__ import annotations

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

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import load_pairs, sample_pairs_for_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import (
    better,
    gate_pass,
    load_policy,
    save_policy,
    score_50d,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
MARK_CACHE_PATH = os.path.join(OUT, "MARK_ORACLE_CACHE__50d.json")


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    pairs = load_pairs()
    tr = sample_pairs_for_days(50, pairs, seed=42, soft_bias=False)
    day_map = {str(d): m1 for d, m1 in days}
    dials = clip_streak_dials(default_streak_dials())
    policy = load_policy(CKPT)
    mark_cache: Dict[str, Dict[str, Any]] = {}
    # warm mark cache from disk if present
    if os.path.isfile(MARK_CACHE_PATH):
        try:
            raw = json.load(open(MARK_CACHE_PATH, encoding="utf-8"))
            mark_cache = {k: v for k, v in raw.items()}
            print(f"loaded mark cache n={len(mark_cache)}", flush=True)
        except Exception as e:
            print(f"cache skip {e}", flush=True)

    print("score…", flush=True)
    pre = score_50d(policy, days, pairs, seed=42, n_days=50, max_entry_samples=16, mark_cache=mark_cache)
    # persist mark cache (without plans if huge — plans needed)
    try:
        # plans are dicts with int keys — json needs str keys
        serial = {}
        for k, v in mark_cache.items():
            vv = dict(v)
            if vv.get("plan") is not None:
                vv["plan"] = {str(kk): int(aa) for kk, aa in vv["plan"].items()}
            serial[k] = vv
        with open(MARK_CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(serial, f)
    except Exception as e:
        print(f"cache write skip {e}", flush=True)

    print(
        f"PRE same={pre['same_outcome']} policy={pre['policy_clear']} mwt={pre['mark_would_take']} breach={pre['n_breach']}",
        flush=True,
    )
    base_clear = int(pre["policy_clear"])
    best = pre
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    cycles = []

    for cyc in range(1, 12):
        if gate_pass(best) and best["policy_clear"] >= base_clear:
            print("GATE HIT", flush=True)
            break
        print(f"\n=== SPRINT {cyc} ===", flush=True)
        policy.load_state_dict(best_state)
        cur = score_50d(
            policy, days, pairs, seed=42, n_days=50, max_entry_samples=16, mark_cache=mark_cache
        )
        mwt_rows = [r for r in cur["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        award_rows = [r for r in cur["rows"] if r["miss_class"] == "AWARD"]
        print(f"  mwt={len(mwt_rows)} award={len(award_rows)}", flush=True)

        xs, ys, ws = [], [], []
        for row in mwt_rows:
            date = str(row["date"])
            t, r = float(row["target_pct"]), float(row["risk_pct"])
            ckey = f"{date}|{t}|{r}"
            if ckey not in mark_cache:
                m = execute_mark_soul_day(day_map[date], date, t, r, max_entry_samples=16)
                mark_cache[ckey] = {k: v for k, v in m.items() if k != "day"}
            mark = mark_cache[ckey]
            plan = mark.get("plan")
            if not plan or mark.get("source") != "soul_plan":
                continue
            # restore plan keys to int
            if plan and isinstance(next(iter(plan.keys()), None), str):
                plan = {int(k): int(v) for k, v in plan.items()}
            day = GoalEquityDay(
                day_map[date],
                target_pct=t,
                risk_pct=r,
                date_str=date,
                eyes_mode="mark_doctrine",
                risk_use_frac=float(mark["risk_use_frac"]),
                per_trade_cap_pct=float(mark["per_trade_cap_pct"]),
                mark_soul=True,
                full_obs=True,
            )
            day._plan_lock_ruf = float(mark["risk_use_frac"])
            day._plan_lock_cap = float(mark["per_trade_cap_pct"])
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
                ncopy = 12 if act != ACTION_HOLD else 2
                w = 20.0 if act != ACTION_HOLD else 2.5
                for _ in range(ncopy):
                    xs.append(np.asarray(obs, np.float32).reshape(-1))
                    ys.append(act)
                    ws.append(w)
                day.step_action(tb, act)

        # Award days: Mark plan craft (dirs) + light self-imitate HOLDs
        for row in award_rows:
            date = str(row["date"])
            t, r = float(row["target_pct"]), float(row["risk_pct"])
            ckey = f"{date}|{t}|{r}"
            if ckey not in mark_cache:
                m = execute_mark_soul_day(day_map[date], date, t, r, max_entry_samples=16)
                mark_cache[ckey] = {k: v for k, v in m.items() if k != "day"}
            mark = mark_cache[ckey]
            plan = mark.get("plan")
            if plan and isinstance(next(iter(plan.keys()), 0), str):
                plan = {int(k): int(v) for k, v in plan.items()}
            if plan and mark.get("source") == "soul_plan":
                day = GoalEquityDay(
                    day_map[date],
                    target_pct=t,
                    risk_pct=r,
                    date_str=date,
                    eyes_mode="mark_doctrine",
                    risk_use_frac=float(mark["risk_use_frac"]),
                    per_trade_cap_pct=float(mark["per_trade_cap_pct"]),
                    mark_soul=True,
                    full_obs=True,
                )
                day._plan_lock_ruf = float(mark["risk_use_frac"])
                day._plan_lock_cap = float(mark["per_trade_cap_pct"])
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
                    if act != ACTION_HOLD:
                        for _ in range(4):
                            xs.append(np.asarray(obs, np.float32).reshape(-1))
                            ys.append(act)
                            ws.append(6.0)
                    elif np.random.random() < 0.35:
                        xs.append(np.asarray(obs, np.float32).reshape(-1))
                        ys.append(act)
                        ws.append(1.5)
                    day.step_action(tb, act)
            else:
                day = GoalEquityDay(
                    day_map[date],
                    target_pct=t,
                    risk_pct=r,
                    date_str=date,
                    eyes_mode="mark_doctrine",
                    mark_soul=True,
                    full_obs=True,
                    mark_align_policy=True,
                )
                i = 0
                for tb in day.runner.decision_indices():
                    if day.dead or day.banked:
                        break
                    obs = day.observe(tb)
                    with torch.no_grad():
                        a, _ = policy.act(obs, greedy=True)
                        a = int(a)
                    if a != ACTION_HOLD or i % 3 == 0:
                        xs.append(np.asarray(obs, np.float32).reshape(-1))
                        ys.append(a)
                        ws.append(1.2 if a == ACTION_HOLD else 2.0)
                    day.step_action(tb, a)
                    i += 1

        if len(ys) < 20:
            print("  no labels", flush=True)
            break
        X = np.stack(xs)
        y = np.asarray(ys, np.int64)
        w = np.asarray(ws, np.float32)
        print(f"  train n={len(y)} dir={int((y != 0).sum())}", flush=True)
        # Alternate: high entry push vs more HOLD/KL protection
        kl = 0.35 if cyc % 2 == 1 else 0.55
        lr = 4e-4 if cyc % 2 == 1 else 2e-4
        pol2, losses = train_bc(
            X,
            y,
            epochs=30 + 2 * cyc,
            hidden=128,
            seed=42 + cyc,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=lr,
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=kl,
        )
        print(f"  match={match_rate(pol2, X, y)}", flush=True)
        post = score_50d(
            pol2, days, pairs, seed=42, n_days=50, max_entry_samples=16, mark_cache=mark_cache
        )
        print(
            f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )
        cycles.append(
            {
                "cycle": cyc,
                "pre": {k: v for k, v in cur.items() if k != "rows"},
                "post": {k: v for k, v in post.items() if k != "rows"},
            }
        )
        if better(post, cur, base_clear) or (
            post["n_breach"] == 0
            and post["policy_clear"] >= base_clear
            and post["same_outcome"] > best["same_outcome"]
        ):
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            save_policy(pol2, note=f"sprint_{cyc}", dials=dials)
            print("  KEEP", flush=True)
            base_clear = max(base_clear, int(post["policy_clear"]))
        else:
            print("  REJECT", flush=True)

    policy.load_state_dict(best_state)
    save_policy(policy, note="sprint_best", dials=dials)
    r1 = score_50d(policy, days, pairs, seed=42, n_days=50, max_entry_samples=16, mark_cache=mark_cache)
    r2 = score_50d(policy, days, pairs, seed=42, n_days=50, max_entry_samples=16, mark_cache=mark_cache)
    assert r1["same_outcome"] == r2["same_outcome"]
    passed = gate_pass(r1)
    out = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "fable_50d_sprint_entry_focus",
        "final_run1": {k: v for k, v in r1.items() if k != "rows"},
        "final_run2": {k: v for k, v in r2.items() if k != "rows"},
        "final_rows": r1["rows"],
        "cycles": cycles,
        "gate_pass": passed,
        "proven_touched": False,
        "shell_touched": False,
    }
    with open(os.path.join(OUT, "FINAL_50D_MATCH__latest.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    with open(os.path.join(OUT, "LOOP_CYCLES_50D__latest.json"), "w", encoding="utf-8") as f:
        json.dump({"cycles": cycles, "final": out["final_run1"], "gate_pass": passed}, f, indent=2)
    print(f"DONE gate_pass={passed} same={r1['same_outcome']}/50 policy={r1['policy_clear']} breach={r1['n_breach']}", flush=True)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
