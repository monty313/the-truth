"""Fast 50d Fable loop: reuse frozen Mark outcomes from baseline; only re-run policy.

Baseline BASELINE_50D__frozen.json already has per-day Mark awards + T/R.
Train still uses Mark full-day plans (oracle cache built lazily only for MWT days).
"""
from __future__ import annotations

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

from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import (
    better,
    gate_pass,
    load_policy,
    save_policy,
)
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
ORACLE = os.path.join(OUT, "MARK_ORACLE_CACHE__50d.json")


def _run_one_policy_day(args: Tuple) -> dict:
    """Worker-friendly single-day policy score (loads days from global path)."""
    date, t, r, mr, state_dict, hidden, obs_dim = args
    # lazy import for process pool safety
    from lineages.adaptive_rl_brain_7_31_26.equity_day import (
        GoalEquityDay,
        load_calendar_days,
    )
    from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

    # cache calendar on worker
    global _WORKER_DAY_MAP
    if "_WORKER_DAY_MAP" not in globals() or _WORKER_DAY_MAP is None:
        _all = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
        _WORKER_DAY_MAP = {str(d): m1 for d, m1 in _all}
    m1 = _WORKER_DAY_MAP[str(date)]
    pol = Channel1Policy(obs_dim=int(obs_dim), hidden=int(hidden))
    pol.load_state_dict(state_dict)
    pol.eval()
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
    res = day.run(greedy_policy=pol, pure_greedy=True, use_heuristic=False)
    pol_award = bool(res.cleared and not res.breached)
    mark_award = bool(mr["mark_award"])
    if pol_award:
        mclass = "AWARD"
    elif res.breached:
        mclass = "POLICY_BREACH"
    elif mark_award and not pol_award:
        mclass = "MARK_WOULD_TAKE"
    else:
        mclass = "NO_OPPORTUNITY" if not mark_award else "BOTH_MISS"
    if mark_award and not pol_award and not res.breached:
        mclass = "MARK_WOULD_TAKE"
    thrash = int(res.n_entries) > int(mr.get("mark_n_entries") or 0) + 1
    return {
        "date": str(date),
        "target_pct": float(t),
        "risk_pct": float(r),
        "mark_cleared": bool(mr.get("mark_cleared", mark_award)),
        "mark_breached": bool(mr.get("mark_breached", False)),
        "mark_award": mark_award,
        "mark_pnl": mr.get("mark_pnl"),
        "mark_n_entries": mr.get("mark_n_entries"),
        "mark_source": mr.get("mark_source"),
        "policy_cleared": bool(res.cleared),
        "policy_breached": bool(res.breached),
        "policy_award": pol_award,
        "policy_pnl": round(float(res.pnl_pct), 4),
        "policy_n_entries": int(res.n_entries),
        "same_outcome": bool(mark_award == pol_award),
        "miss_class": mclass,
        "policy_thrash_vs_mark": thrash,
    }


_WORKER_DAY_MAP = None


def score_policy_vs_mark_rows(
    policy: Channel1Policy,
    days: List[Tuple[str, Any]],
    mark_rows: List[dict],
    *,
    parallel: bool = True,
    workers: int = 4,
) -> Dict[str, Any]:
    """Policy walk only; Mark outcomes taken from frozen baseline rows."""
    state = {k: v.detach().cpu().clone() for k, v in policy.state_dict().items()}
    hidden = 128
    obs_dim = int(getattr(policy, "obs_dim", MARK_FULL_DIM))
    work = [
        (
            str(mr["date"]),
            float(mr["target_pct"]),
            float(mr["risk_pct"]),
            mr,
            state,
            hidden,
            obs_dim,
        )
        for mr in mark_rows
    ]
    rows: List[dict] = []
    if parallel and len(work) > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        print(f"    parallel policy score workers={workers} days={len(work)}", flush=True)
        # Windows spawn: use max_workers carefully
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = {ex.submit(_run_one_policy_day, w): i for i, w in enumerate(work)}
            done_n = 0
            ordered = [None] * len(work)
            for fut in as_completed(futs):
                i = futs[fut]
                ordered[i] = fut.result()
                done_n += 1
                if done_n % 10 == 0 or done_n == len(work):
                    print(f"    policy days done {done_n}/{len(work)}", flush=True)
            rows = list(ordered)
    else:
        day_map = {str(d): m1 for d, m1 in days}
        for i, mr in enumerate(mark_rows):
            date = str(mr["date"])
            t, r = float(mr["target_pct"]), float(mr["risk_pct"])
            if (i + 1) % 10 == 0 or i == 0:
                print(f"    policy day {i+1}/{len(mark_rows)} {date}", flush=True)
            # sequential fallback via worker fn without reloading map each time
            global _WORKER_DAY_MAP
            _WORKER_DAY_MAP = day_map
            rows.append(
                _run_one_policy_day(
                    (date, t, r, mr, state, hidden, obs_dim)
                )
            )
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["miss_class"]] = counts.get(r["miss_class"], 0) + 1
    return {
        "n_days": len(rows),
        "mark_clear": sum(1 for r in rows if r["mark_award"]),
        "policy_clear": sum(1 for r in rows if r["policy_award"]),
        "same_outcome": sum(1 for r in rows if r["same_outcome"]),
        "n_breach": sum(1 for r in rows if r["policy_breached"] or r["mark_breached"]),
        "miss_class_counts": counts,
        "mark_would_take": counts.get("MARK_WOULD_TAKE", 0),
        "no_opportunity": counts.get("NO_OPPORTUNITY", 0),
        "seed": 42,
        "window": "first_50_calendar_loadable",
        "decode": "policy_full_obs_mark_align_pure_greedy",
        "soft_bias": False,
        "rows": rows,
    }


def load_oracle() -> Dict[str, dict]:
    if not os.path.isfile(ORACLE):
        return {}
    raw = json.load(open(ORACLE, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        vv = dict(v)
        if vv.get("plan") is not None:
            vv["plan"] = {int(a): int(b) for a, b in vv["plan"].items()}
        out[k] = vv
    return out


def save_oracle(cache: Dict[str, dict]) -> None:
    serial = {}
    for k, v in cache.items():
        vv = dict(v)
        if vv.get("plan") is not None:
            vv["plan"] = {str(a): int(b) for a, b in vv["plan"].items()}
        serial[k] = vv
    with open(ORACLE, "w", encoding="utf-8") as f:
        json.dump(serial, f)


def ensure_plan(
    cache: Dict[str, dict],
    day_map: Dict[str, Any],
    date: str,
    t: float,
    r: float,
    max_entry_samples: int = 14,
) -> dict:
    ckey = f"{date}|{t}|{r}"
    if ckey in cache and cache[ckey].get("plan") is not None:
        return cache[ckey]
    print(f"    oracle {date} T/R={t}/{r}…", flush=True)
    m = execute_mark_soul_day(
        day_map[date], date, t, r, max_entry_samples=max_entry_samples
    )
    cache[ckey] = {k: v for k, v in m.items() if k != "day"}
    if cache[ckey].get("plan") is not None:
        cache[ckey]["plan"] = {int(a): int(b) for a, b in cache[ckey]["plan"].items()}
    return cache[ckey]


def collect_labels(
    day_map: Dict[str, Any],
    score: dict,
    policy: Channel1Policy,
    oracle: Dict[str, dict],
    *,
    miss_os: int = 4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for row in score["rows"]:
        date = str(row["date"])
        t, r = float(row["target_pct"]), float(row["risk_pct"])
        g = row["miss_class"]
        if g == "MARK_WOULD_TAKE":
            mark = ensure_plan(oracle, day_map, date, t, r)
            plan = mark.get("plan")
            if not plan or mark.get("source") != "soul_plan":
                continue
            for _rep in range(miss_os):
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
                    ncopy = 8 if act != ACTION_HOLD else 2
                    w = 18.0 if act != ACTION_HOLD else 2.0
                    for _ in range(ncopy):
                        xs.append(np.asarray(obs, np.float32).reshape(-1))
                        ys.append(act)
                        ws.append(w)
                    day.step_action(tb, act)
                # DAgger disagree once
                dayp = GoalEquityDay(
                    day_map[date],
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
                        pa, _ = policy.act(obs, greedy=True)
                        pa = int(pa)
                    ma = int(plan.get(int(tb), ACTION_HOLD))
                    if pa != ma or ma != ACTION_HOLD:
                        w = 12.0 if ma != ACTION_HOLD else 3.0
                        ncopy = 3 if ma != ACTION_HOLD else 1
                        for _ in range(ncopy):
                            xs.append(np.asarray(obs, np.float32).reshape(-1))
                            ys.append(ma)
                            ws.append(w)
                    dayp.step_action(tb, pa)
        elif g == "AWARD":
            # Self-imitate only (no Mark re-oracle) — preserves good days, stays fast
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
                    ws.append(2.0 if a != ACTION_HOLD else 1.0)
                day.step_action(tb, a)
                i += 1
    return (
        np.stack(xs) if xs else np.zeros((0, MARK_FULL_DIM), np.float32),
        np.asarray(ys, np.int64) if ys else np.zeros((0,), np.int64),
        np.asarray(ws, np.float32) if ws else np.zeros((0,), np.float32),
    )


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    with open(BASELINE, encoding="utf-8") as f:
        baseline = json.load(f)
    mark_rows = baseline["rows"]
    # force mark_award path: baseline mark_clear was 50
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    dials = clip_streak_dials(default_streak_dials())
    policy = load_policy(CKPT)
    oracle = load_oracle()

    print("Fast policy score (Mark from baseline)…", flush=True)
    pre = score_policy_vs_mark_rows(policy, days, mark_rows)
    print(
        f"PRE same={pre['same_outcome']} policy={pre['policy_clear']} "
        f"mark={pre['mark_clear']} mwt={pre['mark_would_take']} breach={pre['n_breach']}",
        flush=True,
    )
    base_clear = int(baseline.get("policy_clear", pre["policy_clear"]))
    # allow climbing from current best even if higher than original 27
    base_clear = min(base_clear, int(pre["policy_clear"]))
    # actually keep reject vs original baseline floor
    floor_clear = int(baseline["policy_clear"])
    best = pre
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    cycles = []

    for cyc in range(1, 16):
        if gate_pass(best) and int(best["policy_clear"]) >= floor_clear:
            print("*** GATE HIT ***", flush=True)
            break
        print(f"\n===== FAST CYCLE {cyc}/15 =====", flush=True)
        policy.load_state_dict(best_state)
        # reuse last best score as cur when starting cycle 1 (avoid double full score)
        if cyc == 1:
            cur = best
        else:
            cur = score_policy_vs_mark_rows(policy, days, mark_rows)
        print(
            f"  PRE same={cur['same_outcome']} policy={cur['policy_clear']} mwt={cur['mark_would_take']}",
            flush=True,
        )
        print("  labels…", flush=True)
        # Focus labels on MWT only for speed; awards self-imitate inside collect
        mwt_only = {
            "rows": [r for r in cur["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
            + [r for r in cur["rows"] if r["miss_class"] == "AWARD"][:15]
        }
        X, y, w = collect_labels(
            day_map, mwt_only, policy, oracle, miss_os=4 + min(cyc // 2, 4)
        )
        save_oracle(oracle)
        print(f"  n={len(y)} dir={int((y != 0).sum()) if len(y) else 0}", flush=True)
        if len(y) < 30:
            print("  too few", flush=True)
            break
        kl = 0.30 + 0.05 * (cyc % 3)
        pol2, _ = train_bc(
            X,
            y,
            epochs=28 + cyc,
            hidden=128,
            seed=100 + cyc,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=3.5e-4,
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=kl,
        )
        print(f"  match={match_rate(pol2, X, y)}", flush=True)
        post = score_policy_vs_mark_rows(pol2, days, mark_rows)
        print(
            f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )
        keep = (
            post["n_breach"] == 0
            and int(post["policy_clear"]) >= floor_clear
            and int(post["same_outcome"]) >= int(best["same_outcome"])
            and (
                int(post["same_outcome"]) > int(best["same_outcome"])
                or int(post["policy_clear"]) > int(best["policy_clear"])
                or int(post["mark_would_take"]) < int(best["mark_would_take"])
            )
        )
        cycles.append(
            {
                "cycle": cyc,
                "pre": {k: v for k, v in cur.items() if k != "rows"},
                "post": {k: v for k, v in post.items() if k != "rows"},
                "keep": keep,
                "kl": kl,
            }
        )
        if keep:
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            save_policy(pol2, note=f"fast_c{cyc}", dials=dials)
            print("  KEEP", flush=True)
        else:
            print("  REJECT", flush=True)

    policy.load_state_dict(best_state)
    save_policy(policy, note="fast_best", dials=dials)
    r1 = score_policy_vs_mark_rows(policy, days, mark_rows)
    r2 = score_policy_vs_mark_rows(policy, days, mark_rows)
    assert r1["same_outcome"] == r2["same_outcome"] and r1["n_breach"] == r2["n_breach"]
    passed = gate_pass(r1) and int(r1["policy_clear"]) >= floor_clear
    final = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "fable_50d_fast_loop_mark_from_baseline",
        "baseline": {k: v for k, v in baseline.items() if k != "rows"},
        "final_run1": {k: v for k, v in r1.items() if k != "rows"},
        "final_run2": {k: v for k, v in r2.items() if k != "rows"},
        "final_rows": r1["rows"],
        "cycles": cycles,
        "gate_pass": passed,
        "proven_touched": False,
        "shell_touched": False,
        "rewards_penalties_cause": True,
    }
    with open(os.path.join(OUT, "FINAL_50D_MATCH__latest.json"), "w", encoding="utf-8") as f:
        json.dump(final, f, indent=2)
    with open(os.path.join(OUT, "LOOP_CYCLES_50D__latest.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "saved_at": final["saved_at"],
                "baseline_agg": final["baseline"],
                "cycles": cycles,
                "final": final["final_run1"],
                "gate_pass": passed,
            },
            f,
            indent=2,
        )
    with open(os.path.join(OUT, "FINAL_50D_MATCH__latest.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# Fable 50d fast\n\n"
            f"baseline same={baseline['same_outcome']} policy={baseline['policy_clear']}\n\n"
            f"final same=**{r1['same_outcome']}**/50 policy=**{r1['policy_clear']}** "
            f"mwt={r1['mark_would_take']} breach=**{r1['n_breach']}** gate={passed}\n"
        )
    print(
        f"DONE gate_pass={passed} same={r1['same_outcome']}/50 "
        f"policy={r1['policy_clear']} mark={r1['mark_clear']} breach={r1['n_breach']}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
