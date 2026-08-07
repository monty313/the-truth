"""Spine Shadow DAgger climb — TIMING/PATH first (from SPINE_GAP_DIAGNOSIS).

Diagnosis (practice MWT @ same=35):
  - gold plan awards: 15/15
  - policy: 0/15
  - size-lock only +1 convert
  → offline plan-path BC matches labels but ONLINE trajectory diverges.
    Fix = multi-iter DAgger on policy states + Mark/spine labels, then KEEP/REJECT.

Price data: the-truth/data/raw via price_data / load_calendar_days.
No PROVEN. No 3 teachers. No entry-reward crank.
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

from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import (
    compile_spine_from_soul,
    fire_times,
    load_spine,
    load_spine_index,
)
from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    get_plan,
    load_oracle,
    plan_labels,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc
from lineages.adaptive_rl_brain_7_31_26.train_spine_shadow import (
    append_learning_md,
    build_error_card,
)

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
SHADOW = os.path.join(_HERE, "checkpoints", "mark_shadow_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
CYCLE_LOG = os.path.join(OUT, "SPINE_DAGGER_CYCLES__latest.json")
SPINE_INDEX = os.path.join(_HERE, "checkpoints", "spines", "SPINE_INDEX__latest.json")


def dagger_heavy(
    day_map,
    date: str,
    t: float,
    r: float,
    mark: dict,
    policy,
    *,
    fire_boost: int = 14,
    thrash_boost: int = 10,
    near_fire_bars: int = 50,
) -> Tuple[list, list, list]:
    """DAgger on policy path; extreme weight on missing spine fires / thrash."""
    plan = mark.get("plan") or {}
    plan = {int(k): int(v) for k, v in plan.items()}
    fire_ts = sorted(int(k) for k, v in plan.items() if int(v) != ACTION_HOLD)
    fire_set = set(fire_ts)

    day = GoalEquityDay(
        day_map[date],
        target_pct=float(t),
        risk_pct=float(r),
        date_str=date,
        eyes_mode="mark_doctrine",
        mark_soul=True,
        full_obs=True,
        mark_align_policy=True,
    )
    # lock size to Mark soul dials so path error isn't confounded by size
    if mark.get("risk_use_frac") not in (None, "dynamic"):
        day._plan_lock_ruf = float(mark["risk_use_frac"])
        day._plan_lock_cap = float(mark["per_trade_cap_pct"])

    xs, ys, ws = [], [], []
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
        obs = np.asarray(day.observe(tb), np.float32).reshape(-1)
        with torch.no_grad():
            pa, _ = policy.act(obs, greedy=True)
            pa = int(pa)
        ma = int(plan.get(int(tb), ACTION_HOLD))
        near = any(abs(int(tb) - ft) <= near_fire_bars for ft in fire_ts) if fire_ts else False

        if pa != ma:
            if ma != ACTION_HOLD:
                # missed spine fire/add — primary gap
                n, w = fire_boost, 22.0
            elif pa != ACTION_HOLD:
                # thrash fire while spine HOLD
                n, w = thrash_boost, 16.0
            else:
                n, w = 2, 3.0
            if near and ma != ACTION_HOLD:
                n = int(n * 1.5)
                w = w * 1.25
            for _ in range(n):
                xs.append(obs.copy())
                ys.append(ma)
                ws.append(w)
        elif ma != ACTION_HOLD:
            # agree on fire — still reinforce (light)
            for _ in range(3):
                xs.append(obs.copy())
                ys.append(ma)
                ws.append(6.0)
        elif near and ma == ACTION_HOLD:
            # HOLD near fire window — wait_loaded skill
            for _ in range(2):
                xs.append(obs.copy())
                ys.append(ACTION_HOLD)
                ws.append(5.0)

        day.step_action(tb, pa)
    return xs, ys, ws


def load_spines() -> Dict[str, Any]:
    if not os.path.isfile(SPINE_INDEX):
        return {}
    idx = load_spine_index(SPINE_INDEX)
    out = {}
    for it in idx.get("items") or []:
        sp = load_spine(it["path"])
        out[f"{it['day']}|{float(it['target_pct'])}|{float(it['risk_pct'])}"] = sp
    return out


def run(max_rounds: int = 20, dagger_iters: int = 3, keep_floor: int = 33) -> dict:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    # Price from the-truth/data/raw via load_calendar_days → price_data
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    spines = load_spines()
    oracle = load_oracle()
    policy = load_policy(CKPT)
    dials = clip_streak_dials(default_streak_dials())

    print("DAgger climb — initial score (data/raw curriculum)…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    card0 = build_error_card(
        best, spines, cycle=0, decision="BASELINE", change="dagger_timing_path"
    )
    append_learning_md(card0)
    cycles = [
        {
            "round": 0,
            "same": best["same_outcome"],
            "mwt": best["mark_would_take"],
            "breach": best["n_breach"],
            "decision": "BASELINE",
        }
    ]
    fail: Dict[str, int] = {}

    for rnd in range(1, max_rounds + 1):
        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            print("*** 50/50 practice ***", flush=True)
            break
        policy.load_state_dict(best_state)
        mwt = sorted(
            [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"],
            key=lambda r: (fail.get(r["date"], 0), float(r.get("policy_pnl") or 0)),
        )
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            break
        # rotate focus among least-failed
        focus = mwt[rnd % len(mwt)] if fail.get(mwt[0]["date"], 0) >= 3 else mwt[0]
        # Focus-only DAgger (support MWT caused pack 35→27 collapse). One light support max.
        targets = [focus]
        if fail.get(focus["date"], 0) >= 2:
            # stuck on focus: add one support MWT only
            extra = [r for r in mwt if r["date"] != focus["date"]][:1]
            targets.extend(extra)
        print(
            f"\n===== DAGGER {rnd}/{max_rounds} focus={focus['date']} "
            f"targets={len(targets)} fails={fail.get(focus['date'],0)} =====",
            flush=True,
        )

        pol = policy
        for it in range(dagger_iters):
            xs, ys, ws = [], [], []
            for row in targets:
                date = row["date"]
                t, r = float(row["target_pct"]), float(row["risk_pct"])
                mark = get_plan(oracle, day_map, date, t, r)
                n_roll = 3 if date == focus["date"] else 1
                for _ in range(n_roll):
                    a, b, c = dagger_heavy(
                        day_map,
                        date,
                        t,
                        r,
                        mark,
                        pol,
                        fire_boost=10 if it == 0 else 8,
                        thrash_boost=12,  # protect pack from thrash
                    )
                    xs.extend(a)
                    ys.extend(b)
                    ws.extend(c)
                if date == focus["date"]:
                    a, b, c = plan_labels(
                        day_map, date, t, r, mark, dir_copy=6, hold_copy=4
                    )
                    xs.extend(a)
                    ys.extend(b)
                    ws.extend(c)
            # heavy award protect — lesson: pack collapse is worse than slow climb
            for row in awards[:28]:
                a, b, c = award_self(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    pol,
                )
                xs.extend(a)
                ys.extend(b)
                ws.extend([x * 2.4 for x in c])

            if len(ys) < 30:
                print(f"  iter{it+1}: few labels", flush=True)
                break
            X = np.stack(xs)
            y = np.asarray(ys, np.int64)
            w = np.asarray(ws, np.float32)
            n_dir = int((y != 0).sum())
            n_hold = int((y == 0).sum())
            # high KL to freeze pack; lower only if focus failed twice
            kl = 0.55 if fail.get(focus["date"], 0) < 2 else 0.40
            if it > 0:
                kl = min(0.65, kl + 0.05)
            print(
                f"  dagger-iter {it+1}/{dagger_iters} n={len(y)} "
                f"dir={n_dir} hold={n_hold} ratio={n_dir/max(n_hold,1):.2f} kl={kl}",
                flush=True,
            )
            pol, _ = train_bc(
                X,
                y,
                epochs=14 + it * 3,
                hidden=128,
                seed=1200 + rnd * 10 + it,
                warm_state={k: v.detach().clone() for k, v in pol.state_dict().items()},
                obs_dim=MARK_FULL_DIM,
                lr=2.2e-4 if it == 0 else 1.8e-4,
                sample_weights=w,
                kl_anchor_state=best_state,
                kl_coef=kl,
            )
            print(f"  match={match_rate(pol, X, y)}", flush=True)

        post = score_policy(pol, day_map, mark_rows)
        focus_ok = False
        for row in post["rows"]:
            if row["date"] == focus["date"]:
                focus_ok = bool(row["policy_award"])
                print(
                    f"  focus {focus['date']} award={focus_ok} pnl={row['policy_pnl']} "
                    f"n={row['policy_n_entries']}",
                    flush=True,
                )
                break
        print(
            f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )

        # pack-repair if focus ok but pack dropped
        if focus_ok and post["n_breach"] == 0 and post["same_outcome"] < best["same_outcome"]:
            print("  PACK-repair…", flush=True)
            hx, hy, hw = [], [], []
            for row in [r for r in best["rows"] if r.get("policy_award")][:28]:
                a, b, c = award_self(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    pol,
                )
                hx.extend(a)
                hy.extend(b)
                hw.extend([x * 2.3 for x in c])
            if len(hy) >= 40:
                pol, _ = train_bc(
                    np.stack(hx),
                    np.asarray(hy, np.int64),
                    epochs=14,
                    hidden=128,
                    seed=1300 + rnd,
                    warm_state={k: v.detach().clone() for k, v in pol.state_dict().items()},
                    obs_dim=MARK_FULL_DIM,
                    lr=1.6e-4,
                    sample_weights=np.asarray(hw, np.float32),
                    kl_anchor_state=best_state,
                    kl_coef=0.75,
                )
                post = score_policy(pol, day_map, mark_rows)
                focus_ok = any(
                    r["date"] == focus["date"] and r["policy_award"] for r in post["rows"]
                )
                print(
                    f"  REPAIR same={post['same_outcome']} mwt={post['mark_would_take']} "
                    f"focus_ok={focus_ok}",
                    flush=True,
                )

        if post["n_breach"] > 0:
            print("  breach — REJECT restore", flush=True)
            decision = "REJECT"
            fail[focus["date"]] = fail.get(focus["date"], 0) + 1
        else:
            avg_ent = float(np.mean([r["policy_n_entries"] for r in post["rows"]]))
            keep = (
                post["policy_clear"] >= floor_clear
                and post["same_outcome"] >= live_floor
                and avg_ent <= 6.5
                and (
                    post["same_outcome"] > best["same_outcome"]
                    or (
                        focus_ok
                        and post["same_outcome"] >= best["same_outcome"]
                        and post["policy_clear"] >= best["policy_clear"]
                    )
                )
            )
            decision = "KEEP" if keep else "REJECT"
            if keep:
                best = post
                best_state = {k: v.detach().clone() for k, v in pol.state_dict().items()}
                live_floor = max(live_floor, int(post["same_outcome"]))
                save_policy(pol, note=f"spine_dagger_KEEP_{focus['date']}", dials=dials)
                torch.save(
                    {
                        "tag": "mark_shadow_v1",
                        "method": "spine_dagger_timing",
                        "same_outcome": post["same_outcome"],
                        "state_dict": pol.state_dict(),
                        "obs_dim": MARK_FULL_DIM,
                        "hidden": 128,
                        "proven_touched": False,
                        "saved_at": datetime.now(timezone.utc).isoformat(),
                    },
                    SHADOW,
                )
                with open(BEST, "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "same_outcome": post["same_outcome"],
                            "policy_clear": post["policy_clear"],
                            "mwt": post["mark_would_take"],
                            "breach": post["n_breach"],
                            "source": f"spine_dagger_KEEP_{focus['date']}",
                            "updated_at": datetime.now(timezone.utc).isoformat(),
                        },
                        f,
                        indent=2,
                    )
                fail[focus["date"]] = 0
                print(f"  KEEP best_same={best['same_outcome']}", flush=True)
            else:
                fail[focus["date"]] = fail.get(focus["date"], 0) + 1
                print(f"  REJECT fail[{focus['date']}]={fail[focus['date']]}", flush=True)

        card = build_error_card(
            post,
            spines,
            cycle=rnd,
            decision=decision,
            change=f"dagger_x{dagger_iters}:{focus['date']}",
        )
        append_learning_md(card)
        cycles.append(
            {
                "round": rnd,
                "focus": focus["date"],
                "focus_ok": focus_ok,
                "same": post["same_outcome"],
                "mwt": post["mark_would_take"],
                "breach": post["n_breach"],
                "decision": decision,
                "best_same": best["same_outcome"],
            }
        )
        with open(CYCLE_LOG, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "best_same": best["same_outcome"],
                    "best_mwt": best["mark_would_take"],
                    "implication": "TIMING/PATH — multi-iter DAgger",
                    "cycles": cycles,
                    "fail": fail,
                },
                f,
                indent=2,
            )

    summary = {
        "best_same": best["same_outcome"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "cycles": cycles,
    }
    print(f"DONE best same={summary['best_same']} mwt={summary['best_mwt']}", flush=True)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=16)
    ap.add_argument("--dagger-iters", type=int, default=3)
    ap.add_argument("--keep-floor", type=int, default=33)
    args = ap.parse_args(list(argv) if argv is not None else None)
    run(
        max_rounds=args.max_rounds,
        dagger_iters=args.dagger_iters,
        keep_floor=args.keep_floor,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
