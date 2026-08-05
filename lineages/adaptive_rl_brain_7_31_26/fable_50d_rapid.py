"""Rapid 50d match loop: prebuild MWT Mark plans once, then train/rescore fast.

Uses BASELINE_50D Mark awards. Policy score sequential (reliable on Windows).
Oracle only for current MWT days (lazy, cached to disk).
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
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import gate_pass, load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import execute_mark_soul_day
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
ORACLE = os.path.join(OUT, "MARK_ORACLE_CACHE__50d.json")
MAX_ES = 12  # faster soul search


def score_policy(policy: Channel1Policy, day_map: Dict[str, Any], mark_rows: List[dict]) -> dict:
    rows = []
    for i, mr in enumerate(mark_rows):
        date = str(mr["date"])
        t, r = float(mr["target_pct"]), float(mr["risk_pct"])
        if i % 10 == 0:
            print(f"    score {i+1}/{len(mark_rows)} {date}", flush=True)
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
        res = day.run(greedy_policy=policy, pure_greedy=True, use_heuristic=False)
        pol_award = bool(res.cleared and not res.breached)
        mark_award = bool(mr["mark_award"])
        if mark_award and not pol_award and not res.breached:
            mclass = "MARK_WOULD_TAKE"
        elif pol_award:
            mclass = "AWARD"
        elif res.breached:
            mclass = "POLICY_BREACH"
        else:
            mclass = "NO_OPPORTUNITY"
        rows.append(
            {
                "date": date,
                "target_pct": t,
                "risk_pct": r,
                "mark_award": mark_award,
                "mark_n_entries": mr.get("mark_n_entries"),
                "policy_award": pol_award,
                "policy_breached": bool(res.breached),
                "policy_pnl": round(float(res.pnl_pct), 4),
                "policy_n_entries": int(res.n_entries),
                "same_outcome": bool(mark_award == pol_award),
                "miss_class": mclass,
            }
        )
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r["miss_class"]] = counts.get(r["miss_class"], 0) + 1
    return {
        "n_days": len(rows),
        "mark_clear": sum(1 for r in rows if r["mark_award"]),
        "policy_clear": sum(1 for r in rows if r["policy_award"]),
        "same_outcome": sum(1 for r in rows if r["same_outcome"]),
        "n_breach": sum(1 for r in rows if r["policy_breached"]),
        "miss_class_counts": counts,
        "mark_would_take": counts.get("MARK_WOULD_TAKE", 0),
        "no_opportunity": counts.get("NO_OPPORTUNITY", 0),
        "rows": rows,
    }


def load_oracle() -> Dict[str, dict]:
    if not os.path.isfile(ORACLE):
        return {}
    raw = json.load(open(ORACLE, encoding="utf-8"))
    out = {}
    for k, v in raw.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {int(a): int(b) for a, b in vv["plan"].items()}
        out[k] = vv
    return out


def save_oracle(cache: Dict[str, dict]) -> None:
    serial = {}
    for k, v in cache.items():
        vv = dict(v)
        if vv.get("plan"):
            vv["plan"] = {str(a): int(b) for a, b in vv["plan"].items()}
        # drop huge day refs
        serial[k] = {kk: vv[kk] for kk in vv if kk != "day"}
    with open(ORACLE, "w", encoding="utf-8") as f:
        json.dump(serial, f)


def get_plan(oracle, day_map, date, t, r) -> dict:
    key = f"{date}|{t}|{r}"
    if key in oracle and oracle[key].get("plan") is not None:
        return oracle[key]
    print(f"    oracle {date}…", flush=True)
    m = execute_mark_soul_day(day_map[date], date, t, r, max_entry_samples=MAX_ES)
    blob = {k: v for k, v in m.items() if k != "day"}
    if blob.get("plan"):
        blob["plan"] = {int(a): int(b) for a, b in blob["plan"].items()}
    oracle[key] = blob
    save_oracle(oracle)
    return blob


def plan_labels(day_map, date, t, r, mark, *, dir_copy=10, hold_copy=2) -> Tuple[list, list, list]:
    plan = mark.get("plan")
    if not plan or mark.get("source") != "soul_plan":
        return [], [], []
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
        obs = day.observe(tb)
        act = int(plan.get(int(tb), ACTION_HOLD))
        # Dirs: dense. HOLD: sparse subsample so HOLD mass does not drown craft.
        o = np.asarray(obs, np.float32).reshape(-1)
        if act != ACTION_HOLD:
            n, w = max(8, dir_copy), 14.0
            for _ in range(n):
                xs.append(o.copy())
                ys.append(act)
                ws.append(w)
        else:
            # keep ~1/4 of HOLD bars, modest weight
            if (int(tb) // 25) % 4 == 0:
                for _ in range(max(2, hold_copy // 2)):
                    xs.append(o.copy())
                    ys.append(act)
                    ws.append(3.5)
        day.step_action(tb, act)
    return xs, ys, ws


def dagger_labels(day_map, date, t, r, mark, policy) -> Tuple[list, list, list]:
    plan = mark.get("plan") or {}
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
        obs = day.observe(tb)
        with torch.no_grad():
            pa, _ = policy.act(obs, greedy=True)
            pa = int(pa)
        ma = int(plan.get(int(tb), ACTION_HOLD)) if plan else int(day.recommended_action(tb))
        # Strongly correct early thrash: policy fires, Mark HOLDs
        if pa != ma:
            if ma == ACTION_HOLD and pa != ACTION_HOLD:
                n, w = 4, 8.0  # anti-thrash
            elif ma != ACTION_HOLD:
                n, w = 4, 10.0  # miss Mark entry
            else:
                n, w = 2, 3.0
            o = np.asarray(obs, np.float32).reshape(-1)
            for _ in range(n):
                xs.append(o)
                ys.append(ma)
                ws.append(w)
        day.step_action(tb, pa)
    return xs, ys, ws


def award_self(day_map, date, t, r, policy) -> Tuple[list, list, list]:
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
    xs, ys, ws = [], [], []
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
            ws.append(2.5 if a != ACTION_HOLD else 1.0)
        day.step_action(tb, a)
        i += 1
    return xs, ys, ws


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    dials = clip_streak_dials(default_streak_dials())
    policy = load_policy(CKPT)
    oracle = load_oracle()
    print(f"oracle cache size={len(oracle)}", flush=True)

    print("Score…", flush=True)
    cur = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={cur['same_outcome']} policy={cur['policy_clear']} "
        f"mwt={cur['mark_would_take']} breach={cur['n_breach']}",
        flush=True,
    )
    best = cur
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    cycles = [{"cycle": 0, "score": {k: v for k, v in cur.items() if k != "rows"}}]

    for cyc in range(1, 20):
        if gate_pass(best) and best["policy_clear"] >= floor_clear:
            print("*** GATE HIT ***", flush=True)
            break
        print(f"\n===== RAPID {cyc}/19 =====", flush=True)
        policy.load_state_dict(best_state)
        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        print(f"  mwt={len(mwt)} awards={len(awards)}", flush=True)

        xs, ys, ws = [], [], []
        # Prebuild any missing MWT oracles first
        for row in mwt:
            get_plan(oracle, day_map, row["date"], row["target_pct"], row["risk_pct"])

        # Target ~1:1 to 2:1 dir:hold by sparse HOLD + dense dirs
        dir_copy = 8 + min(cyc, 6)
        hold_copy = 2
        for row in mwt:
            mark = get_plan(oracle, day_map, row["date"], row["target_pct"], row["risk_pct"])
            for _rep in range(2 + cyc // 3):
                a, b, c = plan_labels(
                    day_map,
                    row["date"],
                    row["target_pct"],
                    row["risk_pct"],
                    mark,
                    dir_copy=dir_copy,
                    hold_copy=hold_copy,
                )
                xs.extend(a)
                ys.extend(b)
                ws.extend(c)
            a, b, c = dagger_labels(
                day_map,
                row["date"],
                row["target_pct"],
                row["risk_pct"],
                mark,
                policy,
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)

        # protect awards lightly (prefer dirs from self-imitate)
        for row in awards[:12]:
            a, b, c = award_self(
                day_map, row["date"], row["target_pct"], row["risk_pct"], policy
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)

        if len(ys) < 20:
            print("  no labels", flush=True)
            break
        X = np.stack(xs)
        y = np.asarray(ys, np.int64)
        w = np.asarray(ws, np.float32)
        n_dir = int((y != 0).sum())
        n_hold = int((y == 0).sum())
        print(f"  train n={len(y)} dir={n_dir} hold={n_hold} ratio={n_dir/max(n_hold,1):.2f}", flush=True)

        # Mid KL: enough anchor to avoid thrash, enough room to learn entries
        kl = 0.35 if cyc % 2 == 1 else 0.48
        lr = 3.5e-4 if cyc % 2 == 1 else 2.5e-4
        pol2, losses = train_bc(
            X,
            y,
            epochs=32 + cyc,
            hidden=128,
            seed=200 + cyc,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=lr,
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=kl,
        )
        m = match_rate(pol2, X, y)
        print(f"  match={m} kl={kl}", flush=True)

        post = score_policy(pol2, day_map, mark_rows)
        print(
            f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )
        # If breach but better same_outcome, one HOLD-repair pass then re-score
        if post["n_breach"] > 0 and post["same_outcome"] >= best["same_outcome"]:
            print("  HOLD-repair after breach…", flush=True)
            # freeze dirs, boost HOLD labels from Mark plans on thrash days
            hx, hy, hw = [], [], []
            for row in post["rows"]:
                if not row["policy_breached"] and row["policy_n_entries"] < 4:
                    continue
                mark = get_plan(
                    oracle, day_map, row["date"], row["target_pct"], row["risk_pct"]
                )
                a, b, c = plan_labels(
                    day_map,
                    row["date"],
                    row["target_pct"],
                    row["risk_pct"],
                    mark,
                    dir_copy=2,
                    hold_copy=6,
                )
                # invert: only keep HOLD samples for repair
                for o, act, wt in zip(a, b, c):
                    if act == ACTION_HOLD:
                        hx.append(o)
                        hy.append(act)
                        hw.append(8.0)
            if len(hy) >= 20:
                Xh = np.stack(hx)
                yh = np.asarray(hy, np.int64)
                wh = np.asarray(hw, np.float32)
                pol2, _ = train_bc(
                    Xh,
                    yh,
                    epochs=12,
                    hidden=128,
                    seed=300 + cyc,
                    warm_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
                    obs_dim=MARK_FULL_DIM,
                    lr=2e-4,
                    sample_weights=wh,
                    kl_anchor_state=best_state,
                    kl_coef=0.6,
                )
                post = score_policy(pol2, day_map, mark_rows)
                print(
                    f"  REPAIR same={post['same_outcome']} policy={post['policy_clear']} "
                    f"breach={post['n_breach']}",
                    flush=True,
                )
        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor_clear
            and (
                post["same_outcome"] > best["same_outcome"]
                or (
                    post["same_outcome"] == best["same_outcome"]
                    and post["policy_clear"] > best["policy_clear"]
                )
                or (
                    post["same_outcome"] == best["same_outcome"]
                    and post["mark_would_take"] < best["mark_would_take"]
                    and post["policy_clear"] >= best["policy_clear"]
                )
            )
        )
        cycles.append(
            {
                "cycle": cyc,
                "pre_same": best["same_outcome"],
                "post_same": post["same_outcome"],
                "post_policy": post["policy_clear"],
                "post_mwt": post["mark_would_take"],
                "post_breach": post["n_breach"],
                "keep": keep,
                "kl": kl,
                "dir_copy": dir_copy,
            }
        )
        if keep:
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            save_policy(pol2, note=f"rapid_{cyc}", dials=dials)
            print("  KEEP", flush=True)
        else:
            print("  REJECT", flush=True)

    policy.load_state_dict(best_state)
    save_policy(policy, note="rapid_best", dials=dials)
    r1 = score_policy(policy, day_map, mark_rows)
    r2 = score_policy(policy, day_map, mark_rows)
    assert r1["same_outcome"] == r2["same_outcome"] and r1["n_breach"] == r2["n_breach"]
    passed = gate_pass(r1) and r1["policy_clear"] >= floor_clear
    final = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "fable_50d_rapid",
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
                "cycles": cycles,
                "final": final["final_run1"],
                "gate_pass": passed,
            },
            f,
            indent=2,
        )
    with open(os.path.join(OUT, "FINAL_50D_MATCH__latest.md"), "w", encoding="utf-8") as f:
        f.write(
            f"# Fable 50d rapid\n\n"
            f"baseline same={baseline['same_outcome']} policy={baseline['policy_clear']}\n\n"
            f"final same=**{r1['same_outcome']}**/50 policy=**{r1['policy_clear']}** "
            f"mwt={r1['mark_would_take']} breach=**{r1['n_breach']}** gate={passed}\n\n"
            f"cycles={len(cycles)}\n"
        )
    print(
        f"DONE gate_pass={passed} same={r1['same_outcome']}/50 "
        f"policy={r1['policy_clear']} breach={r1['n_breach']}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
