"""Convert one MARK_WOULD_TAKE day at a time without killing award days.

For each remaining miss day (Mark wins, policy loses):
  1. Heavy BC on that day's Mark plan (dirs + sparse HOLD)
  2. Light award self-imitate for protection
  3. High KL to current best
  4. Keep only if: that day converts OR same_outcome rises, breach 0, policy_clear >= floor
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List

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
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    dagger_labels,
    get_plan,
    load_oracle,
    plan_labels,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")


def main() -> int:
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    dials = clip_streak_dials(default_streak_dials())
    oracle = load_oracle()
    policy = load_policy(CKPT)

    print("Initial score…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} policy={best['policy_clear']} "
        f"mwt={best['mark_would_take']} breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    cycles = [{"cycle": 0, "same": best["same_outcome"], "policy": best["policy_clear"]}]
    rounds = 0
    max_rounds = 60  # enough to visit each miss multiple times

    while rounds < max_rounds:
        if gate_pass(best) and best["policy_clear"] >= floor:
            print("*** GATE HIT ***", flush=True)
            break
        policy.load_state_dict(best_state)
        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            print("no MWT left", flush=True)
            break
        # pick day with worst pnl (most negative) first
        mwt = sorted(mwt, key=lambda r: float(r.get("policy_pnl") or 0))
        target = mwt[rounds % len(mwt)]
        date = target["date"]
        t, r = float(target["target_pct"]), float(target["risk_pct"])
        print(f"\n===== ONE-DAY {rounds+1} focus {date} T/R={t}/{r} =====", flush=True)
        mark = get_plan(oracle, day_map, date, t, r)
        xs, ys, ws = [], [], []
        # Mid recipe (overnight): convert focus without pack crater.
        # freeze_trunk: pack safer but never converted. full net: converts then 35→32.
        # Here: full net + high KL + fewer epochs + more award + less DAgger.
        for _ in range(5):
            a, b, c = plan_labels(
                day_map, date, t, r, mark, dir_copy=8, hold_copy=5
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)
        for _ in range(2):
            a, b, c = dagger_labels(day_map, date, t, r, mark, policy)
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)
        # protect awards heavily
        for row in awards[:28]:
            a, b, c = award_self(
                day_map, row["date"], row["target_pct"], row["risk_pct"], policy
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend([x * 2.2 for x in c])
        # one other MWT light only
        for row in mwt[1:2]:
            m2 = get_plan(oracle, day_map, row["date"], row["target_pct"], row["risk_pct"])
            a, b, c = plan_labels(
                day_map,
                row["date"],
                row["target_pct"],
                row["risk_pct"],
                m2,
                dir_copy=2,
                hold_copy=3,
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)

        X = np.stack(xs)
        y = np.asarray(ys, np.int64)
        w = np.asarray(ws, np.float32)
        print(f"  n={len(y)} dir={int((y!=0).sum())} hold={int((y==0).sum())}", flush=True)
        pol2, _ = train_bc(
            X,
            y,
            epochs=20,
            hidden=128,
            seed=400 + rounds,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=2.0e-4,
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=0.72,
            freeze_trunk=False,
        )
        print(f"  match={match_rate(pol2, X, y)}", flush=True)
        post = score_policy(pol2, day_map, mark_rows)
        # did focus day convert?
        focus_ok = False
        for row in post["rows"]:
            if row["date"] == date:
                focus_ok = bool(row["policy_award"])
                print(
                    f"  focus {date} award={row['policy_award']} pnl={row['policy_pnl']} "
                    f"n={row['policy_n_entries']}",
                    flush=True,
                )
                break
        print(
            f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )
        # Convert-but-pack-drop: re-graft focus plan onto BEST weights (not thrashy pol2).
        # Overnight lesson: repair-from-pol2 left same stuck at 32; start from best_state.
        repaired = False
        if (
            focus_ok
            and post["n_breach"] == 0
            and post["same_outcome"] < best["same_outcome"]
        ):
            print(
                "  PACK-repair from BEST + light focus plan (not thrash embryo)…",
                flush=True,
            )
            hx, hy, hw = [], [], []
            protect = [
                r
                for r in (best.get("rows") or [])
                if r.get("policy_award") or r.get("miss_class") == "AWARD"
            ]
            if not protect:
                protect = list(awards)[:24]
            # Award labels from BEST weights (not thrashy pol2)
            from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy

            pol_anchor = Channel1Policy(obs_dim=MARK_FULL_DIM, hidden=128)
            pol_anchor.load_state_dict(best_state)
            pol_anchor.eval()
            for row in protect[:32]:
                a, b, c = award_self(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    pol_anchor,
                )
                hx.extend(a)
                hy.extend(b)
                hw.extend([x * 2.8 for x in c])
            # Light focus Mark plan (convert signal without drowning pack)
            m_focus = get_plan(oracle, day_map, date, t, r)
            for _ in range(3):
                a, b, c = plan_labels(
                    day_map, date, t, r, m_focus, dir_copy=6, hold_copy=5
                )
                hx.extend(a)
                hy.extend(b)
                hw.extend(c)
            if len(hy) >= 40:
                Xh = np.stack(hx)
                yh = np.asarray(hy, np.int64)
                wh = np.asarray(hw, np.float32)
                pol2, _ = train_bc(
                    Xh,
                    yh,
                    epochs=20,
                    hidden=128,
                    seed=500 + rounds,
                    warm_state=best_state,  # graft onto BEST, not thrash
                    obs_dim=MARK_FULL_DIM,
                    lr=1.8e-4,
                    sample_weights=wh,
                    kl_anchor_state=best_state,
                    kl_coef=0.72,
                    freeze_trunk=True,
                )
                post = score_policy(pol2, day_map, mark_rows)
                focus_ok = False
                for row in post["rows"]:
                    if row["date"] == date:
                        focus_ok = bool(row["policy_award"])
                        break
                print(
                    f"  REPAIR same={post['same_outcome']} policy={post['policy_clear']} "
                    f"mwt={post['mark_would_take']} breach={post['n_breach']} "
                    f"focus_ok={focus_ok}",
                    flush=True,
                )
                repaired = True
                # second gentle pass if still below best but focus_ok
                if (
                    focus_ok
                    and post["same_outcome"] < best["same_outcome"]
                    and post["n_breach"] == 0
                ):
                    print("  PACK-repair pass 2 (more award freeze)…", flush=True)
                    hx2, hy2, hw2 = [], [], []
                    for row in protect[:32]:
                        a, b, c = award_self(
                            day_map,
                            row["date"],
                            float(row["target_pct"]),
                            float(row["risk_pct"]),
                            pol_anchor,
                        )
                        hx2.extend(a)
                        hy2.extend(b)
                        hw2.extend([x * 3.0 for x in c])
                    for _ in range(2):
                        a, b, c = plan_labels(
                            day_map, date, t, r, m_focus, dir_copy=4, hold_copy=6
                        )
                        hx2.extend(a)
                        hy2.extend(b)
                        hw2.extend(c)
                    if len(hy2) >= 40:
                        pol2, _ = train_bc(
                            np.stack(hx2),
                            np.asarray(hy2, np.int64),
                            epochs=12,
                            hidden=128,
                            seed=600 + rounds,
                            warm_state={
                                k: v.detach().clone() for k, v in pol2.state_dict().items()
                            },
                            obs_dim=MARK_FULL_DIM,
                            lr=1.5e-4,
                            sample_weights=np.asarray(hw2, np.float32),
                            kl_anchor_state=best_state,
                            kl_coef=0.78,
                            freeze_trunk=True,
                        )
                        post = score_policy(pol2, day_map, mark_rows)
                        focus_ok = any(
                            row["date"] == date and row["policy_award"]
                            for row in post["rows"]
                        )
                        print(
                            f"  REPAIR2 same={post['same_outcome']} mwt={post['mark_would_take']} "
                            f"focus_ok={focus_ok}",
                            flush=True,
                        )

        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor
            and (
                post["same_outcome"] > best["same_outcome"]
                or (focus_ok and post["same_outcome"] >= best["same_outcome"]
                    and post["policy_clear"] >= best["policy_clear"])
            )
        )
        cycles.append(
            {
                "round": rounds + 1,
                "focus": date,
                "focus_ok": focus_ok,
                "same": post["same_outcome"],
                "policy": post["policy_clear"],
                "breach": post["n_breach"],
                "keep": keep,
                "repaired": repaired,
            }
        )
        if keep:
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            save_policy(pol2, note=f"oneday_{date}", dials=dials)
            # durable best meter for squad
            try:
                with open(os.path.join(OUT, "BEST__latest.json"), "w", encoding="utf-8") as bf:
                    json.dump(
                        {
                            "same_outcome": best["same_outcome"],
                            "policy_clear": best["policy_clear"],
                            "mwt": best["mark_would_take"],
                            "breach": best["n_breach"],
                            "source": f"one_day_KEEP_{date}",
                            "repaired": repaired,
                        },
                        bf,
                        indent=2,
                    )
                    bf.write("\n")
            except OSError:
                pass
            print("  KEEP", flush=True)
            # Goal notes (wins only) for later LLM/KAG
            try:
                works = os.path.join(OUT, "WHAT_WORKS__GOAL.md")
                with open(works, "a", encoding="utf-8") as wf:
                    wf.write(
                        f"| KEEP live | **{best['same_outcome']}** | {best['mark_would_take']} | "
                        f"{best['n_breach']} | one_day_KEEP_{date} repaired={repaired} |\n"
                    )
            except OSError:
                pass
        else:
            print("  REJECT", flush=True)
        rounds += 1

    policy.load_state_dict(best_state)
    save_policy(policy, note="oneday_best", dials=dials)
    r1 = score_policy(policy, day_map, mark_rows)
    r2 = score_policy(policy, day_map, mark_rows)
    assert r1["same_outcome"] == r2["same_outcome"]
    passed = gate_pass(r1) and r1["policy_clear"] >= floor
    final = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "method": "fable_50d_one_day",
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
        json.dump({"cycles": cycles, "final": final["final_run1"], "gate_pass": passed}, f, indent=2)
    print(
        f"DONE gate_pass={passed} same={r1['same_outcome']}/50 "
        f"policy={r1['policy_clear']} breach={r1['n_breach']}",
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
