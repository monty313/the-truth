"""Safe spine one-day climb — only levers that already raised same (KEEP path).

Shape that works (from WHAT_WORKS__GOAL.md):
  focus one MWT · Mark plan path heavy · light DAgger · heavy award protect ·
  high KL to best · full pack score · KEEP only if same>=best and breach=0

Spine add-on: weight plan labels near spine fire times (not pack-wide DAgger).
Price: the-truth/data/raw via load_calendar_days.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    dagger_labels,
    get_plan,
    load_oracle,
    plan_labels,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc
from lineages.adaptive_rl_brain_7_31_26.train_spine_shadow import append_learning_md, build_error_card

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
SHADOW = os.path.join(_HERE, "checkpoints", "mark_shadow_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
CYCLE_LOG = os.path.join(OUT, "SPINE_SAFE_ONE_DAY__latest.json")
WORKS = os.path.join(OUT, "WHAT_WORKS__GOAL.md")


def append_works_keep(same: int, mwt: int, focus: str, note: str) -> None:
    line = (
        f"| KEEP live | **{same}** | {mwt} | 0 | spine_safe focus `{focus}` — {note} |\n"
    )
    try:
        with open(WORKS, "a", encoding="utf-8") as f:
            f.write(line)
    except OSError:
        pass


def run(max_rounds: int = 40, keep_floor: int = 33) -> dict:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    policy = load_policy(CKPT)
    dials = clip_streak_dials(default_streak_dials())

    print("SAFE one-day — score…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    fail: Dict[str, int] = {}
    cycles: List[dict] = [
        {
            "round": 0,
            "same": best["same_outcome"],
            "mwt": best["mark_would_take"],
            "breach": best["n_breach"],
            "decision": "BASELINE",
        }
    ]
    card0 = build_error_card(
        best, {}, cycle=0, decision="BASELINE", change="safe_one_day_works_recipe"
    )
    append_learning_md(card0)

    for rnd in range(1, max_rounds + 1):
        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            print("*** PRACTICE 50/50 ***", flush=True)
            break
        policy.load_state_dict(best_state)
        mwt = sorted(
            [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"],
            key=lambda r: (fail.get(r["date"], 0), float(r.get("policy_pnl") or 0)),
        )
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            break
        # rotate after 2 fails on same day
        top = [r for r in mwt if fail.get(r["date"], 0) < 3] or mwt
        focus = top[(rnd - 1) % len(top)]
        date = focus["date"]
        t, r = float(focus["target_pct"]), float(focus["risk_pct"])
        print(
            f"\n===== SAFE {rnd}/{max_rounds} focus={date} fails={fail.get(date,0)} =====",
            flush=True,
        )
        mark = get_plan(oracle, day_map, date, t, r)

        xs, ys, ws = [], [], []
        # Balanced plan path: enough dirs to convert, enough HOLD to protect pack
        # (lesson: pred_hold_rate ~0.27 cratered pack 35→30 even with KL 0.58)
        for _ in range(5):
            a, b, c = plan_labels(
                day_map, date, t, r, mark, dir_copy=6, hold_copy=8
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)
        # One light DAgger pass only
        a, b, c = dagger_labels(day_map, date, t, r, mark, policy)
        xs.extend(a)
        ys.extend(b)
        ws.extend(c)
        # One other MWT light
        for row in mwt[1:2]:
            m2 = get_plan(
                oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
            )
            a, b, c = plan_labels(
                day_map,
                row["date"],
                float(row["target_pct"]),
                float(row["risk_pct"]),
                m2,
                dir_copy=2,
                hold_copy=4,
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)
        # Heavy award protect (pack freeze)
        for row in awards[:30]:
            a, b, c = award_self(
                day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend([x * 2.5 for x in c])

        if len(ys) < 40:
            fail[date] = fail.get(date, 0) + 1
            continue
        X = np.stack(xs)
        y = np.asarray(ys, np.int64)
        w = np.asarray(ws, np.float32)
        n_dir = int((y != 0).sum())
        n_hold = int((y == 0).sum())
        # if still dir-heavy, upsample HOLD by weight
        if n_hold > 0 and n_dir / max(n_hold, 1) > 1.1:
            for i in range(len(y)):
                if int(y[i]) == 0:
                    w[i] *= 1.6
        kl = 0.62 if fail.get(date, 0) < 2 else 0.52
        print(
            f"  n={len(y)} dir={n_dir} hold={n_hold} ratio={n_dir/max(n_hold,1):.2f} kl={kl}",
            flush=True,
        )
        pol2, _ = train_bc(
            X,
            y,
            epochs=28,
            hidden=128,
            seed=900 + rnd,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=2.2e-4,
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=kl,
        )
        print(f"  match={match_rate(pol2, X, y)}", flush=True)
        post = score_policy(pol2, day_map, mark_rows)

        focus_ok = False
        for row in post["rows"]:
            if row["date"] == date:
                focus_ok = bool(row["policy_award"])
                print(
                    f"  focus {date} award={focus_ok} pnl={row['policy_pnl']} "
                    f"n={row['policy_n_entries']}",
                    flush=True,
                )
                break
        print(
            f"  POST same={post['same_outcome']} mwt={post['mark_would_take']} "
            f"breach={post['n_breach']}",
            flush=True,
        )

        # Early abort if pack cratered (>3 days lost) — don't waste pack-repair forever
        if post["same_outcome"] < best["same_outcome"] - 3:
            print("  pack crater — REJECT restore (no repair)", flush=True)
            fail[date] = fail.get(date, 0) + 1
            decision = "REJECT"
            cycles.append(
                {
                    "round": rnd,
                    "focus": date,
                    "focus_ok": focus_ok,
                    "same": post["same_outcome"],
                    "mwt": post["mark_would_take"],
                    "breach": post["n_breach"],
                    "decision": decision,
                    "best_same": best["same_outcome"],
                    "note": "crater",
                }
            )
            card = build_error_card(
                post, {}, cycle=rnd, decision=decision, change=f"safe_crater:{date}"
            )
            append_learning_md(card)
            with open(CYCLE_LOG, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                        "best_same": best["same_outcome"],
                        "best_mwt": best["mark_would_take"],
                        "cycles": cycles,
                        "fail": fail,
                    },
                    f,
                    indent=2,
                )
            continue

        if (
            focus_ok
            and post["n_breach"] == 0
            and post["same_outcome"] < best["same_outcome"]
        ):
            print("  PACK-repair…", flush=True)
            hx, hy, hw = [], [], []
            for row in [r for r in best["rows"] if r.get("policy_award")][:28]:
                a, b, c = award_self(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    pol2,
                )
                hx.extend(a)
                hy.extend(b)
                hw.extend([x * 2.2 for x in c])
            if len(hy) >= 40:
                pol2, _ = train_bc(
                    np.stack(hx),
                    np.asarray(hy, np.int64),
                    epochs=16,
                    hidden=128,
                    seed=950 + rnd,
                    warm_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
                    obs_dim=MARK_FULL_DIM,
                    lr=1.8e-4,
                    sample_weights=np.asarray(hw, np.float32),
                    kl_anchor_state=best_state,
                    kl_coef=0.72,
                )
                post = score_policy(pol2, day_map, mark_rows)
                focus_ok = any(
                    r["date"] == date and r["policy_award"] for r in post["rows"]
                )
                print(
                    f"  REPAIR same={post['same_outcome']} mwt={post['mark_would_take']} "
                    f"focus_ok={focus_ok}",
                    flush=True,
                )

        avg_ent = float(np.mean([r["policy_n_entries"] for r in post["rows"]]))
        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor_clear
            and post["same_outcome"] >= live_floor
            and avg_ent <= 6.0
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
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(pol2, note=f"spine_safe_KEEP_{date}", dials=dials)
            torch.save(
                {
                    "tag": "mark_shadow_v1",
                    "method": "spine_safe_one_day",
                    "same_outcome": post["same_outcome"],
                    "state_dict": pol2.state_dict(),
                    "obs_dim": MARK_FULL_DIM,
                    "hidden": 128,
                    "proven_touched": False,
                    "focus": date,
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
                        "source": f"spine_safe_KEEP_{date}",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            fail[date] = 0
            append_works_keep(
                post["same_outcome"],
                post["mark_would_take"],
                date,
                "plan_path+light_dagger+award_protect+high_KL",
            )
            print(f"  KEEP best_same={best['same_outcome']}", flush=True)
        else:
            fail[date] = fail.get(date, 0) + 1
            print(f"  REJECT fail[{date}]={fail[date]}", flush=True)

        card = build_error_card(
            post, {}, cycle=rnd, decision=decision, change=f"safe:{date}"
        )
        append_learning_md(card)
        cycles.append(
            {
                "round": rnd,
                "focus": date,
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
                    "best_breach": best["n_breach"],
                    "method": "spine_safe_one_day",
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
    print(
        f"DONE best same={summary['best_same']} mwt={summary['best_mwt']} "
        f"breach={summary['best_breach']}",
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=36)
    ap.add_argument("--keep-floor", type=int, default=33)
    args = ap.parse_args(list(argv) if argv is not None else None)
    run(max_rounds=args.max_rounds, keep_floor=args.keep_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
