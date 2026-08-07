"""Spine Shadow one-day surgical loop (intelligent climb).

Why this shape (not pack-wide BC every cycle):
  - 50d score is the bottleneck (~5–8 min). One focus day + light award
    protect converts MWT without drowning the pack (proven: one_day KEEP → 33→35).
  - Error class is PER DAY, not pack-mode. false_fire needs wait_loaded;
    wrong_size needs t1/t2 plan path — pack-dominant class mis-prescribes fix.
  - Oracle spines are green (50/50). Net only shadows spine events under force-gate.

Laws: no PROVEN write, no 3 teachers, no entry-reward crank, KEEP/REJECT conscience.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
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
    DaySpine,
    classify_spine_error,
    compile_spine_from_soul,
    load_spine,
    load_spine_index,
)
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    dagger_labels,
    get_plan,
    load_oracle,
    plan_labels,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc
from lineages.adaptive_rl_brain_7_31_26.train_spine_shadow import (
    append_learning_md,
    build_error_card,
    spine_event_labels,
    surgical_fix_for,
)

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
SHADOW_CKPT = os.path.join(_HERE, "checkpoints", "mark_shadow_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
CYCLE_LOG = os.path.join(OUT, "SPINE_ONE_DAY_CYCLES__latest.json")
SPINE_INDEX = os.path.join(_HERE, "checkpoints", "spines", "SPINE_INDEX__latest.json")

KEEP_FLOOR = 33


def load_spine_map() -> Dict[str, DaySpine]:
    if not os.path.isfile(SPINE_INDEX):
        return {}
    idx = load_spine_index(SPINE_INDEX)
    out: Dict[str, DaySpine] = {}
    for it in idx.get("items") or []:
        key = f"{it['day']}|{float(it['target_pct'])}|{float(it['risk_pct'])}"
        # also plain float formats used in rows
        out[key] = load_spine(it["path"])
        out[f"{it['day']}|{it['target_pct']}|{it['risk_pct']}"] = out[key]
    return out


def day_error_class(row: dict, spine: Optional[DaySpine]) -> str:
    if spine is None:
        # crude from n_entries alone
        n = int(row.get("policy_n_entries") or 0)
        if n == 0:
            return "false_hold"
        if n >= 4:
            return "false_fire"
        return "wrong_size_or_timing"
    return classify_spine_error(
        spine=spine,
        policy_fire_ts=[],
        policy_n_entries=int(row.get("policy_n_entries") or 0),
        policy_award=bool(row.get("policy_award")),
        policy_breached=bool(row.get("policy_breached")),
    )


def pick_focus(
    mwt: List[dict],
    *,
    round_i: int,
    spine_map: Dict[str, DaySpine],
    fail_counts: Dict[str, int],
) -> Tuple[dict, str]:
    """Pick focus day: worst pnl among least-failed class-aware candidates.

    Avoid hammering the same day after 3 fails; rotate to next-worst.
    Prefer wrong_size over false_fire when both present (size/timing was
    autopsy dominant historically; false_fire thrash is higher risk of breach).
    """
    scored = []
    for r in mwt:
        key = f"{r['date']}|{r['target_pct']}|{r['risk_pct']}"
        spine = spine_map.get(key) or spine_map.get(
            f"{r['date']}|{float(r['target_pct'])}|{float(r['risk_pct'])}"
        )
        cls = day_error_class(r, spine)
        fails = int(fail_counts.get(r["date"], 0))
        # priority: fewer fails, prefer wrong_size, worse pnl
        class_pref = 0 if cls == "wrong_size_or_timing" else (1 if cls == "false_hold" else 2)
        scored.append((fails, class_pref, float(r.get("policy_pnl") or 0), r, cls))
    scored.sort(key=lambda x: (x[0], x[1], x[2]))
    # rotate among top-3 least-failed if stuck
    top = scored[: max(3, min(5, len(scored)))]
    choice = top[round_i % len(top)]
    return choice[3], choice[4]


def collect_focus_batch(
    day_map,
    oracle,
    spine_map,
    focus: dict,
    focus_cls: str,
    mwt: List[dict],
    awards: List[dict],
    policy,
    fix: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    date = focus["date"]
    t, r = float(focus["target_pct"]), float(focus["risk_pct"])
    mark = get_plan(oracle, day_map, date, t, r)
    key = f"{date}|{t}|{r}"
    spine = spine_map.get(key) or spine_map.get(f"{date}|{focus['target_pct']}|{focus['risk_pct']}")
    if spine is None:
        spine = compile_spine_from_soul(date, t, r, mark)
        spine_map[key] = spine

    xs, ys, ws = [], [], []

    # Class-specific spine labels on focus (heavy)
    for _ in range(8):
        a, b, c = spine_event_labels(
            day_map,
            spine,
            mark,
            dir_copy=int(fix["dir_copy"]),
            wait_copy=int(fix["wait_copy"]),
            hold_copy=int(fix["hold_copy"]),
        )
        xs.extend(a)
        ys.extend(b)
        ws.extend(c)

    # Plan path denser for wrong_size / false_hold; lighter for false_fire
    if focus_cls in ("wrong_size_or_timing", "false_hold", "late_entry"):
        plan_reps, dir_c, hold_c = 6, max(14, int(fix["dir_copy"])), 3
    elif focus_cls == "false_fire":
        plan_reps, dir_c, hold_c = 3, 4, 10
    else:
        plan_reps, dir_c, hold_c = 4, 10, 4

    for _ in range(plan_reps):
        a, b, c = plan_labels(
            day_map, date, t, r, mark, dir_copy=dir_c, hold_copy=hold_c
        )
        xs.extend(a)
        ys.extend(b)
        ws.extend(c)

    if fix.get("dagger", True):
        for _ in range(5):
            a, b, c = dagger_labels(day_map, date, t, r, mark, policy)
            xs.extend(a)
            ys.extend(b)
            ws.extend(c)

    # Light other MWT (top 2 by pnl besides focus)
    others = [x for x in mwt if x["date"] != date][:2]
    for row in others:
        m2 = get_plan(
            oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
        )
        a, b, c = plan_labels(
            day_map,
            row["date"],
            float(row["target_pct"]),
            float(row["risk_pct"]),
            m2,
            dir_copy=4,
            hold_copy=2,
        )
        xs.extend(a)
        ys.extend(b)
        ws.extend(c)

    # Award protect (heavier — pack slip is the #1 KEEP killer)
    for row in awards[:24]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        xs.extend(a)
        ys.extend(b)
        ws.extend([x * 1.8 for x in c])

    if not ys:
        return (
            np.zeros((0, MARK_FULL_DIM), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
        )
    return np.stack(xs), np.asarray(ys, np.int64), np.asarray(ws, np.float32)


def pack_repair(pol2, best, day_map, oracle, mwt, focus):
    hx, hy, hw = [], [], []
    protect = [r for r in best["rows"] if r.get("policy_award")][:28]
    for row in protect:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), pol2
        )
        hx.extend(a)
        hy.extend(b)
        hw.extend([x * 2.2 for x in c])
    for row in [focus] + mwt[:3]:
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
            hold_copy=8,
        )
        for o, act, _ in zip(a, b, c):
            if int(act) == ACTION_HOLD:
                hx.append(o)
                hy.append(ACTION_HOLD)
                hw.append(7.0)
            else:
                hx.append(o)
                hy.append(int(act))
                hw.append(2.5)
    if len(hy) < 40:
        return pol2
    pol2, _ = train_bc(
        np.stack(hx),
        np.asarray(hy, np.int64),
        epochs=16,
        hidden=128,
        seed=911,
        warm_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
        obs_dim=MARK_FULL_DIM,
        lr=1.8e-4,
        sample_weights=np.asarray(hw, np.float32),
        kl_anchor_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
        kl_coef=0.72,
    )
    return pol2


def run(max_rounds: int = 40, keep_floor: int = KEEP_FLOOR) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    spine_map = load_spine_map()
    oracle = load_oracle()
    policy = load_policy(CKPT)
    dials = clip_streak_dials(default_streak_dials())

    print("Initial score…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} policy={best['policy_clear']} "
        f"mwt={best['mark_would_take']} breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))

    card0 = build_error_card(
        best, spine_map, cycle=0, decision="BASELINE", change="spine_one_day_start"
    )
    append_learning_md(card0)

    fail_counts: Dict[str, int] = {}
    cycles: List[Dict[str, Any]] = [
        {
            "round": 0,
            "same": best["same_outcome"],
            "mwt": best["mark_would_take"],
            "breach": best["n_breach"],
            "decision": "BASELINE",
            "top_error": card0["dominant_spine_error"],
        }
    ]

    for rnd in range(1, max_rounds + 1):
        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            print("*** PRACTICE 50/50 ***", flush=True)
            break

        policy.load_state_dict(best_state)
        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            print("no MWT left", flush=True)
            break

        focus, focus_cls = pick_focus(
            mwt, round_i=rnd - 1, spine_map=spine_map, fail_counts=fail_counts
        )
        fix = surgical_fix_for(focus_cls)
        # mild escalate if this day failed twice already
        if fail_counts.get(focus["date"], 0) >= 2:
            fix = dict(fix)
            fix["kl"] = max(0.15, float(fix["kl"]) - 0.1)
            fix["epochs"] = int(fix["epochs"]) + 8
            fix["dir_copy"] = int(fix["dir_copy"]) + (4 if focus_cls != "false_fire" else 0)
            fix["wait_copy"] = int(fix["wait_copy"]) + (4 if focus_cls == "false_fire" else 0)
            fix["label"] = fix["label"] + "_retry"

        print(
            f"\n===== SPINE-1D {rnd}/{max_rounds} focus={focus['date']} "
            f"cls={focus_cls} fix={fix['label']} fails={fail_counts.get(focus['date'],0)} =====",
            flush=True,
        )

        X, y, w = collect_focus_batch(
            day_map, oracle, spine_map, focus, focus_cls, mwt, awards, policy, fix
        )
        n_dir = int((y != 0).sum()) if len(y) else 0
        n_hold = int((y == 0).sum()) if len(y) else 0
        print(
            f"  n={len(y)} dir={n_dir} hold={n_hold} ratio={n_dir/max(n_hold,1):.2f} "
            f"kl={fix['kl']} epochs={fix['epochs']}",
            flush=True,
        )
        if len(y) < 40:
            print("  too few labels", flush=True)
            fail_counts[focus["date"]] = fail_counts.get(focus["date"], 0) + 1
            continue

        pol2, _ = train_bc(
            X,
            y,
            epochs=int(fix["epochs"]),
            hidden=128,
            seed=800 + rnd,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=float(fix["lr"]),
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=float(fix["kl"]),
        )
        print(f"  match={match_rate(pol2, X, y)}", flush=True)
        post = score_policy(pol2, day_map, mark_rows)

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

        # Pack repair if focus won but pack fell
        if focus_ok and post["n_breach"] == 0 and post["same_outcome"] < best["same_outcome"]:
            print("  PACK-repair…", flush=True)
            pol2 = pack_repair(pol2, best, day_map, oracle, mwt, focus)
            post = score_policy(pol2, day_map, mark_rows)
            focus_ok = False
            for row in post["rows"]:
                if row["date"] == focus["date"]:
                    focus_ok = bool(row["policy_award"])
                    break
            print(
                f"  REPAIR same={post['same_outcome']} mwt={post['mark_would_take']} "
                f"breach={post['n_breach']} focus_ok={focus_ok}",
                flush=True,
            )

        # Breach HOLD repair
        if post["n_breach"] > 0:
            print("  HOLD-repair (breach)…", flush=True)
            hx, hy, hw = [], [], []
            for row in post["rows"]:
                if not row.get("policy_breached") and int(row.get("policy_n_entries") or 0) < 5:
                    continue
                m2 = get_plan(
                    oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
                )
                a, b, c = plan_labels(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    m2,
                    dir_copy=1,
                    hold_copy=10,
                )
                for o, act, _ in zip(a, b, c):
                    if int(act) == ACTION_HOLD:
                        hx.append(o)
                        hy.append(ACTION_HOLD)
                        hw.append(9.0)
            if len(hy) >= 20:
                pol2, _ = train_bc(
                    np.stack(hx),
                    np.asarray(hy, np.int64),
                    epochs=12,
                    hidden=128,
                    seed=900 + rnd,
                    warm_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
                    obs_dim=MARK_FULL_DIM,
                    lr=1.5e-4,
                    sample_weights=np.asarray(hw, np.float32),
                    kl_anchor_state=best_state,
                    kl_coef=0.65,
                )
                post = score_policy(pol2, day_map, mark_rows)
                print(
                    f"  BREACH-REPAIR same={post['same_outcome']} breach={post['n_breach']}",
                    flush=True,
                )

        avg_ent = float(np.mean([r["policy_n_entries"] for r in post["rows"]])) if post["rows"] else 0
        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor_clear
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
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(
                pol2,
                note=f"spine_1d_KEEP_{focus['date']}_{fix['label']}",
                dials=dials,
            )
            torch.save(
                {
                    "tag": "mark_shadow_v1",
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "state_dict": pol2.state_dict(),
                    "hidden": 128,
                    "obs_dim": MARK_FULL_DIM,
                    "round": rnd,
                    "focus": focus["date"],
                    "same_outcome": post["same_outcome"],
                    "proven_touched": False,
                    "method": "spine_one_day",
                    "focus_cls": focus_cls,
                },
                SHADOW_CKPT,
            )
            with open(BEST, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "same_outcome": post["same_outcome"],
                        "policy_clear": post["policy_clear"],
                        "mwt": post["mark_would_take"],
                        "breach": post["n_breach"],
                        "source": f"spine_one_day_KEEP_{focus['date']}",
                        "focus_cls": focus_cls,
                        "fix": fix["label"],
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            fail_counts[focus["date"]] = 0
            print(f"  KEEP → best same={best['same_outcome']} mwt={best['mark_would_take']}", flush=True)
        else:
            fail_counts[focus["date"]] = fail_counts.get(focus["date"], 0) + 1
            print(f"  REJECT (fail_count[{focus['date']}]={fail_counts[focus['date']]})", flush=True)

        card = build_error_card(
            post if keep else best if not keep and post["same_outcome"] < best["same_outcome"] else post,
            spine_map,
            cycle=rnd,
            decision=decision,
            change=f"{focus['date']}:{focus_cls}:{fix['label']}",
        )
        # always log POST meters for KAG
        card["same"] = post["same_outcome"]
        card["mwt"] = post["mark_would_take"]
        card["breach"] = post["n_breach"]
        card["policy_clear"] = post["policy_clear"]
        append_learning_md(card)

        cycles.append(
            {
                "round": rnd,
                "focus": focus["date"],
                "focus_cls": focus_cls,
                "focus_ok": focus_ok,
                "same": post["same_outcome"],
                "policy": post["policy_clear"],
                "mwt": post["mark_would_take"],
                "breach": post["n_breach"],
                "decision": decision,
                "fix": fix["label"],
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
                    "cycles": cycles,
                    "fail_counts": fail_counts,
                },
                f,
                indent=2,
            )

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "best_same": best["same_outcome"],
        "best_policy": best["policy_clear"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "n_rounds": len(cycles) - 1,
        "cycles": cycles,
    }
    with open(CYCLE_LOG, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(
        f"DONE best same={summary['best_same']} mwt={summary['best_mwt']} "
        f"breach={summary['best_breach']}",
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=30)
    ap.add_argument("--keep-floor", type=int, default=KEEP_FLOOR)
    args = ap.parse_args(list(argv) if argv is not None else None)
    run(max_rounds=args.max_rounds, keep_floor=args.keep_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
