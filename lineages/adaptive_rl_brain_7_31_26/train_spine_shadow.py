"""S2–S6 — Spine Shadow train: miss-first on MWT spine events, KEEP/REJECT, error cards.

Does NOT open 3 teachers. Does NOT crank entry rewards as primary.
Does NOT touch PROVEN. Uses mark_aligned_decode path via GoalEquityDay.mark_align_policy.
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
    SPINE_DIR,
    SPINE_INDEX,
    DaySpine,
    classify_spine_error,
    fire_times,
    load_spine,
    load_spine_index,
    wait_times,
)
from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
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
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.spine_oracle_score import compile_practice_spines
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import match_rate, train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
SHADOW_CKPT = os.path.join(_HERE, "checkpoints", "mark_shadow_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
LEARNING_MD = os.path.join(OUT, "LEARNING_50D_MATCH.md")
SPINE_LEARNING = os.path.join(OUT, "SPINE_SHADOW_LEARNING.md")
ERROR_CARD = os.path.join(OUT, "SPINE_ERROR_CARD__latest.md")
CYCLE_LOG = os.path.join(OUT, "SPINE_SHADOW_CYCLES__latest.json")
SESSION_NOTE = os.path.join(OUT, "SPINE_SHADOW_SESSION__latest.md")

KEEP_FLOOR_SAME = 33


def _spine_map_from_index() -> Dict[str, DaySpine]:
    if not os.path.isfile(SPINE_INDEX):
        return {}
    idx = load_spine_index()
    out: Dict[str, DaySpine] = {}
    for it in idx.get("items") or []:
        key = f"{it['day']}|{it['target_pct']}|{it['risk_pct']}"
        out[key] = load_spine(it["path"])
    return out


def spine_event_labels(
    day_map: Dict[str, Any],
    spine: DaySpine,
    mark: dict,
    *,
    dir_copy: int = 12,
    wait_copy: int = 6,
    hold_copy: int = 2,
) -> Tuple[list, list, list]:
    """BC labels weighted toward spine fire/add/wait_loaded (not dense bar CE)."""
    plan = mark.get("plan") or spine.plan
    if not plan:
        return [], [], []
    plan = {int(k): int(v) for k, v in plan.items()}
    fire_set = set(fire_times(spine))
    wait_set = set(wait_times(spine))
    day = GoalEquityDay(
        day_map[spine.day],
        target_pct=float(spine.target_pct),
        risk_pct=float(spine.risk_pct),
        date_str=spine.day,
        eyes_mode="mark_doctrine",
        risk_use_frac=float(spine.risk_use_frac),
        per_trade_cap_pct=float(spine.per_trade_cap_pct),
        mark_soul=True,
        full_obs=True,
    )
    day._plan_lock_ruf = float(spine.risk_use_frac)
    day._plan_lock_cap = float(spine.per_trade_cap_pct)
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
        act = int(plan.get(int(tb), ACTION_HOLD))
        tb_i = int(tb)
        if tb_i in fire_set or act != ACTION_HOLD:
            n, w = max(10, dir_copy), 16.0
            for _ in range(n):
                xs.append(obs.copy())
                ys.append(act)
                ws.append(w)
        elif tb_i in wait_set:
            n, w = max(4, wait_copy), 10.0  # HOLD-on-spine skill
            for _ in range(n):
                xs.append(obs.copy())
                ys.append(ACTION_HOLD)
                ws.append(w)
        else:
            if (tb_i // 25) % 4 == 0:
                for _ in range(max(1, hold_copy // 2)):
                    xs.append(obs.copy())
                    ys.append(ACTION_HOLD)
                    ws.append(3.0)
        day.step_action(tb, act)
    return xs, ys, ws


def build_error_card(
    score: dict,
    spine_map: Dict[str, DaySpine],
    *,
    cycle: int,
    decision: str,
    change: str,
) -> Dict[str, Any]:
    mwt = [r for r in score["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
    classes: Counter = Counter()
    details = []
    for r in mwt:
        key = f"{r['date']}|{r['target_pct']}|{r['risk_pct']}"
        spine = spine_map.get(key)
        if spine is None:
            cls = "unknown_no_spine"
        else:
            cls = classify_spine_error(
                spine=spine,
                policy_fire_ts=[],  # entry times not logged in compact score; use n_entries proxy
                policy_n_entries=int(r.get("policy_n_entries") or 0),
                policy_award=bool(r["policy_award"]),
                policy_breached=bool(r.get("policy_breached")),
            )
        classes[cls] += 1
        details.append({"date": r["date"], "class": cls, "pnl": r.get("policy_pnl"), "n": r.get("policy_n_entries")})
    top = classes.most_common(1)[0][0] if classes else "none"
    card = {
        "cycle": cycle,
        "decision": decision,
        "change": change,
        "same": score["same_outcome"],
        "policy_clear": score["policy_clear"],
        "mwt": score["mark_would_take"],
        "breach": score["n_breach"],
        "dominant_spine_error": top,
        "error_class_counts": dict(classes),
        "mwt_details": details[:20],
        "written_at": datetime.now(timezone.utc).isoformat(),
    }
    lines = [
        f"# SPINE ERROR CARD — cycle {cycle}",
        "",
        f"**Decision:** {decision}",
        f"**Change tried:** {change}",
        f"**Meters:** same={card['same']} policy={card['policy_clear']} mwt={card['mwt']} breach={card['breach']}",
        f"**Dominant spine error:** `{top}`",
        "",
        "## Error class counts (MWT)",
        "",
    ]
    for k, v in classes.most_common():
        lines.append(f"- `{k}`: {v}")
    lines += ["", "## MWT sample", ""]
    for d in details[:12]:
        lines.append(f"- {d['date']}: {d['class']} pnl={d['pnl']} n_entries={d['n']}")
    lines += [
        "",
        "## Surgical fix next",
        "",
    ]
    if top == "false_hold":
        lines.append("- Raise fire/add spine weight; DAgger on false_hold days; do not crank global entry reward.")
    elif top == "false_fire":
        lines.append("- Raise wait_loaded / HOLD-on-spine weight; cut fire oversample.")
    elif top == "late_entry":
        lines.append("- Weight bars near spine fire window earlier; miss-first those days only.")
    elif top == "early_entry":
        lines.append("- Weight wait_loaded before t1; anti-thrash HOLD.")
    elif top == "wrong_size_or_timing":
        lines.append("- Keep side; re-BC plan path with higher dir_copy near t1/t2; size dials from spine.")
    elif top == "breach_thrash":
        lines.append("- REJECT path: HOLD repair, raise kl_coef, never keep breach.")
    else:
        lines.append("- Re-open error card after next score; no blind epochs.")
    with open(ERROR_CARD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    with open(os.path.join(OUT, f"SPINE_ERROR_CARD__cycle{cycle:03d}.json"), "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    return card


def append_learning_md(card: Dict[str, Any]) -> None:
    line = (
        f"| spine-shadow {card['cycle']} | {card['same']} | {card['policy_clear']} | "
        f"{card['mwt']} | {card['breach']} | **{card['decision']}** · {card['dominant_spine_error']} · {card['change'][:40]} |"
    )
    # SPINE_SHADOW_LEARNING
    if not os.path.isfile(SPINE_LEARNING):
        header = [
            "# SPINE SHADOW LEARNING LOG",
            "",
            "Goal: same_outcome 50/50 on held-out (new data), breach 0. Practice keep-floor 33.",
            "Method: Day Spine compile → oracle green → miss-first shadow train → KEEP/REJECT → error card.",
            "",
            "| Cycle | same | policy | mwt | breach | decision / top error / change |",
            "|------:|-----:|-------:|----:|-------:|-------------------------------|",
            "",
        ]
        with open(SPINE_LEARNING, "w", encoding="utf-8") as f:
            f.write("\n".join(header) + "\n")
    with open(SPINE_LEARNING, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    # also mirror a short line into LEARNING_50D_MATCH for KAG continuity
    if os.path.isfile(LEARNING_MD):
        with open(LEARNING_MD, "a", encoding="utf-8") as f:
            f.write(
                f"\n| spine-shadow {card['cycle']} | {card['same']} | {card['policy_clear']} | "
                f"{card['mwt']} | {card['breach']} | **{card['decision']}** ({card['dominant_spine_error']}) |\n"
            )


def surgical_fix_for(top: str) -> Dict[str, Any]:
    """One top-error fix dials — not entry-reward crank."""
    if top == "false_hold":
        return {"dir_copy": 14, "wait_copy": 4, "hold_copy": 2, "kl": 0.32, "lr": 3.5e-4, "epochs": 40, "dagger": True, "label": "boost_fire_false_hold"}
    if top == "false_fire":
        return {"dir_copy": 6, "wait_copy": 10, "hold_copy": 6, "kl": 0.55, "lr": 2e-4, "epochs": 28, "dagger": True, "label": "boost_wait_false_fire"}
    if top == "late_entry":
        return {"dir_copy": 16, "wait_copy": 5, "hold_copy": 2, "kl": 0.30, "lr": 3.5e-4, "epochs": 42, "dagger": True, "label": "earlier_fire_window"}
    if top == "early_entry":
        return {"dir_copy": 8, "wait_copy": 12, "hold_copy": 5, "kl": 0.50, "lr": 2.2e-4, "epochs": 30, "dagger": True, "label": "wait_before_t1"}
    if top == "wrong_size_or_timing":
        return {"dir_copy": 12, "wait_copy": 6, "hold_copy": 3, "kl": 0.40, "lr": 2.8e-4, "epochs": 36, "dagger": True, "label": "plan_path_size_timing"}
    if top == "breach_thrash":
        return {"dir_copy": 4, "wait_copy": 12, "hold_copy": 8, "kl": 0.65, "lr": 1.5e-4, "epochs": 20, "dagger": False, "label": "hold_repair"}
    return {"dir_copy": 10, "wait_copy": 6, "hold_copy": 3, "kl": 0.40, "lr": 2.8e-4, "epochs": 36, "dagger": True, "label": "default_miss_first"}


def run_cycles(
    *,
    max_cycles: int = 8,
    keep_floor: int = KEEP_FLOOR_SAME,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}

    # Ensure spines exist
    spine_map = _spine_map_from_index()
    if len(spine_map) < 40:
        print("Compiling spines (missing index)…", flush=True)
        oracle0 = load_oracle()
        spines, _ = compile_practice_spines(mark_rows, day_map, oracle0)
        spine_map = {f"{s.day}|{s.target_pct}|{s.risk_pct}": s for s in spines}

    oracle = load_oracle()
    policy = load_policy(CKPT)
    print("Initial practice score…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} policy={best['policy_clear']} "
        f"mwt={best['mark_would_take']} breach={best['n_breach']}",
        flush=True,
    )
    if best["same_outcome"] < keep_floor:
        print(
            f"WARNING: start same {best['same_outcome']} < keep_floor {keep_floor}; "
            "will not accept KEEP below floor.",
            flush=True,
        )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    # freeze start if better than disk BEST
    disk_best = json.load(open(BEST, encoding="utf-8")) if os.path.isfile(BEST) else {}
    session_best_same = max(int(disk_best.get("same_outcome", 0)), int(best["same_outcome"]), keep_floor)

    card0 = build_error_card(
        best,
        spine_map,
        cycle=0,
        decision="BASELINE",
        change="freeze_start_meters",
    )
    append_learning_md(card0)
    cycles: List[Dict[str, Any]] = [
        {
            "cycle": 0,
            "same": best["same_outcome"],
            "policy": best["policy_clear"],
            "mwt": best["mark_would_take"],
            "breach": best["n_breach"],
            "decision": "BASELINE",
            "top_error": card0["dominant_spine_error"],
        }
    ]

    top = card0["dominant_spine_error"]
    for cyc in range(1, max_cycles + 1):
        if best["same_outcome"] >= 50 and best["n_breach"] == 0:
            print("*** PRACTICE 50/50 HIT ***", flush=True)
            break
        print(f"\n===== SPINE SHADOW {cyc}/{max_cycles} top={top} =====", flush=True)
        policy.load_state_dict(best_state)
        fix = surgical_fix_for(top)
        # Escalate after flat rejects: lower KL, raise dir focus (not entry-reward crank)
        n_reject_flat = sum(
            1
            for c in cycles
            if c.get("decision") == "REJECT"
            and int(c.get("same", 0)) == int(best["same_outcome"])
        )
        if n_reject_flat >= 2:
            fix = dict(fix)
            fix["dir_copy"] = int(fix["dir_copy"]) + 6
            fix["kl"] = max(0.18, float(fix["kl"]) - 0.12)
            fix["epochs"] = int(fix["epochs"]) + 12
            fix["lr"] = min(4.5e-4, float(fix["lr"]) * 1.15)
            fix["label"] = str(fix["label"]) + "_escalate"
        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        # worst pnl first — focus top day heavily (one_day KEEP style)
        mwt = sorted(mwt, key=lambda r: float(r.get("policy_pnl") or 0))
        # rotate focus across MWT so we don't thrash the same day forever
        focus_idx = (cyc - 1) % max(len(mwt), 1)
        if mwt:
            focus = mwt[focus_idx]
            mwt = [focus] + [r for j, r in enumerate(mwt) if j != focus_idx]
        print(
            f"  mwt={len(mwt)} fix={fix['label']} focus={mwt[0]['date'] if mwt else None}",
            flush=True,
        )

        xs, ys, ws = [], [], []
        for i, row in enumerate(mwt):
            date, t, r = row["date"], float(row["target_pct"]), float(row["risk_pct"])
            mark = get_plan(oracle, day_map, date, t, r)
            key = f"{date}|{t}|{r}"
            spine = spine_map.get(key)
            if spine is None:
                from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import compile_spine_from_soul

                spine = compile_spine_from_soul(date, t, r, mark)
                spine_map[key] = spine
            if i == 0:
                reps, dmul = 8, 1.7
            elif i < 3:
                reps, dmul = 3, 1.25
            else:
                reps, dmul = 1, 1.0
            for _rep in range(reps):
                a, b, c = spine_event_labels(
                    day_map,
                    spine,
                    mark,
                    dir_copy=max(8, int(int(fix["dir_copy"]) * dmul)),
                    wait_copy=int(fix["wait_copy"]),
                    hold_copy=int(fix["hold_copy"]),
                )
                xs.extend(a)
                ys.extend(b)
                ws.extend(c)
            if fix.get("dagger"):
                for _ in range(4 if i == 0 else 1):
                    a, b, c = dagger_labels(day_map, date, t, r, mark, policy)
                    xs.extend(a)
                    ys.extend(b)
                    ws.extend(c)
            # classic dense plan path on focus (proven one_day KEEP)
            if i == 0:
                for _ in range(5):
                    a, b, c = plan_labels(
                        day_map,
                        date,
                        t,
                        r,
                        mark,
                        dir_copy=max(14, int(fix["dir_copy"])),
                        hold_copy=max(2, int(fix["hold_copy"])),
                    )
                    xs.extend(a)
                    ys.extend(b)
                    ws.extend(c)
        # protect awards
        for row in awards[:22]:
            a, b, c = award_self(
                day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
            )
            xs.extend(a)
            ys.extend(b)
            ws.extend([x * 1.7 for x in c])

        if len(ys) < 30:
            print("  insufficient labels — stop", flush=True)
            break
        X = np.stack(xs)
        y = np.asarray(ys, np.int64)
        w = np.asarray(ws, np.float32)
        n_dir = int((y != 0).sum())
        n_hold = int((y == 0).sum())
        print(
            f"  train n={len(y)} dir={n_dir} hold={n_hold} "
            f"ratio={n_dir/max(n_hold,1):.2f} kl={fix['kl']}",
            flush=True,
        )
        pol2, _ = train_bc(
            X,
            y,
            epochs=int(fix["epochs"]),
            hidden=128,
            seed=500 + cyc,
            warm_state=best_state,
            obs_dim=MARK_FULL_DIM,
            lr=float(fix["lr"]),
            sample_weights=w,
            kl_anchor_state=best_state,
            kl_coef=float(fix["kl"]),
        )
        print(f"  match={match_rate(pol2, X, y)}", flush=True)
        post = score_policy(pol2, day_map, mark_rows)
        # focus convert check early for pack-repair path
        focus_date = mwt[0]["date"] if mwt else None
        focus_ok = False
        if focus_date:
            for row in post["rows"]:
                if row["date"] == focus_date:
                    focus_ok = bool(row.get("policy_award"))
                    print(
                        f"  focus {focus_date} award={focus_ok} pnl={row.get('policy_pnl')} "
                        f"n={row.get('policy_n_entries')}",
                        flush=True,
                    )
                    break
        print(
            f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
            f"mwt={post['mark_would_take']} breach={post['n_breach']}",
            flush=True,
        )

        # Pack-repair: focus converted but pack slipped (one_day lesson)
        if (
            focus_ok
            and post["n_breach"] == 0
            and post["same_outcome"] < best["same_outcome"]
        ):
            print("  PACK-repair (focus ok, pack slipped)…", flush=True)
            hx, hy, hw = [], [], []
            protect = [r for r in best["rows"] if r.get("policy_award")][:28]
            for row in protect:
                a, b, c = award_self(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    pol2,
                )
                hx.extend(a)
                hy.extend(b)
                hw.extend([x * 2.0 for x in c])
            for row in mwt[:4]:
                mark = get_plan(
                    oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
                )
                a, b, c = plan_labels(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    mark,
                    dir_copy=2,
                    hold_copy=8,
                )
                for o, act, wt in zip(a, b, c):
                    if int(act) == ACTION_HOLD:
                        hx.append(o)
                        hy.append(ACTION_HOLD)
                        hw.append(6.0)
            if len(hy) >= 40:
                pol2, _ = train_bc(
                    np.stack(hx),
                    np.asarray(hy, np.int64),
                    epochs=16,
                    hidden=128,
                    seed=700 + cyc,
                    warm_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
                    obs_dim=MARK_FULL_DIM,
                    lr=2e-4,
                    sample_weights=np.asarray(hw, np.float32),
                    kl_anchor_state=best_state,
                    kl_coef=0.70,
                )
                post = score_policy(pol2, day_map, mark_rows)
                focus_ok = False
                if focus_date:
                    for row in post["rows"]:
                        if row["date"] == focus_date:
                            focus_ok = bool(row.get("policy_award"))
                            break
                print(
                    f"  REPAIR same={post['same_outcome']} mwt={post['mark_would_take']} "
                    f"breach={post['n_breach']} focus_ok={focus_ok}",
                    flush=True,
                )

        # HOLD repair if breach
        if post["n_breach"] > 0:
            print("  HOLD-repair…", flush=True)
            hx, hy, hw = [], [], []
            for row in post["rows"]:
                if not row.get("policy_breached") and int(row.get("policy_n_entries") or 0) < 5:
                    continue
                mark = get_plan(
                    oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
                )
                a, b, c = plan_labels(
                    day_map,
                    row["date"],
                    float(row["target_pct"]),
                    float(row["risk_pct"]),
                    mark,
                    dir_copy=2,
                    hold_copy=8,
                )
                for o, act, wt in zip(a, b, c):
                    if act == ACTION_HOLD:
                        hx.append(o)
                        hy.append(act)
                        hw.append(9.0)
            if len(hy) >= 20:
                pol2, _ = train_bc(
                    np.stack(hx),
                    np.asarray(hy, np.int64),
                    epochs=14,
                    hidden=128,
                    seed=600 + cyc,
                    warm_state={k: v.detach().clone() for k, v in pol2.state_dict().items()},
                    obs_dim=MARK_FULL_DIM,
                    lr=1.8e-4,
                    sample_weights=np.asarray(hw, np.float32),
                    kl_anchor_state=best_state,
                    kl_coef=0.62,
                )
                post = score_policy(pol2, day_map, mark_rows)
                print(
                    f"  REPAIR same={post['same_outcome']} breach={post['n_breach']}",
                    flush=True,
                )

        # KEEP rules: not worse than session best floor, breach 0, hold not collapsed
        improved = post["same_outcome"] > best["same_outcome"]
        not_worse = (
            post["n_breach"] == 0
            and post["same_outcome"] >= keep_floor
            and post["same_outcome"] >= best["same_outcome"]
            and post["policy_clear"] >= floor_clear
        )
        avg_ent = float(np.mean([r["policy_n_entries"] for r in post["rows"]])) if post["rows"] else 0
        thrash = avg_ent > 6.0
        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor_clear
            and post["same_outcome"] >= keep_floor
            and not thrash
            and (
                improved
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
            session_best_same = max(session_best_same, int(post["same_outcome"]))
            save_policy(pol2, note=f"spine_shadow_keep_c{cyc}_{fix['label']}", dials=clip_streak_dials(default_streak_dials()))
            torch.save(
                {
                    "tag": "mark_shadow_v1",
                    "saved_at": datetime.now(timezone.utc).isoformat(),
                    "state_dict": pol2.state_dict(),
                    "hidden": 128,
                    "obs_dim": MARK_FULL_DIM,
                    "cycle": cyc,
                    "same_outcome": post["same_outcome"],
                    "proven_touched": False,
                    "method": "spine_shadow",
                    "fix": fix["label"],
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
                        "source": f"spine_shadow_KEEP_c{cyc}",
                        "updated_note": fix["label"],
                    },
                    f,
                    indent=2,
                )
        else:
            policy.load_state_dict(best_state)

        card = build_error_card(
            post if keep else best,
            spine_map,
            cycle=cyc,
            decision=decision,
            change=fix["label"],
        )
        # if reject, still write card from POST for diagnosis
        if not keep:
            card = build_error_card(
                post,
                spine_map,
                cycle=cyc,
                decision=decision,
                change=fix["label"],
            )
        append_learning_md(card)
        top = card["dominant_spine_error"]
        cycles.append(
            {
                "cycle": cyc,
                "same": post["same_outcome"],
                "policy": post["policy_clear"],
                "mwt": post["mark_would_take"],
                "breach": post["n_breach"],
                "decision": decision,
                "top_error": top,
                "fix": fix["label"],
                "avg_entries": avg_ent,
            }
        )
        print(f"  → {decision} best_same={best['same_outcome']} next_top={top}", flush=True)

        # note stall but keep rotating focus (do not stop — objective requires climb)
        recent = [c for c in cycles if c.get("decision") == "REJECT"][-3:]
        if (
            len(recent) == 3
            and len({c.get("top_error") for c in recent}) == 1
            and all(int(c.get("same", 0)) == int(best["same_outcome"]) for c in recent)
        ):
            print(
                "  stall note: 3 flat rejects same class — rotating focus + escalate next",
                flush=True,
            )

    summary = {
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "best_same": best["same_outcome"],
        "best_policy": best["policy_clear"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "session_best_same": session_best_same,
        "cycles": cycles,
        "ckpt": CKPT,
        "shadow_ckpt": SHADOW_CKPT if os.path.isfile(SHADOW_CKPT) else None,
    }
    with open(CYCLE_LOG, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    with open(SESSION_NOTE, "w", encoding="utf-8") as f:
        f.write(
            f"# Spine Shadow session\n\n"
            f"- best_same: {summary['best_same']}\n"
            f"- best_mwt: {summary['best_mwt']}\n"
            f"- breach: {summary['best_breach']}\n"
            f"- cycles: {len(cycles)}\n"
            f"- log: `{CYCLE_LOG}`\n"
            f"- error card: `{ERROR_CARD}`\n"
            f"- learning: `{SPINE_LEARNING}`\n"
        )
    print(
        f"DONE best same={summary['best_same']} mwt={summary['best_mwt']} "
        f"breach={summary['best_breach']}",
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-cycles", type=int, default=6)
    ap.add_argument("--keep-floor", type=int, default=KEEP_FLOOR_SAME)
    args = ap.parse_args(list(argv) if argv is not None else None)
    run_cycles(max_cycles=args.max_cycles, keep_floor=args.keep_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
