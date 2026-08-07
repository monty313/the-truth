"""F3 full Spine Shadow train: phase+event+size+clue_gate + learn≠copy + KEEP/REJECT.

Fast path: one-day focus MWT + award protect; multi-head loss; export Channel1 for score.
Doctrine: Fable 5 Alternate — Spine Shadow.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.compile_day_spine import size_bucket_for
from lineages.adaptive_rl_brain_7_31_26.equity_day import GoalEquityDay, load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    get_plan,
    load_oracle,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.kag_teachers.student_interface import check_learn_not_copy
from lineages.adaptive_rl_brain_7_31_26.mark_shadow_policy import (
    EVENT_I,
    EVENTS,
    PHASE_I,
    PHASES,
    SIZE_I,
    SIZES,
    SpineShadowNet,
    as_channel1,
    event_at_t,
    event_to_action,
    phase_at_t,
    size_at_event,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
SHADOW = os.path.join(_HERE, "checkpoints", "mark_shadow_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
REPORT = os.path.join(OUT, "SPINE_SHADOW_FULL__latest.md")
STATE = os.path.join(OUT, "SPINE_SHADOW_FULL__latest.json")


def collect_shadow_labels(
    day_map,
    date: str,
    t: float,
    r: float,
    mark: dict,
    policy: Channel1Policy,
    *,
    mwt: bool = True,
) -> Tuple[list, list, list, list, list, list]:
    """Collect obs + phase/event/size/act labels from Mark plan + policy path (DAgger)."""
    plan = mark.get("plan") or {}
    plan = {int(k): int(v) for k, v in plan.items()}
    t1 = mark.get("t1")
    t2 = mark.get("t2")
    if t1 is None:
        fires = [k for k, v in plan.items() if int(v) != ACTION_HOLD]
        t1 = min(fires) if fires else None
    side = mark.get("side")
    ruf = mark.get("risk_use_frac")
    cap = mark.get("per_trade_cap_pct")
    if ruf in (None, "dynamic"):
        bucket = "base"
    else:
        bucket = size_bucket_for(float(ruf), float(cap if cap not in (None, "dynamic") else 0.25))

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
    if ruf not in (None, "dynamic"):
        day._plan_lock_ruf = float(ruf)
        day._plan_lock_cap = float(cap)

    xs, yp, ye, ys, ya, ww = [], [], [], [], [], []
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
        ph = phase_at_t(int(tb), int(t1) if t1 is not None else None, int(t2) if t2 is not None else None)
        ev = event_at_t(int(tb), plan, int(t1) if t1 is not None else None, int(t2) if t2 is not None else None)
        sz = size_at_event(ev, bucket)
        # sample weight: MWT near events heavy
        if pa != ma:
            n, w = (4, 14.0) if mwt else (2, 6.0)
        elif ev in ("fire", "add", "wait_loaded"):
            n, w = (2, 8.0) if mwt else (1, 3.0)
        else:
            n, w = (1, 2.5) if (int(tb) // 25) % 3 == 0 else (0, 0.0)
        for _ in range(n):
            xs.append(obs.copy())
            yp.append(PHASE_I[ph])
            ye.append(EVENT_I[ev])
            ys.append(SIZE_I.get(sz, 0))
            ya.append(ma)
            ww.append(w)
        day.step_action(tb, pa)
    return xs, yp, ye, ys, ya, ww


def train_shadow(
    X,
    y_phase,
    y_event,
    y_size,
    y_act,
    sample_weights,
    *,
    warm: dict,
    epochs: int = 20,
    lr: float = 2e-4,
    kl_coef: float = 0.60,
    seed: int = 0,
) -> Tuple[SpineShadowNet, Dict[str, float]]:
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = SpineShadowNet(obs_dim=MARK_FULL_DIM, hidden=128)
    net.load_from_channel1(warm)
    anchor = Channel1Policy(obs_dim=MARK_FULL_DIM, hidden=128)
    anchor.load_state_dict(warm)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    Xt = torch.tensor(X, dtype=torch.float32)
    yp = torch.tensor(y_phase, dtype=torch.long)
    ye = torch.tensor(y_event, dtype=torch.long)
    ys = torch.tensor(y_size, dtype=torch.long)
    ya = torch.tensor(y_act, dtype=torch.long)
    sw = np.asarray(sample_weights, np.float32)
    sw = np.maximum(sw, 1e-6)
    sw = sw / float(sw.mean())
    sw_t = torch.tensor(sw, dtype=torch.float32)
    n = len(y_act)

    for ep in range(epochs):
        perm = np.random.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, 256):
            idx = perm[i : i + 256]
            ph, ev, sz, act, gate = net(Xt[idx])
            # multi-head CE (spine shadow F3)
            lp = F.cross_entropy(ph, yp[idx], reduction="none")
            le = F.cross_entropy(ev, ye[idx], reduction="none")
            ls = F.cross_entropy(sz, ys[idx], reduction="none")
            la = F.cross_entropy(act, ya[idx], reduction="none")
            w = sw_t[idx]
            loss = (w * (0.35 * lp + 0.40 * le + 0.15 * ls + 0.45 * la)).mean()
            # light clue_gate: prefer not all-zero / not all-one (entropy toward mid)
            gmean = gate.mean()
            loss = loss + 0.02 * ((gmean - 0.55) ** 2)
            # KL on act to BEST
            with torch.no_grad():
                a_log = anchor(Xt[idx])
                a_p = F.softmax(a_log, dim=-1)
                a_lp = F.log_softmax(a_log, dim=-1)
            n_lp = F.log_softmax(act, dim=-1)
            kl = (a_p * (a_lp - n_lp)).sum(-1).mean()
            loss = loss + kl_coef * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  shadow epoch {ep+1}/{epochs} loss={ep_loss/max(nb,1):.4f}", flush=True)

    net.eval()
    with torch.no_grad():
        ph, ev, sz, act, _ = net(Xt)
        pred_a = act.argmax(-1).numpy()
        pred_e = ev.argmax(-1).numpy()
        pred_p = ph.argmax(-1).numpy()
    act_m = float((pred_a == y_act).mean())
    evt_m = float((pred_e == y_event).mean())
    ph_m = float((pred_p == y_phase).mean())
    # learn≠copy: act without event/phase = copying
    gate = check_learn_not_copy(
        act_match=act_m, topology_match=0.5 * (evt_m + ph_m), role_map_match=evt_m
    )
    metrics = {
        "act_match": act_m,
        "event_match": evt_m,
        "phase_match": ph_m,
        "learn_not_copy_pass": bool(gate["pass"]),
        "copying": bool(gate["copying"]),
    }
    print(f"  shadow metrics {metrics}", flush=True)
    return net, metrics


def run(max_rounds: int = 8, keep_floor: int = 33) -> dict:
    os.makedirs(OUT, exist_ok=True)
    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    policy = load_policy(CKPT)
    dials = clip_streak_dials(default_streak_dials())

    print("Spine Shadow FULL (phase+event+size+clue) — score…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    cycles = []

    for rnd in range(1, max_rounds + 1):
        if best["same_outcome"] >= 50:
            break
        policy.load_state_dict(best_state)
        mwt = sorted(
            [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"],
            key=lambda r: float(r.get("policy_pnl") or 0),
        )
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            break
        focus = mwt[(rnd - 1) % len(mwt)]
        print(f"\n===== SHADOW-FULL {rnd}/{max_rounds} focus={focus['date']} =====", flush=True)

        xs, yp, ye, ys, ya, ww = [], [], [], [], [], []
        for row in [focus] + mwt[1:2]:
            mark = get_plan(
                oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
            )
            a, b, c, d, e, f = collect_shadow_labels(
                day_map,
                row["date"],
                float(row["target_pct"]),
                float(row["risk_pct"]),
                mark,
                policy,
                mwt=True,
            )
            xs.extend(a)
            yp.extend(b)
            ye.extend(c)
            ys.extend(d)
            ya.extend(e)
            ww.extend(f)
        for row in awards[:20]:
            a, b, c = award_self(
                day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
            )
            for o, act, w in zip(a, b, c):
                xs.append(o)
                yp.append(PHASE_I["in_trade"] if int(act) != ACTION_HOLD else PHASE_I["before_first_fire"])
                ye.append(EVENT_I["hold_on_spine"])
                ys.append(SIZE_I["none"])
                ya.append(int(act))
                ww.append(float(w) * 2.0)

        if len(ya) < 40:
            print("  few labels", flush=True)
            continue
        X = np.stack(xs)
        print(
            f"  n={len(ya)} dir={int((np.asarray(ya)!=0).sum())} "
            f"hold={int((np.asarray(ya)==0).sum())}",
            flush=True,
        )
        net, metrics = train_shadow(
            X,
            np.asarray(yp, np.int64),
            np.asarray(ye, np.int64),
            np.asarray(ys, np.int64),
            np.asarray(ya, np.int64),
            np.asarray(ww, np.float32),
            warm=best_state,
            epochs=18,
            lr=2e-4,
            kl_coef=0.62,
            seed=800 + rnd,
        )
        if metrics.get("copying"):
            print("  learn≠copy FAIL — REJECT (no pack thrash)", flush=True)
            cycles.append({"round": rnd, "decision": "REJECT", "reason": "copying", **metrics})
            continue

        pol2 = as_channel1(net)
        post = score_policy(pol2, day_map, mark_rows)
        focus_ok = any(
            r["date"] == focus["date"] and r["policy_award"] for r in post["rows"]
        )
        print(
            f"  focus_ok={focus_ok} same={post['same_outcome']} mwt={post['mark_would_take']} "
            f"breach={post['n_breach']}",
            flush=True,
        )
        if post["same_outcome"] < best["same_outcome"] - 3:
            decision = "REJECT"
            print("  pack crater REJECT", flush=True)
        else:
            keep = (
                post["n_breach"] == 0
                and post["policy_clear"] >= floor_clear
                and post["same_outcome"] >= live_floor
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

        if decision == "KEEP":
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(pol2, note=f"spine_shadow_full_KEEP_{focus['date']}", dials=dials)
            torch.save(
                {
                    "tag": "mark_shadow_v1",
                    "method": "spine_shadow_full",
                    "heads": ["phase", "event", "size", "act", "clue_gate"],
                    "phases": PHASES,
                    "events": EVENTS,
                    "sizes": SIZES,
                    "state_dict": net.state_dict(),
                    "channel1_state": best_state,
                    "same_outcome": post["same_outcome"],
                    "proven_touched": False,
                    "metrics": metrics,
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
                        "source": f"spine_shadow_full_KEEP_{focus['date']}",
                        "updated_at": datetime.now(timezone.utc).isoformat(),
                    },
                    f,
                    indent=2,
                )
            print(f"  KEEP same={best['same_outcome']}", flush=True)
            try:
                with open(os.path.join(OUT, "WHAT_WORKS__GOAL.md"), "a", encoding="utf-8") as wf:
                    wf.write(
                        f"| KEEP live | **{best['same_outcome']}** | {best['mark_would_take']} | "
                        f"0 | spine_shadow_full phase+event+size+clue |\n"
                    )
            except OSError:
                pass
        else:
            print("  REJECT", flush=True)

        cycles.append(
            {
                "round": rnd,
                "focus": focus["date"],
                "focus_ok": focus_ok,
                "decision": decision,
                "same": post["same_outcome"],
                "mwt": post["mark_would_take"],
                "breach": post["n_breach"],
                "best_same": best["same_outcome"],
                "metrics": metrics,
            }
        )
        with open(STATE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "best_same": best["same_outcome"],
                    "heads": ["phase", "event", "size", "act", "clue_gate"],
                    "cycles": cycles,
                },
                f,
                indent=2,
            )
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# Spine Shadow FULL\n\n"
                f"Heads: phase + event + size + act + clue_gate\n\n"
                f"best_same={best['same_outcome']} mwt={best['mark_would_take']}\n\n"
                f"last={cycles[-1]}\n"
            )

    print(f"DONE best same={best['same_outcome']}", flush=True)
    return {"best_same": best["same_outcome"], "cycles": cycles}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--keep-floor", type=int, default=33)
    args = ap.parse_args()
    run(max_rounds=args.max_rounds, keep_floor=args.keep_floor)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
