"""RSI+BB L2L skill — learn load/release under HTF mass (not day memos).

Step-back (why this exists)
---------------------------
35→36 used multi-day Mark *fire_skill* BC. That is calendar-free but still
underspecified: the model did not explicitly learn Mark's instrument —
RSI(5)+BB **on RSI** for LTF timing while HTF price-BB mass confirms trend.

This module makes that instrument the **skill id** for learn-to-learn:
  • HTF mass  = both set HTFs close vs price BB mid (strong tide permission)
  • LTF time  = RSI+BB geometry → pullback_load (WAIT) vs continuation_release (FIRE)
  • Concurrence = prefer bars where Mark plan agrees with RSI-BB act
  • Fable gate = full 50d KEEP only if same rises, breach 0

Never strategy-only sole teacher (proved pack crater). Always KL + awards.

Code twin: strategies/rsi_bb_pullback_continuation.py
KAG: data/knowledge/army/RSI_BB_L2L_SKILL.md
"""
from __future__ import annotations

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

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    get_plan,
    load_oracle,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_HOLD,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.strategies.rsi_bb_pullback_continuation import (
    ALL_SET_IDS,
    collect_rsi_bb_dataset,
    kag_lesson_row,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc, train_bc_multitask

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
TEEN = os.path.join(_HERE, "checkpoints", "TEEN_STAGE_same36_fable_kag_fire_skill.pt")
CHILD = os.path.join(_HERE, "checkpoints", "CHILD_STAGE_same35_mark_clone_full_obs.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
HARNESS = os.path.join(OUT, "RSI_BB_L2L_HARNESS__latest.json")
REPORT = os.path.join(OUT, "RSI_BB_L2L__latest.md")
MEM = os.path.join(OUT, "RSI_BB_L2L_MEMORY.jsonl")

PRINCIPLE_IDS = (
    "htf_price_bb_mass",
    "ltf_rsi_bb_geometry",
    "pullback_load_then_release",
    "ltf_never_defines_side",
    "learn_not_copy",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_live() -> Tuple[Channel1Policy, str]:
    if os.path.isfile(CKPT):
        return load_policy(CKPT), CKPT
    if os.path.isfile(TEEN):
        return load_policy(TEEN), TEEN
    return load_policy(CHILD), CHILD


def build_rsi_bb_skill_batch(
    days: Sequence[Tuple[str, Any]],
    *,
    kind: str = "both",  # pullback | continuation | both
    max_days: int = 40,
    decide_every: int = 22,
    seed: int = 42,
    mark_rows: Optional[List[dict]] = None,
    day_map: Optional[Dict[str, Any]] = None,
    oracle: Optional[dict] = None,
    concurrence_boost: float = 2.2,
    base_w_pull: float = 1.4,
    base_w_cont: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Build multi-head skill batch from RSI-BB geometry (all sets 1–4).

    kind filters pullback_load vs continuation_release.
    If Mark oracle+plans available, upweight bars where Mark.act == skill.act
    (concurrence = principle agreement, not calendar memo).
    """
    ds = collect_rsi_bb_dataset(
        days,
        max_days=max_days,
        decide_every=decide_every,
        full_obs=True,
        seed=seed,
        set_ids=ALL_SET_IDS,
        history_days=min(120, max(40, len(list(days)) - 2)),
        light_obs=True,
        neg_per_pos=0.35,
    )
    n = int(ds.get("n") or 0)
    meta: Dict[str, Any] = {
        "principle_ids": list(PRINCIPLE_IDS),
        "strategy": "rsi_bb_pullback_continuation_v1",
        "kind_filter": kind,
        "n_raw": n,
        "n_pos": ds.get("n_pos"),
        "n_pull": ds.get("n_pullback"),
        "n_cont": ds.get("n_continuation"),
        "hits_by_set": ds.get("hits_by_set"),
        "sets_with_hits": ds.get("sets_with_hits"),
        "concurrence_n": 0,
        "n_keep": 0,
    }
    if n == 0:
        z = np.zeros((0, MARK_FULL_DIM), np.float32)
        e = np.zeros((0,), np.int64)
        return z, e, e, e, np.zeros((0,), np.float32), meta

    X = np.asarray(ds["X"], np.float32)
    y_act = np.asarray(ds["y_act"], np.int64)
    y_topo = np.asarray(ds.get("y_topology", ds.get("y_topo")), np.int64)
    y_wait = np.asarray(ds["y_wait"], np.int64)
    rows = list(ds.get("meta") or ds.get("meta_rows") or [])

    # Map day → Mark plan for concurrence (optional)
    plans: Dict[str, Dict[int, int]] = {}
    if mark_rows and day_map is not None and oracle is not None:
        for mr in mark_rows[:50]:
            d = str(mr["date"])
            mark = get_plan(
                oracle, day_map, d, float(mr["target_pct"]), float(mr["risk_pct"])
            )
            if mark and mark.get("plan"):
                plans[d] = {int(k): int(v) for k, v in mark["plan"].items()}

    xs, ya, yt, yw, ws = [], [], [], [], []
    conc = 0
    for i in range(n):
        row = rows[i] if i < len(rows) else {}
        k = str(row.get("kind") or "")
        # meta_rows kind is "positive"/"negative"; geometry kind in hit fields
        geo = str(row.get("kind") if row.get("side") else "")  # pullback/continuation on positives
        if "pullback" in str(row.get("topology") or "") or str(row.get("act")) == "wait_loaded":
            geo_kind = "pullback"
        elif "continuation" in str(row.get("topology") or "") or str(row.get("act") or "").startswith(
            "fire"
        ):
            geo_kind = "continuation"
        else:
            # fall back: HOLD → pullback-ish, fire → continuation
            geo_kind = "pullback" if int(y_act[i]) == ACTION_HOLD else "continuation"

        # Prefer explicit hit kind from kag_lesson_row
        if row.get("kind") in ("pullback", "continuation"):
            geo_kind = str(row["kind"])
        elif row.get("topology") == "pullback_load":
            geo_kind = "pullback"
        elif row.get("topology") == "continuation_release":
            geo_kind = "continuation"

        if kind == "pullback" and geo_kind != "pullback":
            continue
        if kind == "continuation" and geo_kind != "continuation":
            continue
        # skip pure negatives for skill focus (keep some HOLD from pullbacks only)
        if row.get("kind") == "negative" and kind != "both":
            continue

        w = base_w_pull if geo_kind == "pullback" else base_w_cont
        # Concurrence with Mark
        day = str(row.get("day") or "")
        bi = int(row.get("bar_index") or -1)
        if day in plans and bi >= 0:
            ma = plans[day].get(bi, plans[day].get(int(bi), None))
            if ma is not None and int(ma) == int(y_act[i]):
                w *= concurrence_boost
                conc += 1
        xs.append(X[i])
        ya.append(int(y_act[i]))
        yt.append(int(y_topo[i]))
        yw.append(int(y_wait[i]))
        ws.append(float(w))

    meta["concurrence_n"] = conc
    meta["n_keep"] = len(ya)
    if not ya:
        z = np.zeros((0, MARK_FULL_DIM), np.float32)
        e = np.zeros((0,), np.int64)
        return z, e, e, e, np.zeros((0,), np.float32), meta

    w_arr = np.asarray(ws, np.float32)
    w_arr = np.maximum(w_arr, 1e-6)
    w_arr = w_arr / float(w_arr.mean())
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ya, np.int64),
        np.asarray(yt, np.int64),
        np.asarray(yw, np.int64),
        w_arr,
        meta,
    )


def inject_awards(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    day_map: Dict[str, Any],
    awards: List[dict],
    policy: Channel1Policy,
    *,
    y_topo: Optional[np.ndarray] = None,
    y_wait: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    xs, ys, ws = list(X), list(y), list(w)
    tps = list(y_topo) if y_topo is not None else None
    wts = list(y_wait) if y_wait is not None else None
    for row in awards[:28]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        for o, act, ww in zip(a, b, c):
            xs.append(o)
            ys.append(int(act))
            ws.append(float(ww) * 2.8)
            if tps is not None:
                tps.append(0)  # pullback/hold topology soft
            if wts is not None:
                wts.append(0 if int(act) == ACTION_HOLD else 1)
    X2 = np.stack(xs).astype(np.float32)
    y2 = np.asarray(ys, np.int64)
    w2 = np.asarray(ws, np.float32)
    w2 = w2 / float(w2.mean())
    yt2 = np.asarray(tps, np.int64) if tps is not None else None
    yw2 = np.asarray(wts, np.int64) if wts is not None else None
    return X2, y2, w2, yt2, yw2


def run_rsi_bb_l2l(
    *,
    max_rounds: int = 6,
    keep_floor: int = 36,
    kl_coef: float = 1.20,
    epochs: int = 14,
    lr: float = 1.0e-4,
    use_multitask: bool = False,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    print("=== RSI+BB L2L SKILL (HTF mass → LTF RSI-BB load/release) ===", flush=True)
    print(f"  principles: {PRINCIPLE_IDS}", flush=True)

    policy, src = _load_live()
    print(f"  load {os.path.basename(src)}", flush=True)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    dials = clip_streak_dials(default_streak_dials())

    print("Score base…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(keep_floor, int(best["same_outcome"]))
    cycles: List[dict] = []

    # Round schedule: continuation first (like 36 KEEP), then pullback, then both
    schedule = ["continuation", "pullback", "both", "continuation", "both", "pullback"]

    hid = 128
    try:
        w0 = best_state.get("net.0.weight") or best_state.get("trunk.0.weight")
        if w0 is not None:
            hid = int(w0.shape[0])
    except Exception:
        hid = 128

    for rnd in range(1, max_rounds + 1):
        kind = schedule[(rnd - 1) % len(schedule)]
        print(f"\n===== RSI-BB L2L {rnd}/{max_rounds} skill_kind={kind} =====", flush=True)
        policy.load_state_dict(best_state)

        X, y_act, y_topo, y_wait, w, meta = build_rsi_bb_skill_batch(
            days,
            kind=kind,
            max_days=45,
            seed=800 + rnd,
            mark_rows=mark_rows,
            day_map=day_map,
            oracle=oracle,
        )
        print(f"  skill batch meta={meta}", flush=True)
        if meta["n_keep"] < 40:
            print("  sparse skill batch — skip", flush=True)
            continue

        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        X, y_act, w, y_topo, y_wait = inject_awards(
            X, y_act, w, day_map, awards, policy, y_topo=y_topo, y_wait=y_wait
        )
        hold_frac = float((y_act == 0).mean())
        print(
            f"  n={len(y_act)} hold_frac={hold_frac:.2f} dir={int((y_act!=0).sum())} "
            f"concurrence≈{meta.get('concurrence_n')}",
            flush=True,
        )

        # Default single-head: preserves teen Channel1 geometry (multi-head-only
        # strategy teach cratered pack historically). Multitask optional.
        if use_multitask and y_topo is not None and y_wait is not None:
            pol2, loss_hist = train_bc_multitask(
                X,
                y_act,
                y_topo,
                y_wait,
                epochs=epochs,
                lr=lr,
                hidden=hid,
                seed=900 + rnd,
                warm_state=best_state,
                kl_anchor_state=best_state,
                kl_coef=float(kl_coef),
            )
            last_loss = float(loss_hist[-1]["loss_total"]) if loss_hist else 0.0
        else:
            pol2, losses = train_bc(
                X,
                y_act,
                sample_weights=w,
                epochs=epochs,
                lr=lr,
                hidden=hid,
                seed=900 + rnd,
                warm_state=best_state,
                kl_anchor_state=best_state,
                kl_coef=float(kl_coef),
                freeze_trunk=(kind == "pullback"),
            )
            last_loss = float(losses[-1]) if losses else 0.0

        with torch.no_grad():
            pred = pol2(torch.tensor(X, dtype=torch.float32))
            if isinstance(pred, dict):
                pred_a = pred["action"].argmax(-1).numpy()
            else:
                pred_a = pred.argmax(-1).numpy()
        act_match = float((pred_a == y_act).mean())
        print(f"  train act_match={act_match:.3f} loss={last_loss:.4f}", flush=True)

        post = score_policy(pol2, day_map, mark_rows)
        print(
            f"  POST same={post['same_outcome']} mwt={post['mark_would_take']} "
            f"breach={post['n_breach']}",
            flush=True,
        )

        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= floor_clear
            and post["same_outcome"] >= live_floor
            and post["same_outcome"] > best["same_outcome"]
        )
        decision = "KEEP" if keep else "REJECT"
        if post["same_outcome"] < best["same_outcome"] - 2:
            decision = "REJECT"
            print("  pack crater — REJECT", flush=True)
        elif not keep:
            print("  REJECT (no same raise / gate)", flush=True)

        if decision == "KEEP":
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            save_policy(
                pol2,
                note=f"rsi_bb_l2l_KEEP_r{rnd}_{kind}",
                dials=dials,
            )
            with open(BEST, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "same_outcome": post["same_outcome"],
                        "policy_clear": post["policy_clear"],
                        "mwt": post["mark_would_take"],
                        "breach": post["n_breach"],
                        "source": f"rsi_bb_l2l_KEEP_{kind}_r{rnd}",
                        "stage": "teen" if post["same_outcome"] > 35 else "child",
                        "growth_method": "rsi_bb_l2l_skill",
                        "core_skill": "LTF RSI+BB load/release under HTF price-BB mass",
                        "principle_ids": list(PRINCIPLE_IDS),
                        "skill_kind": kind,
                        "note": "RSI-BB geometry skill + Mark concurrence; not day memo",
                        "ts": _utcnow(),
                    },
                    f,
                    indent=2,
                )
            print(f"  KEEP best_same={best['same_outcome']} skill={kind}", flush=True)
        else:
            policy.load_state_dict(best_state)

        row = {
            "ts": _utcnow(),
            "round": rnd,
            "skill_kind": kind,
            "decision": decision,
            "same": post["same_outcome"],
            "mwt": post["mark_would_take"],
            "breach": post["n_breach"],
            "best_same": best["same_outcome"],
            "act_match": act_match,
            "meta": meta,
            "principles": list(PRINCIPLE_IDS),
            "method": "rsi_bb_l2l_skill",
        }
        cycles.append(row)
        with open(MEM, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        with open(HARNESS, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": _utcnow(),
                    "best_same": best["same_outcome"],
                    "best_mwt": best["mark_would_take"],
                    "best_breach": best["n_breach"],
                    "method": "rsi_bb_l2l_skill",
                    "principles": list(PRINCIPLE_IDS),
                    "cycles": cycles[-20:],
                },
                f,
                indent=2,
            )
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# RSI+BB L2L skill report\n\n"
                f"- best_same: **{best['same_outcome']}**\n"
                f"- last: {decision} kind={kind} same={post['same_outcome']}\n"
                f"- skill: HTF price-BB mass + LTF RSI(5)+BB on RSI load/release\n"
                f"- principles: {', '.join(PRINCIPLE_IDS)}\n"
            )

    summary = {
        "best_same": best["same_outcome"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "cycles": len(cycles),
        "method": "rsi_bb_l2l_skill",
        "principles": list(PRINCIPLE_IDS),
    }
    print(f"DONE rsi_bb_l2l {summary}", flush=True)
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="RSI+BB L2L skill under HTF mass")
    ap.add_argument("--max-rounds", type=int, default=6)
    ap.add_argument("--keep-floor", type=int, default=36)
    ap.add_argument("--kl-coef", type=float, default=1.20)
    ap.add_argument("--epochs", type=int, default=14)
    ap.add_argument("--lr", type=float, default=1.0e-4)
    ap.add_argument("--multitask", action="store_true")
    args = ap.parse_args()
    run_rsi_bb_l2l(
        max_rounds=args.max_rounds,
        keep_floor=args.keep_floor,
        kl_coef=args.kl_coef,
        epochs=args.epochs,
        lr=args.lr,
        use_multitask=args.multitask,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
