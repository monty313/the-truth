"""Combined knowledge climb past high-water (Mark path-family + pack protect + light strategy).

PRIMARY knowledge (what raised 35→36): multi-day path-family BC from fable_kag_l2l
  (fire_skill / wait_skill clusters across MWT days — not calendar memos).
PROTECT: KL to BEST + award self-imitate + HOLD-floor inject.
AUX: light strategy fire samples (CCI / RSI-BB) at low weight — never sole teacher.
BANNED: strategy-only multi-head full replace (proved 35→15).

KEEP only if frozen 50d same_outcome strictly rises and n_breach == 0.

Usage (repo root the-truth):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/climb_35_combined_knowledge.py
  python lineages/adaptive_rl_brain_7_31_26/climb_35_combined_knowledge.py --max-rounds 10 --kl-coef 1.15
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
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

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import load_policy, save_policy
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import (
    award_self,
    load_oracle,
    save_oracle,
    score_policy,
)
from lineages.adaptive_rl_brain_7_31_26.fable_kag_l2l import (
    build_pattern_bank,
    choose_family,
    collect_cluster_labels,
    load_pattern_mem,
    walk_day_labels,
)
from lineages.adaptive_rl_brain_7_31_26.fable_50d_rapid import get_plan
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import ACTION_HOLD, Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.strategies.cci_dual_level_continuation import (
    collect_continuation_dataset,
)
from lineages.adaptive_rl_brain_7_31_26.strategies.rsi_bb_pullback_continuation import (
    ALL_SET_IDS,
    collect_rsi_bb_dataset,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc
from collections import Counter

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST_JSON = os.path.join(OUT, "BEST__latest.json")
REPORT = os.path.join(OUT, "CLIMB35_COMBINED__latest.json")
CANDIDATE = os.path.join(_HERE, "checkpoints", "mark_clone_candidate_combined_v1.pt")
MIX_RECIPE = os.path.join(OUT, "MIX_RECIPE_CLIMB35.md")

SOURCES = (
    "mark_path_family_fire_skill",  # fable_kag_l2l multi-day clusters
    "mark_path_family_wait_skill",
    "award_self_protect",
    "kl_anchor_best",
    "hold_floor_inject",
    "strategy_aux_light",
)


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def meters(score: Dict[str, Any]) -> Dict[str, int]:
    return {
        "same_outcome": int(score["same_outcome"]),
        "policy_clear": int(score["policy_clear"]),
        "mark_would_take": int(score.get("mark_would_take") or 0),
        "n_breach": int(score["n_breach"]),
    }


def keep_gate(
    post: Dict[str, Any],
    pre: Dict[str, Any],
    *,
    baseline_clear: int = 27,
) -> Tuple[bool, str]:
    """Strict KEEP: same must rise, breach 0, clear ≥ baseline floor."""
    post_m = meters(post) if "same_outcome" in post and "rows" in post else post
    pre_m = meters(pre) if "same_outcome" in pre and ("rows" in pre or "mark_would_take" in pre) else pre
    # accept plain meters dicts
    def _g(d: Dict[str, Any], k: str, default: int = 0) -> int:
        return int(d.get(k, default))

    if _g(post_m, "n_breach") != 0:
        return False, "breach"
    if _g(post_m, "policy_clear") < int(baseline_clear):
        return False, "below_baseline_clear"
    if _g(post_m, "same_outcome") <= _g(pre_m, "same_outcome"):
        return False, "same_not_up"
    if _g(post_m, "policy_clear") < _g(pre_m, "policy_clear"):
        return False, "clear_down"
    return True, "same_up_breach0"


def dated_backup(policy: Channel1Policy, meters_d: Dict[str, int], note: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    same = meters_d["same_outcome"]
    bak_dir = os.path.join(OUT, f"backups/KEEP{same}__{stamp}")
    os.makedirs(bak_dir, exist_ok=True)
    pt = os.path.join(bak_dir, "mark_clone_full_obs_v1.pt")
    torch.save(
        {
            "tag": "mark_clone_full_obs_v1",
            "saved_at": _utcnow(),
            "state_dict": policy.state_dict(),
            "hidden": 128,
            "obs_dim": MARK_FULL_DIM,
            "multi_head": False,
            "meters": meters_d,
            "train_note": note,
            "source": "climb_35_combined_knowledge",
            "proven_touched": False,
            "shell_touched": False,
        },
        pt,
    )
    with open(os.path.join(bak_dir, "METERS.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "meters": meters_d,
                "note": note,
                "source": "climb_35_combined_knowledge",
                "ts": _utcnow(),
            },
            f,
            indent=2,
        )
    if os.path.isfile(MIX_RECIPE):
        shutil.copy2(MIX_RECIPE, os.path.join(bak_dir, "MIX_RECIPE_CLIMB35.md"))
    return bak_dir


def collect_strategy_aux_fire(
    days: Sequence[Tuple[str, Any]],
    *,
    max_days: int = 12,
    decide_every: int = 22,
    seed: int = 42,
    max_samples: int = 180,
    weight: float = 0.08,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """AUX: only directional strategy fires at low weight (no chop flood)."""
    ds_c = collect_continuation_dataset(
        days,
        max_days=max_days,
        decide_every=decide_every,
        full_obs=True,
        seed=seed,
        set_ids=ALL_SET_IDS,
        history_days=18,
        light_obs=True,
        neg_per_pos=0.5,
    )
    ds_r = collect_rsi_bb_dataset(
        days,
        max_days=max_days,
        decide_every=decide_every,
        full_obs=True,
        seed=seed + 1,
        set_ids=ALL_SET_IDS,
        history_days=min(80, max(0, len(list(days)) - 2)),
        light_obs=True,
        neg_per_pos=0.5,
    )
    xs: List[np.ndarray] = []
    ys: List[int] = []
    ws: List[float] = []
    for ds in (ds_c, ds_r):
        if int(ds.get("n") or 0) == 0:
            continue
        for i in range(int(ds["n"])):
            act = int(ds["y_act"][i])
            if act == ACTION_HOLD:
                continue  # no HOLD flood from strategies
            xs.append(ds["X"][i])
            ys.append(act)
            ws.append(float(weight))
    meta = {"n": len(ys), "cci_pos": ds_c.get("n_pos"), "rsi_pos": ds_r.get("n_pos"), "weight": weight}
    if not ys:
        return (
            np.zeros((0, MARK_FULL_DIM), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            meta,
        )
    if len(ys) > max_samples:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(ys), size=max_samples, replace=False)
        xs = [xs[i] for i in idx]
        ys = [ys[i] for i in idx]
        ws = [ws[i] for i in idx]
        meta["n"] = len(ys)
    return (
        np.stack(xs).astype(np.float32),
        np.asarray(ys, np.int64),
        np.asarray(ws, np.float32),
        meta,
    )


def collect_surgical_miss_fire(
    day_map: Dict[str, Any],
    mwt_rows: List[dict],
    awards: List[dict],
    policy: Channel1Policy,
    oracle: dict,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
    """Ultra-surgical: only miss_continuation disagrees (Mark fires, policy HOLD).

    Plus heavy award HOLD protect — pack-safe fire teaching without thrash.
    """
    xs, ys, ws = [], [], []
    n_miss = 0
    for row in mwt_rows:
        mark = get_plan(
            oracle, day_map, str(row["date"]), float(row["target_pct"]), float(row["risk_pct"])
        )
        if not mark or not mark.get("plan"):
            continue
        labs = walk_day_labels(
            day_map,
            str(row["date"]),
            float(row["target_pct"]),
            float(row["risk_pct"]),
            mark,
            policy,
            only_disagree=True,
        )
        for obs, ma, law, _fp, w in labs:
            if law != "miss_continuation" or int(ma) == ACTION_HOLD:
                continue
            # Mark fire labels on policy-path states only
            for _ in range(4):
                xs.append(obs)
                ys.append(int(ma))
                ws.append(float(w) * 1.5)
            n_miss += 1
    for row in awards[:36]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        for o, act, w in zip(a, b, c):
            xs.append(o)
            ys.append(int(act))
            # protect awards: HOLD higher
            ws.append(float(w) * (3.2 if int(act) == ACTION_HOLD else 2.0))
    meta = {
        "mode": "surgical_miss_fire",
        "n": len(ys),
        "n_miss_bars": n_miss,
        "dir": int(sum(1 for y in ys if y != 0)),
        "hold": int(sum(1 for y in ys if y == 0)),
    }
    if len(ys) < 40:
        return (
            np.zeros((0, MARK_FULL_DIM), np.float32),
            np.zeros((0,), np.int64),
            np.zeros((0,), np.float32),
            meta,
        )
    X = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, np.int64)
    w = np.asarray(ws, np.float32)
    w = np.maximum(w, 1e-6)
    w = w / float(w.mean())
    return X, y, w, meta


def inject_hold(
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    day_map: Dict[str, Any],
    awards: List[dict],
    policy: Channel1Policy,
    *,
    min_hold: float = 0.35,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    hold_frac = float((y == 0).mean()) if len(y) else 0.0
    if hold_frac >= min_hold:
        return X, y, w, hold_frac
    extra_x, extra_y, extra_w = [], [], []
    for row in awards[:30]:
        a, b, c = award_self(
            day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
        )
        for o, act, ww in zip(a, b, c):
            if int(act) == ACTION_HOLD:
                extra_x.append(o)
                extra_y.append(0)
                extra_w.append(float(ww) * 3.0)
    if not extra_x:
        return X, y, w, hold_frac
    X = np.concatenate([X, np.stack(extra_x)], axis=0)
    y = np.concatenate([y, np.asarray(extra_y, np.int64)])
    w = np.concatenate([w, np.asarray(extra_w, np.float32)])
    w = np.maximum(w, 1e-6)
    w = w / float(w.mean())
    return X, y, w, float((y == 0).mean())


def run_climb(
    *,
    max_rounds: int = 10,
    epochs: int = 18,
    kl_coef: float = 1.25,
    lr: float = 1.2e-4,
    strategy_weight: float = 0.08,
    seed: int = 42,
    scratch: Optional[str] = None,
    skip_strategy_aux: bool = False,
    warm_ckpt: Optional[str] = None,
    force_family: Optional[str] = None,
    write_live_on_keep: bool = True,
    surgical: bool = False,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    scratch = scratch or os.path.join(OUT, "scratch_climb")
    os.makedirs(scratch, exist_ok=True)

    print("=== climb_35_combined_knowledge (path-family + KL + strategy aux) ===", flush=True)
    print(f"sources={SOURCES}", flush=True)

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline.get("policy_clear", 27))
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    oracle = load_oracle()
    dials = clip_streak_dials(default_streak_dials())
    mem = load_pattern_mem()

    warm_path = warm_ckpt or CKPT
    print(f"  warm_ckpt={warm_path}", flush=True)
    policy = load_policy(warm_path)
    assert not getattr(policy, "multi_head", False), "embryo must be single-head"
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}

    print("PRE-score frozen 50d (score_policy + baseline mark_rows)…", flush=True)
    pre_score = score_policy(policy, day_map, mark_rows)
    pre_m = meters(pre_score)
    print(
        f"  PRE same={pre_m['same_outcome']} policy={pre_m['policy_clear']} "
        f"mwt={pre_m['mark_would_take']} breach={pre_m['n_breach']}",
        flush=True,
    )
    with open(os.path.join(scratch, "climb_pre.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "phase": "PRE_TRAIN",
                "meters": pre_m,
                "sources": list(SOURCES),
                "path": "climb_35_combined_knowledge",
                "ts": _utcnow(),
            },
            f,
            indent=2,
        )
    if pre_m["n_breach"] != 0:
        return {"ok": False, "error": "pre_breach", "pre": pre_m, "kept": False}

    # optional strategy aux once
    strat_X = strat_y = strat_w = None
    strat_meta: Dict[str, Any] = {"n": 0, "skipped": skip_strategy_aux}
    if not skip_strategy_aux:
        print("Collecting light strategy FIRE aux…", flush=True)
        strat_X, strat_y, strat_w, strat_meta = collect_strategy_aux_fire(
            all_days,
            max_days=12,
            decide_every=22,
            seed=seed,
            max_samples=160,
            weight=strategy_weight,
        )
        print(f"  strategy fire aux n={strat_meta.get('n')}", flush=True)

    best = pre_score
    live_floor = int(pre_m["same_outcome"])
    cycles: List[Dict[str, Any]] = [
        {"cycle": 0, "decision": "PRE", "meters": pre_m, "phase": "baseline"}
    ]
    kept = False
    keep_backup = None
    mwt = [r for r in pre_score["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
    awards = [r for r in pre_score["rows"] if r["miss_class"] == "AWARD"]

    for rnd in range(1, max_rounds + 1):
        policy.load_state_dict(best_state)
        policy.eval()
        mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
        awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
        if not mwt:
            print("no MWT left", flush=True)
            break

        # Probe laws for family choice (Fable ONE intervention)
        law_probe: Counter = Counter()
        for row in mwt[:10]:
            mark = get_plan(
                oracle, day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"])
            )
            if not mark:
                continue
            labs = walk_day_labels(
                day_map,
                row["date"],
                float(row["target_pct"]),
                float(row["risk_pct"]),
                mark,
                policy,
            )
            for _o, _ma, law, _fp, _w in labs:
                law_probe[law] += 1
        # Climb order: fire_skill first (MWT = miss fire); skip no_invent early —
        # it already cratered pack (36→29 breach) on round-1 probes.
        if surgical:
            family = "surgical_miss_fire"
            print(
                f"\n===== COMBINED {rnd}/{max_rounds} family={family} "
                f"(miss_continuation only + award protect) =====",
                flush=True,
            )
            X, y, w, meta = collect_surgical_miss_fire(
                day_map, mwt, awards, policy, oracle
            )
            print(f"  surgical meta={meta}", flush=True)
            freeze = True
            use_kl = max(float(kl_coef), 1.4)
            use_epochs = min(int(epochs), 14)
            use_lr = min(float(lr), 1.0e-4)
        else:
            if force_family:
                family = str(force_family)
            else:
                family = choose_family(law_probe, mem + cycles, round_i=rnd)
                forced = ["fire_skill", "fire_skill", "all", "wait_skill", "fire_skill"]
                if rnd <= len(forced):
                    family = forced[rnd - 1]
                if family == "no_invent" and rnd <= 6:
                    family = "fire_skill"
            print(
                f"\n===== COMBINED {rnd}/{max_rounds} family={family} "
                f"probe={dict(law_probe.most_common(5))} =====",
                flush=True,
            )
            X, y, w, meta = collect_cluster_labels(
                day_map, mwt, awards, policy, oracle, target_family=family
            )
            print(f"  cluster meta={meta}", flush=True)
            if meta["n"] < 40:
                X, y, w, meta = collect_cluster_labels(
                    day_map, mwt, awards, policy, oracle, target_family="all"
                )
                family = "all"
                print(f"  fallback all meta={meta}", flush=True)
            freeze = family in ("wait_skill", "no_invent")
            use_kl = float(kl_coef)
            use_epochs = int(epochs)
            use_lr = float(lr)

        if meta["n"] < 40:
            print("  sparse — skip round", flush=True)
            continue

        # HOLD floor inject
        X, y, w, hold_frac = inject_hold(
            X, y, w, day_map, awards, policy, min_hold=0.45 if surgical else 0.35
        )
        print(f"  hold_frac={hold_frac:.3f} n={len(y)}", flush=True)

        # blend strategy fire aux at low mass (never in surgical mode)
        if (
            not surgical
            and strat_X is not None
            and len(strat_y) > 0
            and family in ("fire_skill", "all")
        ):
            X = np.concatenate([X, strat_X], axis=0)
            y = np.concatenate([y, strat_y], axis=0)
            w = np.concatenate([w, strat_w], axis=0)
            w = np.maximum(w, 1e-6)
            w = w / float(w.mean())
            print(f"  +strategy fire aux → n={len(y)}", flush=True)

        hid = 128
        w0 = best_state.get("net.0.weight")
        if w0 is not None:
            hid = int(w0.shape[0])

        pol2, losses = train_bc(
            X,
            y,
            sample_weights=w,
            epochs=use_epochs,
            lr=use_lr,
            hidden=hid,
            seed=seed + rnd,
            warm_state=best_state,
            kl_anchor_state=best_state,
            kl_coef=use_kl,
            freeze_trunk=freeze,
            multi_head=False,
            obs_dim=MARK_FULL_DIM,
        )
        with torch.no_grad():
            pred = pol2(torch.tensor(X, dtype=torch.float32)).argmax(dim=-1).numpy()
        act_match = float((pred == y).mean())
        print(
            f"  train act_match={act_match:.3f} loss={losses[-1] if losses else 0:.4f}",
            flush=True,
        )
        # learn≠copy heuristic from fable_kag_l2l
        if act_match > 0.95:
            print("  learn≠copy warn: very high act_match — still score pack", flush=True)

        torch.save(
            {
                "tag": "mark_clone_candidate_combined_v1",
                "saved_at": _utcnow(),
                "state_dict": pol2.state_dict(),
                "hidden": hid,
                "obs_dim": MARK_FULL_DIM,
                "multi_head": False,
                "family": family,
                "round": rnd,
                "proven_touched": False,
            },
            CANDIDATE,
        )

        print("  POST-score…", flush=True)
        post = score_policy(pol2, day_map, mark_rows)
        post_m = meters(post)
        print(
            f"  POST same={post_m['same_outcome']} policy={post_m['policy_clear']} "
            f"mwt={post_m['mark_would_take']} breach={post_m['n_breach']}",
            flush=True,
        )
        ok, reason = keep_gate(post, best, baseline_clear=floor_clear)
        # also require beat live floor
        if ok and int(post_m["same_outcome"]) <= live_floor:
            ok, reason = False, "not_above_live_floor"
        decision = "KEEP" if ok else "REJECT"
        print(f"  DECISION {decision} ({reason})", flush=True)

        row = {
            "cycle": rnd,
            "family": family,
            "decision": decision,
            "reason": reason,
            "meters": post_m,
            "meta": meta,
            "act_match": act_match,
            "sources": list(SOURCES),
            "ts": _utcnow(),
        }
        cycles.append(row)

        if decision == "KEEP":
            if write_live_on_keep:
                save_policy(
                    pol2,
                    note=f"climb_35_combined_KEEP_r{rnd}_{family}_same{post_m['same_outcome']}",
                    dials=dials,
                )
            else:
                # still persist candidate as the KEEP embryo artifact
                torch.save(
                    {
                        "tag": "mark_clone_full_obs_v1",
                        "saved_at": _utcnow(),
                        "state_dict": pol2.state_dict(),
                        "hidden": hid,
                        "obs_dim": MARK_FULL_DIM,
                        "multi_head": False,
                        "train_note": f"KEEP candidate r{rnd}_{family}",
                        "proven_touched": False,
                    },
                    CANDIDATE,
                )
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post_m["same_outcome"]))
            keep_backup = dated_backup(
                pol2,
                post_m,
                note=f"climb_35_combined_knowledge KEEP r{rnd} family={family}",
            )
            best_blob = {
                "same_outcome": post_m["same_outcome"],
                "policy_clear": post_m["policy_clear"],
                "mwt": post_m["mark_would_take"],
                "breach": post_m["n_breach"],
                "source": f"climb_35_combined_knowledge_KEEP_{family}_r{rnd}",
                "stage": "teen" if post_m["same_outcome"] > 35 else "child",
                "growth_method": "combined_path_family_kl_strategy_aux",
                "core_skill": f"pattern family {family} multi-day + pack protect",
                "prior_same": pre_m["same_outcome"],
                "backup": keep_backup,
                "multi_head": False,
                "warm_ckpt": warm_path,
                "ts": _utcnow(),
            }
            if write_live_on_keep:
                with open(BEST_JSON, "w", encoding="utf-8") as f:
                    json.dump(best_blob, f, indent=2)
            else:
                with open(os.path.join(scratch, "BEST_AFTER_KEEP.json"), "w", encoding="utf-8") as f:
                    json.dump(best_blob, f, indent=2)
            kept = True
            print(f"  KEEP backup={keep_backup}", flush=True)
            # write honest POST after KEEP
            with open(os.path.join(scratch, "climb_post.json"), "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "phase": "POST_TRAIN_KEEP",
                        "meters": post_m,
                        "pre": pre_m,
                        "kept": True,
                        "family": family,
                        "round": rnd,
                        "backup": keep_backup,
                        "path": "climb_35_combined_knowledge",
                        "ts": _utcnow(),
                    },
                    f,
                    indent=2,
                )
            break  # one successful KEEP is the goal bar
        else:
            policy.load_state_dict(best_state)

    save_oracle(oracle)
    final_m = meters(best)
    if not kept:
        with open(os.path.join(scratch, "climb_post.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "phase": "POST_TRAIN_NO_KEEP",
                    "meters": final_m,
                    "pre": pre_m,
                    "kept": False,
                    "note": "BEST unchanged; all rounds REJECT",
                    "path": "climb_35_combined_knowledge",
                    "ts": _utcnow(),
                },
                f,
                indent=2,
            )

    report = {
        "ts": _utcnow(),
        "path": "climb_35_combined_knowledge",
        "pre": pre_m,
        "final": final_m,
        "kept": kept,
        "backup": keep_backup,
        "cycles": cycles,
        "sources": list(SOURCES),
        "strategy_aux": strat_meta,
        "baseline_clear_floor": floor_clear,
        "ckpt": CKPT,
        "candidate": CANDIDATE,
        "mix_recipe": MIX_RECIPE,
        "ban": "strategy_only_multi_head_full_replace",
        "proven_touched": False,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    with open(os.path.join(scratch, "climb_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(
        f"=== DONE kept={kept} final same={final_m['same_outcome']} "
        f"(pre {pre_m['same_outcome']}) breach={final_m['n_breach']} ===",
        flush=True,
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode",
        type=str,
        default="path_family",
        choices=["path_family", "distill_teen", "blend_teen"],
        help="path_family|distill_teen|blend_teen (CHILD floor + TEEN fire_skill fusion)",
    )
    ap.add_argument("--max-rounds", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=18)
    ap.add_argument("--kl-coef", type=float, default=1.25)
    ap.add_argument("--lr", type=float, default=1.2e-4)
    ap.add_argument("--strategy-weight", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--scratch", type=str, default="")
    ap.add_argument("--skip-strategy-aux", action="store_true")
    ap.add_argument(
        "--warm-ckpt",
        type=str,
        default="",
        help="optional embryo path (e.g. CHILD_STAGE_same35_*.pt) instead of live BEST",
    )
    ap.add_argument(
        "--force-family",
        type=str,
        default="",
        help="force path family: fire_skill|wait_skill|all|no_invent",
    )
    ap.add_argument(
        "--no-write-live",
        action="store_true",
        help="KEEP writes backup+candidate only (does not overwrite live BEST/ckpt)",
    )
    ap.add_argument(
        "--surgical",
        action="store_true",
        help="miss_continuation-only + award protect + freeze_trunk (pack-safe micro climb)",
    )
    args = ap.parse_args(list(argv) if argv is not None else None)
    scratch = args.scratch or os.environ.get(
        "CLIMB_SCRATCH",
        os.path.join(OUT, "scratch_climb"),
    )
    if args.mode == "distill_teen":
        from lineages.adaptive_rl_brain_7_31_26 import climb_35_distill_teen as _dist

        os.environ["CLIMB_SCRATCH"] = scratch
        sys.argv = [sys.argv[0], f"--scratch={scratch}"]
        return int(_dist.main())
    if args.mode == "blend_teen":
        from lineages.adaptive_rl_brain_7_31_26 import climb_35_blend_teen as _blend

        os.environ["CLIMB_SCRATCH"] = scratch
        return int(_blend.main())

    report = run_climb(
        max_rounds=args.max_rounds,
        epochs=args.epochs,
        kl_coef=args.kl_coef,
        lr=args.lr,
        strategy_weight=args.strategy_weight,
        seed=args.seed,
        scratch=scratch,
        skip_strategy_aux=bool(args.skip_strategy_aux),
        warm_ckpt=args.warm_ckpt or None,
        force_family=args.force_family or None,
        write_live_on_keep=not bool(args.no_write_live),
        surgical=bool(args.surgical),
    )
    if report.get("error"):
        return 2
    return 0 if report.get("kept") else 1


if __name__ == "__main__":
    raise SystemExit(main())
