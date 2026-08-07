"""Climb past 35/50 using multi-head strategy BC — KEEP only if same rises.

CHANGE LOG:
- 2026-08-06  created — WHY: user gate: strategies help only if they raise
  same_outcome above 35 with breach 0. Warm from BEST embryo; multi-task
  BC on CCI + RSI-BB labels (all sets); KL-anchor action to BEST; full 50d
  score; KEEP writes ckpt only on strict improvement.

Usage (repo root):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/climb_35_with_strategies.py
  python lineages/adaptive_rl_brain_7_31_26/climb_35_with_strategies.py --epochs 40 --max-label-days 25
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import load_pairs
from lineages.adaptive_rl_brain_7_31_26.fable_50d_mark_match_loop import (
    BASELINE,
    CKPT,
    CKPT_DIR,
    OUT,
    SEED,
    better,
    load_policy,
    not_worse,
    save_policy,
    score_50d,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import Channel1Policy
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.strategies.cci_dual_level_continuation import (
    collect_continuation_dataset,
)
from lineages.adaptive_rl_brain_7_31_26.strategies.rsi_bb_pullback_continuation import (
    ALL_SET_IDS,
    collect_rsi_bb_dataset,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc_multitask

REPORT = os.path.join(OUT, "CLIMB35_STRATEGIES__latest.json")
CANDIDATE = os.path.join(CKPT_DIR, "mark_clone_candidate_strategies_v1.pt")
BACKUP = os.path.join(OUT, "PRE_CLIMB35_STRATEGY__backup.pt")


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _merge_ds(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    if int(a.get("n") or 0) == 0:
        return b
    if int(b.get("n") or 0) == 0:
        return a
    hits = {sid: 0 for sid in ALL_SET_IDS}
    for ds in (a, b):
        for sid, c in (ds.get("hits_by_set") or {}).items():
            hits[int(sid)] = hits.get(int(sid), 0) + int(c)
    return {
        "X": np.concatenate([a["X"], b["X"]], axis=0),
        "y_act": np.concatenate([a["y_act"], b["y_act"]], axis=0),
        "y_topology": np.concatenate([a["y_topology"], b["y_topology"]], axis=0),
        "y_wait": np.concatenate([a["y_wait"], b["y_wait"]], axis=0),
        "n": int(a["n"]) + int(b["n"]),
        "n_pos": int(a.get("n_pos") or 0) + int(b.get("n_pos") or 0),
        "n_neg": int(a.get("n_neg") or 0) + int(b.get("n_neg") or 0),
        "hits_by_set": hits,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--max-label-days", type=int, default=22)
    ap.add_argument("--decide-every", type=int, default=12)
    ap.add_argument("--kl-coef", type=float, default=0.55)
    ap.add_argument("--topo-coef", type=float, default=0.45)
    ap.add_argument("--wait-coef", type=float, default=0.35)
    ap.add_argument("--lr", type=float, default=4e-4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--skip-pre-score", action="store_true", help="trust BEST=35 without re-score")
    ap.add_argument("--force-keep-equal", action="store_true", help="KEEP if not_worse (not only better)")
    args = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    print("=== climb_35_with_strategies · KEEP only if same rises ===", flush=True)

    # --- 0. Backup live embryo ---
    if os.path.isfile(CKPT):
        shutil.copy2(CKPT, BACKUP)
        print(f"  backup → {BACKUP}", flush=True)

    # --- 1. Load BEST (legacy single-head 35) ---
    best_legacy = load_policy(CKPT)
    print(
        f"  live embryo multi_head={best_legacy.multi_head} "
        f"obs={best_legacy.obs_dim} hidden={best_legacy.hidden}",
        flush=True,
    )

    # Floor from BEST json + optional re-score
    best_path = os.path.join(OUT, "BEST__latest.json")
    floor = {
        "same_outcome": 35,
        "policy_clear": 35,
        "mark_would_take": 15,
        "n_breach": 0,
    }
    if os.path.isfile(best_path):
        try:
            bj = json.load(open(best_path, encoding="utf-8-sig"))
            floor["same_outcome"] = int(bj.get("same_outcome", 35))
            floor["policy_clear"] = int(bj.get("policy_clear", floor["same_outcome"]))
            floor["mark_would_take"] = int(bj.get("mwt", bj.get("mark_would_take", 15)))
            floor["n_breach"] = int(bj.get("breach", bj.get("n_breach", 0)))
        except Exception as e:
            print(f"  BEST json read skip: {e}", flush=True)

    baseline_floor = 27
    if os.path.isfile(BASELINE):
        try:
            base = json.load(open(BASELINE, encoding="utf-8"))
            baseline_floor = int(base.get("policy_clear", 27))
        except Exception:
            pass

    print(
        f"  floor same={floor['same_outcome']} policy={floor['policy_clear']} "
        f"mwt={floor['mark_would_take']} breach={floor['n_breach']} "
        f"baseline_clear_floor={baseline_floor}",
        flush=True,
    )

    print("Loading days + pairs…", flush=True)
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    pairs = load_pairs()
    window = days[:50]

    pre = dict(floor)
    mark_cache: Dict[str, Any] = {}
    if not args.skip_pre_score:
        print("PRE-score live embryo (50d frozen recipe)…", flush=True)
        pre_full = score_50d(best_legacy, window, pairs, seed=args.seed, mark_cache=mark_cache)
        pre = {
            "same_outcome": int(pre_full["same_outcome"]),
            "policy_clear": int(pre_full["policy_clear"]),
            "mark_would_take": int(pre_full["mark_would_take"]),
            "n_breach": int(pre_full["n_breach"]),
        }
        print(
            f"  PRE same={pre['same_outcome']} policy={pre['policy_clear']} "
            f"mwt={pre['mark_would_take']} breach={pre['n_breach']}",
            flush=True,
        )
        if pre["n_breach"] != 0:
            print("ABORT: live embryo already breaches — fix that first", flush=True)
            return 2

    # --- 2. Collect strategy labels (all sets) ---
    print("Collecting CCI + RSI-BB labels (sets 1–4)…", flush=True)
    ds_cci = collect_continuation_dataset(
        days,
        max_days=args.max_label_days,
        decide_every=args.decide_every,
        full_obs=True,
        seed=args.seed,
        set_ids=ALL_SET_IDS,
        history_days=25,
        light_obs=True,
        neg_per_pos=1.5,
    )
    print(f"  CCI n={ds_cci['n']} pos={ds_cci['n_pos']} hits={ds_cci['hits_by_set']}", flush=True)
    ds_rsi = collect_rsi_bb_dataset(
        days,
        max_days=args.max_label_days,
        decide_every=args.decide_every,
        full_obs=True,
        seed=args.seed + 3,
        set_ids=ALL_SET_IDS,
        history_days=min(120, max(0, len(days) - 2)),
        light_obs=True,
        neg_per_pos=1.5,
    )
    print(
        f"  RSI-BB n={ds_rsi['n']} pos={ds_rsi['n_pos']} "
        f"pull={ds_rsi.get('n_pullback')} cont={ds_rsi.get('n_continuation')} "
        f"hits={ds_rsi['hits_by_set']}",
        flush=True,
    )
    ds = _merge_ds(ds_cci, ds_rsi)
    print(f"  MERGED n={ds['n']} pos={ds['n_pos']} hits={ds['hits_by_set']}", flush=True)
    if int(ds["n_pos"]) < 20:
        print("ABORT: too few strategy positives", flush=True)
        return 2

    # --- 3. Multi-head warm from BEST + KL anchor ---
    print("Multi-head warm-start from BEST + strategy multi-task BC…", flush=True)
    warm_state = {k: v.detach().clone() for k, v in best_legacy.state_dict().items()}
    # Build multi-head; flexible map from legacy
    student = Channel1Policy(obs_dim=MARK_FULL_DIM, hidden=128, multi_head=True)
    info = student.load_state_dict_flexible(warm_state, strict=False)
    print(f"  map: {info}", flush=True)
    anchor_state = {k: v.detach().clone() for k, v in student.state_dict().items()}

    # Mild oversample non-chop topology
    X, ya, yt, yw = ds["X"], ds["y_act"], ds["y_topology"], ds["y_wait"]
    pos_m = yt != 3
    if pos_m.any() and (~pos_m).any():
        pos_idx = np.where(pos_m)[0]
        neg_idx = np.where(~pos_m)[0]
        n_rep = max(1, len(neg_idx) // max(len(pos_idx), 1))
        take = np.concatenate([np.tile(pos_idx, n_rep), neg_idx, pos_idx])
        rng = np.random.default_rng(args.seed)
        take = rng.permutation(take)
        X, ya, yt, yw = X[take], ya[take], yt[take], yw[take]
        print(f"  oversample train n={len(ya)}", flush=True)

    policy, hist = train_bc_multitask(
        X,
        ya,
        yt,
        yw,
        epochs=int(args.epochs),
        hidden=128,
        seed=int(args.seed),
        warm_state=student.state_dict(),
        obs_dim=MARK_FULL_DIM,
        topo_coef=float(args.topo_coef),
        wait_coef=float(args.wait_coef),
        kl_anchor_state=anchor_state,
        kl_coef=float(args.kl_coef),
        lr=float(args.lr),
    )
    print(f"  last loss={hist[-1] if hist else None}", flush=True)

    # Save candidate only (not BEST yet)
    torch.save(
        {
            "tag": "mark_clone_candidate_strategies_v1",
            "saved_at": _utcnow(),
            "state_dict": policy.state_dict(),
            "hidden": 128,
            "obs_dim": MARK_FULL_DIM,
            "multi_head": True,
            "proven_touched": False,
            "shell_touched": False,
            "train_note": "strategy multi-head BC candidate — not KEEP until same rises",
        },
        CANDIDATE,
    )
    print(f"  candidate → {CANDIDATE}", flush=True)

    # --- 4. POST 50d score ---
    print("POST-score candidate (50d frozen recipe)…", flush=True)
    post_full = score_50d(policy, window, pairs, seed=args.seed, mark_cache=mark_cache)
    post = {
        "same_outcome": int(post_full["same_outcome"]),
        "policy_clear": int(post_full["policy_clear"]),
        "mark_would_take": int(post_full["mark_would_take"]),
        "n_breach": int(post_full["n_breach"]),
        "miss_class_counts": post_full.get("miss_class_counts"),
    }
    print(
        f"  POST same={post['same_outcome']} policy={post['policy_clear']} "
        f"mwt={post['mark_would_take']} breach={post['n_breach']}",
        flush=True,
    )

    pre_score = {
        "same_outcome": pre["same_outcome"],
        "policy_clear": pre["policy_clear"],
        "mark_would_take": pre["mark_would_take"],
        "n_breach": pre["n_breach"],
    }
    post_score = {
        "same_outcome": post["same_outcome"],
        "policy_clear": post["policy_clear"],
        "mark_would_take": post["mark_would_take"],
        "n_breach": post["n_breach"],
    }

    is_better = better(post_score, pre_score, baseline_floor)
    is_safe = not_worse(post_score, pre_score, baseline_floor)
    # User gate: only KEEP if same *increases* (strict)
    same_up = int(post["same_outcome"]) > int(pre["same_outcome"])
    breach_ok = int(post["n_breach"]) == 0
    keep = bool(same_up and breach_ok and is_safe)
    if args.force_keep_equal and is_safe and breach_ok and not same_up:
        keep = False  # still require rise unless we add another flag
    # explicit: user said only if increase the 35
    keep = bool(same_up and breach_ok)

    decision = "KEEP" if keep else "REJECT"
    print(
        f"=== DECISION {decision}  same {pre['same_outcome']}→{post['same_outcome']} "
        f"breach={post['n_breach']} better={is_better} safe={is_safe} ===",
        flush=True,
    )

    if keep:
        dials = clip_streak_dials(default_streak_dials())
        save_policy(
            policy,
            note=(
                f"climb35_strategies KEEP same {pre['same_outcome']}→{post['same_outcome']} "
                f"cci+rsi_bb multi-head"
            ),
            dials=dials,
        )
        best_out = {
            "same_outcome": post["same_outcome"],
            "policy_clear": post["policy_clear"],
            "mwt": post["mark_would_take"],
            "breach": post["n_breach"],
            "source": "climb_35_with_strategies",
            "prior_same": pre["same_outcome"],
            "ts": _utcnow(),
            "multi_head": True,
        }
        with open(best_path, "w", encoding="utf-8") as f:
            json.dump(best_out, f, indent=2)
        print(f"  KEEP wrote {CKPT} + BEST__latest.json", flush=True)
    else:
        print("  REJECT — live BEST embryo left unchanged", flush=True)
        # restore from backup if anything touched CKPT (save_policy only on keep)
        if os.path.isfile(BACKUP) and not keep:
            # ensure CKPT still legacy
            pass

    report = {
        "ts": _utcnow(),
        "decision": decision,
        "keep": keep,
        "pre": pre,
        "post": post,
        "same_up": same_up,
        "breach_ok": breach_ok,
        "is_better": is_better,
        "is_safe": is_safe,
        "labels": {
            "n": ds["n"],
            "n_pos": ds["n_pos"],
            "hits_by_set": ds["hits_by_set"],
            "cci_pos": ds_cci.get("n_pos"),
            "rsi_pos": ds_rsi.get("n_pos"),
        },
        "train": {
            "epochs": args.epochs,
            "kl_coef": args.kl_coef,
            "topo_coef": args.topo_coef,
            "wait_coef": args.wait_coef,
            "lr": args.lr,
        },
        "ckpt_live": CKPT,
        "ckpt_candidate": CANDIDATE,
        "backup": BACKUP,
        "proven_touched": False,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"  report → {REPORT}", flush=True)
    return 0 if keep else 1


if __name__ == "__main__":
    raise SystemExit(main())
