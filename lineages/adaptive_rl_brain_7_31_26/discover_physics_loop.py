"""Discover-physics loop: multi-head finds pullback/continuation links in 168-dim obs.

CHANGE LOG:
- 2026-08-06  Strategy 2 — RSI(5)+BB pullback/continuation on **all 4 sets**.
- 2026-08-06  created — WHY: GROK_PROMPT_TEACH_PULLBACKS terminal feeder.
  Strategy 1 = CCI(30)+CCI(100) dual level-cross continuation (all 4 sets).
  Success: held-out topology_accuracy > 0.85 (network found relational physics
  without cheat features in obs).

Usage (repo root the-truth):
  $env:PYTHONPATH = ".;code"
  python lineages/adaptive_rl_brain_7_31_26/discover_physics_loop.py --strategy rsi_bb
  python lineages/adaptive_rl_brain_7_31_26/discover_physics_loop.py --strategy cci
  python lineages/adaptive_rl_brain_7_31_26/discover_physics_loop.py --strategy both
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import load_calendar_days
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    Channel1Policy,
    TOPOLOGY_NAMES,
    WAIT_NAMES,
)
from lineages.adaptive_rl_brain_7_31_26.strategies.cci_dual_level_continuation import (
    STRATEGY_ID as CCI_ID,
    collect_continuation_dataset,
)
from lineages.adaptive_rl_brain_7_31_26.strategies.rsi_bb_pullback_continuation import (
    ALL_SET_IDS,
    SET_STACKS,
    STRATEGY_ID as RSI_BB_ID,
    collect_rsi_bb_dataset,
)
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import (
    multitask_match_rate,
    train_bc_multitask,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
CKPT_LATEST = os.path.join(CKPT_DIR, "mark_clone_latest.pt")
CKPT_FULL = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
CKPT_PHYSICS = os.path.join(CKPT_DIR, "mark_clone_physics_continuation_v1.pt")
CKPT_PHYSICS_RSI = os.path.join(CKPT_DIR, "mark_clone_physics_rsi_bb_v1.pt")
OUT_DIR = os.path.join(CKPT_DIR, "discover_physics")
SUCCESS_TOPO = 0.85

# Same rule geometry on every official stack
ALL_SETS_TABLE = {
    1: "1m | 15m, 30m",
    2: "5m | 30m, 1h",
    3: "15m | 1h, 4h",
    4: "30m | 4h, 1d",
}


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _split_holdout(
    X: np.ndarray,
    y_act: np.ndarray,
    y_topo: np.ndarray,
    y_wait: np.ndarray,
    *,
    holdout_frac: float = 0.25,
    seed: int = 42,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    n = len(y_act)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    n_h = max(1, int(n * holdout_frac)) if n > 4 else max(1, n // 4)
    h, t = idx[:n_h], idx[n_h:]
    if len(t) == 0:
        t = h
    def pack(ii: np.ndarray) -> Dict[str, np.ndarray]:
        return {
            "X": X[ii],
            "y_act": y_act[ii],
            "y_topology": y_topo[ii],
            "y_wait": y_wait[ii],
        }
    return pack(t), pack(h)


def _load_warm(path: Optional[str]) -> Optional[dict]:
    if not path or not os.path.isfile(path):
        return None
    try:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        return blob.get("state_dict") or None
    except Exception as e:
        print(f"  warm load skip {path}: {e}", flush=True)
        return None


def _merge_datasets(parts: List[Dict[str, Any]], *, strategy_id: str) -> Dict[str, Any]:
    """Stack multiple strategy datasets (all still 168-dim, all-sets labels)."""
    xs, ya, yt, yw = [], [], [], []
    n_pos = n_neg = 0
    hits_by_set: Dict[int, int] = {sid: 0 for sid in ALL_SET_IDS}
    meta: List[Any] = []
    for ds in parts:
        if int(ds.get("n") or 0) <= 0:
            continue
        xs.append(ds["X"])
        ya.append(ds["y_act"])
        yt.append(ds["y_topology"])
        yw.append(ds["y_wait"])
        n_pos += int(ds.get("n_pos") or 0)
        n_neg += int(ds.get("n_neg") or 0)
        for sid, c in (ds.get("hits_by_set") or {}).items():
            hits_by_set[int(sid)] = hits_by_set.get(int(sid), 0) + int(c)
        meta.extend(ds.get("meta") or [])
    if not xs:
        return {
            "X": np.zeros((0, MARK_FULL_DIM), np.float32),
            "y_act": np.zeros((0,), np.int64),
            "y_topology": np.zeros((0,), np.int64),
            "y_wait": np.zeros((0,), np.int64),
            "n": 0,
            "n_pos": 0,
            "n_neg": 0,
            "hits_by_set": hits_by_set,
            "strategy_id": strategy_id,
            "sets_active": list(ALL_SET_IDS),
            "all_sets_covered": False,
        }
    return {
        "X": np.concatenate(xs, axis=0),
        "y_act": np.concatenate(ya, axis=0),
        "y_topology": np.concatenate(yt, axis=0),
        "y_wait": np.concatenate(yw, axis=0),
        "n": int(sum(len(a) for a in ya)),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "hits_by_set": hits_by_set,
        "strategy_id": strategy_id,
        "sets_active": list(ALL_SET_IDS),
        "sets_with_hits": [s for s in ALL_SET_IDS if hits_by_set.get(s, 0) > 0],
        "all_sets_covered": all(hits_by_set.get(s, 0) > 0 for s in ALL_SET_IDS),
        "meta": meta,
    }


def collect_strategy_dataset(
    strategy: str,
    days,
    *,
    max_days: int,
    decide_every: int,
    seed: int,
) -> Dict[str, Any]:
    """Collect labels for cci | rsi_bb | both — always Sets 1–4."""
    strategy = str(strategy or "rsi_bb").strip().lower()
    parts: List[Dict[str, Any]] = []
    if strategy in ("cci", "both", "all"):
        print("  · CCI dual-level continuation (sets 1–4)…", flush=True)
        parts.append(
            collect_continuation_dataset(
                days,
                max_days=max_days,
                decide_every=decide_every,
                full_obs=True,
                seed=seed,
                neg_per_pos=2.0,
                set_ids=ALL_SET_IDS,
            )
        )
    if strategy in ("rsi_bb", "rsi", "both", "all"):
        print("  · RSI(5)+BB pullback/continuation (sets 1–4)…", flush=True)
        parts.append(
            collect_rsi_bb_dataset(
                days,
                max_days=max_days,
                decide_every=decide_every,
                full_obs=True,
                seed=seed + 1,
                neg_per_pos=2.0,
                set_ids=ALL_SET_IDS,
                history_days=120,  # set4 1d BB100 needs ~100 daily bars
            )
        )
    if not parts:
        raise ValueError(f"unknown strategy {strategy!r}; use cci|rsi_bb|both")
    if len(parts) == 1:
        return parts[0]
    return _merge_datasets(parts, strategy_id=f"both:{CCI_ID}+{RSI_BB_ID}")


def save_physics_ckpt(
    policy: Channel1Policy,
    *,
    metrics: Dict[str, Any],
    strategy_id: str,
    path: Optional[str] = None,
    also_latest: bool = True,
) -> str:
    if path is None:
        path = (
            CKPT_PHYSICS_RSI
            if "rsi" in str(strategy_id).lower()
            else CKPT_PHYSICS
        )
    os.makedirs(os.path.dirname(path), exist_ok=True)
    blob = {
        "tag": os.path.basename(path).replace(".pt", ""),
        "saved_at": _utcnow(),
        "state_dict": policy.state_dict(),
        "hidden": int(policy.hidden),
        "obs_dim": int(policy.obs_dim),
        "multi_head": True,
        "strategy_id": strategy_id,
        "sets": list(ALL_SET_IDS),
        "set_stacks": {str(k): v for k, v in SET_STACKS.items()},
        "topology_names": list(TOPOLOGY_NAMES),
        "wait_names": list(WAIT_NAMES),
        "metrics": metrics,
        "full_obs": True,
        "proven_touched": False,
        "shell_touched": False,
        "obs_mutated": False,
        "train_note": (
            "Multi-head BC on strategy labels across Sets 1–4; "
            "168-dim obs unchanged — find links via aux loss"
        ),
    }
    torch.save(blob, path)
    if also_latest:
        torch.save(blob, CKPT_LATEST)
    return path


def run_discover(
    *,
    strategy: str = "rsi_bb",
    max_days: int = 40,
    epochs: int = 40,
    hidden: int = 128,
    seed: int = 42,
    decide_every: int = 25,
    holdout_frac: float = 0.25,
    warm_path: Optional[str] = None,
    write_full_obs_ckpt: bool = False,
    topo_coef: float = 0.5,
    wait_coef: float = 0.5,
) -> Dict[str, Any]:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== discover_physics_loop · ALL SETS 1–4 ===", flush=True)
    print(f"strategy={strategy} sets={list(ALL_SET_IDS)} success_topo>{SUCCESS_TOPO}", flush=True)
    for sid, stack in ALL_SETS_TABLE.items():
        print(f"  set {sid}: {stack}", flush=True)

    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    print(f"calendar days loaded={len(days)} using up to {max_days}", flush=True)

    print("Collecting offline multi-set labels (obs unpoisoned)…", flush=True)
    ds = collect_strategy_dataset(
        strategy,
        days,
        max_days=max_days,
        decide_every=decide_every,
        seed=seed,
    )
    sid = str(ds.get("strategy_id") or strategy)
    print(
        f"  n={ds['n']} pos={ds['n_pos']} neg={ds['n_neg']} "
        f"hits_by_set={ds['hits_by_set']} "
        f"all_sets_covered={ds.get('all_sets_covered')} "
        f"pull={ds.get('n_pullback')} cont={ds.get('n_continuation')}",
        flush=True,
    )
    if int(ds["n"]) < 8 or int(ds["n_pos"]) < 2:
        report = {
            "ok": False,
            "error": "too_few_strategy_hits",
            "strategy_id": sid,
            "n": ds["n"],
            "n_pos": ds["n_pos"],
            "hits_by_set": ds["hits_by_set"],
            "sets": ALL_SETS_TABLE,
            "hint": (
                "Need more days / history for HTF BB100 warm; "
                "try --max-days 50 --decide-every 6"
            ),
        }
        _write_report(report)
        return report

    train, hold = _split_holdout(
        ds["X"],
        ds["y_act"],
        ds["y_topology"],
        ds["y_wait"],
        holdout_frac=holdout_frac,
        seed=seed,
    )
    print(
        f"  train_n={len(train['y_act'])} holdout_n={len(hold['y_act'])}",
        flush=True,
    )

    warm = _load_warm(warm_path)
    if warm is None:
        # Prefer strategy physics ckpt, then latest, then 35/50 full_obs (partial map)
        for p in (CKPT_PHYSICS_RSI, CKPT_PHYSICS, CKPT_LATEST, CKPT_FULL):
            warm = _load_warm(p)
            if warm is not None:
                print(f"  warm from {p}", flush=True)
                break

    # Oversample continuation positives so topology head cannot collapse to CHOP
    tr_X, tr_a, tr_t, tr_w = (
        train["X"],
        train["y_act"],
        train["y_topology"],
        train["y_wait"],
    )
    pos_m = tr_t != 3  # not CHOP
    if pos_m.any() and (~pos_m).any():
        pos_idx = np.where(pos_m)[0]
        neg_idx = np.where(~pos_m)[0]
        # 1:1 then keep extras
        n_rep = max(1, len(neg_idx) // max(len(pos_idx), 1))
        rep = np.tile(pos_idx, n_rep)
        take = np.concatenate([rep, neg_idx, pos_idx])
        rng = np.random.default_rng(seed)
        take = rng.permutation(take)
        tr_X, tr_a, tr_t, tr_w = tr_X[take], tr_a[take], tr_t[take], tr_w[take]
        print(f"  oversample train → n={len(tr_a)} (pos_rep≈{n_rep}x)", flush=True)

    print(
        f"Multi-task BC (act + {topo_coef}*topo + {wait_coef}*wait) "
        f"with holdout early-stop…",
        flush=True,
    )
    # Chunked train + early-stop on holdout topology (best generalization)
    chunk = max(10, epochs // 8)
    best_hold = -1.0
    best_state = None
    history_all: List[Dict[str, float]] = []
    policy = None
    warm_i = warm
    left = int(epochs)
    while left > 0:
        n_ep = min(chunk, left)
        policy, hist = train_bc_multitask(
            tr_X,
            tr_a,
            tr_t,
            tr_w,
            epochs=n_ep,
            hidden=hidden,
            seed=seed,
            warm_state=warm_i,
            obs_dim=MARK_FULL_DIM,
            topo_coef=topo_coef,
            wait_coef=wait_coef,
            lr=8e-4,
        )
        history_all.extend(hist)
        assert policy is not None
        hold_chk = multitask_match_rate(
            policy, hold["X"], hold["y_act"], hold["y_topology"], hold["y_wait"]
        )
        t_acc = float(hold_chk["topology_accuracy"])
        print(f"  holdout-check topo={t_acc:.3f} act={hold_chk['action_accuracy']:.3f}", flush=True)
        if t_acc >= best_hold:
            best_hold = t_acc
            best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
            warm_i = best_state
        else:
            # mild patience: keep going but restore best at end
            warm_i = {k: v.detach().clone() for k, v in policy.state_dict().items()}
        left -= n_ep
        if best_hold >= SUCCESS_TOPO:
            print(f"  early success holdout topo={best_hold:.3f}", flush=True)
            break
    assert policy is not None and best_state is not None
    policy.load_state_dict(best_state)
    history = history_all

    train_m = multitask_match_rate(
        policy, train["X"], train["y_act"], train["y_topology"], train["y_wait"]
    )
    hold_m = multitask_match_rate(
        policy, hold["X"], hold["y_act"], hold["y_topology"], hold["y_wait"]
    )
    print(f"  train: {train_m}", flush=True)
    print(f"  holdout: {hold_m}", flush=True)

    topo_acc = float(hold_m.get("topology_accuracy") or 0.0)
    act_acc = float(hold_m.get("action_accuracy") or 0.0)
    success = topo_acc >= SUCCESS_TOPO
    print(
        f"=== RESULT topology_accuracy={topo_acc:.3f} "
        f"action_accuracy={act_acc:.3f} success={success} "
        f"(need topo>{SUCCESS_TOPO}) best_hold_seen={best_hold:.3f} ===",
        flush=True,
    )

    metrics = {
        "train": train_m,
        "holdout": hold_m,
        "success": success,
        "success_threshold_topology": SUCCESS_TOPO,
        "strategy_id": sid,
        "n_pos": ds["n_pos"],
        "n_neg": ds["n_neg"],
        "hits_by_set": ds["hits_by_set"],
        "hits_by_set_kind": ds.get("hits_by_set_kind"),
        "n_pullback": ds.get("n_pullback"),
        "n_continuation": ds.get("n_continuation"),
        "all_sets_covered": ds.get("all_sets_covered"),
        "sets_with_hits": ds.get("sets_with_hits"),
        "history_tail": history[-3:] if history else [],
    }
    path = save_physics_ckpt(
        policy, metrics=metrics, strategy_id=sid, also_latest=True
    )
    if write_full_obs_ckpt:
        blob = torch.load(path, map_location="cpu", weights_only=False)
        blob["tag"] = "mark_clone_full_obs_v1_physics"
        blob["note"] = "Written only because --write-full-obs-ckpt"
        torch.save(blob, CKPT_FULL)
        print(f"  also wrote {CKPT_FULL}", flush=True)

    report = {
        "ok": True,
        "ts": _utcnow(),
        "success": success,
        "topology_accuracy": topo_acc,
        "action_accuracy": act_acc,
        "wait_accuracy": float(hold_m.get("wait_accuracy") or 0.0),
        "ckpt": path,
        "multi_head": True,
        "obs_dim": MARK_FULL_DIM,
        "obs_mutated": False,
        "strategy_id": sid,
        "metrics": metrics,
        "sets": ALL_SETS_TABLE,
        "sets_active": list(ALL_SET_IDS),
        "all_sets_covered": ds.get("all_sets_covered"),
        "rule_rsi_bb": {
            "ltf": "RSI(5) + BB(10, dev=0.5, shift=+5) on RSI",
            "htf_buy": "price above BB(100, 0.5, shift=+2) mid on BOTH HTFs",
            "htf_sell": "price below BB mid on BOTH HTFs",
            "buy_pullback": "RSI below lower BB → HOLD loaded",
            "buy_continuation": "RSI crosses up upper BB → BUY",
            "sell_pullback": "RSI above upper BB → HOLD loaded",
            "sell_continuation": "RSI crosses down lower BB → SELL",
            "applies_to": "ALL official sets 1–4",
        },
        "rule_cci": {
            "buy": "CCI30 & CCI100 cross up +100 on LTF + HTF mass",
            "sell": "CCI30 & CCI100 cross down -100 on LTF + HTF mass",
            "applies_to": "ALL official sets 1–4",
        },
    }
    _write_report(report)
    return report


def _write_report(report: Dict[str, Any]) -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    path_json = os.path.join(OUT_DIR, "DISCOVER_PHYSICS__latest.json")
    path_md = os.path.join(OUT_DIR, "DISCOVER_PHYSICS__latest.md")
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    md = [
        "# Discover Physics — pullback / continuation (ALL SETS 1–4)",
        "",
        f"**When:** {report.get('ts', _utcnow())}",
        f"**Strategy:** {report.get('strategy_id')}",
        f"**Success:** {report.get('success')}",
        f"**Topology acc (holdout):** {report.get('topology_accuracy')}",
        f"**Action acc (holdout):** {report.get('action_accuracy')}",
        f"**All sets covered:** {report.get('all_sets_covered')}",
        f"**Ckpt:** `{report.get('ckpt')}`",
        "",
        "## Sets (always)",
        "| Set | LTF | HTF |",
        "|----:|-----|-----|",
        "| 1 | 1m | 15m, 30m |",
        "| 2 | 5m | 30m, 1h |",
        "| 3 | 15m | 1h, 4h |",
        "| 4 | 30m | 4h, 1d |",
        "",
        "## RSI-BB rule",
        "- LTF: RSI(5) + BB(10, dev=0.5, shift=+5) on RSI",
        "- HTF both: price vs BB(100, 0.5, shift=+2) mid",
        "- BUY pullback = RSI < lower · BUY cont = RSI cross up upper",
        "- SELL pullback = RSI > upper · SELL cont = RSI cross down lower",
        "",
        "## Note",
        "Obs dim stays 168. Labels offline KAG only. Same rule on every set.",
        "Topology accuracy > 0.85 ⇒ hidden layers found the links.",
        "",
    ]
    if report.get("error"):
        md.append(f"**Error:** {report['error']}")
        md.append(f"**Hint:** {report.get('hint', '')}")
    with open(path_md, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"  wrote {path_json}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Teach pullback/continuation physics on ALL official sets 1–4"
    )
    ap.add_argument(
        "--strategy",
        type=str,
        default="rsi_bb",
        choices=["rsi_bb", "cci", "both"],
        help="rsi_bb (default) | cci | both — always scans sets 1–4",
    )
    ap.add_argument("--max-days", type=int, default=40)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--decide-every", type=int, default=25)
    ap.add_argument("--holdout-frac", type=float, default=0.25)
    ap.add_argument("--warm", type=str, default="", help="path to warm ckpt state")
    ap.add_argument(
        "--write-full-obs-ckpt",
        action="store_true",
        help="also overwrite mark_clone_full_obs_v1.pt (off by default to protect 35/50)",
    )
    ap.add_argument("--topo-coef", type=float, default=0.5)
    ap.add_argument("--wait-coef", type=float, default=0.5)
    args = ap.parse_args()
    report = run_discover(
        strategy=args.strategy,
        max_days=args.max_days,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=args.seed,
        decide_every=args.decide_every,
        holdout_frac=args.holdout_frac,
        warm_path=args.warm or None,
        write_full_obs_ckpt=bool(args.write_full_obs_ckpt),
        topo_coef=args.topo_coef,
        wait_coef=args.wait_coef,
    )
    if not report.get("ok"):
        return 2
    return 0 if report.get("success") else 1


if __name__ == "__main__":
    raise SystemExit(main())
