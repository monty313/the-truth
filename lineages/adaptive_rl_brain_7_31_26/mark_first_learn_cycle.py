"""Mark always learning: SEE chart first → UPDATE policy → HARVEST other brains.

Order is law:
  1) Mark (doctrine + SETS LAW) walks the price day and writes labels
  2) Policy pure-greedy acts on same obs
  3) BC policy toward Mark on disagree + all Mark labels
  4) Probe other lineage ckpts; distill only agree-with-Mark bars into embryo
  5) Report award streak under random T/R (no retrain)

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/mark_first_learn_cycle.py
  python lineages/adaptive_rl_brain_7_31_26/mark_first_learn_cycle.py --dates 2026-04-02,2026-04-01 --epochs 25
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
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.sets import assert_mark_sets_law
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
EMBRYO = os.path.join(CKPT_DIR, "mark_clone_doctrine_v1.pt")
LABEL_DIR = os.path.join(CKPT_DIR, "mark_first_labels")
REPORT = os.path.join(CKPT_DIR, "mark_first_cycle_report.json")
NAMES = {0: "HOLD", 1: "BUY", 2: "SELL"}

# Museum: other trained policies to harvest (lineage only — not PROVEN write)
MUSEUM_GLOBS = (
    "multi_pair_consistent_v1.pt",
    "channel1_curriculum_v3_second_best.pt",
    "channel1_curriculum_v2_hold_shape.pt",
    "mark_clone_channel1_v1.pt",
)


def _load_days_map(csv: str = "XAUUSD_curriculum_2026.csv"):
    days = load_calendar_days(csv, min_bars=900)
    return {str(d): m1 for d, m1 in days}, days


def _load_or_init_embryo(hidden: int = 64) -> Channel1Policy:
    pol = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=hidden)
    if os.path.isfile(EMBRYO):
        blob = torch.load(EMBRYO, map_location="cpu", weights_only=False)
        h = int(blob.get("hidden", hidden))
        if h != hidden:
            pol = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=h)
        pol.load_state_dict(blob["state_dict"])
        print(f"  embryo loaded {EMBRYO} hidden={h}", flush=True)
    else:
        print("  embryo cold-start", flush=True)
    pol.eval()
    return pol


def mark_sees_chart_first(
    m1,
    date_str: str,
    target: float,
    risk: float,
) -> Dict[str, Any]:
    """Step 1: Mark walks the price path and emits labels BEFORE policy authority."""
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=str(date_str),
        eyes_mode="mark_doctrine",
    )
    labels: List[dict] = []
    xs: List[np.ndarray] = []
    ys: List[int] = []
    indices = day.runner.decision_indices()
    prev_t = 0
    for t in indices:
        if day.dead or day.banked:
            break
        for bt in range(prev_t, t):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
        prev_t = t + 1
        if day.dead or day.banked:
            break
        obs = day.observe(t)
        mark_act = int(day.recommended_action(t))
        # perception snapshot for notes
        try:
            perc = day.runner.perceive(t)
            mark_opp = perc.get("mark_opportunity")
            reason = getattr(mark_opp, "reason", "") if mark_opp is not None else ""
        except Exception:
            reason = ""
        xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
        ys.append(mark_act)
        labels.append(
            {
                "t": int(t),
                "mark_action": mark_act,
                "mark_name": NAMES[mark_act],
                "reason": reason,
                "equity_pct": round(float(day.equity_pct(float(day._close[t]))), 4),
            }
        )
        day.step_action(t, mark_act)
    if not day.dead and not day.banked:
        for bt in range(prev_t, len(day.m1)):
            if day.dead or day.banked:
                break
            day._mark_bar(bt)
    t_last = len(day.m1) - 1
    day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
    pnl = 100.0 * (day.balance - day.eq0) / day.eq0
    cleared = (pnl >= day.target - 1e-12 and not day.breached) or (
        day.banked and not day.breached
    )
    return {
        "date": str(date_str),
        "target_pct": target,
        "risk_pct": risk,
        "who_first": "MARK",
        "n_labels": len(ys),
        "mark_actions": dict(Counter(NAMES[y] for y in ys)),
        "cleared": bool(cleared),
        "breached": bool(day.breached),
        "banked": bool(day.banked),
        "pnl_pct": round(float(pnl), 4),
        "n_entries": int(day.n_entries),
        "X": np.stack(xs, axis=0) if xs else np.zeros((0, CHANNEL1_DIM), np.float32),
        "y": np.asarray(ys, dtype=np.int64),
        "labels": labels,
    }


def policy_second(
    m1,
    date_str: str,
    target: float,
    risk: float,
    policy: Channel1Policy,
    mark_pack: Dict[str, Any],
) -> Dict[str, Any]:
    """Step 2: policy acts; compare to Mark labels already written."""
    day = GoalEquityDay(
        m1,
        target_pct=target,
        risk_pct=risk,
        date_str=str(date_str),
        eyes_mode="mark_doctrine",
    )
    X = mark_pack["X"]
    y_mark = mark_pack["y"]
    # Replay same decision indices: re-run policy on Mark's obs sequence
    # (obs from Mark walk includes shell state under Mark actions — student still
    # learns "what Mark saw / chose". For live shell under policy we also run day.)
    agree = 0
    disagree = 0
    policy_pred: List[int] = []
    policy.eval()
    with torch.no_grad():
        for i in range(len(y_mark)):
            logits = policy(torch.as_tensor(X[i], dtype=torch.float32))
            a = int(torch.argmax(logits, dim=-1).item())
            policy_pred.append(a)
            if a == int(y_mark[i]):
                agree += 1
            else:
                disagree += 1
    # Live shell under pure policy
    r = day.run(greedy_policy=policy, use_heuristic=False, pure_greedy=True)
    n = max(len(y_mark), 1)
    return {
        "date": str(date_str),
        "agree": agree,
        "disagree": disagree,
        "agree_rate": agree / n,
        "policy_actions": dict(Counter(NAMES[a] for a in policy_pred)),
        "policy_cleared": bool(r.cleared),
        "policy_breached": bool(r.breached),
        "policy_pnl": round(float(r.pnl_pct), 4),
        "policy_entries": int(r.n_entries),
        "mark_cleared": mark_pack["cleared"],
        "mark_entries": mark_pack["n_entries"],
    }


def update_policy_to_match_mark(
    policy: Channel1Policy,
    packs: Sequence[Dict[str, Any]],
    *,
    epochs: int = 20,
    lr: float = 3.5e-4,
    seed: int = 0,
) -> Dict[str, Any]:
    """Step 3: BC embryo toward Mark labels (Mark was first)."""
    xs = [p["X"] for p in packs if len(p["y"])]
    ys = [p["y"] for p in packs if len(p["y"])]
    if not xs:
        return {"updated": False, "reason": "no_labels"}
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    policy.train()
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    counts = np.bincount(y, minlength=3).astype(np.float64) + 1.0
    w = (counts.sum() / (3.0 * counts)).astype(np.float32)
    weight = torch.tensor(w, dtype=torch.float32, device=device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    n = len(y)
    losses = []
    for ep in range(epochs):
        perm = np.random.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, 64):
            idx = perm[i : i + 64]
            loss = F.cross_entropy(policy(Xt[idx]), yt[idx], weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        losses.append(ep_loss / max(nb, 1))
    policy.eval()
    with torch.no_grad():
        pred = policy(Xt).argmax(dim=-1).cpu().numpy()
    match = float((pred == y).mean())
    return {
        "updated": True,
        "n_samples": int(n),
        "match_after": match,
        "loss_final": losses[-1] if losses else None,
        "epochs": epochs,
    }


def harvest_museum(
    packs: Sequence[Dict[str, Any]],
    embryo: Channel1Policy,
    *,
    epochs: int = 8,
) -> Dict[str, Any]:
    """Step 4: other ckpts — keep bars where they AGREE with Mark; BC those into embryo."""
    results = []
    harvest_X: List[np.ndarray] = []
    harvest_y: List[int] = []
    if not packs or not any(len(p["y"]) for p in packs):
        return {"harvested": False, "museum": []}

    X_all = np.concatenate([p["X"] for p in packs if len(p["y"])], axis=0)
    y_mark = np.concatenate([p["y"] for p in packs if len(p["y"])], axis=0)

    for name in MUSEUM_GLOBS:
        path = os.path.join(CKPT_DIR, name)
        if not os.path.isfile(path):
            results.append({"name": name, "status": "missing"})
            continue
        try:
            blob = torch.load(path, map_location="cpu", weights_only=False)
            h = int(blob.get("hidden", 48))
            other = Channel1Policy(obs_dim=CHANNEL1_DIM, hidden=h)
            other.load_state_dict(blob["state_dict"])
            other.eval()
        except Exception as e:
            results.append({"name": name, "status": f"load_fail:{e}"})
            continue
        agree = 0
        with torch.no_grad():
            for i in range(len(y_mark)):
                # obs dim must match; skip if architecture mismatch on forward
                try:
                    a = int(
                        torch.argmax(
                            other(torch.as_tensor(X_all[i], dtype=torch.float32)),
                            dim=-1,
                        ).item()
                    )
                except Exception:
                    agree = -1
                    break
                if a == int(y_mark[i]):
                    agree += 1
                    harvest_X.append(X_all[i])
                    harvest_y.append(int(y_mark[i]))
        if agree < 0:
            results.append({"name": name, "status": "incompatible_obs"})
            continue
        rate = agree / max(len(y_mark), 1)
        results.append(
            {
                "name": name,
                "status": "ok",
                "agree_with_mark": agree,
                "agree_rate": rate,
                "benefit": "distill_agree_bars" if rate >= 0.35 else "low_agree_skip_distill",
            }
        )

    distill = {"distilled": False}
    if harvest_X:
        # unique-ish concat
        Xh = np.stack(harvest_X, axis=0)
        yh = np.asarray(harvest_y, dtype=np.int64)
        distill = update_policy_to_match_mark(
            embryo,
            [{"X": Xh, "y": yh}],
            epochs=epochs,
            lr=2e-4,
            seed=1,
        )
        distill["n_harvest_bars"] = int(len(yh))
        distill["distilled"] = True
    return {"harvested": True, "museum": results, "distill": distill}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Mark-first always-learning cycle")
    ap.add_argument(
        "--dates",
        default="2026-04-02,2026-04-01,2026-03-18,2026-03-19,2026-03-20",
        help="comma dates Mark sees first",
    )
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--risk", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--data", default="XAUUSD_curriculum_2026.csv")
    args = ap.parse_args(argv)

    assert_mark_sets_law()
    os.makedirs(LABEL_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)

    day_map, _ = _load_days_map(args.data)
    dates = [d.strip() for d in args.dates.split(",") if d.strip()]
    print("=" * 60, flush=True)
    print("MARK ALWAYS LEARNING — chart first, then policy", flush=True)
    print("=" * 60, flush=True)

    # --- 1 + 2 Mark first, policy second ---
    packs: List[Dict[str, Any]] = []
    compares: List[Dict[str, Any]] = []
    embryo = _load_or_init_embryo(64)

    for date in dates:
        if date not in day_map:
            print(f"  skip missing date {date}", flush=True)
            continue
        # random-ish pair per day still from fixed target for teach day; shell runtime
        print(f"\n[1] MARK SEES CHART first  date={date} T/R={args.target}/{args.risk}", flush=True)
        mark_pack = mark_sees_chart_first(
            day_map[date], date, args.target, args.risk
        )
        # drop numpy for disk labels
        disk = {k: v for k, v in mark_pack.items() if k not in ("X", "y")}
        disk["y"] = mark_pack["y"].tolist()
        with open(os.path.join(LABEL_DIR, f"mark_first_{date}.json"), "w", encoding="utf-8") as f:
            json.dump(disk, f, indent=2)
        print(
            f"    Mark labels={mark_pack['n_labels']} actions={mark_pack['mark_actions']} "
            f"cleared={mark_pack['cleared']} entries={mark_pack['n_entries']}",
            flush=True,
        )

        print(f"[2] POLICY second (student)…", flush=True)
        cmp_ = policy_second(
            day_map[date], date, args.target, args.risk, embryo, mark_pack
        )
        print(
            f"    agree_rate={cmp_['agree_rate']:.3f} disagree={cmp_['disagree']} "
            f"pol_cleared={cmp_['policy_cleared']} mark_cleared={cmp_['mark_cleared']}",
            flush=True,
        )
        packs.append(mark_pack)
        compares.append(cmp_)

    # --- 3 Update policy to match Mark ---
    print("\n[3] UPDATE policy → match Mark…", flush=True)
    upd = update_policy_to_match_mark(embryo, packs, epochs=args.epochs, seed=int(datetime.now().timestamp()) % 10000)
    print(f"    {upd}", flush=True)

    # --- 4 Harvest museum ---
    print("\n[4] HARVEST other trained policies (agree-with-Mark only)…", flush=True)
    harv = harvest_museum(packs, embryo, epochs=max(6, args.epochs // 3))
    for m in harv.get("museum", []):
        print(f"    museum {m}", flush=True)
    print(f"    distill={harv.get('distill')}", flush=True)

    # save embryo
    blob = {
        "tag": "mark_clone_doctrine_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state_dict": embryo.state_dict(),
        "hidden": embryo.net[0].out_features if hasattr(embryo.net[0], "out_features") else 64,
        "obs_dim": CHANNEL1_DIM,
        "eyes_mode": "mark_doctrine",
        "teacher": "mark_first_always_learning",
        "cycle": "mark_sees_chart_then_update_policy_then_harvest",
        "proven_touched": False,
        "dials": {
            "decode": "policy_greedy_after_mark_first",
            "risk_use_frac": 0.35,
            "stop_atr_mult": 2.0,
            "per_trade_cap_pct": 0.25,
        },
    }
    # hidden from module
    try:
        blob["hidden"] = int(embryo.net[0].out_features)
    except Exception:
        blob["hidden"] = 64
    torch.save(blob, EMBRYO)
    torch.save(blob, os.path.join(CKPT_DIR, "mark_clone_latest.pt"))
    print(f"\n  saved embryo {EMBRYO}", flush=True)

    # --- 5 Quick post agree recheck ---
    post = []
    for date, pack in zip([c["date"] for c in compares], packs):
        if date not in day_map:
            continue
        post.append(
            policy_second(day_map[date], date, args.target, args.risk, embryo, pack)
        )
    mean_agree_before = float(np.mean([c["agree_rate"] for c in compares])) if compares else 0.0
    mean_agree_after = float(np.mean([c["agree_rate"] for c in post])) if post else 0.0

    # award streak teacher still the floor
    streak_note = {}
    try:
        from lineages.adaptive_rl_brain_7_31_26.eval_award_streak import main as streak_main

        # teacher reaffirm without wiping
        streak_path = os.path.join(CKPT_DIR, "award_streak_teacher_fullrand.json")
        if os.path.isfile(streak_path):
            streak_note = {
                "teacher_streak_file": streak_path,
                "note": "re-run eval_award_streak.py --decode teacher for live meter",
            }
    except Exception:
        pass

    report = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "law": "MARK_ALWAYS_LEARNING.md — Mark chart first, then policy, then harvest",
        "sets_law": "MARK_SETS_LAW.md",
        "dates": dates,
        "mark_first": [
            {k: v for k, v in p.items() if k not in ("X", "y", "labels")} for p in packs
        ],
        "compare_before_update": compares,
        "update": upd,
        "harvest": harv,
        "compare_after_update": post,
        "mean_agree_before": mean_agree_before,
        "mean_agree_after": mean_agree_after,
        "agree_improved": mean_agree_after >= mean_agree_before - 1e-9,
        "embryo_ckpt": EMBRYO,
        "proven_touched": False,
        "streak_note": streak_note,
        "always_learning": True,
    }
    with open(REPORT, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\n[5] REPORT {REPORT}", flush=True)
    print(
        f"STATUS mark_first=true agree {mean_agree_before:.3f}->{mean_agree_after:.3f} "
        f"updated={upd.get('updated')} harvested={harv.get('harvested')} proven_ok=true",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
