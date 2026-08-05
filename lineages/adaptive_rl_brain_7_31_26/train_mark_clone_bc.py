"""Train a NEW Channel1 policy to clone Mark doctrine (same obs, new brain).

Teacher = five-law doctrine (FORCE→REGIME→VELOCITY→ENTRY) + MARK SOUL:
  goal-relative lot size + force-aligned adds (shell, mark_doctrine path).
  Actions still HOLD/BUY/SELL; soul lives in when teacher fires add + size math.
PROVEN never touched. Not the banned trail+cushion+scale-in package.

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py
  python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 20 --practice-n 50
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
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.perception.observation import CHANNEL1_DIM
from lineages.adaptive_rl_brain_7_31_26.perception.observation_full import MARK_FULL_DIM
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
    Channel1Policy,
)

CKPT_DIR = os.path.join(_HERE, "checkpoints")
CKPT_PATH = os.path.join(CKPT_DIR, "mark_clone_doctrine_v1.pt")
CKPT_SOUL_PATH = os.path.join(CKPT_DIR, "mark_clone_soul_v1.pt")
CKPT_FULL_OBS_PATH = os.path.join(CKPT_DIR, "mark_clone_full_obs_v1.pt")
REPORT_PATH = os.path.join(CKPT_DIR, "mark_clone_bc_report.json")


def _load_ten_pairs() -> List[Tuple[float, float]]:
    path = os.path.join(_HERE, "ten_pairs.json")
    if not os.path.isfile(path):
        return [(1.0, 2.0), (1.5, 2.5), (2.0, 3.0), (2.5, 3.5), (3.0, 3.5)]
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)["pairs"]
    return [(float(p["target_pct"]), float(p["risk_pct"])) for p in raw]


def collect_teacher_dataset(
    days: List[Tuple[str, Any]],
    *,
    target: float = 2.0,
    risk: float = 3.0,
    decide_every: int = 25,
    max_days: int = 50,
    multi_pair: bool = True,
    seed: int = 42,
    soul_plans: bool = True,
    full_obs: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """Walk days with doctrine eyes; store (obs, teacher_action) pairs.

    multi_pair=True: each day gets a random (target,risk) from ten_pairs so
    progress/danger slots teach meta (any pair, no retrain).

    soul_plans=True (default): full-chart Mark soul plans (size+adds) as labels.
    full_obs=True: 168-dim Mark board (sets + doctrine + 92 agents + self).
    """
    pair_pool = _load_ten_pairs() if multi_pair else [(target, risk)]
    if soul_plans:
        from lineages.adaptive_rl_brain_7_31_26.mark_soul_plan import (
            collect_soul_plan_labels,
        )

        X, y, act_counts, meta = collect_soul_plan_labels(
            days,
            target=target,
            risk=risk,
            max_days=max_days,
            multi_pair=multi_pair,
            seed=seed,
            pairs=pair_pool,
            full_obs=full_obs,
        )
        print(f"  soul_plan_meta={meta}", flush=True)
        return X, y, act_counts

    xs: List[np.ndarray] = []
    ys: List[int] = []
    act_counts = {ACTION_HOLD: 0, ACTION_BUY: 0, ACTION_SELL: 0}
    rng = np.random.default_rng(seed)
    for i, (date_str, m1) in enumerate(days[:max_days]):
        if multi_pair:
            t, r = pair_pool[int(rng.integers(0, len(pair_pool)))]
        else:
            t, r = target, risk
        day = GoalEquityDay(
            m1,
            target_pct=t,
            risk_pct=r,
            date_str=str(date_str),
            decide_every=decide_every,
            eyes_mode="mark_doctrine",
            mark_clone=False,
            mark_soul=True,  # soul: goal size + force adds in teacher labels
            full_obs=full_obs,
        )
        for t_bar in day.runner.decision_indices():
            if day.banked or day.dead:
                break
            obs = day.observe(t_bar)
            teacher = int(day.recommended_action(t_bar))
            xs.append(np.asarray(obs, dtype=np.float32).reshape(-1))
            ys.append(teacher)
            act_counts[teacher] = act_counts.get(teacher, 0) + 1
            day.step_action(t_bar, teacher)
    obs_dim = MARK_FULL_DIM if full_obs else CHANNEL1_DIM
    if not xs:
        return (
            np.zeros((0, obs_dim), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            act_counts,
        )
    X = np.stack(xs, axis=0)
    y = np.asarray(ys, dtype=np.int64)
    return X, y, act_counts


def train_bc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 25,
    batch: int = 256,
    lr: float = 1e-3,
    hidden: int = 64,
    seed: int = 42,
    warm_state: Optional[dict] = None,
    obs_dim: Optional[int] = None,
    sample_weights: Optional[np.ndarray] = None,
    kl_anchor_state: Optional[dict] = None,
    kl_coef: float = 0.0,
) -> Tuple[Channel1Policy, List[float]]:
    """BC clone. Optional per-sample weights (streak/gap rewards → importance).

    sample_weights: shape (n,) from reward dials — MARK_WOULD_TAKE / soul-side
    labels get higher weight so rewards/penalties cause the update.

    kl_anchor_state + kl_coef: keep new policy close to a prior good embryo
    so miss-day corrections do not destroy award-day behavior.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cpu")
    dim = int(obs_dim) if obs_dim is not None else int(X.shape[-1] if len(X) else CHANNEL1_DIM)
    policy = Channel1Policy(obs_dim=dim, hidden=hidden).to(device)
    if warm_state is not None:
        try:
            policy.load_state_dict(warm_state)
            print("  warm-start: loaded prior embryo weights", flush=True)
        except Exception as e:
            print(f"  warm-start skip ({e})", flush=True)
    anchor = None
    if kl_anchor_state is not None and float(kl_coef) > 0.0:
        anchor = Channel1Policy(obs_dim=dim, hidden=hidden).to(device)
        anchor.load_state_dict(kl_anchor_state)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)
        print(f"  KL anchor on (coef={kl_coef})", flush=True)
    # Slightly lower LR when warm-starting so we polish, not thrash
    use_lr = float(lr) * (0.35 if warm_state is not None else 1.0)
    opt = torch.optim.Adam(policy.parameters(), lr=use_lr)
    n = len(y)
    losses: List[float] = []
    if n == 0:
        return policy, losses
    # Inverse-frequency class weights.
    # Mark soul: HOLD is deliberate wait (HITL: policy fired early). Boost HOLD
    # enough to stop reverse-thrash without killing dir_match.
    counts = np.bincount(y, minlength=3).astype(np.float64) + 1.0
    w = (counts.sum() / (3.0 * counts)).astype(np.float32)
    w[ACTION_HOLD] = float(max(w[ACTION_HOLD], 1.35))
    # Keep directional at least as strong as HOLD for sparse entries
    w[ACTION_BUY] = float(max(w[ACTION_BUY], w[ACTION_HOLD] * 0.95))
    w[ACTION_SELL] = float(max(w[ACTION_SELL], w[ACTION_HOLD] * 0.95))
    class_weight = torch.tensor(w, dtype=torch.float32, device=device)
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    yt = torch.tensor(y, dtype=torch.long, device=device)
    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=np.float32).reshape(-1)
        if sw.shape[0] != n:
            raise ValueError(f"sample_weights len {sw.shape[0]} != n {n}")
        sw = np.maximum(sw, 1e-6)
        sw = sw / float(sw.mean())
        sw_t = torch.tensor(sw, dtype=torch.float32, device=device)
    else:
        sw_t = None
    for ep in range(epochs):
        perm = np.random.permutation(n)
        ep_loss = 0.0
        nb = 0
        for i in range(0, n, batch):
            idx = perm[i : i + batch]
            logits = policy(Xt[idx])
            if sw_t is None:
                loss = F.cross_entropy(logits, yt[idx], weight=class_weight)
            else:
                # per-sample CE × reward-derived weight × class weight via nll
                logp = F.log_softmax(logits, dim=-1)
                nll = F.nll_loss(logp, yt[idx], weight=class_weight, reduction="none")
                loss = (nll * sw_t[idx]).mean()
            if anchor is not None:
                with torch.no_grad():
                    a_logits = anchor(Xt[idx])
                    a_logp = F.log_softmax(a_logits, dim=-1)
                    a_p = a_logp.exp()
                logp_new = F.log_softmax(logits, dim=-1)
                # KL(anchor || new) keeps new near old good policy
                kl = (a_p * (a_logp - logp_new)).sum(dim=-1).mean()
                loss = loss + float(kl_coef) * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
            ep_loss += float(loss.item())
            nb += 1
        losses.append(ep_loss / max(nb, 1))
        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"  bc epoch {ep+1}/{epochs} loss={losses[-1]:.4f}", flush=True)
    return policy, losses


@torch.no_grad()
def match_rate(policy: Channel1Policy, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    if len(y) == 0:
        return {"match": 0.0, "n": 0}
    device = next(policy.parameters()).device
    logits = policy(torch.tensor(X, dtype=torch.float32, device=device))
    pred = logits.argmax(dim=-1).cpu().numpy()
    match = float((pred == y).mean())
    # match when teacher is directional
    mask = y != ACTION_HOLD
    dir_match = float((pred[mask] == y[mask]).mean()) if mask.any() else 0.0
    hold_rate = float((pred == ACTION_HOLD).mean())
    return {
        "match": match,
        "dir_match": dir_match,
        "pred_hold_rate": hold_rate,
        "n": int(len(y)),
    }


def eval_greedy_days(
    policy: Channel1Policy,
    days: List[Tuple[str, Any]],
    *,
    target: float,
    risk: float,
    max_days: int = 20,
    full_obs: bool = False,
) -> Dict[str, Any]:
    """Run doctrine shell with pure greedy policy (no teacher override)."""
    cleared = 0
    breached = 0
    entries = []
    teacher_match_steps = 0
    total_steps = 0
    for date_str, m1 in days[:max_days]:
        day = GoalEquityDay(
            m1,
            target_pct=target,
            risk_pct=risk,
            date_str=str(date_str),
            eyes_mode="mark_doctrine",
            mark_soul=True,
            full_obs=full_obs,
        )
        # Custom run: greedy policy for action, track match vs teacher
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
            teacher = int(day.recommended_action(t))
            act, _ = policy.act(obs, greedy=True)
            act = int(act)
            if act == teacher:
                teacher_match_steps += 1
            total_steps += 1
            day.step_action(t, act)
        if not day.dead and not day.banked:
            for bt in range(prev_t, len(day.m1)):
                if day.dead or day.banked:
                    break
                day._mark_bar(bt)
        # EOD flatten score
        t_last = len(day.m1) - 1
        price = float(day._close[t_last])
        sp = float(day._spread_px[t_last])
        day._flatten(price, sp)
        pnl = 100.0 * (day.balance - day.eq0) / day.eq0
        day.min_eq_pct = min(day.min_eq_pct, pnl)
        if pnl <= -day.risk + 1e-12:
            day.breached = True
        goal_hit = (pnl >= day.target - 1e-12) and (not day.breached)
        if day.banked and not day.breached:
            goal_hit = True
        if goal_hit:
            cleared += 1
        if day.breached:
            breached += 1
        entries.append(day.n_entries)
    n = min(len(days), max_days)
    return {
        "n_days": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "mean_entries": float(np.mean(entries)) if entries else 0.0,
        "step_match_rate": teacher_match_steps / max(total_steps, 1),
        "total_steps": total_steps,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--practice-n", type=int, default=50)
    ap.add_argument("--max-train-days", type=int, default=50)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--risk", type=float, default=3.0)
    ap.add_argument(
        "--ab-after",
        action="store_true",
        default=True,
        help="After train, run hard+soft policy A/B (default on)",
    )
    ap.add_argument("--seed", type=int, default=None, help="train seed (default: rotate)")
    ap.add_argument("--no-warmstart", action="store_true", help="cold start from scratch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument(
        "--full-obs",
        action="store_true",
        help="168-dim Mark board: sets+doctrine+92 agents+self (full clone eyes)",
    )
    args = ap.parse_args(argv)

    os.makedirs(CKPT_DIR, exist_ok=True)
    obs_dim = MARK_FULL_DIM if args.full_obs else CHANNEL1_DIM
    if args.full_obs and args.hidden < 64:
        args.hidden = 128
    ckpt_out = CKPT_FULL_OBS_PATH if args.full_obs else CKPT_PATH
    seed_path = os.path.join(CKPT_DIR, "mark_clone_bc_seed.json")
    if args.seed is not None:
        train_seed = int(args.seed)
    else:
        prev = 41
        if os.path.isfile(seed_path):
            try:
                prev = int(json.load(open(seed_path, "r")).get("seed", 41))
            except Exception:
                prev = 41
        train_seed = prev + 1
    with open(seed_path, "w", encoding="utf-8") as f:
        json.dump({"seed": train_seed, "saved_at": datetime.now(timezone.utc).isoformat()}, f)

    print("Loading calendar days…", flush=True)
    all_days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=args.practice_n)
    print(
        f"practice={len(practice)} forward={len(forward)} "
        f"teacher=mark_doctrine target={args.target}/{args.risk} seed={train_seed} "
        f"full_obs={args.full_obs} obs_dim={obs_dim}",
        flush=True,
    )

    print(
        "Collecting Mark SOUL plan dataset (practice, multi-pair random T/R)…",
        flush=True,
    )
    X, y, counts = collect_teacher_dataset(
        practice,
        target=args.target,
        risk=args.risk,
        max_days=args.max_train_days,
        multi_pair=True,
        seed=train_seed,
        soul_plans=True,
        full_obs=bool(args.full_obs),
    )
    print(
        f"  samples={len(y)} action_counts={counts} multi_pair=True soul_plans=True "
        f"full_obs={args.full_obs} X_dim={X.shape[-1] if len(y) else 0}",
        flush=True,
    )
    if len(y) < 50:
        print("Too few samples — abort", flush=True)
        return 2

    warm_state = None
    warm_path = ckpt_out if os.path.isfile(ckpt_out) else CKPT_PATH
    if not args.no_warmstart and os.path.isfile(warm_path) and not args.full_obs:
        # full_obs dim change → cold start (cannot load 32-dim into 168)
        try:
            blob0 = torch.load(warm_path, map_location="cpu", weights_only=False)
            if (
                int(blob0.get("hidden", args.hidden)) == int(args.hidden)
                and int(blob0.get("obs_dim", CHANNEL1_DIM)) == int(obs_dim)
            ):
                warm_state = blob0["state_dict"]
            else:
                print("  warm-start skip: hidden/obs_dim mismatch", flush=True)
        except Exception as e:
            print(f"  warm-start skip: {e}", flush=True)
    elif args.full_obs:
        print("  full_obs: cold start new 168-dim Mark clone brain", flush=True)

    print("BC training…", flush=True)
    policy, losses = train_bc(
        X,
        y,
        epochs=args.epochs,
        hidden=args.hidden,
        seed=train_seed,
        lr=args.lr,
        warm_state=warm_state,
        obs_dim=obs_dim,
    )
    metrics = match_rate(policy, X, y)
    print(f"  train match={metrics}", flush=True)

    # Holdout teacher labels on a few forward days
    Xf, yf, _ = collect_teacher_dataset(
        forward,
        target=args.target,
        risk=args.risk,
        max_days=15,
        soul_plans=True,
        full_obs=bool(args.full_obs),
        seed=train_seed + 7,
    )
    fwd_match = match_rate(policy, Xf, yf) if len(yf) else {}
    print(f"  forward label match={fwd_match}", flush=True)

    print("Greedy eval practice (20d)…", flush=True)
    ev_p = eval_greedy_days(
        policy,
        practice,
        target=args.target,
        risk=args.risk,
        max_days=20,
        full_obs=bool(args.full_obs),
    )
    print(f"  {ev_p}", flush=True)
    print("Greedy eval forward (20d)…", flush=True)
    ev_f = eval_greedy_days(
        policy,
        forward,
        target=args.target,
        risk=args.risk,
        max_days=20,
        full_obs=bool(args.full_obs),
    )
    print(f"  {ev_f}", flush=True)

    # Teacher-only baseline on same windows (decode=heuristic doctrine)
    def teacher_score(days, max_days=20):
        c = b = 0
        ent = []
        for date_str, m1 in days[:max_days]:
            day = GoalEquityDay(
                m1,
                target_pct=args.target,
                risk_pct=args.risk,
                date_str=str(date_str),
                eyes_mode="mark_doctrine",
                mark_soul=True,
            )
            r = day.run(use_heuristic=True)
            c += int(r.cleared)
            b += int(r.breached)
            ent.append(r.n_entries)
        n = min(len(days), max_days)
        return {
            "cleared": c,
            "breached": b,
            "clear_pct": 100.0 * c / max(n, 1),
            "mean_entries": float(np.mean(ent)) if ent else 0.0,
            "n_days": n,
        }

    print("Teacher decode practice/forward…", flush=True)
    t_p = teacher_score(practice)
    t_f = teacher_score(forward)
    print(f"  teacher practice {t_p}", flush=True)
    print(f"  teacher forward  {t_f}", flush=True)

    blob = {
        "tag": "mark_clone_full_obs_v1" if args.full_obs else "mark_clone_soul_v1",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "state_dict": policy.state_dict(),
        "hidden": args.hidden,
        "obs_dim": int(obs_dim),
        "eyes_mode": "mark_doctrine",
        "teacher": "five_laws_plus_mark_soul",
        "full_obs": bool(args.full_obs),
        "mark_soul": {
            "goal_relative_size": True,
            "force_aligned_adds": True,
            "max_units": 3,
            "not_banned_package": "no trail+cushion; adds only under force+heat",
        },
        "obs_blocks": (
            "channel1+doctrine+majority+92_agents+self"
            if args.full_obs
            else "channel1_32"
        ),
        "dials": {
            "decode": "policy_greedy_mark_soul_shell",
            "risk_use_frac": "goal_relative_mark_soul",
            "stop_atr_mult": 2.0,
            "per_trade_cap_pct": "goal_relative_mark_soul",
            "base_risk_use_frac": 0.35,
            "base_per_trade_cap_pct": 0.25,
        },
        "meta_role": "clone_of_mark_attention_over_full_eyes",
        "proven_touched": False,
    }
    torch.save(blob, ckpt_out)
    if not args.full_obs:
        torch.save(blob, CKPT_PATH)
        torch.save(blob, CKPT_SOUL_PATH)
    else:
        torch.save(blob, CKPT_FULL_OBS_PATH)
    # also latest pointer
    latest = os.path.join(CKPT_DIR, "mark_clone_latest.pt")
    torch.save(blob, latest)

    report = {
        "saved_at": blob["saved_at"],
        "ckpt": CKPT_PATH,
        "teacher_action_counts": {str(k): v for k, v in counts.items()},
        "train_match": metrics,
        "forward_label_match": fwd_match,
        "bc_losses": losses,
        "eval_greedy_practice": ev_p,
        "eval_greedy_forward": ev_f,
        "teacher_practice": t_p,
        "teacher_forward": t_f,
        "doctrine": "MARK_DOCTRINE_FIVE_LAWS.md",
        "proven_touched": False,
        "clone_ready_heuristic": (
            t_f.get("breached", 1) == 0
            and t_p.get("breached", 1) == 0
            and t_f.get("clear_pct", 0) >= 25
        ),
        "clone_ready_policy": False,  # filled after A/B below
        "next_morning": (
            "Read report + mark_clone_policy_ab_hard_soft.json; "
            "if pass_gates and clone_ready_policy, promote Mark decode; "
            "else more BC epochs / doctrine tweak. Never PROVEN."
        ),
    }

    ab_path = os.path.join(CKPT_DIR, "mark_clone_policy_ab_hard_soft.json")
    ab_block: Dict[str, Any] = {}
    if args.ab_after:
        print("Policy A/B hard 3.0/3.5 + soft 1.0/2.0 (forward)…", flush=True)
        from lineages.adaptive_rl_brain_7_31_26.compare_mark_clone_attention import (
            ab_one_pair,
        )

        hard = ab_one_pair(
            forward,
            3.0,
            3.5,
            policy=policy,
            eyes_mode="mark_doctrine",
            mode_label="forward",
            full_obs=bool(args.full_obs),
        )
        soft = ab_one_pair(
            forward,
            1.0,
            2.0,
            policy=policy,
            eyes_mode="mark_doctrine",
            mode_label="forward",
            full_obs=bool(args.full_obs),
        )
        ab_block = {
            "scored_at": datetime.now(timezone.utc).isoformat(),
            "kind": "policy_weight_ab_hard_soft",
            "policy_ckpt": ckpt_out,
            "full_obs": bool(args.full_obs),
            "eyes_mode": "mark_doctrine",
            "mode": "forward",
            "hard_3_0_3_5": hard,
            "soft_1_0_2_0": soft,
            "pass_gates": {
                "hard_breach_0": hard.get("breach_ok_policy", False),
                "soft_breach_0": soft.get("breach_ok_policy", False),
                "soft_no_collapse": not soft.get("soft_clear_collapse", True),
                "hard_thrash_improved": hard.get("thrash_improved", False),
            },
            "proven_touched": False,
        }
        with open(ab_path, "w", encoding="utf-8") as f:
            json.dump(ab_block, f, indent=2)
        print(f"wrote {ab_path}", flush=True)
        print("pass_gates", ab_block["pass_gates"], flush=True)
        report["policy_ab"] = {
            "hard_clear_pct": hard.get("policy", {}).get("clear_pct"),
            "hard_mean_entries": hard.get("policy", {}).get("mean_entries"),
            "base_hard_clear_pct": hard.get("baseline", {}).get("clear_pct"),
            "base_hard_mean_entries": hard.get("baseline", {}).get("mean_entries"),
            "soft_clear_pct": soft.get("policy", {}).get("clear_pct"),
            "soft_mean_entries": soft.get("policy", {}).get("mean_entries"),
            "base_soft_clear_pct": soft.get("baseline", {}).get("clear_pct"),
            "pass_gates": ab_block["pass_gates"],
        }

    report["clone_ready_policy"] = bool(
        metrics.get("match", 0) >= 0.70
        and metrics.get("dir_match", 0) >= 0.85
        and ev_f.get("breached", 1) == 0
        and (not ab_block or ab_block.get("pass_gates", {}).get("soft_no_collapse", False))
        and (not ab_block or ab_block.get("pass_gates", {}).get("hard_breach_0", False))
    )
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"wrote {CKPT_PATH}", flush=True)
    print(f"wrote {REPORT_PATH}", flush=True)
    print(
        f"READY flags: heuristic={report['clone_ready_heuristic']} "
        f"policy={report['clone_ready_policy']}",
        flush=True,
    )
    return 0 if report["clone_ready_policy"] or report["clone_ready_heuristic"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
