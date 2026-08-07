"""MAML / Reptile learn-to-learn on the proven fable_kag fire_skill path.

Background
----------
Teen 36 was earned by ``fable_kag_l2l``:
  multi-day **fire_skill** pattern pool BC
  + high KL to embryo
  + award protect
  + full 50d KEEP only if same rises and breach==0
  skill id = path family, NOT a calendar day.

This module adds true meta-learning on top of that recipe so the policy
learns *how to adapt* fire_skill (miss_continuation / LTF continuation)
across day-tasks (distribution shift), then still uses the same pack KEEP.

L2L binding
-----------
- Child SHA 9BDCEAAE… is floor history (never demote BEST below live best).
- Live BEST / teen 36 is the climb base.
- Tasks = fire_skill across multi-day MWT fingerprints (not day memos).
- ANIL-style: freeze trunk (features); fast-adapt action head only.
- learn≠copy: reject if act-only costume without hold topology.
- KEEP only same > live_floor and breach==0.

Algorithms
----------
1. **Reptile** (default): stable first-order meta — after k inner steps on a
   support day-cluster, move meta weights toward the adapted weights.
2. **FOMAML** (optional): first-order MAML outer grads on query loss.
3. **Polish**: one multi-day fire_skill BC + KL (the KEEP36 method) before score.

Optional: if ``learn2learn`` is installed, Reptile still uses our pure torch
path (API-compatible spirit); no hard dependency (Py3.14 often lacks wheels).

Run
---
  python -u lineages/adaptive_rl_brain_7_31_26/maml_fire_skill_meta.py \\
    --max-meta-iters 24 --keep-floor 36 --goal-same 37
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

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
from lineages.adaptive_rl_brain_7_31_26.fable_kag_l2l import (
    CHILD_SHA,
    collect_cluster_labels,
    walk_day_labels,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_HOLD,
    Channel1Policy,
)
from lineages.adaptive_rl_brain_7_31_26.rewards import clip_streak_dials, default_streak_dials
from lineages.adaptive_rl_brain_7_31_26.train_mark_clone_bc import train_bc

OUT = os.path.join(_HERE, "checkpoints", "fable_50d_match")
CKPT = os.path.join(_HERE, "checkpoints", "mark_clone_full_obs_v1.pt")
CHILD = os.path.join(_HERE, "checkpoints", "CHILD_STAGE_same35_mark_clone_full_obs.pt")
TEEN = os.path.join(_HERE, "checkpoints", "TEEN_STAGE_same36_fable_kag_fire_skill.pt")
BASELINE = os.path.join(OUT, "BASELINE_50D__frozen.json")
BEST = os.path.join(OUT, "BEST__latest.json")
HARNESS = os.path.join(OUT, "MAML_FIRE_SKILL_HARNESS__latest.json")
MEM = os.path.join(OUT, "MAML_FIRE_SKILL_MEMORY.jsonl")
REPORT = os.path.join(OUT, "MAML_FIRE_SKILL__latest.md")
WHAT_WORKS = os.path.join(OUT, "WHAT_WORKS__GOAL.md")
METHOD = "maml_fire_skill_meta"

try:
    import learn2learn as l2l  # type: ignore

    HAS_L2L = True
    L2L_VER = getattr(l2l, "__version__", "unknown")
except Exception:
    HAS_L2L = False
    L2L_VER = None


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest().upper()


def _sha16(path: str) -> str:
    return _sha256(path)[:16] if os.path.isfile(path) else "?"


def append_mem(row: dict) -> None:
    os.makedirs(OUT, exist_ok=True)
    with open(MEM, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, default=str) + "\n")


def resolve_src() -> str:
    """Prefer live BEST ckpt, then teen 36, never force demote to child."""
    if os.path.isfile(CKPT):
        return CKPT
    if os.path.isfile(TEEN):
        return TEEN
    if os.path.isfile(CHILD):
        return CHILD
    return CKPT


def hidden_from_state(state: dict) -> int:
    w0 = state.get("net.0.weight")
    if w0 is None:
        w0 = state.get("trunk.0.weight")
    if w0 is not None:
        return int(w0.shape[0])
    return 128


def clone_policy(policy: Channel1Policy) -> Channel1Policy:
    c = Channel1Policy(
        obs_dim=int(policy.obs_dim),
        hidden=int(policy.hidden),
        multi_head=bool(policy.multi_head),
    )
    c.load_state_dict(copy.deepcopy(policy.state_dict()))
    return c


def freeze_trunk_(policy: Channel1Policy) -> None:
    for p in policy.trunk_parameters():
        p.requires_grad_(False)


def trainable_params(policy: Channel1Policy) -> List[nn.Parameter]:
    return [p for p in policy.parameters() if p.requires_grad]


def batch_loss(
    policy: Channel1Policy,
    X: torch.Tensor,
    y: torch.Tensor,
    w: Optional[torch.Tensor],
    *,
    anchor: Optional[Channel1Policy] = None,
    kl_coef: float = 0.0,
    class_weight: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    logits = policy(X)
    if class_weight is None:
        counts = torch.bincount(y, minlength=3).float() + 1.0
        cw = (counts.sum() / (3.0 * counts)).clamp(min=0.5)
        cw[ACTION_HOLD] = max(float(cw[ACTION_HOLD]), 1.35)
    else:
        cw = class_weight
    logp = F.log_softmax(logits, dim=-1)
    nll = F.nll_loss(logp, y, weight=cw, reduction="none")
    if w is not None:
        loss = (nll * w).mean()
    else:
        loss = nll.mean()
    if anchor is not None and kl_coef > 0:
        with torch.no_grad():
            a_logits = anchor(X)
            a_logp = F.log_softmax(a_logits, dim=-1)
            a_p = a_logp.exp()
        logp_new = F.log_softmax(logits, dim=-1)
        kl = (a_p * (a_logp - logp_new)).sum(dim=-1).mean()
        loss = loss + float(kl_coef) * kl
    return loss


def inner_adapt(
    policy: Channel1Policy,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    *,
    steps: int = 3,
    lr: float = 2e-3,
    anchor: Optional[Channel1Policy] = None,
    kl_coef: float = 0.85,
    freeze_trunk: bool = True,
) -> Channel1Policy:
    """k-step head-only SGD on support (fast adapt)."""
    learner = clone_policy(policy)
    if freeze_trunk:
        freeze_trunk_(learner)
    if len(y) < 8:
        return learner
    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.long)
    wt = torch.tensor(w, dtype=torch.float32)
    wt = wt / wt.mean().clamp(min=1e-6)
    opt = torch.optim.SGD(trainable_params(learner), lr=lr)
    learner.train()
    for _ in range(max(1, steps)):
        # mini batches if large
        n = len(y)
        if n > 256:
            idx = np.random.choice(n, size=256, replace=False)
            loss = batch_loss(
                learner, Xt[idx], yt[idx], wt[idx], anchor=anchor, kl_coef=kl_coef
            )
        else:
            loss = batch_loss(learner, Xt, yt, wt, anchor=anchor, kl_coef=kl_coef)
        opt.zero_grad()
        loss.backward()
        opt.step()
    learner.eval()
    return learner


def reptile_meta_update(
    meta: Channel1Policy,
    adapted: Channel1Policy,
    *,
    epsilon: float = 0.15,
    head_only: bool = True,
) -> None:
    """meta ← meta + ε (adapted − meta). Reptile first-order meta-learn."""
    with torch.no_grad():
        msd = meta.state_dict()
        asd = adapted.state_dict()
        for k in msd:
            if head_only and ("net.0" in k or "trunk" in k):
                # keep feature trunk from meta (ANIL spirit)
                continue
            if not torch.is_floating_point(msd[k]):
                continue
            msd[k].copy_(msd[k] + float(epsilon) * (asd[k] - msd[k]))
        meta.load_state_dict(msd)


def fomaml_outer_step(
    meta: Channel1Policy,
    support: Tuple[np.ndarray, np.ndarray, np.ndarray],
    query: Tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    adapt_steps: int = 2,
    inner_lr: float = 2e-3,
    outer_lr: float = 5e-4,
    anchor: Optional[Channel1Policy] = None,
    kl_coef: float = 0.85,
) -> float:
    """First-order MAML: adapt on support, query loss grads → meta head."""
    Xs, ys, ws = support
    Xq, yq, wq = query
    learner = clone_policy(meta)
    freeze_trunk_(learner)
    if len(ys) < 8 or len(yq) < 8:
        return 0.0

    Xt = torch.tensor(Xs, dtype=torch.float32)
    yt = torch.tensor(ys, dtype=torch.long)
    wt = torch.tensor(ws, dtype=torch.float32)
    wt = wt / wt.mean().clamp(min=1e-6)

    # inner loop (no create_graph — FOMAML)
    for _ in range(adapt_steps):
        loss = batch_loss(learner, Xt, yt, wt, anchor=anchor, kl_coef=kl_coef)
        grads = torch.autograd.grad(
            loss,
            trainable_params(learner),
            create_graph=False,
            allow_unused=True,
        )
        with torch.no_grad():
            for p, g in zip(trainable_params(learner), grads):
                if g is not None:
                    p.sub_(float(inner_lr) * g)

    Xqt = torch.tensor(Xq, dtype=torch.float32)
    yqt = torch.tensor(yq, dtype=torch.long)
    wqt = torch.tensor(wq, dtype=torch.float32)
    wqt = wqt / wqt.mean().clamp(min=1e-6)

    # re-enable grad path from query through current learner params
    for p in learner.parameters():
        p.requires_grad_(True)
    freeze_trunk_(learner)
    q_loss = batch_loss(learner, Xqt, yqt, wqt, anchor=anchor, kl_coef=kl_coef * 0.5)
    q_loss.backward()

    # copy head grads onto meta and step
    meta_train = []
    learner_train = trainable_params(learner)
    # map by name
    lmap = {n: p for n, p in learner.named_parameters() if p.requires_grad}
    with torch.no_grad():
        for n, p in meta.named_parameters():
            if n in lmap and lmap[n].grad is not None and p.requires_grad:
                if p.grad is None:
                    p.grad = lmap[n].grad.detach().clone()
                else:
                    p.grad.copy_(lmap[n].grad.detach())
                meta_train.append(p)
    if meta_train:
        opt = torch.optim.SGD(meta_train, lr=outer_lr)
        opt.step()
        opt.zero_grad()
    return float(q_loss.detach().item())


# ---------------------------------------------------------------------------
# Task construction — multi-day fire_skill (KEEP36 method), split by day
# ---------------------------------------------------------------------------


def collect_fire_tasks_by_day(
    day_map: Dict[str, Any],
    mwt_rows: List[dict],
    awards: List[dict],
    policy: Channel1Policy,
    oracle: dict,
) -> List[Dict[str, Any]]:
    """One meta-task per MWT day with fire/structure bars.

    Teen-36 already agrees more → pure fire_skill disagree bars can be sparse.
    Include: fire_skill family, miss/continuation laws, and any Mark *directional*
    bar on an MWT day (still multi-day transfer, skill id stays fire_skill).
    """
    tasks: List[Dict[str, Any]] = []
    fire_laws = {"miss_continuation", "ltf_continuation_htf_strong"}

    for row in mwt_rows:
        date = str(row["date"])
        mark = get_plan(
            oracle, day_map, date, float(row["target_pct"]), float(row["risk_pct"])
        )
        if not mark or not mark.get("plan"):
            continue
        labs = walk_day_labels(
            day_map,
            date,
            float(row["target_pct"]),
            float(row["risk_pct"]),
            mark,
            policy,
        )
        xs, ys, ws = [], [], []
        law_c: Counter = Counter()
        for obs, ma, law, fp, w in labs:
            fam = fp.split("|", 1)[0]
            keep = (
                fam == "fire_skill"
                or law in fire_laws
                or ma != ACTION_HOLD  # Mark fire/entry on MWT day
            )
            if not keep:
                continue
            xs.append(obs)
            ys.append(ma)
            # upweight true fire_skill / miss
            ww = float(w)
            if law in fire_laws or fam == "fire_skill":
                ww *= 1.4
            ws.append(ww)
            law_c[law] += 1
        if len(ys) < 6:
            continue
        # No per-task award_self here (too slow: N_mwt × N_award full day walks).
        # HOLD floor is applied once in polish via collect_cluster_labels + inject.
        X = np.stack(xs).astype(np.float32)
        y = np.asarray(ys, np.int64)
        w = np.asarray(ws, np.float32)
        w = np.maximum(w, 1e-6)
        w = w / float(w.mean())
        tasks.append(
            {
                "date": date,  # provenance only
                "family": "fire_skill",
                "X": X,
                "y": y,
                "w": w,
                "laws": dict(law_c),
                "n": len(y),
                "skill_id": "fire_skill_multi_day",  # never use date as skill
            }
        )
    print(
        f"  fire tasks built={len(tasks)} "
        f"n_bars={[t['n'] for t in tasks[:10]]}",
        flush=True,
    )
    return tasks


def split_support_query(
    task: Dict[str, Any], *, support_frac: float = 0.55, seed: int = 0
) -> Tuple[Tuple[np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rng = np.random.default_rng(seed)
    n = int(task["n"])
    idx = rng.permutation(n)
    cut = max(8, int(n * support_frac))
    cut = min(cut, n - 4) if n > 12 else max(1, n // 2)
    s, q = idx[:cut], idx[cut:]
    if len(q) == 0:
        q = idx[-max(1, n // 4) :]
    X, y, w = task["X"], task["y"], task["w"]
    return (X[s], y[s], w[s]), (X[q], y[q], w[q])


def pool_tasks(
    tasks: Sequence[Dict[str, Any]], indices: Sequence[int]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, ws = [], [], []
    for i in indices:
        t = tasks[i]
        xs.append(t["X"])
        ys.append(t["y"])
        ws.append(t["w"])
    X = np.concatenate(xs, axis=0)
    y = np.concatenate(ys, axis=0)
    w = np.concatenate(ws, axis=0)
    w = w / float(w.mean())
    return X, y, w


# ---------------------------------------------------------------------------
# Main climb
# ---------------------------------------------------------------------------


def run_maml_fire_skill(
    *,
    max_meta_iters: int = 24,
    keep_floor: int = 36,
    goal_same: int = 37,
    algo: str = "reptile",  # reptile | fomaml
    adapt_steps: int = 3,
    inner_lr: float = 2e-3,
    reptile_eps: float = 0.12,
    polish_epochs: int = 14,
    kl_coef: float = 1.15,
    tasks_per_iter: int = 3,
) -> Dict[str, Any]:
    os.makedirs(OUT, exist_ok=True)
    print("=== MAML FIRE-SKILL META (on KEEP36 recipe) ===", flush=True)
    print(f"  learn2learn installed: {HAS_L2L} ({L2L_VER})", flush=True)
    print(f"  algo={algo} adapt_steps={adapt_steps} keep_floor={keep_floor} goal={goal_same}", flush=True)
    print(
        "  base method: multi-day fire_skill + KL + award protect + pack KEEP",
        flush=True,
    )

    src = resolve_src()
    print(f"  load src={os.path.basename(src)} sha16={_sha16(src)}", flush=True)
    if os.path.isfile(CHILD):
        print(
            f"  child floor sha16={_sha16(CHILD)} expected={CHILD_SHA[:16]} (history only)",
            flush=True,
        )

    baseline = json.load(open(BASELINE, encoding="utf-8"))
    mark_rows = baseline["rows"]
    floor_clear = int(baseline["policy_clear"])
    days = load_calendar_days("XAUUSD_curriculum_2026.csv", min_bars=900)[:50]
    day_map = {str(d): m1 for d, m1 in days}
    oracle = load_oracle()
    dials = clip_streak_dials(default_streak_dials())

    policy = load_policy(src)
    print("Score base (teen/live BEST)…", flush=True)
    best = score_policy(policy, day_map, mark_rows)
    print(
        f"START same={best['same_outcome']} mwt={best['mark_would_take']} "
        f"breach={best['n_breach']} clear={best['policy_clear']}",
        flush=True,
    )
    best_state = {k: v.detach().clone() for k, v in policy.state_dict().items()}
    live_floor = max(int(keep_floor), int(best["same_outcome"]))
    cycles: List[dict] = []

    # Anchor = current BEST (teen 36) for KL — do not anchor to child when teen exists
    anchor = clone_policy(policy)
    anchor.eval()
    for p in anchor.parameters():
        p.requires_grad_(False)

    mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
    awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
    print(f"  MWT days={len(mwt)} awards={len(awards)}", flush=True)

    print("Build per-day fire_skill tasks (multi-day transfer substrate)…", flush=True)
    tasks = collect_fire_tasks_by_day(day_map, mwt, awards, policy, oracle)
    print(
        f"  tasks={len(tasks)} sizes={[t['n'] for t in tasks[:12]]}…",
        flush=True,
    )
    if len(tasks) < 2:
        print("  WARN: too few fire tasks — falling back to pooled fire_skill only", flush=True)

    meta = clone_policy(policy)
    freeze_trunk_(meta)

    # ----- Meta loop -----
    for it in range(1, max_meta_iters + 1):
        if best["same_outcome"] >= goal_same and best["n_breach"] == 0:
            break
        meta.load_state_dict(copy.deepcopy(best_state))
        freeze_trunk_(meta)

        if len(tasks) >= 2:
            # sample task indices
            rng = np.random.default_rng(1000 + it)
            order = rng.permutation(len(tasks))
            picked = order[: max(1, min(tasks_per_iter, len(tasks)))]
            q_losses = []
            for j, ti in enumerate(picked):
                task = tasks[int(ti)]
                support, query = split_support_query(task, seed=it * 17 + j)
                if algo == "fomaml":
                    ql = fomaml_outer_step(
                        meta,
                        support,
                        query,
                        adapt_steps=adapt_steps,
                        inner_lr=inner_lr,
                        outer_lr=5e-4,
                        anchor=anchor,
                        kl_coef=kl_coef * 0.7,
                    )
                    q_losses.append(ql)
                else:
                    # Reptile: adapt on support, pull meta toward adapted
                    Xs, ys, ws = support
                    adapted = inner_adapt(
                        meta,
                        Xs,
                        ys,
                        ws,
                        steps=adapt_steps,
                        lr=inner_lr,
                        anchor=anchor,
                        kl_coef=kl_coef * 0.7,
                        freeze_trunk=True,
                    )
                    reptile_meta_update(
                        meta, adapted, epsilon=reptile_eps, head_only=True
                    )
                    # diagnostic query loss
                    Xq, yq, wq = query
                    with torch.no_grad():
                        ql = float(
                            batch_loss(
                                adapted,
                                torch.tensor(Xq, dtype=torch.float32),
                                torch.tensor(yq, dtype=torch.long),
                                torch.tensor(wq, dtype=torch.float32),
                            ).item()
                        )
                    q_losses.append(ql)
            mean_q = float(np.mean(q_losses)) if q_losses else 0.0
            print(
                f"\n----- meta {it}/{max_meta_iters} algo={algo} "
                f"tasks={list(picked)} mean_q={mean_q:.4f} -----",
                flush=True,
            )
        else:
            print(f"\n----- meta {it}/{max_meta_iters} pooled fire only -----", flush=True)

        # ----- Polish: proven KEEP36 multi-day fire_skill BC + KL -----
        # Meta init → short BC on FULL multi-day fire pool (the method that made 36)
        X, y, w, meta_info = collect_cluster_labels(
            day_map,
            mwt,
            awards,
            # score labels vs current meta (DAgger trajectory)
            meta if len(tasks) >= 2 else policy,
            oracle,
            target_family="fire_skill",
            multi_day_boost=1.7,
        )
        print(f"  polish cluster meta={meta_info}", flush=True)
        if meta_info["n"] < 40:
            X, y, w, meta_info = collect_cluster_labels(
                day_map, mwt, awards, policy, oracle, target_family="fire_skill"
            )
        if meta_info["n"] < 40:
            print("  sparse fire pool — skip round", flush=True)
            continue

        hold_frac = float((y == 0).mean())
        if hold_frac < 0.32:
            print(f"  hold_frac={hold_frac:.2f} inject award HOLD", flush=True)
            for row in awards[:28]:
                a, b, c = award_self(
                    day_map, row["date"], float(row["target_pct"]), float(row["risk_pct"]), policy
                )
                ex, ey, ew = [], [], []
                for o, act, ww in zip(a, b, c):
                    if int(act) == ACTION_HOLD:
                        ex.append(o)
                        ey.append(0)
                        ew.append(float(ww) * 3.0)
                if ex:
                    X = np.concatenate([X, np.stack(ex)], axis=0)
                    y = np.concatenate([y, np.asarray(ey, np.int64)])
                    w = np.concatenate([w, np.asarray(ew, np.float32)])
            w = w / float(w.mean())
            hold_frac = float((y == 0).mean())
            print(f"  hold_frac={hold_frac:.2f} n={len(y)}", flush=True)

        hid = hidden_from_state(best_state)
        # Warm from meta-adapted init (learn-to-learn), KL to live BEST (36)
        warm = {k: v.detach().clone() for k, v in meta.state_dict().items()}
        pol2, losses = train_bc(
            X,
            y,
            sample_weights=w,
            epochs=polish_epochs,
            lr=1.0e-4,
            hidden=hid,
            seed=7000 + it,
            warm_state=warm,
            kl_anchor_state=best_state,
            kl_coef=float(kl_coef),
            freeze_trunk=False,  # fire_skill KEEP36 used light full train
        )
        with torch.no_grad():
            pred = pol2(torch.tensor(X, dtype=torch.float32)).argmax(-1).numpy()
        act_match = float((pred == y).mean())
        topo = float(((pred == 0) == (y == 0)).mean())
        copying = act_match > 0.93 and topo < 0.55
        print(
            f"  polish act_match={act_match:.3f} hold_topo={topo:.3f} "
            f"copying={copying} loss={losses[-1] if losses else 0:.4f}",
            flush=True,
        )
        if copying:
            print("  learn≠copy FAIL — REJECT", flush=True)
            row = {
                "ts": _utcnow(),
                "iter": it,
                "decision": "REJECT",
                "reason": "learn_not_copy",
                "act_match": act_match,
                "hold_topo": topo,
                "best_same": best["same_outcome"],
                "method": METHOD,
                "algo": algo,
            }
            append_mem(row)
            cycles.append(row)
            continue

        post = score_policy(pol2, day_map, mark_rows)
        print(
            f"  POST same={post['same_outcome']} mwt={post['mark_would_take']} "
            f"breach={post['n_breach']} clear={post['policy_clear']}",
            flush=True,
        )

        keep = (
            post["n_breach"] == 0
            and post["policy_clear"] >= max(floor_clear - 2, 27)
            and post["same_outcome"] >= live_floor
            and post["same_outcome"] > best["same_outcome"]
        )
        decision = "KEEP" if keep else "REJECT"
        if keep:
            best = post
            best_state = {k: v.detach().clone() for k, v in pol2.state_dict().items()}
            live_floor = max(live_floor, int(post["same_outcome"]))
            meta.load_state_dict(best_state)
            # refresh tasks / anchor on new trajectory
            anchor = clone_policy(pol2)
            for p in anchor.parameters():
                p.requires_grad_(False)
            policy = pol2
            mwt = [r for r in best["rows"] if r["miss_class"] == "MARK_WOULD_TAKE"]
            awards = [r for r in best["rows"] if r["miss_class"] == "AWARD"]
            tasks = collect_fire_tasks_by_day(day_map, mwt, awards, policy, oracle)

            save_policy(
                pol2,
                note=f"{METHOD}_KEEP_i{it}_fire_skill",
                dials=dials,
            )
            teen_out = os.path.join(
                _HERE,
                "checkpoints",
                f"TEEN_STAGE_same{post['same_outcome']}_{METHOD}.pt",
            )
            torch.save(
                {
                    "tag": f"teen_same{post['same_outcome']}_{METHOD}",
                    "state_dict": pol2.state_dict(),
                    "hidden": hid,
                    "obs_dim": 168,
                    "multi_head": False,
                    "child_frozen_sha256": CHILD_SHA,
                    "growth_method": METHOD,
                    "core_skill": "fire_skill multi-day + MAML/Reptile head adapt",
                    "saved_at": _utcnow(),
                },
                teen_out,
            )
            with open(BEST, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "same_outcome": post["same_outcome"],
                        "policy_clear": post["policy_clear"],
                        "mwt": post["mark_would_take"],
                        "breach": post["n_breach"],
                        "source": f"{METHOD}_KEEP_fire_skill_i{it}",
                        "stage": "teen",
                        "child_frozen_sha256": CHILD_SHA,
                        "growth_method": METHOD,
                        "core_skill": (
                            "fire_skill multi-day transfer + "
                            f"{algo} meta-adapt (learn-to-learn)"
                        ),
                        "skill_class": "path_family_fire_skill_not_day_memo",
                        "base_recipe": "fable_kag_l2l_KEEP36",
                        "note": (
                            "MAML/Reptile on multi-day fire_skill; "
                            "polish=BC+KL award protect; pack KEEP"
                        ),
                        "ts": _utcnow(),
                    },
                    f,
                    indent=2,
                )
            print(f"  KEEP best_same={best['same_outcome']}", flush=True)
            try:
                with open(WHAT_WORKS, "a", encoding="utf-8") as wf:
                    wf.write(
                        f"| KEEP {METHOD} | **{best['same_outcome']}** | "
                        f"{best['mark_would_take']} | {best['n_breach']} | "
                        f"fire_skill+{algo} meta |\n"
                    )
            except OSError:
                pass
        else:
            print("  REJECT (restore BEST state)", flush=True)
            meta.load_state_dict(best_state)

        row = {
            "ts": _utcnow(),
            "iter": it,
            "decision": decision,
            "same": post["same_outcome"],
            "mwt": post["mark_would_take"],
            "breach": post["n_breach"],
            "clear": post["policy_clear"],
            "best_same": best["same_outcome"],
            "act_match": act_match,
            "hold_topo": topo,
            "cluster": meta_info,
            "method": METHOD,
            "algo": algo,
            "base_recipe": "fable_kag_l2l_fire_skill",
        }
        append_mem(row)
        cycles.append(row)

        with open(HARNESS, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated_at": _utcnow(),
                    "best_same": best["same_outcome"],
                    "best_mwt": best["mark_would_take"],
                    "best_breach": best["n_breach"],
                    "method": METHOD,
                    "algo": algo,
                    "learn2learn": HAS_L2L,
                    "base_recipe": "fable_kag_l2l fire_skill multi-day → 36",
                    "keep_floor": keep_floor,
                    "goal_same": goal_same,
                    "child_sha": CHILD_SHA,
                    "cycles": cycles[-24:],
                    "passed_goal": best["same_outcome"] >= goal_same,
                },
                f,
                indent=2,
            )
        with open(REPORT, "w", encoding="utf-8") as f:
            f.write(
                f"# MAML fire-skill meta report\n\n"
                f"- best_same: **{best['same_outcome']}**\n"
                f"- goal: {goal_same} · floor: {keep_floor}\n"
                f"- last: {decision} same={post['same_outcome']}\n"
                f"- algo: {algo} (ANIL head + multi-day fire_skill)\n"
                f"- base: fable_kag_l2l KEEP36 fire_skill\n"
                f"- child SHA: `{CHILD_SHA[:16]}…` (floor history)\n"
                f"- learn2learn pkg: {HAS_L2L}\n"
            )

        if best["same_outcome"] >= goal_same and best["n_breach"] == 0:
            print(f"GOAL same={best['same_outcome']} >= {goal_same}", flush=True)
            break

    summary = {
        "best_same": best["same_outcome"],
        "best_mwt": best["mark_would_take"],
        "best_breach": best["n_breach"],
        "cycles": len(cycles),
        "method": METHOD,
        "algo": algo,
        "passed_goal": best["same_outcome"] >= goal_same,
        "learn2learn": HAS_L2L,
    }
    print(f"DONE {METHOD} {summary}", flush=True)
    return summary


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="MAML/Reptile meta on multi-day fire_skill (KEEP36 recipe)"
    )
    ap.add_argument("--max-meta-iters", type=int, default=20)
    ap.add_argument("--keep-floor", type=int, default=36)
    ap.add_argument("--goal-same", type=int, default=37)
    ap.add_argument("--algo", choices=("reptile", "fomaml"), default="reptile")
    ap.add_argument("--adapt-steps", type=int, default=3)
    ap.add_argument("--inner-lr", type=float, default=2e-3)
    ap.add_argument("--reptile-eps", type=float, default=0.12)
    ap.add_argument("--polish-epochs", type=int, default=14)
    ap.add_argument("--kl-coef", type=float, default=1.15)
    ap.add_argument("--tasks-per-iter", type=int, default=3)
    args = ap.parse_args()
    run_maml_fire_skill(
        max_meta_iters=args.max_meta_iters,
        keep_floor=args.keep_floor,
        goal_same=args.goal_same,
        algo=args.algo,
        adapt_steps=args.adapt_steps,
        inner_lr=args.inner_lr,
        reptile_eps=args.reptile_eps,
        polish_epochs=args.polish_epochs,
        kl_coef=args.kl_coef,
        tasks_per_iter=args.tasks_per_iter,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
