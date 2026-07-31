"""Consistency scorer — the self-tuner's TRUE judge + a smooth climb METER.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (self-tuning meta-loop, Phase 0, 2026-07-20).
WHAT:  Two numbers from ONE batched greedy rollout over random (day, target, risk):
       - consistency = fraction of episodes that CLEAR (hit the handed target AND
         never breach). This is the ratchet's TRUE judge — adopt a change only if
         THIS improves by more than its own error bar.
       - surrogate  = a smooth, climb-shaped METER of "getting there" (additive:
         participation ramp + target-crossing sigmoid − intrabar breach barrier).
         It is the SHAPE the meta-tuner's estimated gradient (ES / finite-diff) pushes
         on. NOTE: it is a scalar meter, NOT autograd-through-the-sim — the tuned knobs
         act on TRAINING, so the meta-gradient is estimated, and true consistency stays
         the final gate.
WHY:   Consistency is a hard non-differentiable count; the tuner needs a smooth shape
       to climb and the true count to judge.
INVARIANT (Monty): the ONLY user inputs are daily target% and risk%. The envelope and
       focus here are AUTO-DERIVED from those two numbers (auto_ranges) — no hidden knob.
REVIEW-HARDENED 2026-07-20 (correctness/durability/two-inputs/speed team):
       + common-random-numbers seeding (pass `gen`) so champion & challenger score on the
         SAME days -> the ratchet can't drift upward on luck;
       + days sampled WITHOUT replacement + honest SE on distinct days;
       + surrogate reshaped additive (strong slope from the timid zero-consistency start),
         breach term uses the intrabar worst (min_worst), same basis as the breach law;
       + envelope + focus auto-bound to goal_pct/floor_pct.
WHEN:  2026-07-20.
INTERCONNECTED WITH: training/fastsim (results: goal_hit/breached/min_eq/min_worst/
       target/risk), training/gpu_rollout (rollout), core/configs (goals), the meta-loop.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-30  side_bias_bull/bear + wrong_side_rate in evaluate() via
  mind_probe.side_metrics_from_decisions; side_metrics= flag to disable cost —
  WHY: meta can see WrongSide without expanding obs/dims (Vector teaching Step 1).
- 2026-07-25  auto_ranges reads goals.yaml goal_conditioning (focus_frac 0.40,
  focus 2.5/3.5, ranges) — WHY: single source for SIGON sampling law.
- 2026-07-20  review-hardened: CRN seeding, without-replacement days + honest SE,
  additive climb surrogate on intrabar-worst, auto-bound envelope/focus  — WHY: 4-agent
  review (noise-ratchet, flat surrogate, breach basis, two-inputs invariant).
- 2026-07-20  created — WHY: Phase 0 of the self-tuner (true judge + smooth meter).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import math
import os
import sys

import torch
import torch.nn.functional as F

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.gpu_rollout import rollout        # noqa: E402
from core.configs import goals_cfg              # noqa: E402
from training.fastsim import FRAME, SELF_DIM    # noqa: E402  # same 10 / 12 mind_probe uses


# Machine policy (NOT human knobs): the practice envelope is a FIXED wide band so one brain
# generalizes to any handed X, and it is always stretched to CONTAIN the user's two numbers.
# The 60/40 focus split is fixed policy too. The ONLY numbers a human ever sets are
# goal_pct + floor_pct (configs/goals.yaml). Everything below derives from those.
_FOCUS_FRAC = 0.40
_ENVELOPE = {"goal_range": (2.5, 70.5), "floor_range": (1.0, 4.0)}


def auto_ranges() -> dict:
    """40% focus pair from goals.yaml; 60% uniform in goal_range x floor_range."""
    g = goals_cfg()
    gc = g.get("goal_conditioning", {}) or {}
    goal = float(gc.get("focus_target", g.get("goal_pct", 2.5)))
    floor = float(gc.get("focus_risk", g.get("floor_pct", 3.5)))
    gr = gc.get("goal_range", _ENVELOPE["goal_range"])
    fr = gc.get("floor_range", _ENVELOPE["floor_range"])
    ff = float(gc.get("focus_frac", _FOCUS_FRAC))
    return {"tgt_lo": float(gr[0]), "tgt_hi": float(gr[1]),
            "risk_lo": float(fr[0]), "risk_hi": float(fr[1]),
            "focus_target": goal, "focus_risk": floor, "focus_frac": ff}


_SIDE_ZEROS = {"side_bias_bull": 0.0, "side_bias_bear": 0.0, "wrong_side_rate": 0.0}


@torch.no_grad()
def _side_metrics_on_days(brain, sim, day_idx, decide_every: int = 5) -> dict:
    """Side scoreboard for meta: reuses mind_probe.side_metrics_from_decisions.

    Walks firm-cont bars only (placeholder self-state, same spirit as Mind Probe).
    FRAME=10 / SELF_DIM=12 match fastsim + mind_probe. Soft-fails to zeros.
    """
    from telemetry.mind_probe import (
        DecisionRecord, side_metrics_from_decisions, LONG_OPS, SHORT_OPS, N_OPS,
    )
    try:
        if not hasattr(sim, "days_obs") or not hasattr(sim, "cont_buy_idx"):
            return dict(_SIDE_ZEROS)
        brain.eval()
        dev = next(brain.parameters()).device
        if torch.is_tensor(day_idx):
            days = [int(x) for x in day_idx.detach().cpu().tolist()]
        else:
            days = [int(x) for x in day_idx]
        seen, decisions = set(), []
        de = max(1, int(decide_every))
        for d in days:
            if d in seen:
                continue
            seen.add(d)
            L = int(sim.day_lens[d].item()) if hasattr(sim, "day_lens") else int(sim.Lmax)
            L = min(L, int(sim.Lmax))
            day_obs = sim.days_obs[d]  # (Lmax, C)
            cb_idx, cs_idx = sim.cont_buy_idx, sim.cont_sell_idx
            t = FRAME
            while t < L:
                row = day_obs[t]
                cont_buy = bool((row[cb_idx] > 0).any().item()) if cb_idx.numel() else False
                cont_sell = bool((row[cs_idx] > 0).any().item()) if cs_idx.numel() else False
                if not cont_buy and not cont_sell:
                    t += de
                    continue
                # frame window + placeholder self (goal/floor only) — no dim expand
                sl = max(0, t - FRAME + 1)
                window = day_obs[sl: t + 1]
                if window.shape[0] < FRAME:
                    pad = window[:1].repeat(FRAME - int(window.shape[0]), 1)
                    window = torch.cat([pad, window], dim=0)
                self_pad = torch.zeros(FRAME, SELF_DIM, device=day_obs.device, dtype=day_obs.dtype)
                self_pad[:, 0] = 3.0
                self_pad[:, 1] = 3.5
                obs = torch.cat([window, self_pad], dim=-1).reshape(1, -1).to(dev)
                result = brain(obs)
                op_part = result[0] if isinstance(result, (tuple, list)) else result
                if hasattr(op_part, "probs"):
                    probs = op_part.probs.detach().reshape(-1)
                else:
                    probs = torch.softmax(op_part.logits.detach().reshape(-1), dim=-1)
                probs = probs[:N_OPS]
                chosen = int(probs.argmax().item())
                p_long = float(probs[list(LONG_OPS)].sum().item())
                p_short = float(probs[list(SHORT_OPS)].sum().item())
                decisions.append(DecisionRecord(
                    t=int(t),
                    op_probs=[float(x) for x in probs.detach().cpu().tolist()],
                    chosen_op=chosen,
                    chosen_op_name="",
                    chosen_size=0.0,
                    value=0.0,
                    cont_buy=cont_buy,
                    cont_sell=cont_sell,
                    p_long=p_long,
                    p_short=p_short,
                    p_hold=float(probs[0].item()),
                ))
                t += de
        if not decisions:
            return dict(_SIDE_ZEROS)
        sm = side_metrics_from_decisions(decisions)
        n_cont = int(sm["n_cont_buy_only"] + sm["n_cont_sell_only"])
        n_wrong = int(sm["n_wrong_side_under_bull"] + sm["n_wrong_side_under_bear"])
        return {
            "side_bias_bull": float(sm["side_bias_bull"]),
            "side_bias_bear": float(sm["side_bias_bear"]),
            "wrong_side_rate": float(n_wrong) / float(max(n_cont, 1)),
        }
    except Exception:
        return dict(_SIDE_ZEROS)


@torch.no_grad()
def evaluate(brain, sim, day_pool, n_episodes: int = 512, focus_frac: float = 0.6,
             decide_every: int = 5, gen: "torch.Generator | None" = None,
             ranges: dict | None = None, side_metrics: bool = True) -> dict:
    """Greedy-score `brain` over n random (day, target, risk) episodes from `day_pool`.
    Pass the SAME `gen` (a seeded torch.Generator on sim.dev) to two brains to compare
    them on identical episodes (common random numbers -> no luck drift). Returns the true
    consistency (+ honest SE + per-episode cleared mask) and the smooth climb surrogate.

    Also returns side_bias_bull, side_bias_bear, wrong_side_rate (Vector teaching Step 1)
    when side_metrics=True (default). Pass side_metrics=False to skip the extra firm-cont
    brain walk if meta-loop runtime becomes painful — keys still present as 0.0.

    Every parameter here is internal; only target%/risk% (via auto_ranges) come from the user."""
    dev = sim.dev
    r = ranges or auto_ranges()
    pool = torch.as_tensor(day_pool, dtype=torch.long, device=dev)
    assert pool.numel() > 0, "consistency.evaluate: empty day_pool"
    n = int(n_episodes)

    # days WITHOUT replacement where the pool allows (effective sample = distinct days)
    if pool.numel() >= n:
        di = pool[torch.randperm(pool.numel(), device=dev, generator=gen)[:n]]
    else:
        di = pool[torch.randint(pool.numel(), (n,), device=dev, generator=gen)]
    tg = torch.empty(n, device=dev).uniform_(r["tgt_lo"], r["tgt_hi"], generator=gen)
    rk = torch.empty(n, device=dev).uniform_(r["risk_lo"], r["risk_hi"], generator=gen)
    ff = float(r.get("focus_frac", focus_frac))          # single source (auto_ranges) — not a second copy
    if ff > 0:                                            # same focus mix training uses
        m = torch.rand(n, device=dev, generator=gen) < ff
        tg = torch.where(m, torch.full_like(tg, r["focus_target"]), tg)
        rk = torch.where(m, torch.full_like(rk, r["focus_risk"]), rk)

    res = rollout(brain, sim, di, tg, rk, greedy=True, collect=False, decide_every=decide_every)
    cleared = res["goal_hit"].float()                     # hit target AND not breached
    breached = res["breached"].float()
    day_pnl = torch.nan_to_num(res["day_pnl"], nan=0.0, posinf=0.0, neginf=0.0)
    min_worst = torch.nan_to_num(res.get("min_worst", res["min_eq"]), nan=0.0, posinf=0.0, neginf=0.0)
    target, risk = res["target"], res["risk"]

    # ---- TRUE metric + honest error bar (on distinct days) ----
    consistency = float(cleared.mean().item())
    n_distinct = int(torch.unique(di).numel())
    p = consistency
    se = math.sqrt(max(p * (1.0 - p), 1e-9) / max(n_distinct, 1))

    # ---- smooth CLIMB meter (additive; strong slope from a timid, clears-nothing start) ----
    s_t, s_r = 0.5, 0.5
    clear = torch.sigmoid((day_pnl - target) / s_t)                       # crosses at the target
    participate = torch.minimum(torch.clamp(day_pnl, min=0.0), target) / torch.clamp(target, min=1e-6)
    barrier = F.softplus((-risk - min_worst) / s_r)                       # ~0 while safe, bites near floor
    surrogate = float((clear + participate - barrier).mean().item())

    # ---- side scoreboard (optional cost; keys always present for stable dict shape) ----
    side = _side_metrics_on_days(brain, sim, di, decide_every=decide_every) if side_metrics else dict(_SIDE_ZEROS)

    return {"consistency": consistency, "surrogate": surrogate,
            "breach_rate": float(breached.mean().item()),
            "cleared": int(cleared.sum().item()), "n": n, "n_distinct_days": n_distinct,
            "se": se, "cleared_mask": cleared.bool(),
            "day_idx": di.detach().clone(),  # per-episode day -> gate can cluster by DAY
            # Vector teaching Step 1 — future code must not assume old dict-only shape:
            # always includes side_bias_bull, side_bias_bear, wrong_side_rate
            "side_bias_bull": float(side["side_bias_bull"]),
            "side_bias_bear": float(side["side_bias_bear"]),
            "wrong_side_rate": float(side["wrong_side_rate"])}


def split_days(n_days: int, holdout_frac: float = 0.15):
    """Fixed train / honesty-check split. The most-recent slice is held out and NEVER
    trained on — so the consistency number reflects days the brain didn't practice.
    Guards a non-empty train slice for tiny universes (review fix)."""
    if n_days <= 1:
        return list(range(n_days)), list(range(n_days))
    n_hold = min(max(1, int(round(n_days * holdout_frac))), n_days - 1)
    all_idx = list(range(n_days))
    return all_idx[:-n_hold], all_idx[-n_hold:]


# --------------------- smoke test ---------------------
if __name__ == "__main__":
    from core.configs import path as rpath
    from training.gpu_data import build_day_tensors
    from training.fastsim import FastSim
    from inference.loader import load_brain

    src = sys.argv[1] if len(sys.argv) > 1 else rpath("data", "raw", "XAUUSD_M1_drill.csv")
    tag = os.path.splitext(os.path.basename(src))[0]
    do, dp, dl, dates, cols = build_day_tensors(src, cache_path=rpath("artifacts", "gpu_cache_%s.npz" % tag))
    sim = FastSim(do, dp, dl, cols, device="cpu", K=24)
    brain, meta = load_brain("PROVEN_2x_2026-07-19")
    if brain is None:
        print("no brain to score"); sys.exit(0)
    train, hold = split_days(do.shape[0])
    g = torch.Generator(device="cpu").manual_seed(0)
    a = evaluate(brain, sim, hold, n_episodes=24, gen=g)
    g2 = torch.Generator(device="cpu").manual_seed(0)
    b = evaluate(brain, sim, hold, n_episodes=24, gen=g2)
    print("CRN check (same seed -> identical): consistency %.3f vs %.3f  surrogate %.3f vs %.3f"
          % (a["consistency"], b["consistency"], a["surrogate"], b["surrogate"]))
    print("holdout | consistency %.3f (+/- %.3f on %d days) | surrogate %.3f | breach %.3f | cleared %d/%d"
          % (a["consistency"], a["se"], a["n_distinct_days"], a["surrogate"], a["breach_rate"], a["cleared"], a["n"]))
