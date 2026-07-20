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

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.gpu_rollout import rollout        # noqa: E402
from core.configs import goals_cfg              # noqa: E402


# Machine policy (NOT human knobs): the practice envelope is a FIXED wide band so one brain
# generalizes to any handed X, and it is always stretched to CONTAIN the user's two numbers.
# The 60/40 focus split is fixed policy too. The ONLY numbers a human ever sets are
# goal_pct + floor_pct (configs/goals.yaml). Everything below derives from those.
_FOCUS_FRAC = 0.6                                          # 60% of practice pinned to the user's exact X, 40% random
_ENVELOPE = {"goal_range": (0.5, 4.0), "floor_range": (1.0, 6.0)}   # fixed band; always widened to hold the user's X


def auto_ranges() -> dict:
    """Single source of truth, from the user's TWO inputs (goal_pct, floor_pct). The envelope
    is FIXED machine policy (a wide band so one brain handles any X) always stretched to contain
    the user's numbers; the focus point IS the user's numbers; the focus split is fixed policy.
    A human sets ONLY goal_pct + floor_pct — nothing else here is a knob."""
    g = goals_cfg()
    goal = float(g.get("goal_pct", 2.5))
    floor = float(g.get("floor_pct", 4.0))
    gc = g.get("goal_conditioning", {})
    gr = gc.get("goal_range", _ENVELOPE["goal_range"])    # fixed policy band (may be mirrored in goals.yaml)
    fr = gc.get("floor_range", _ENVELOPE["floor_range"])
    return {"tgt_lo": min(float(gr[0]), goal), "tgt_hi": max(float(gr[1]), goal),
            "risk_lo": min(float(fr[0]), floor), "risk_hi": max(float(fr[1]), floor),
            "focus_target": goal, "focus_risk": floor, "focus_frac": _FOCUS_FRAC}


@torch.no_grad()
def evaluate(brain, sim, day_pool, n_episodes: int = 512, focus_frac: float = 0.6,
             decide_every: int = 5, gen: "torch.Generator | None" = None,
             ranges: dict | None = None) -> dict:
    """Greedy-score `brain` over n random (day, target, risk) episodes from `day_pool`.
    Pass the SAME `gen` (a seeded torch.Generator on sim.dev) to two brains to compare
    them on identical episodes (common random numbers -> no luck drift). Returns the true
    consistency (+ honest SE + per-episode cleared mask) and the smooth climb surrogate.
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

    return {"consistency": consistency, "surrogate": surrogate,
            "breach_rate": float(breached.mean().item()),
            "cleared": int(cleared.sum().item()), "n": n, "n_distinct_days": n_distinct,
            "se": se, "cleared_mask": cleared.bool(),
            "day_idx": di.detach().clone()}    # per-episode day -> lets the gate cluster by DAY (kills pseudo-replication)


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

    src = sys.argv[1] if len(sys.argv) > 1 else rpath("data", "XAUUSD_M1_drill.csv")
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
