"""The self-tuner (the heart) — evolve rewards/penalties/hyperparameters toward
CONSISTENCY = clearing the target without breaching, day after day, in a row.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (self-tuning meta-loop, Phase 1, 2026-07-20).
WHAT:  A generation loop: propose candidate configs (mutate the champion's knobs) ->
       PROBE each (short train with that config) -> SCORE on true consistency (CRN, so
       every candidate is judged on the SAME days) -> ADOPT a candidate ONLY if it is
       statistically MORE consistent than the champion (paired McNemar gate, self-
       calibrating margin — no human knob) -> always keep the champion. On a plateau,
       widen exploration; the champion is never lost. Headline = longest day-after-day
       clean streak.
WHY:   Monty's north star is consistency, and he wants it self-completing/self-correcting
       with only two human inputs ever: daily target% and risk%.
SAFETY (the ratchet): adopt requires (c - b) >= z*sqrt(b+c) on the SAME episodes, where
       c = episodes the candidate cleared but the champion didn't, b = the reverse. So a
       change is kept only on a REAL, paired improvement — it can climb or hold, never
       backslide, and it can't win by luck (CRN removes the luck).
WHEN:  2026-07-20.
INTERCONNECTED WITH: evaluation/consistency (evaluate + auto_ranges), training/fastsim,
       training/gpu_rollout (rollout+ppo_update), training/policy.Brain, configs/rewards
       + training (base weights the knobs perturb).
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  review-hardened (4-agent: durability/correctness/two-inputs/speed): adopt_gate
  gets a continuity correction + strict z=2.33 + a min-disagreements floor (kills the
  winner's-curse ratchet-drift); mutate is CUDA-generator safe; probe snapshots & restores
  sim.w (no leak between candidates); base values now come from the config one-door
  (rewards.yaml + training.yaml) so there is no second copy to drift — WHY: the ratchet is
  the whole safety story; it must not creep on noise, and the two-inputs invariant needs a
  single source of truth.
- 2026-07-20  created — WHY: Phase 1, the self-tuner. Core = the paired 'more-consistent'
  adopt gate (the ratchet) + champion-keeping + plateau-explore + day-after-day streak.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import math
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.policy import Brain                           # noqa: E402
from training.gpu_rollout import rollout, ppo_update        # noqa: E402
from evaluation.consistency import evaluate, auto_ranges    # noqa: E402
from core.configs import load, training_cfg                 # noqa: E402


# The knobs the tuner may move. BOUNDS keep every proposal sane and live HERE; the base
# VALUES do NOT — they come from the config one-door (rewards.yaml for the reward/penalty
# weights, training.yaml ppo: for lr/entropy) via base_config(), so the champion starts
# from exactly the numbers on disk and there is no second copy to drift out of sync.
BOUNDS = {
    "w_death_penalty":     (-40.0, -1.0),
    "w_did_nothing":       (-25.0,  0.0),
    "w_idleness_hunger":   (-0.05,  0.0),
    "w_day_goal_hit":      ( 0.0,  12.0),
    "w_streak_per_day":    ( 0.0,   3.0),
    "w_trade_consistency": ( 0.0,   3.0),
    "w_net_profit":        ( 0.5,  25.0),
    "w_no_drawdown_close": ( 0.0,   1.0),
    "lr":                  (1e-5,  3e-3),
    "entropy_coef":        ( 0.0,   0.1),
}
_PPO_KEYS = {"lr", "entropy_coef"}   # these live under training.yaml ppo:, the rest in rewards.yaml
# last-resort defaults, used ONLY if a key is absent from BOTH config files (so a stripped
# config still boots). configs/ remains the real source when the keys are present.
_FALLBACK = {"w_death_penalty": -10.0, "w_did_nothing": -6.0, "w_idleness_hunger": -0.002,
             "w_day_goal_hit": 2.0, "w_streak_per_day": 0.15, "w_trade_consistency": 0.10,
             "w_net_profit": 6.0, "w_no_drawdown_close": 0.02, "lr": 3e-4, "entropy_coef": 0.01}


def base_config() -> dict:
    """The champion's starting knobs, READ FROM THE CONFIG ONE-DOOR (rewards.yaml +
    training.yaml). Bounds are applied so a hand-edited config can never start out of range."""
    rw = load("rewards")
    ppo = (training_cfg().get("ppo") or {})
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        src = ppo if k in _PPO_KEYS else rw
        v = float(src.get(k, _FALLBACK[k]))
        out[k] = float(min(max(v, lo), hi))
    return out


def mutate(config: dict, scale: float, gen: "torch.Generator | None" = None) -> dict:
    """Propose a candidate: jitter each knob by a log-normal step of size `scale`
    (grows on plateau), clamped to its bounds. scale=0 => the champion unchanged.
    CUDA-safe: the random step is drawn on the generator's own device."""
    out = dict(config)
    dev = gen.device if gen is not None else "cpu"
    for k, (lo, hi) in BOUNDS.items():
        step = float(torch.randn(1, generator=gen, device=dev).item()) * scale
        v = config[k] * math.exp(step) if config[k] != 0 else \
            (lo + hi) * 0.5 * (math.exp(step) - 1.0)   # let a zeroed knob leave zero
        out[k] = float(min(max(v, lo), hi))
    return out


def adopt_gate(champ_mask: torch.Tensor, cand_mask: torch.Tensor,
               z: float = 2.33, d_min: int = 12) -> tuple:
    """THE RATCHET. champ_mask/cand_mask: booleans over the SAME episodes (common random
    numbers). Adopt iff the candidate clears significantly MORE of them than it loses:
      c = cand cleared & champ didn't ; b = champ cleared & cand didn't
      adopt iff  (c - b - 1) / sqrt(b + c) >= z   AND  (b + c) >= d_min   AND  c > b
    The -1 is Edwards' CONTINUITY CORRECTION — the plain McNemar z is anti-conservative at
    small counts (it would wave through lucky candidates); the correction + the d_min floor
    (need at least d_min differing days before the gate may fire) stop the ratchet from
    creeping upward on noise. z defaults to 2.33 (~1% one-sided) for the ADOPT decision; the
    caller may pass a looser z (e.g. 1.64) for a cheap first-round SCREEN before confirming
    the single best at the strict z on a FRESH seed. Returns (adopt: bool, info)."""
    champ = champ_mask.bool(); cand = cand_mask.bool()
    b = int((champ & ~cand).sum().item())
    c = int((~champ & cand).sum().item())
    disagree = b + c
    stat = (c - b - 1) / math.sqrt(disagree) if disagree > 0 else 0.0
    adopt = (disagree >= d_min) and (c > b) and (stat >= z)
    return adopt, {"b": b, "c": c, "stat": round(stat, 3), "disagree": disagree,
                   "z": z, "d_min": d_min}


@torch.no_grad()
def day_after_day_streak(brain, sim, ordered_days, gen=None, focus_frac=0.6,
                         decide_every=5, ranges=None) -> dict:
    """HEADLINE metric: walk the days in calendar order (each once, random X) and return
    the longest run of consecutive CLEARED days — 'day after day, in a row'."""
    r = ranges or auto_ranges()
    dev = sim.dev
    di = torch.as_tensor(ordered_days, dtype=torch.long, device=dev)
    n = int(di.numel())
    tg = torch.empty(n, device=dev).uniform_(r["tgt_lo"], r["tgt_hi"], generator=gen)
    rk = torch.empty(n, device=dev).uniform_(r["risk_lo"], r["risk_hi"], generator=gen)
    if focus_frac > 0:
        m = torch.rand(n, device=dev, generator=gen) < focus_frac
        tg = torch.where(m, torch.full_like(tg, r["focus_target"]), tg)
        rk = torch.where(m, torch.full_like(rk, r["focus_risk"]), rk)
    res = rollout(brain, sim, di, tg, rk, greedy=True, collect=False, decide_every=decide_every)
    cleared = res["goal_hit"].bool().tolist()      # in calendar order
    best = cur = 0
    for cl in cleared:
        cur = cur + 1 if cl else 0
        best = max(best, cur)
    return {"longest_streak": best, "clear_rate": float(sum(cleared) / max(n, 1)), "n": n}


def probe(config: dict, sim, train_pool, brain0, n_updates: int, obs_dim: int,
          instances: int = 2048, decide_every: int = 5, gamma: float = 0.999,
          lam: float = 0.95, clip: float = 0.2, env_mb: int = 256, gen=None) -> Brain:
    """Short training run of a CLONE of the champion brain under `config`. Returns the
    trained candidate brain. (Reward knobs go into the sim; lr/entropy into PPO.)
    The sim is SHARED across candidates, so this snapshots sim.w and restores it in a
    finally — a candidate's reward weights never leak into the next probe or the champion."""
    dev = sim.dev
    w_backup = dict(sim.w)                                 # snapshot the shared reward weights
    try:
        sim.w.update({k: float(v) for k, v in config.items() if k in sim.w})   # candidate rewards
        brain = Brain(obs_dim, hidden=128).to(dev)
        brain.load_state_dict(brain0.state_dict())                              # warm-start from champion
        opt = torch.optim.Adam(brain.parameters(), lr=float(config.get("lr", 3e-4)))
        r = auto_ranges()
        pool = torch.as_tensor(train_pool, dtype=torch.long, device=dev)
        for _ in range(n_updates):
            di = pool[torch.randint(pool.numel(), (instances,), device=dev, generator=gen)]
            tg = torch.empty(instances, device=dev).uniform_(r["tgt_lo"], r["tgt_hi"], generator=gen)
            rk = torch.empty(instances, device=dev).uniform_(r["risk_lo"], r["risk_hi"], generator=gen)
            m = torch.rand(instances, device=dev, generator=gen) < 0.6
            tg = torch.where(m, torch.full_like(tg, r["focus_target"]), tg)
            rk = torch.where(m, torch.full_like(rk, r["focus_risk"]), rk)
            stored = rollout(brain, sim, di, tg, rk, greedy=False, collect=True, decide_every=decide_every)
            ppo_update(brain, opt, stored, sim.days_obs, gamma=gamma, lam=lam, clip=clip,
                       epochs=1, ent_coef=float(config.get("entropy_coef", 0.01)), env_mb=env_mb)
        return brain
    finally:
        sim.w.clear(); sim.w.update(w_backup)             # restore — no cross-candidate leakage


# --------------------- adopt-gate test (fast, no training) ---------------------
if __name__ == "__main__":
    def mask(cleared_idx, n):
        m = torch.zeros(n, dtype=torch.bool); m[list(cleared_idx)] = True; return m

    n = 200
    champ       = mask(range(0, 100), n)                                  # champion clears the first 100
    cand_better = mask(range(55, 195), n)                                 # gains many, loses few -> ADOPT
    cand_lucky  = mask(list(range(0, 95)) + list(range(100, 115)), n)     # a small edge (c=15,b=5) the OLD
                                                                          # lenient gate waved through
                                                                          # (winner's curse) -> now REJECT
    cand_noise  = mask(range(3, 103), n)                                  # tiny shuffle, too few disagreements -> REJECT (d_min)
    cand_worse  = mask(range(140, 200), n)                                # clears far fewer -> REJECT

    print("adopt gate (z=2.33, continuity-corrected, d_min=12):")
    for name, cand, want in [("clearly better", cand_better, True),
                             ("lucky small edge", cand_lucky, False),
                             ("within noise", cand_noise, False),
                             ("worse", cand_worse, False),
                             ("identical", champ.clone(), False)]:
        adopt, info = adopt_gate(champ, cand)
        ok = "PASS" if adopt == want else "FAIL"
        print("  %-17s -> adopt=%-5s want=%-5s  (c=%d gained, b=%d lost, disagree=%d, stat=%s)  [%s]"
              % (name, adopt, want, info["c"], info["b"], info["disagree"], info["stat"], ok))

    # mutate sanity: champion unchanged at scale 0, moves at scale>0, always in bounds;
    # base_config now reads its starting values from the config one-door.
    g = torch.Generator().manual_seed(0)
    b0 = base_config()
    same = mutate(b0, 0.0, g)
    moved = mutate(b0, 0.3, g)
    in_bounds = all(BOUNDS[k][0] <= moved[k] <= BOUNDS[k][1] for k in BOUNDS)
    print("base_config (read from configs/):", {k: round(v, 4) for k, v in b0.items()})
    print("mutate scale0 == base:", all(abs(same[k]-b0[k]) < 1e-9 for k in b0),
          "| scale.3 in bounds:", in_bounds)
