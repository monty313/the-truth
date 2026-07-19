"""Meta-optimizer — PROPOSES weight/hparam configs; NEVER adopts (ADR-0007).
5W+I: WHO Claude; adoption ONLY by Monty's recorded OK. WHAT random+evolution
search over reward weights and PPO hparams; each trial = short training probe;
proposals + evidence written to artifacts/proposals/. WHEN 2026-07-19.
WHY 'a ML to help optimize the weights... set it up right the first time'.
INTERCONNECTED: rewards.load_weights, ppo, tracker, docs/adr (adoption ADRs).

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import json, os, random, copy
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "artifacts", "proposals"); os.makedirs(OUT, exist_ok=True)

SEARCH = {  # (min, max) multipliers on current value
    "w_day_goal_hit": (0.5, 3.0), "w_streak_per_day": (0.3, 3.0),
    "w_trade_consistency": (0.3, 3.0), "w_idleness_hunger": (0.2, 5.0),
    "w_death_penalty": (0.5, 2.0), "lr": (0.3, 3.0), "entropy_coef": (0.5, 4.0),
}

def propose(base_weights: dict, base_hp: dict, n: int = 8, seed: int = 1):
    rng = random.Random(seed)
    out = []
    for i in range(n):
        w, hp = copy.deepcopy(base_weights), copy.deepcopy(base_hp)
        for k, (lo, hi) in SEARCH.items():
            m = rng.uniform(lo, hi)
            if k in w: w[k] = round(w[k] * m, 5)
            elif k in hp: hp[k] = round(hp[k] * m, 6)
        out.append({"trial": i, "weights": w, "hparams": hp})
    path = os.path.join(OUT, f"proposal_batch_{seed}.json")
    json.dump({"note": "PROPOSALS ONLY — Monty approves adoption (ADR-0007)",
               "trials": out}, open(path, "w"), indent=2)
    return path
