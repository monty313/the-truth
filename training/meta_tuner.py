"""Self-tuner: rewards/hypers toward consistency. Target% and risk% are RUNTIME inputs only.

CHANGE LOG:
- 2026-07-25  restore on main; w_pullback_with_htf fallback 0.25
- 2026-07-24  unlock w_pullback_with_htf in BOUNDS
- 2026-07-20  Phase 1 self-tuner
"""
from __future__ import annotations
import math
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.policy import Brain  # noqa: E402
from training.gpu_rollout import rollout, ppo_update  # noqa: E402
from evaluation.consistency import evaluate, auto_ranges  # noqa: E402
from core.configs import load, training_cfg, path, policy_hidden  # noqa: E402

BOUNDS = {
    "w_death_penalty": (-40.0, -1.0),
    "w_did_nothing": (-25.0, 0.0),
    "w_idleness_hunger": (-0.05, 0.0),
    "w_day_goal_hit": (0.0, 12.0),
    "w_streak_per_day": (0.0, 3.0),
    "w_trade_consistency": (0.0, 3.0),
    "w_net_profit": (0.5, 25.0),
    "w_no_drawdown_close": (0.0, 1.0),
    "w_pullback_with_htf": (0.0, 1.0),
    "lr": (1e-5, 3e-3),
    "entropy_coef": (0.0, 0.1),
}
_PPO_KEYS = {"lr", "entropy_coef"}
_FALLBACK = {
    "w_death_penalty": -10.0, "w_did_nothing": -6.0, "w_idleness_hunger": -0.002,
    "w_day_goal_hit": 2.0, "w_streak_per_day": 0.15, "w_trade_consistency": 0.10,
    "w_net_profit": 6.0, "w_no_drawdown_close": 0.02, "w_pullback_with_htf": 0.25,
    "lr": 3e-4, "entropy_coef": 0.01,
}

def base_config() -> dict:
    rw = load("rewards")
    ppo = training_cfg().get("ppo") or {}
    out = {}
    for k, (lo, hi) in BOUNDS.items():
        src = ppo if k in _PPO_KEYS else rw
        v = float(src.get(k, _FALLBACK[k]))
        out[k] = float(min(max(v, lo), hi))
    return out

def mutate(config: dict, scale: float, gen=None, max_knobs: int = 3) -> dict:
    keys = list(BOUNDS.keys())
    out = dict(config)
    for _ in range(max(1, min(max_knobs, len(keys)))):
        j = int(torch.randint(0, len(keys), (1,), generator=gen).item())
        k = keys[j]
        lo, hi = BOUNDS[k]
        noise = float(torch.randn(1, generator=gen).item()) * scale * (hi - lo)
        out[k] = float(min(max(out[k] + noise, lo), hi))
    return out

def adopt_gate(b: int, c: int, z: float = 2.33, min_disagreements: int = 5) -> bool:
    if (b + c) < min_disagreements:
        return False
    return (c - b) >= z * math.sqrt(b + c + 1e-9)

def day_after_day_streak(brain, sim, ordered_days, gen=None, focus_frac=0.6,
                         decide_every=1, ranges=None):
    r = ranges or auto_ranges()
    ff = float(r.get("focus_frac", focus_frac))
    device = next(brain.parameters()).device
    gen = gen or torch.Generator(device="cpu").manual_seed(0)
    cleared = []
    for d in ordered_days:
        di = torch.tensor([int(d)], device=device)
        tg = torch.full((1,), float(r.get("tgt_lo", r.get("focus_target", 3.0))), device=device)
        rk = torch.full((1,), float(r.get("risk_lo", r.get("focus_risk", 3.5))), device=device)
        m = torch.rand(1, generator=gen) < ff
        if "focus_target" in r:
            tg = torch.where(m, torch.full_like(tg, float(r["focus_target"])), tg)
            rk = torch.where(m, torch.full_like(rk, float(r["focus_risk"])), rk)
        out = rollout(brain, sim, di, tg, rk, greedy=True, collect=False, decide_every=decide_every)
        pnl = float(out["day_pnl"][0].item())
        cleared.append(1 if (pnl >= float(tg[0]) and pnl > -float(rk[0])) else 0)
    longest = cur = 0
    for x in cleared:
        cur = cur + 1 if x else 0
        longest = max(longest, cur)
    return {"longest_streak": longest, "cleared": cleared}

def apply_config_to_sim(sim, config: dict) -> None:
    if hasattr(sim, "w") and isinstance(sim.w, dict):
        for k, v in config.items():
            if k not in _PPO_KEYS:
                sim.w[k] = float(v)

def probe(brain, sim, work_days, config, *, decide_every=1, steps=32):
    apply_config_to_sim(sim, config)
    device = next(brain.parameters()).device
    opt = torch.optim.Adam(brain.parameters(), lr=float(config.get("lr", 3e-4)))
    for i in range(max(1, steps)):
        di = torch.tensor([int(work_days[j % len(work_days)]) for j in range(min(8, len(work_days)))], device=device)
        tg = torch.empty(len(di), device=device).uniform_(2.5, 5.0)
        rk = torch.empty(len(di), device=device).uniform_(1.5, 4.0)
        batch = rollout(brain, sim, di, tg, rk, greedy=False, collect=True, decide_every=decide_every)
        ppo_update(brain, opt, batch, entropy_coef=float(config.get("entropy_coef", 0.01)))
    return brain

def run(sim, obs_dim: int, work_days, audit_days, *, minutes: float = 60.0,
        decide_every: int = 1, device: str = "cpu"):
    t_end = time.time() + minutes * 60.0
    champion_cfg = base_config()
    brain = Brain(obs_dim, hidden=policy_hidden()).to(device)
    for name in ("PROVEN_SPRINT_row04_clear24_2026-07-20", "lift_best", "PROVEN_LIFT_2026-07-20",
                 "gpu_best", "PROVEN_2x_2026-07-19", "best_trading"):
        p = path("artifacts", f"{name}.pt")
        if os.path.isfile(p):
            try:
                ck = torch.load(p, map_location=device)
                brain.load_state_dict(ck.get("model", ck), strict=False)
                break
            except Exception:
                pass
    gen = torch.Generator(device="cpu").manual_seed(0)
    ranges = auto_ranges()
    champ_res = evaluate(brain, sim, list(audit_days), n_episodes=len(audit_days),
        focus_frac=float(ranges.get("focus_frac", 0.6)), decide_every=decide_every, gen=gen, ranges=ranges)
    champion_score = float(champ_res.get("consistency", 0.0))
    history = []
    gen_id = 0
    while time.time() < t_end:
        gen_id += 1
        cand_cfg = mutate(champion_cfg, scale=0.15, gen=gen)
        w_snap = dict(getattr(sim, "w", {}) or {})
        try:
            probe(brain, sim, work_days, cand_cfg, decide_every=decide_every, steps=32)
            gen2 = torch.Generator(device="cpu").manual_seed(0)
            res = evaluate(brain, sim, list(audit_days), n_episodes=len(audit_days),
                focus_frac=float(ranges.get("focus_frac", 0.6)), decide_every=decide_every, gen=gen2, ranges=ranges)
            c = int(max(0, round((res.get("consistency", 0) - champion_score) * len(audit_days))))
            b = int(max(0, round((champion_score - res.get("consistency", 0)) * len(audit_days))))
            if adopt_gate(b, c) and float(res.get("breach_rate", 1)) <= float(champ_res.get("breach_rate", 1)) + 1e-9:
                champion_cfg, champion_score, champ_res = cand_cfg, float(res.get("consistency", 0)), res
                history.append({"gen": gen_id, "adopted": True, "consistency": champion_score})
            else:
                history.append({"gen": gen_id, "adopted": False, "consistency": float(res.get("consistency", 0))})
        finally:
            if hasattr(sim, "w") and w_snap:
                sim.w.update(w_snap)
    return {"champion_config": champion_cfg, "champion_score": champion_score, "history": history, "champ_res": champ_res}

if __name__ == "__main__":
    print("meta_tuner OK", list(BOUNDS.keys()))
    print("Target/risk are runtime inputs: prove_it.py <brain> <tgt> <risk>")
