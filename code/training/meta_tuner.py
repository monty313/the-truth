"""Self-tuner: rewards/hypers toward consistency. Target% and risk% are RUNTIME inputs only.

CHANGE LOG:
- 2026-08-04  Mark-on-chart meta: TREND_KNOBS = force/pullback/quick_pull/setup_skip;
  fallbacks aligned to rewards.yaml Mark law starts; search still never freezes
  answers — WHY: policy=Mark on chart (POLICY_EQUALS_MARK_ON_CHART.md).
- 2026-07-30  Step 2 adaptive de-timid: when WrongSide hot, scale 0.35 +
  force >=2 trend-knob mutations; sticky focus until rulers clear —
  WHY: meta learns where/how hard to search; never freezes dial values.
- 2026-07-30  secondary adopt veto: flat consistency + worse wrong_side_rate —
  WHY: evaluate() now returns side metrics; do not adopt side regressions when clear% flat.
- 2026-07-30  self-heal dials in BOUNDS (with/against trend, quick_pull, setup_skip) — WHY: meta searches side-bias cures; defaults 0.
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

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
    # Self-heal toolkit — meta may raise from 0; never a frozen human answer
    "w_with_trend_close": (0.0, 1.0),
    "w_against_trend_close": (-1.0, 0.0),
    "w_quick_pull_close": (0.0, 0.5),
    "w_setup_skip": (-0.5, 0.0),
    "lr": (1e-5, 3e-3),
    "entropy_coef": (0.0, 0.1),
}
_PPO_KEYS = {"lr", "entropy_coef"}
# Fallbacks = Mark-on-chart starting personality (same spirit as rewards.yaml).
# Meta mutates within BOUNDS; never treat these as frozen cures.
_FALLBACK = {
    "w_death_penalty": -10.0, "w_did_nothing": -6.0, "w_idleness_hunger": -0.002,
    "w_day_goal_hit": 2.0, "w_streak_per_day": 0.15, "w_trade_consistency": 0.10,
    "w_net_profit": 6.0, "w_no_drawdown_close": 0.02, "w_pullback_with_htf": 0.25,
    "w_with_trend_close": 0.05, "w_against_trend_close": -0.05,
    "w_quick_pull_close": 0.08, "w_setup_skip": -0.02,
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

def mutate(config: dict, scale: float, gen=None, max_knobs: int = 3,
           force_keys: list | tuple | None = None, min_force: int = 0) -> dict:
    """Mutate up to max_knobs. If force_keys/min_force set, reserve that many
    slots for the force group (e.g. trend dials under WrongSide pressure).
    Never writes fixed answers — only noise within BOUNDS."""
    keys = list(BOUNDS.keys())
    out = dict(config)
    n = max(1, min(max_knobs, len(keys)))
    force = [k for k in (force_keys or []) if k in BOUNDS]
    n_force = min(max(0, int(min_force)), len(force), n) if force else 0
    picked: list[str] = []
    # forced slots first (with replacement if group smaller than n_force)
    for _ in range(n_force):
        avail = [k for k in force if k not in picked] or force
        j = int(torch.randint(0, len(avail), (1,), generator=gen).item())
        picked.append(avail[j])
    # remaining slots: any knob
    for _ in range(n - n_force):
        j = int(torch.randint(0, len(keys), (1,), generator=gen).item())
        picked.append(keys[j])
    for k in picked:
        lo, hi = BOUNDS[k]
        noise = float(torch.randn(1, generator=gen).item()) * scale * (hi - lo)
        out[k] = float(min(max(out[k] + noise, lo), hi))
    return out

def adopt_gate(b: int, c: int, z: float = 2.33, min_disagreements: int = 5) -> bool:
    if (b + c) < min_disagreements:
        return False
    return (c - b) >= z * math.sqrt(b + c + 1e-9)


# Consistency delta below this is "flat" for the secondary WrongSide veto.
FLAT_CONS_EPS = 1e-3


def side_adopt_ok(cand_cons: float, champ_cons: float,
                  cand_wsr: float, champ_wsr: float,
                  flat_eps: float = FLAT_CONS_EPS) -> bool:
    """Secondary gate: reject if consistency gain is flat AND wrong_side_rate worsens.

    Primary (adopt_gate + breach) still decides statistical clear/breach wins.
    This only vetoes side-regressions when clear% barely moved.
    """
    flat = (float(cand_cons) - float(champ_cons)) < float(flat_eps)
    worse_side = float(cand_wsr) > float(champ_wsr) + 1e-12
    return not (flat and worse_side)


# ---------------------------------------------------------------------------
# Adaptive search (Step 2) — learn-to-learn WHERE and HOW HARD to search.
# Rulers only: if WrongSide is hot, mutate bigger steps on trend dials.
# Does NOT hard-code dial values; primary/secondary adopt gates unchanged.
# Extend TREND_KNOBS later with Vector-specific dials (min_force still applies).
# ---------------------------------------------------------------------------
SCALE_NORMAL = 0.15
SCALE_AGGRESSIVE = 0.35
# Mark-on-chart force group (Law 1–2–5): with/against force, slingshot pullback,
# quick release after breath, skip when setup visible. Meta searches; never freezes.
# min_force <= len(group). Append Vector dials later without removing these.
TREND_KNOBS = (
    "w_with_trend_close",
    "w_against_trend_close",
    "w_pullback_with_htf",
    "w_quick_pull_close",
    "w_setup_skip",
)
WSR_HOT = 0.15
SIDE_BIAS_HOT = -0.03
# policy_hold disease: high hold on visible setups (Mind Probe / IRAC language)
HOLD_ON_SETUP_HOT = 0.35
# Sticky focus: while hot, refresh; after clear, hold several gens then relax.
FOCUS_HOLD_GENS = 5


def wrong_side_hot(res: dict) -> bool:
    """True when any WrongSide ruler is breached (from evaluate() side keys)."""
    wsr = float(res.get("wrong_side_rate", 0.0) or 0.0)
    sb = float(res.get("side_bias_bull", 0.0) or 0.0)
    ss = float(res.get("side_bias_bear", 0.0) or 0.0)
    return (wsr > WSR_HOT) or (sb < SIDE_BIAS_HOT) or (ss < SIDE_BIAS_HOT)


def mark_chart_disease_hot(res: dict) -> bool:
    """True when metrics say the brain is not trading like Mark on chart.

    Extends WrongSide with optional policy_hold / setup-skip style keys when
    evaluate() provides them; missing keys → not hot (no false focus).
    """
    if wrong_side_hot(res):
        return True
    # Optional IRAC / Mind Probe style meters (0 if absent)
    hold_setup = float(res.get("policy_hold_on_setup_rate", 0.0) or 0.0)
    if hold_setup > HOLD_ON_SETUP_HOT:
        return True
    pull_miss = float(res.get("high_miss_pull_rate", 0.0) or 0.0)
    if pull_miss > 0.20:
        return True
    return False


def search_plan(res: dict, focus_left: int) -> tuple[float, list | None, int, int, bool]:
    """Return (scale, force_keys|None, min_force, new_focus_left, focused).

    Adaptive: Mark-on-chart disease (wrong side / policy_hold) → aggressive
    scale + force Mark TREND_KNOBS; sticky focus_left keeps pressure after.
    Does not hard-code dial values — only WHERE/HOW HARD to search.
    """
    hot = mark_chart_disease_hot(res)
    if hot:
        focus_left = FOCUS_HOLD_GENS
    elif focus_left > 0:
        focus_left -= 1
    focused = hot or focus_left > 0
    if focused:
        # Force at least 2 Mark knobs so search cannot ignore force/pullback
        return (SCALE_AGGRESSIVE, list(TREND_KNOBS), 2, focus_left, True)
    return (SCALE_NORMAL, None, 0, focus_left, False)

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
    focus_left = 0  # sticky aggressive generations remaining
    was_focused = False  # for enter/exit history markers
    while time.time() < t_end:
        gen_id += 1
        # Adaptive search from champion side metrics (disease of the incumbent).
        scale, force_keys, min_force, focus_left, focused = search_plan(
            champ_res, focus_left)
        # One-line history marker when focused mode enters or exits.
        focus_event = None
        if focused and not was_focused:
            focus_event = "enter_focused"
        elif (not focused) and was_focused:
            focus_event = "exit_focused"
        was_focused = focused
        cand_cfg = mutate(
            champion_cfg, scale=scale, gen=gen,
            force_keys=force_keys, min_force=min_force)
        w_snap = dict(getattr(sim, "w", {}) or {})
        try:
            probe(brain, sim, work_days, cand_cfg, decide_every=decide_every, steps=32)
            gen2 = torch.Generator(device="cpu").manual_seed(0)
            res = evaluate(brain, sim, list(audit_days), n_episodes=len(audit_days),
                focus_frac=float(ranges.get("focus_frac", 0.6)), decide_every=decide_every, gen=gen2, ranges=ranges)
            c = int(max(0, round((res.get("consistency", 0) - champion_score) * len(audit_days))))
            b = int(max(0, round((champion_score - res.get("consistency", 0)) * len(audit_days))))
            primary = (
                adopt_gate(b, c)
                and float(res.get("breach_rate", 1))
                <= float(champ_res.get("breach_rate", 1)) + 1e-9
            )
            # evaluate() always returns side_bias_* / wrong_side_rate (Step 1 dict shape)
            secondary = side_adopt_ok(
                float(res.get("consistency", 0)), champion_score,
                float(res.get("wrong_side_rate", 0.0)),
                float(champ_res.get("wrong_side_rate", 0.0)),
            )
            row = {
                "gen": gen_id,
                "consistency": float(res.get("consistency", 0)),
                "wrong_side_rate": float(res.get("wrong_side_rate", 0.0)),
                "search": "aggressive" if focused else "normal",
                "scale": scale,
                "focus_left": focus_left,
            }
            if focus_event:
                row["focus_event"] = focus_event
            if primary and secondary:
                champion_cfg, champion_score, champ_res = cand_cfg, float(res.get("consistency", 0)), res
                row.update({
                    "adopted": True,
                    "consistency": champion_score,
                    "side_bias_bull": float(res.get("side_bias_bull", 0.0)),
                })
                history.append(row)
            else:
                row.update({
                    "adopted": False,
                    "primary": primary,
                    "secondary": secondary,
                })
                history.append(row)
        finally:
            if hasattr(sim, "w") and w_snap:
                sim.w.update(w_snap)
    return {"champion_config": champion_cfg, "champion_score": champion_score, "history": history, "champ_res": champ_res}

if __name__ == "__main__":
    print("meta_tuner OK", list(BOUNDS.keys()))
    print("Target/risk are runtime inputs: prove_it.py <brain> <tgt> <risk>")
