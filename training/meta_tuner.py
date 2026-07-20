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
import datetime
import hashlib
import json
import math
import os
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from training.policy import Brain                           # noqa: E402
from training.gpu_rollout import rollout, ppo_update        # noqa: E402
from evaluation.consistency import evaluate, auto_ranges    # noqa: E402
from core.configs import load, training_cfg, path            # noqa: E402


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


def mutate(config: dict, scale: float, gen: "torch.Generator | None" = None,
           max_knobs: int = 3) -> dict:
    """Propose a candidate: jitter up to `max_knobs` RANDOMLY-CHOSEN knobs by a log-normal
    step of size `scale` (grows on plateau), clamped to bounds. Moving only a FEW knobs at a
    time (not all 10) keeps proposals near the champion, avoids degenerate box-corner configs
    that could game a small day sample, and makes the search cheaper. scale<=0 => unchanged.
    CUDA-safe: every draw is on the generator's own device."""
    out = dict(config)
    if scale <= 0:
        return out
    dev = gen.device if gen is not None else "cpu"
    keys = list(BOUNDS.keys())
    k_n = min(max_knobs, len(keys))
    pick = torch.randperm(len(keys), generator=gen, device=dev)[:k_n].tolist()
    for j in pick:
        k = keys[j]; lo, hi = BOUNDS[k]
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
    ff = float(r.get("focus_frac", focus_frac))            # single source (auto_ranges) — no hardcoded copy
    dev = sim.dev
    di = torch.as_tensor(ordered_days, dtype=torch.long, device=dev)
    n = int(di.numel())
    tg = torch.empty(n, device=dev).uniform_(r["tgt_lo"], r["tgt_hi"], generator=gen)
    rk = torch.empty(n, device=dev).uniform_(r["risk_lo"], r["risk_hi"], generator=gen)
    if ff > 0:
        m = torch.rand(n, device=dev, generator=gen) < ff
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
          lam: float = 0.95, clip: float = 0.2, env_mb: int = 256, gen=None,
          ranges: dict | None = None) -> Brain:
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
        r = ranges or auto_ranges()
        ff = float(r.get("focus_frac", 0.6))               # single source (auto_ranges) — no hardcoded copy
        pool = torch.as_tensor(train_pool, dtype=torch.long, device=dev)
        for _ in range(n_updates):
            di = pool[torch.randint(pool.numel(), (instances,), device=dev, generator=gen)]
            tg = torch.empty(instances, device=dev).uniform_(r["tgt_lo"], r["tgt_hi"], generator=gen)
            rk = torch.empty(instances, device=dev).uniform_(r["risk_lo"], r["risk_hi"], generator=gen)
            m = torch.rand(instances, device=dev, generator=gen) < ff
            tg = torch.where(m, torch.full_like(tg, r["focus_target"]), tg)
            rk = torch.where(m, torch.full_like(rk, r["focus_risk"]), rk)
            stored = rollout(brain, sim, di, tg, rk, greedy=False, collect=True, decide_every=decide_every)
            ppo_update(brain, opt, stored, sim.days_obs, gamma=gamma, lam=lam, clip=clip,
                       epochs=1, ent_coef=float(config.get("entropy_coef", 0.01)), env_mb=env_mb)
        return brain
    finally:
        sim.w.clear(); sim.w.update(w_backup)             # restore — no cross-candidate leakage


# ============================ the autonomous loop ============================
# CRN discipline: every draw is seeded from (base_seed, generation, purpose) so a run is
# fully reproducible AND champion vs candidate are scored on IDENTICAL episodes within a
# generation (common random numbers -> the gate can't win on luck), while each new
# generation gets FRESH episodes (so a lucky screen can't survive a fresh confirm).
_I63 = (1 << 63) - 1


def _seed_from(base: int, *parts) -> int:
    h = hashlib.sha256(("%d|" % int(base) + "|".join(str(p) for p in parts)).encode()).hexdigest()
    return int(h[:15], 16) % _I63


def _gen(dev, seed: int):
    return torch.Generator(device=dev).manual_seed(int(seed) % _I63)


def hold_out_audit(n_days: int, audit_frac: float = 0.15):
    """Split the FIXED day pool into a PERMANENT held-out AUDIT slice (never used to pick or
    adopt a change — only the honest headline streak + the out-of-sample adopt tripwire read
    it) and the WORK pool (train+select). Same days, only role assignment."""
    if n_days <= 2:
        idx = list(range(n_days)); return idx, idx
    n_aud = min(max(1, int(round(n_days * audit_frac))), n_days - 1)
    idx = list(range(n_days))
    return idx[:-n_aud], idx[-n_aud:]                       # work, audit


def _rotate_split(work_days, base_seed: int, gen_idx: int, sel_frac: float = 0.30):
    """Each generation, re-partition the WORK pool into DISJOINT (train, select) with a fresh
    seeded shuffle. Rotating which days select vs train stops the tuner from overfitting knobs
    to one fixed handful of price paths, WITHOUT adding new days (Monty's fixed-pool rule)."""
    work = list(work_days)
    if len(work) <= 2:
        return work, work
    gg = torch.Generator().manual_seed(_seed_from(base_seed, gen_idx, "split"))
    perm = torch.randperm(len(work), generator=gg).tolist()
    n_sel = min(max(1, int(round(len(work) * sel_frac))), len(work) - 1)
    sel = [work[i] for i in perm[:n_sel]]
    train = [work[i] for i in perm[n_sel:]]
    return train, sel


def _day_paired(champ_cleared: torch.Tensor, cand_cleared: torch.Tensor, day_idx: torch.Tensor):
    """Collapse per-episode clears to ONE paired outcome per DISTINCT DAY — the fix for the
    pseudo-replication that made the gate's strictness illusory (many correlated episodes on
    the same day were being counted as independent). Returns boolean masks over distinct days:
    champ_wins[k]=champ cleared a strictly higher fraction of day k's episodes; cand_wins the
    reverse; a tied day is in neither. Feed these to adopt_gate so b,c count DAYS, not episodes."""
    ch = champ_cleared.detach().float().cpu()
    ca = cand_cleared.detach().float().cpu()
    di = day_idx.detach().cpu()
    days = torch.unique(di)
    champ_wins = torch.zeros(days.numel(), dtype=torch.bool)
    cand_wins = torch.zeros(days.numel(), dtype=torch.bool)
    for k in range(days.numel()):
        m = (di == days[k])
        a = float(ch[m].mean()); b = float(ca[m].mean())
        if b > a:
            cand_wins[k] = True
        elif a > b:
            champ_wins[k] = True
    return champ_wins, cand_wins


def save_champion(pth: str, brain, config: dict, meta: dict):
    """Resumable champion = brain weights + its tuned knobs + loop bookkeeping. One file,
    overwritten each generation; it rides the Drive symlink, so a Colab restart resumes here."""
    tmp = pth + ".tmp"
    torch.save({"model": brain.state_dict(), "config": dict(config), "meta": dict(meta),
                "obs_dim": int(meta.get("obs_dim", 0)), "seed": int(meta.get("base_seed", 0))}, tmp)
    os.replace(tmp, pth)                                   # atomic: a crash mid-save can't corrupt the champion


def load_champion(pth: str, obs_dim: int, dev):
    if not os.path.exists(pth):
        return None
    try:
        d = torch.load(pth, weights_only=False, map_location=dev)
    except Exception:
        return None
    return d if int(d.get("obs_dim", -1)) == int(obs_dim) else None


def _save_record(hist_dir: str, ckpt_dir: str, brain, config: dict, streak: int,
                 obs_dim: int, base_seed: int, gen: int):
    """FROZEN, revertible per-best snapshot with the passed-day count in the filename
    (Monty's spec) + a plain-language history line. Same payload shape as the GPU trainer,
    so a frozen brain plugs straight into the real bot."""
    payload = {"model": brain.state_dict(), "config": dict(config), "obs_dim": int(obs_dim),
               "reward_state": {k: float(v) for k, v in config.items()}, "seed": int(base_seed)}
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    tmp = os.path.join(hist_dir, "_writing.pt"); torch.save(payload, tmp)
    hh = hashlib.sha256(open(tmp, "rb").read()).hexdigest()[:12]
    frozen = "momentum_meta_pass%04d_%s_%s.pt" % (int(streak), stamp, hh)
    os.replace(tmp, os.path.join(hist_dir, frozen))
    mb_tmp = os.path.join(ckpt_dir, "meta_best.pt.tmp")          # newest best, easy to grab (atomic)
    torch.save(payload, mb_tmp); os.replace(mb_tmp, os.path.join(ckpt_dir, "meta_best.pt"))
    md = os.path.join(os.path.dirname(ckpt_dir), "META_TRAINING_HISTORY.md")
    first = not os.path.exists(md)
    with open(md, "a") as fh:
        if first:
            fh.write("# Self-tuner — record clean streaks (newest at bottom)\n"
                     "Each entry is a FROZEN, revertible brain + the tuned knobs that made it.\n"
                     "Same shape as the real bot, so it plugs straight in.\n\n")
        fh.write("## %d cleared days in a row — %s (gen %d)\n"
                 "- frozen: artifacts/checkpoints/history/%s (sha256[:12] %s)\n"
                 "- also: artifacts/checkpoints/meta_best.pt\n\n"
                 % (int(streak), stamp, gen, frozen, hh))
    return frozen


_EXPLORE_CAP = 0.8      # widen only up to here (below 1.0 avoids the worst degenerate corners)


def run(sim, obs_dim: int, work_days, audit_days, *, minutes: float = 1440.0,
        n_updates: int = 4, instances: int = 2048, decide_every: int = 5,
        candidates: int | None = None, n_eval: int | None = None,
        ckpt_dir: str | None = None, base_seed: int | None = None, log=print) -> dict:
    """THE self-tuning loop. Each generation:
      1) ROTATE the fixed work pool into fresh disjoint (train, select) days — rotating stops
         the tuner overfitting knobs to one handful of price paths (no new days added);
      2) propose K candidate knob-sets — index 0 is the champion UNCHANGED (an equal-budget
         'just more training' control), 1..K-1 mutate a FEW knobs by `explore`;
      3) PROBE each (short train of a champion CLONE under its knobs) -> a candidate brain;
      4) SCREEN on the select days (one shared seed): a mutation is eligible only if it beats
         the equal-budget control by a PAIRED, DAY-CLUSTERED margin (screen_z) — credit the
         change, not the extra steps; pick the single best by consistency;
      5) CONFIRM that best vs the FROZEN champion on TWO independent fresh seeds, DAY-CLUSTERED,
         at confirm_z (two fresh wins square the per-gen false-adopt rate -> O(1) over 1000s of gens);
      6) NON-BACKSLIDE: score it on the PERMANENT held-out audit days at a FIXED reference and
         block the adopt if the frozen ANCHOR (best-ever held-out champion) paired-beats it — so
         an adopt can never drop the champion significantly below its best out-of-sample level;
      ADOPT only if BOTH confirms AND the audit floor pass (else keep the champion). These fresh,
      day-clustered hurdles (day-clustering kills pseudo-replication) make a ratchet that climbs
      or holds, never backslides, and can't win on luck. The anchor rises ONLY on a paired OOS win.
      7) read the honest streak on the audit days ONLY when the champion changed; save a new
         best (passed-days in the name); checkpoint so a Colab restart resumes exactly here.
    On `plateau_patience` no-adopt gens, widen exploration to a cap, then reset small (sawtooth,
    never pinned wide); the champion is never lost. Only human inputs remain target%/risk%."""
    dev = sim.dev
    tc = training_cfg(); st = dict(tc.get("self_tuner", {})); ppo = dict(tc.get("ppo", {}))
    K = int(candidates if candidates is not None else st.get("candidates_per_gen", 24))
    confirm_z = float(st.get("confirm_z", 2.0)); screen_z = float(st.get("screen_z", 1.64))
    d_min = int(st.get("min_disagreements", 4))            # in DISTINCT DAYS now (day-clustered gate)
    patience = int(st.get("plateau_patience", 3)); scale0 = float(st.get("explore_scale", 0.25))
    sel_frac = float(st.get("sel_frac", 0.30)); mut_knobs = int(st.get("mutate_knobs", 3))
    n_eval = int(n_eval if n_eval is not None else st.get("eval_episodes", 512))
    gamma = float(ppo.get("gamma", 0.999)); lam = float(ppo.get("lam", 0.95)); clip = float(ppo.get("clip", 0.2))
    base_seed = int(base_seed if base_seed is not None else tc.get("seed", 20260718))
    ranges = auto_ranges()                                  # frozen envelope for the whole run

    ckpt_dir = ckpt_dir or path("artifacts", "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    hist_dir = os.path.join(ckpt_dir, "history"); os.makedirs(hist_dir, exist_ok=True)
    champ_path = os.path.join(ckpt_dir, "meta_champion.pt")

    champ_brain = Brain(obs_dim, hidden=128).to(dev)
    champ_cfg = base_config()
    gen0 = 0; best_streak = 0; explore = scale0; stale = 0
    ck = load_champion(champ_path, obs_dim, dev)
    if ck is not None:                                      # RESUME (Drive-persisted)
        champ_brain.load_state_dict(ck["model"]); champ_cfg = {**champ_cfg, **ck.get("config", {})}
        m = ck.get("meta", {}); gen0 = int(m.get("gen", 0)); best_streak = int(m.get("best_streak", 0))
        explore = float(m.get("explore", scale0)); stale = int(m.get("stale", 0))
        log("resume: gen %d | best streak %d | explore %.3f" % (gen0, best_streak, explore))
    else:                                                  # else warm-start from the best PROFITABLE brain first
        # lift_best = the brain that PROVED it banks the target (2026-07-20 lift demo);
        # seeding the tuner from a brain that already makes money beats seeding from a
        # flat-but-safe one — the ratchet then hones consistency instead of digging out of zero.
        for name in ("lift_best", "gpu_best", "PROVEN_2x_2026-07-19", "best_trading"):
            p = os.path.join(ckpt_dir, name + ".pt")
            if os.path.exists(p):
                d = torch.load(p, weights_only=False, map_location=dev)
                if int(d.get("obs_dim", -1)) == obs_dim:
                    try:
                        champ_brain.load_state_dict(d["model"]); log("warm-start champion from %s" % name); break
                    except Exception:
                        pass

    def _score(brain, days, seed):
        return evaluate(brain, sim, days, n_episodes=n_eval, gen=_gen(dev, seed),
                        decide_every=decide_every, ranges=ranges)

    def _beats(a_res, b_res, z):
        """day-clustered paired test: does b clear significantly MORE DAYS than a? (CRN: a,b
        scored on identical episodes, so their day_idx align)."""
        assert torch.equal(a_res["day_idx"], b_res["day_idx"]), "CRN broken: day_idx misaligned"
        aw, bw = _day_paired(a_res["cleared_mask"], b_res["cleared_mask"], a_res["day_idx"])
        ok, info = adopt_gate(aw, bw, z=z, d_min=d_min)
        return ok, info

    # OUT-OF-SAMPLE ANCHOR (non-backslide): a FROZEN best-held-out champion + its scorecard on
    # a FIXED audit reference (same days, same X every time). An adopt is blocked if the anchor
    # PAIRED-BEATS the candidate out of sample (day-clustered) -> the champion can never drop
    # significantly below its best held-out level. The anchor is raised ONLY when a candidate
    # PAIRED-BEATS it out of sample, so it can't inflate on a lucky point estimate.
    audit_ref = _seed_from(base_seed, "audit-ref")
    anchor_path = os.path.join(ckpt_dir, "meta_anchor.pt")
    anchor_brain = Brain(obs_dim, hidden=128).to(dev)
    anc = load_champion(anchor_path, obs_dim, dev)
    if anc is not None:
        anchor_brain.load_state_dict(anc["model"])
    else:                                                  # first ever: anchor = the warm-start champion
        anchor_brain.load_state_dict(champ_brain.state_dict())
        save_champion(anchor_path, anchor_brain, champ_cfg, {"obs_dim": obs_dim})
    anchor_aud = _score(anchor_brain, audit_days, audit_ref)          # cached held-out scorecard (fixed seed)
    anchor_cons = anchor_aud["consistency"]
    log("audit anchor: consistency %.3f on %d held-out days (fixed reference)" % (anchor_cons, len(audit_days)))

    t0 = time.time(); g = gen0
    while time.time() - t0 < minutes * 60.0:
        g += 1
        train_days, sel_days = _rotate_split(work_days, base_seed, g, sel_frac)   # fresh disjoint split
        mgen = _gen(dev, _seed_from(base_seed, g, "mutate"))
        configs = [dict(champ_cfg)] + [mutate(champ_cfg, explore, mgen, max_knobs=mut_knobs)
                                       for _ in range(max(K - 1, 1))]

        brains = []                                        # PROBE (the expensive part)
        p_seed = _seed_from(base_seed, g, "probe")         # SAME train sample for every candidate this gen ->
        for cfg in configs:                                # the equal-budget control differs ONLY by config
            brains.append(probe(cfg, sim, train_days, champ_brain, n_updates, obs_dim,
                                instances=instances, decide_every=decide_every, gamma=gamma,
                                lam=lam, clip=clip, ranges=ranges, gen=_gen(dev, p_seed)))

        s_seed = _seed_from(base_seed, g, "screen")        # SCREEN on ONE shared seed (CRN)
        screens = [_score(b, sel_days, s_seed) for b in brains]
        ctrl = screens[0]                                  # index 0 = equal-budget 'more training' control
        eligible = [i for i in range(1, len(brains)) if _beats(ctrl, screens[i], screen_z)[0]]
        if eligible:
            best_i = max(eligible, key=lambda i: (screens[i]["consistency"], screens[i]["surrogate"]))
            took = "config"
        else:
            best_i = 0; took = "more-training"             # no knob beat 'just more steps' -> test that

        # HURDLE 1 — TWO independent fresh confirms vs the FROZEN champion on the rotating
        # select days. Requiring a win on TWO fresh seeds squares the per-gen false-adopt rate,
        # so cumulative false adopts stay O(1) over thousands of generations (alpha control).
        c1 = _seed_from(base_seed, g, "confirm")
        champ_conf = _score(champ_brain, sel_days, c1)
        cand_conf = _score(brains[best_i], sel_days, c1)
        passed_sel, info = _beats(champ_conf, cand_conf, confirm_z)
        if passed_sel:
            c2 = _seed_from(base_seed, g, "confirm2")
            passed_sel = _beats(_score(champ_brain, sel_days, c2),
                                _score(brains[best_i], sel_days, c2), confirm_z)[0]

        # HURDLE 2 — out-of-sample NON-BACKSLIDE: score the candidate on the FIXED audit reference
        # and block the adopt if the frozen ANCHOR paired-beats it out of sample (day-clustered).
        # So an adopt can never push the champion significantly below its best held-out level.
        adopt = False; cand_aud = None; anchor_blocks = False
        if passed_sel:
            cand_aud = _score(brains[best_i], audit_days, audit_ref)
            anchor_blocks = _beats(cand_aud, anchor_aud, screen_z)[0]     # does the anchor win out of sample?
            adopt = not anchor_blocks

        if adopt:
            champ_brain = brains[best_i]; champ_cfg = configs[best_i]
            explore = scale0; stale = 0
            cand_reanchors = _beats(anchor_aud, cand_aud, screen_z)[0]    # candidate paired-BEATS anchor OOS?
            if cand_reanchors:                              # conservative re-anchor -> the safety net can only rise
                anchor_brain.load_state_dict(champ_brain.state_dict())
                anchor_aud = cand_aud; anchor_cons = cand_aud["consistency"]
                save_champion(anchor_path, champ_brain, champ_cfg, {"obs_dim": obs_dim, "anchor_cons": anchor_cons})
            log("gen %d ADOPT (%s) | 2x fresh confirm (c=%d b=%d z=%.2f) | audit %.3f vs anchor %.3f%s"
                % (g, took, info["c"], info["b"], info["stat"], cand_aud["consistency"],
                   anchor_cons, " NEW ANCHOR" if cand_reanchors else ""))
        elif not passed_sel:                                # a real SELECT miss counts toward plateau-widening
            stale += 1
            if stale >= patience:
                stale = 0
                if explore >= _EXPLORE_CAP - 1e-9:
                    explore = scale0                       # was pinned wide with no luck -> restart small
                    log("gen %d plateau: reset explore to %.3f (restart search) | champion kept" % (g, explore))
                else:
                    explore = min(explore * 1.5, _EXPLORE_CAP)
                    log("gen %d plateau: widen explore to %.3f | champion kept" % (g, explore))
            else:
                log("gen %d keep champion (sel not proven 2x) | cand sel %.3f vs %.3f"
                    % (g, cand_conf["consistency"], champ_conf["consistency"]))
        else:                                               # cleared select but not the audit floor -> near miss
            log("gen %d keep champion (audit non-backslide) | candidate cleared select, not the held-out floor" % g)

        if adopt or g == gen0 + 1:                         # streak only when the champion CHANGED (or first gen)
            stk = day_after_day_streak(champ_brain, sim, audit_days,   # FIXED seed -> reflects the champion, not X-luck
                                       gen=_gen(dev, _seed_from(base_seed, "audit-streak")),
                                       decide_every=decide_every, ranges=ranges)
            if stk["longest_streak"] > best_streak:
                best_streak = stk["longest_streak"]
                frozen = _save_record(hist_dir, ckpt_dir, champ_brain, champ_cfg, best_streak, obs_dim, base_seed, g)
                log("gen %d NEW BEST STREAK: %d cleared days in a row -> %s" % (g, best_streak, frozen))

        save_champion(champ_path, champ_brain, champ_cfg,   # checkpoint (resumable) every generation
                      {"obs_dim": obs_dim, "gen": g, "best_streak": best_streak,
                       "explore": explore, "stale": stale, "base_seed": base_seed,
                       "anchor_cons": anchor_cons})
        json.dump({"gen": g, "best_streak": int(best_streak), "explore": round(explore, 4),
                   "stale": stale, "last_adopt": bool(adopt), "took": took,
                   "champ_sel_consistency": round(champ_conf["consistency"], 4),
                   "audit_anchor_consistency": round(anchor_cons, 4)},
                  open(os.path.join(ckpt_dir, "meta_progress.json"), "w"), indent=2)

    log("meta run chunk done | gen %d | best streak %d cleared-in-a-row" % (g, best_streak))
    return {"gen": g, "best_streak": int(best_streak), "config": champ_cfg}


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

    # ---- DURABILITY: day-clustering must defeat pseudo-replication (the review's CRITICAL) ----
    # One lucky day, heavily duplicated: 22 identical episodes on day 7 that the candidate clears
    # and the champion doesn't; everything else identical. Episode-level counting is fooled
    # (b+c=22, stat huge); day-clustered counting sees ONE differing day -> below d_min -> REJECT.
    E = 220
    day_idx = torch.arange(E) % 10                     # 10 distinct days, 22 episodes each
    champ_e = torch.ones(E, dtype=torch.bool)          # champion clears everything...
    cand_e = torch.ones(E, dtype=torch.bool)
    champ_e[day_idx == 7] = False                      # ...except day 7; candidate clears day 7 too
    ep_adopt, ep_info = adopt_gate(champ_e, cand_e)                       # WRONG unit (episodes)
    cw, dw = _day_paired(champ_e, cand_e, day_idx)
    day_adopt, day_info = adopt_gate(cw, dw, z=2.0, d_min=4)              # RIGHT unit (days)
    print("\npseudo-replication (1 lucky day x22 episodes):")
    print("  episode-level gate -> adopt=%-5s (disagree=%d)  [illusory significance]"
          % (ep_adopt, ep_info["disagree"]))
    print("  day-clustered gate -> adopt=%-5s (differing days=%d, need >= 4)  [%s]"
          % (day_adopt, day_info["disagree"], "PASS" if not day_adopt else "FAIL"))

    # mutate sanity: champion unchanged at scale 0, moves at scale>0, always in bounds;
    # base_config now reads its starting values from the config one-door.
    g = torch.Generator().manual_seed(0)
    b0 = base_config()
    same = mutate(b0, 0.0, g)
    moved = mutate(b0, 0.4, g)
    in_bounds = all(BOUNDS[k][0] <= moved[k] <= BOUNDS[k][1] for k in BOUNDS)
    n_moved = sum(1 for k in b0 if abs(moved[k] - b0[k]) > 1e-9)
    print("\nbase_config (read from configs/):", {k: round(v, 4) for k, v in b0.items()})
    print("mutate scale0 == base:", all(abs(same[k]-b0[k]) < 1e-9 for k in b0),
          "| scale.4 in bounds:", in_bounds, "| knobs moved:", n_moved, "(<= mutate_knobs)")
