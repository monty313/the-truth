"""Bot 1.5 GPU Edition — 8,000-at-once meta trainer (Monty's spec, 2026-07-20).

WHAT (plain words): one brain, thousands of markets at once. Every instance, every
day, gets a RANDOM target% and RANDOM risk% from your ranges. The brain has to
learn to hit whatever number it's handed without ever crossing the risk line.
It keeps a record of the longest run of CLEARED days in a row (cleared = hit the
target, no breach). Whenever it beats that record it SAVES itself, with the number
of passed days right in the filename. Goal: 365 in a row.

Strategy + lot size are the brain's own free choices; the masks are the one wall.

USAGE (Colab L4):  python scripts/gpu_train.py --instances 8000 --minutes 600
Smaller test:      python scripts/gpu_train.py --instances 64 --max-updates 3 --device cpu

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  created — WHY: Monty's GPU Edition: 8,000 random-X instances, streak
  record auto-save (passed-days in the name), stop at 365 or on plateau.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import argparse
import datetime
import hashlib
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.configs import path as rpath, training_cfg           # noqa: E402
from training.policy import Brain                              # noqa: E402
from training.fastsim import FastSim, SELF_DIM                 # noqa: E402
from training.gpu_rollout import rollout, ppo_update           # noqa: E402
from training.gpu_data import build_day_tensors                # noqa: E402


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=8000)      # Monty: 8000
    ap.add_argument("--minutes", type=float, default=1440.0)    # hardcoded: run until Colab stops it
    ap.add_argument("--max-updates", type=int, default=0)       # 0 = unlimited (time-bound)
    ap.add_argument("--csv", default=rpath("data", "XAUUSD_curriculum_2026.csv"))
    ap.add_argument("--target-lo", type=float, default=2.5)     # Monty's ranges
    ap.add_argument("--target-hi", type=float, default=70.3)
    ap.add_argument("--risk-lo", type=float, default=1.0)
    ap.add_argument("--risk-hi", type=float, default=4.4)
    ap.add_argument("--focus-frac", type=float, default=0.6)    # 2026-07-20 Monty: 60% of practice on one realistic pair
    ap.add_argument("--focus-target", type=float, default=3.0)  # ...target 3%
    ap.add_argument("--focus-risk", type=float, default=3.5)    # ...risk 3.5% (rest stays random across the ranges)
    ap.add_argument("--decide-every", type=int, default=5)      # hardcoded: decide every 5 min (speed)
    ap.add_argument("--target-days", type=int, default=365)     # the finish line
    ap.add_argument("--eval-every", type=int, default=30)       # hardcoded: check record ~once an hour
    ap.add_argument("--eval-envs", type=int, default=512)
    ap.add_argument("--eval-rounds", type=int, default=24)      # grows toward 365 as brain improves
    ap.add_argument("--patience", type=int, default=100000)     # hardcoded: don't plateau-stop; run until Colab stops
    ap.add_argument("--env-mb", type=int, default=256)          # PPO update env minibatch (L4-safe)
    ap.add_argument("--epochs", type=int, default=1)            # hardcoded: leaner/faster update
    ap.add_argument("--K", type=int, default=24)               # hardcoded: fewer position slots (faster)
    ap.add_argument("--warm", default="best_trading")           # COPY of the proof (never overwritten)
    ap.add_argument("--ckpt", default="gpu_live")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    tc = training_cfg(); p = dict(tc.get("ppo", {}))
    gamma = float(p.get("gamma", 0.999)); lam = float(p.get("lam", 0.95))
    clip = float(p.get("clip", 0.2)); lr = float(p.get("lr", 3e-4))
    ent = float(p.get("entropy_coef", 0.01)); seed = int(tc.get("seed", 20260718))
    torch.manual_seed(seed); np.random.seed(seed)

    tag = os.path.splitext(os.path.basename(a.csv))[0]
    do, dp, dl, dates, cols = build_day_tensors(a.csv, cache_path=rpath("artifacts", "gpu_cache_%s.npz" % tag))
    obs_dim = 10 * (len(cols) + SELF_DIM)
    D = do.shape[0]
    print("=" * 68, flush=True)
    print("BOT 1.5 GPU EDITION | device=%s | instances=%d | days-pool=%d" % (dev, a.instances, D), flush=True)
    print("target %.1f%%..%.1f%% | risk %.1f%%..%.1f%% | finish line = %d cleared days in a row"
          % (a.target_lo, a.target_hi, a.risk_lo, a.risk_hi, a.target_days), flush=True)
    print("focus: %.0f%% of practice pinned to target %.1f%% / risk %.1f%% (rest random) | decide every %d bar(s)"
          % (a.focus_frac * 100, a.focus_target, a.focus_risk, a.decide_every), flush=True)
    print("obs_dim=%d (same shape as PROVEN) | masks=LAW, strategy+size=free" % obs_dim, flush=True)
    print("=" * 68, flush=True)

    sim = FastSim(do, dp, dl, cols, device=dev, K=a.K)
    brain = Brain(obs_dim, hidden=128).to(dev)
    opt = torch.optim.Adam(brain.parameters(), lr=lr)

    # warm-start from a COPY of the proof; the original file is never touched
    loaded = None
    for name in ("gpu_best", a.ckpt, "lift_best", a.warm, "PROVEN_2x_2026-07-19", "best_meta"):  # resume best, then the proven-profitable seed
        pth = rpath("artifacts", "checkpoints", name + ".pt")
        if os.path.exists(pth):
            d = torch.load(pth, weights_only=False, map_location=dev)
            if d.get("obs_dim") == obs_dim:
                try:
                    brain.load_state_dict(d["model"]); loaded = name; break
                except Exception:
                    continue
    print("warm-start: %s" % (loaded or "fresh (no matching checkpoint)"), flush=True)

    histdir = rpath("artifacts", "checkpoints", "history"); os.makedirs(histdir, exist_ok=True)
    histmd = rpath("artifacts", "GPU_TRAINING_HISTORY.md")
    prog = rpath("artifacts", "checkpoints", "gpu_progress.json")  # in checkpoints/ -> rides the Drive symlink, so it persists

    def rand_x(n):
        tg = torch.empty(n, device=dev).uniform_(a.target_lo, a.target_hi)
        rk = torch.empty(n, device=dev).uniform_(a.risk_lo, a.risk_hi)
        if a.focus_frac > 0:                       # pin a fraction on one realistic pair
            m = torch.rand(n, device=dev) < a.focus_frac
            tg = torch.where(m, torch.full_like(tg, a.focus_target), tg)
            rk = torch.where(m, torch.full_like(rk, a.focus_risk), rk)
        return tg, rk

    @torch.no_grad()
    def eval_streak(rounds):
        streaks = torch.zeros(a.eval_envs, device=dev)
        records = torch.zeros(a.eval_envs, device=dev)
        best = 0.0
        for _ in range(rounds):
            di = torch.randint(0, D, (a.eval_envs,), device=dev)
            tg, rk = rand_x(a.eval_envs)
            res = rollout(brain, sim, di, tg, rk, greedy=True, collect=False,
                          streak_in=streaks, record_in=records, decide_every=a.decide_every)
            streaks = res["streak"]; records = res["record"]
            best = max(best, float(streaks.max().item()))
            if best >= a.target_days:
                break
        return best

    def save_record(streak_count, eval_rounds, upd):
        payload = {"model": brain.state_dict(), "obs_dim": obs_dim,
                   "reward_state": {}, "seed": seed}
        live = rpath("artifacts", "checkpoints", a.ckpt + ".pt")
        torch.save(payload, live)
        torch.save(payload, rpath("artifacts", "checkpoints", "gpu_best.pt"))
        hh = hashlib.sha256(open(live, "rb").read()).hexdigest()[:12]
        stamp = now()
        frozen = "momentum_gpu_pass%04d_%s_%s.pt" % (int(streak_count), stamp, hh)
        torch.save(payload, os.path.join(histdir, frozen))
        first = not os.path.exists(histmd)
        with open(histmd, "a") as fh:
            if first:
                fh.write("# GPU Edition (Bot 1.5) — record streaks, newest at bottom\n"
                         "Each entry is a FROZEN, revertible brain. It plugs straight into the\n"
                         "real bot (same shape). Load it in the real sim to verify:\n"
                         "  python scripts/replay_best.py --ckpt history/<name-without-.pt>\n\n")
            fh.write("## %d cleared days in a row  —  %s\n" % (int(streak_count), stamp))
            fh.write("- frozen: artifacts/checkpoints/history/%s  (sha256[:12] %s)\n" % (frozen, hh))
            fh.write("- also saved as: artifacts/checkpoints/gpu_best.pt (+ %s.pt)\n" % a.ckpt)
            fh.write("- random target %.1f-%.1f%%, risk %.1f-%.1f%% | update %d, eval depth %d rounds\n\n"
                     % (a.target_lo, a.target_hi, a.risk_lo, a.risk_hi, upd, eval_rounds))
        print("   *** NEW RECORD: %d cleared days in a row -> saved history/%s" % (int(streak_count), frozen), flush=True)

    import time
    t0 = time.time()
    best_streak = -1.0
    eval_rounds = a.eval_rounds
    if os.path.exists(prog):                       # RESUME across sessions (Drive-persisted)
        try:
            _pr = json.load(open(prog))
            best_streak = float(_pr.get("best_streak", -1.0))
            eval_rounds = int(_pr.get("eval_rounds", a.eval_rounds))
            print("resume: prior best streak = %d (eval depth %d)" % (int(best_streak), eval_rounds), flush=True)
        except Exception:
            pass
    evals_since_record = 0
    upd = 0
    while time.time() - t0 < a.minutes * 60:
        if a.max_updates and upd >= a.max_updates:
            break
        # ---- one training rollout: N instances, each a random day + random X ----
        di = torch.randint(0, D, (a.instances,), device=dev)
        tg, rk = rand_x(a.instances)
        stored = rollout(brain, sim, di, tg, rk, greedy=False, collect=True,
                         decide_every=a.decide_every)
        stats = ppo_update(brain, opt, stored, sim.days_obs, gamma=gamma, lam=lam,
                           clip=clip, epochs=a.epochs, ent_coef=ent, env_mb=a.env_mb)
        upd += 1
        res = stored["results"]
        gh = float(res["goal_hit"].float().mean().item()) * 100
        br = float(res["breached"].float().mean().item()) * 100
        mp = float(res["day_pnl"].mean().item())
        tr = float(sim.trades_used.float().mean().item())
        # NOTE: ploss ~ 0.000 is NORMAL at epochs=1 (loss measured at ratio=1 on mean-0
        # advantages) — the gradient is real. Watch ENTROPY (falling = converging) and
        # TRADES/DAY (rising activity) for the true learning signal.
        print("upd %4d | %.0fs | pnl %+.2f%% | hit %.1f%% | breach %.1f%% | trades/day %.1f | entropy %.2f"
              % (upd, time.time() - t0, mp, gh, br, tr, stats.get("entropy", 0.0)), flush=True)

        if upd % a.eval_every == 0:
            best = eval_streak(eval_rounds)
            print("   eval: best streak this check = %d cleared-in-a-row (depth %d rounds)"
                  % (int(best), eval_rounds), flush=True)
            if best > best_streak:
                best_streak = best
                save_record(best_streak, eval_rounds, upd)
                evals_since_record = 0
            else:
                evals_since_record += 1
            # grow eval depth as the brain saturates the current ceiling
            if best >= eval_rounds - 1 and eval_rounds < a.target_days:
                eval_rounds = min(a.target_days, int(np.ceil(eval_rounds * 1.5)))
                print("   (brain saturated the check — raising eval depth to %d rounds)" % eval_rounds, flush=True)
            json.dump({"update": upd, "best_streak": int(best_streak), "eval_rounds": eval_rounds,
                       "last_eval": int(best), "evals_since_record": evals_since_record,
                       "rollout_hit_target_pct": round(gh, 2), "rollout_breach_pct": round(br, 2),
                       "target_days": a.target_days, "instances": a.instances,
                       "target_range": [a.target_lo, a.target_hi], "risk_range": [a.risk_lo, a.risk_hi]},
                      open(prog, "w"), indent=2)
            if best_streak >= a.target_days:
                print("\n*** FINISH LINE: %d cleared days in a row. Stopping (success)." % int(best_streak), flush=True)
                break
            if evals_since_record >= a.patience:
                print("\n*** PLATEAU: no new record in %d checks. Best = %d cleared-in-a-row." % (a.patience, int(best_streak)), flush=True)
                print("    (Stopping so no hours are burned stuck. See gpu_progress.json for where streaks keep ending.)", flush=True)
                break

    print("\nGPU chunk done | best streak reached: %d cleared days in a row (finish line %d)"
          % (int(max(best_streak, 0)), a.target_days), flush=True)


if __name__ == "__main__":
    main()
