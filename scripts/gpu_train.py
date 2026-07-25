"""Bot 1.5 GPU Edition — multi-symbol SIGON trainer (Monty 2026-07-25).

USAGE (Colab L4):
  python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600

OOM step-down: 8000 → 4000 → 2000 → 1024 (see configs/sigon_train.yaml)

HARD RULES:
- Signal-ON obs_dim (~6820) = NEW lineage. Never load PROVEN_* 1820 weights.
- On record: best_sigon.pt only if consistency improves under low breach.
- Day fail → retry SAME day up to max_day_retries (3) per instance.

CHANGE LOG:
- 2026-07-25  SIGON complete path: multi-symbol, 3 retries, day_board, best_sigon,
  auto_ranges 40% focus — WHY: Colab end-to-end.
- 2026-07-20  created — WHY: 8,000 random-X instances, streak record auto-save.
"""
from __future__ import annotations
import argparse
import datetime
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.configs import path as rpath, training_cfg, decide_every as cfg_decide, load as load_cfg
from training.policy import Brain
from training.fastsim import FastSim, SELF_DIM
from training.gpu_rollout import rollout, ppo_update
from training.gpu_data import build_day_tensors, load_multi_symbol_pool
from evaluation.consistency import auto_ranges
from training.day_board import write_day_board
from training.signal_accuracy import write_placeholder_accuracy


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _sigon_cfg() -> dict:
    try:
        return load_cfg("sigon_train") or {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description="SIGON multi-symbol GPU trainer")
    ap.add_argument("--instances", type=int, default=None,
                    help="Parallel markets (default 8000; OOM: 4000/2000/1024)")
    ap.add_argument("--minutes", type=float, default=1440.0)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--csv", default=None, help="Single CSV (legacy)")
    ap.add_argument("--csv-dir", default=None, help="Multi-symbol directory (preferred)")
    ap.add_argument("--symbols", default="XAUUSD,EURUSD,GBPUSD,US30")
    ap.add_argument("--target-lo", type=float, default=None)
    ap.add_argument("--target-hi", type=float, default=None)
    ap.add_argument("--risk-lo", type=float, default=None)
    ap.add_argument("--risk-hi", type=float, default=None)
    ap.add_argument("--focus-frac", type=float, default=None)
    ap.add_argument("--focus-target", type=float, default=None)
    ap.add_argument("--focus-risk", type=float, default=None)
    ap.add_argument("--decide-every", type=int, default=cfg_decide())
    ap.add_argument("--target-days", type=int, default=365)
    ap.add_argument("--eval-every", type=int, default=30)
    ap.add_argument("--eval-envs", type=int, default=512)
    ap.add_argument("--eval-rounds", type=int, default=24)
    ap.add_argument("--patience", type=int, default=100000)
    ap.add_argument("--env-mb", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--warm", default="")
    ap.add_argument("--ckpt", default="gpu_live")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-day-retries", type=int, default=None)
    a = ap.parse_args()

    sc = _sigon_cfg()
    if a.instances is None:
        a.instances = int(sc.get("instances_default") or 8000)
    max_day_retries = int(
        a.max_day_retries if a.max_day_retries is not None else sc.get("max_day_retries") or 3
    )
    oom_fb = sc.get("oom_instances_fallback") or [4000, 2000, 1024]

    # Sampling law from goals.yaml via auto_ranges (40% focus 2.5/3.5, 60% random)
    _r = auto_ranges()
    if a.target_lo is None:
        a.target_lo = _r["tgt_lo"]
    if a.target_hi is None:
        a.target_hi = _r["tgt_hi"]
    if a.risk_lo is None:
        a.risk_lo = _r["risk_lo"]
    if a.risk_hi is None:
        a.risk_hi = _r["risk_hi"]
    if a.focus_frac is None:
        a.focus_frac = _r["focus_frac"]
    if a.focus_target is None:
        a.focus_target = _r["focus_target"]
    if a.focus_risk is None:
        a.focus_risk = _r["focus_risk"]

    print("=" * 64, flush=True)
    print("SIGON GPU TRAIN | focus %.0f%% @ %.1f/%.1f | random range tgt[%.1f,%.1f] risk[%.1f,%.1f]"
          % (100 * a.focus_frac, a.focus_target, a.focus_risk,
             a.target_lo, a.target_hi, a.risk_lo, a.risk_hi), flush=True)
    print("OOM fallback instances: %s" % oom_fb, flush=True)
    print("CACHE: if signals just flipped ON, delete artifacts/gpu_cache_*.npz and "
          "artifacts/symbol_cache/* once (never delete .pt brains).", flush=True)
    print("=" * 64, flush=True)

    symbol_names = None
    sym_list = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    csv_dir = a.csv_dir
    if csv_dir is None and a.csv is None:
        cand = rpath("data")
        if os.path.isdir(cand):
            csv_dir = cand
    if csv_dir and os.path.isdir(csv_dir):
        print("MULTI-SYMBOL pool from %s | symbols=%s" % (csv_dir, sym_list), flush=True)
        do, dp, dl, dates, cols, symbol_names = load_multi_symbol_pool(
            csv_dir, symbols=sym_list, verbose=True
        )
    else:
        src = a.csv or rpath("data", "XAUUSD_curriculum_2026.csv")
        tag = os.path.splitext(os.path.basename(src))[0]
        cache = rpath("artifacts", "gpu_cache_%s.npz" % tag)
        do, dp, dl, dates, cols = build_day_tensors(src, cache_path=cache, verbose=True)
        symbol_names = ["XAUUSD"] * int(do.shape[0])

    D = int(do.shape[0])
    market_cols = int(do.shape[2])
    obs_dim = 10 * (market_cols + SELF_DIM)
    print("BOT 1.5 GPU | instances=%d | days=%d | cols=%d | obs_dim=%d | day_retries=%d"
          % (a.instances, D, market_cols, obs_dim, max_day_retries), flush=True)
    if market_cols >= 600:
        print("obs_dim=%d → SIGON lineage (signal slots ON). Do NOT load PROVEN 1820 weights."
              % obs_dim, flush=True)
    else:
        print("WARNING: market_cols=%d (obs_dim=%d). If you expected ~6820, set "
              "include_signal_agent_slots: true and delete feature caches."
              % (market_cols, obs_dim), flush=True)

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    print("device=%s" % dev, flush=True)

    sim = FastSim(do, dp, dl, cols, device=dev, K=a.K)
    brain = Brain(obs_dim).to(dev)
    loaded = "fresh_sigon"
    for name in (a.warm, "best_sigon", "gpu_best", a.ckpt):
        if not name:
            continue
        pth = rpath("artifacts", "checkpoints", "%s.pt" % name)
        if not os.path.exists(pth):
            continue
        try:
            blob = torch.load(pth, map_location=dev, weights_only=False)
            od = int(blob.get("obs_dim", -1))
            if od != obs_dim:
                print("skip warm %s (obs_dim %s != %d) — PROVEN/wrong lineage blocked"
                      % (name, od, obs_dim), flush=True)
                continue
            brain.load_state_dict(blob["model"])
            loaded = name
            print("warm-start %s (obs_dim match %d)" % (name, obs_dim), flush=True)
            break
        except Exception as e:
            print("warm skip %s: %s" % (name, e), flush=True)
    if loaded == "fresh_sigon":
        print("NEW Brain(%d) SIGON lineage (no matching warm checkpoint)" % obs_dim, flush=True)

    tc = training_cfg() or {}
    gamma = float(tc.get("gamma", 0.99))
    lam = float(tc.get("gae_lambda", 0.95))
    clip = float(tc.get("clip_eps", 0.2))
    ent = float(tc.get("entropy_coef", 0.01))
    lr = float(tc.get("lr", 3e-4))
    opt = torch.optim.Adam(brain.parameters(), lr=lr)

    def rand_x(n):
        tg = torch.empty(n, device=dev).uniform_(a.target_lo, a.target_hi)
        rk = torch.empty(n, device=dev).uniform_(a.risk_lo, a.risk_hi)
        if a.focus_frac > 0:
            m = torch.rand(n, device=dev) < a.focus_frac
            tg = torch.where(m, torch.full_like(tg, a.focus_target), tg)
            rk = torch.where(m, torch.full_like(rk, a.focus_risk), rk)
        return tg, rk

    def eval_streak(rounds):
        best = 0
        for _ in range(max(1, rounds // max(1, a.eval_envs)) + 1):
            di = torch.randint(0, D, (a.eval_envs,), device=dev)
            tg, rk = rand_x(a.eval_envs)
            r = rollout(brain, sim, di, tg, rk, greedy=True, collect=False,
                        decide_every=a.decide_every)
            h = (r["goal_hit"].bool() & ~r["breached"].bool()).cpu().numpy().astype(np.int32)
            run = cur = 0
            for v in h:
                if v:
                    cur += 1
                    run = max(run, cur)
                else:
                    cur = 0
            best = max(best, run)
        return float(best)

    histdir = rpath("artifacts", "checkpoints", "history")
    os.makedirs(histdir, exist_ok=True)
    prog = rpath("artifacts", "checkpoints", "gpu_progress.json")
    best_clear = -1.0
    best_breach = 1.0
    best_streak = -1.0

    def save_record(streak_count, upd, clear_rate=None, breach_rate=None):
        # Highest consistency under low breach wins
        payload = {
            "model": brain.state_dict(),
            "obs_dim": obs_dim,
            "market_cols": market_cols,
            "symbols": sorted(set(symbol_names)) if symbol_names else [],
            "streak": int(streak_count),
            "clear_rate": clear_rate,
            "breach_rate": breach_rate,
            "update": upd,
            "saved_at": now(),
            "lineage": "SIGON",
        }
        serial = hashlib.sha256(str(payload["saved_at"]).encode()).hexdigest()[:12]
        frozen = "SIGON_row%02d_obs%d_SN-%s.pt" % (int(streak_count), obs_dim, serial)
        for path in (
            rpath("artifacts", "checkpoints", a.ckpt + ".pt"),
            rpath("artifacts", "checkpoints", "gpu_best.pt"),
            rpath("artifacts", "checkpoints", "best_sigon.pt"),
            os.path.join(histdir, frozen),
        ):
            tmp = path + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, path)
        with open(rpath("artifacts", "checkpoints", "best_sigon_record.json"), "w") as f:
            json.dump({
                "clear_rate": clear_rate,
                "breach_rate": breach_rate,
                "row": int(streak_count),
                "obs_dim": obs_dim,
                "updated_at": payload["saved_at"],
                "serial": serial,
            }, f, indent=2)
        print("   *** RECORD best_sigon + history/%s (clear=%.1f%% breach=%.2f%%)"
              % (frozen, 100 * (clear_rate or 0), 100 * (breach_rate or 0)), flush=True)

    write_placeholder_accuracy(500)
    retry_left = torch.zeros(a.instances, dtype=torch.long, device=dev)
    sticky_di = torch.randint(0, D, (a.instances,), device=dev)
    t0 = time.time()
    eval_rounds = a.eval_rounds
    upd = 0

    while time.time() - t0 < a.minutes * 60:
        if a.max_updates and upd >= a.max_updates:
            break
        tg, rk = rand_x(a.instances)
        di = sticky_di.clone()
        # New day only when retries exhausted
        fresh = retry_left <= 0
        if fresh.any():
            n_fresh = int(fresh.sum().item())
            di[fresh] = torch.randint(0, D, (n_fresh,), device=dev)
            sticky_di[fresh] = di[fresh]
            retry_left[fresh] = max_day_retries

        stored = rollout(brain, sim, di, tg, rk, greedy=False, collect=True,
                         decide_every=a.decide_every)
        stats = ppo_update(brain, opt, stored, sim.days_obs, gamma=gamma, lam=lam,
                           clip=clip, epochs=a.epochs, ent_coef=ent, env_mb=a.env_mb)
        upd += 1
        res = stored["results"]
        hit = res["goal_hit"].bool() & ~res["breached"].bool()
        failed = ~hit
        # On fail: decrement retries (stay on same day). On clear: zero retries → new day next.
        retry_left = torch.where(
            failed,
            torch.clamp(retry_left - 1, min=0),
            torch.zeros_like(retry_left),
        )

        gh = float(res["goal_hit"].float().mean().item()) * 100
        br = float(res["breached"].float().mean().item()) * 100
        mp = float(res["day_pnl"].mean().item())
        clear_frac = float(hit.float().mean().item())
        retries_active = int((retry_left > 0).sum().item())
        print(
            "upd %4d | %.0fs | pnl %+.2f%% | hit %.1f%% | breach %.1f%% | clear_batch %.1f%% | "
            "entropy %.2f | retries_active %d"
            % (upd, time.time() - t0, mp, gh, br, clear_frac * 100,
               stats.get("entropy", 0.0), retries_active),
            flush=True,
        )

        n_board = min(96, a.instances)
        syms = None
        if symbol_names is not None:
            idx = di[:n_board].detach().cpu().numpy()
            syms = [symbol_names[int(i) % len(symbol_names)] for i in idx]
        write_day_board(
            res["day_pnl"][:n_board].detach().cpu().numpy(),
            tg[:n_board].detach().cpu().numpy(),
            rk[:n_board].detach().cpu().numpy(),
            breached=res["breached"][:n_board].detach().cpu().numpy().astype(bool),
            symbols=syms,
            clear_rate=clear_frac,
            breach_rate=br / 100.0,
            row=int(max(best_streak, 0)),
            obs_dim=obs_dim,
            extra={"update": upd, "retries_active": retries_active},
        )

        if upd % a.eval_every == 0:
            best = eval_streak(eval_rounds)
            # Record if streak up, or clear up with breach still tiny
            improved = (
                best > best_streak
                or (clear_frac > best_clear + 1e-6 and br <= 0.5)
            )
            if improved:
                best_streak = max(best_streak, best)
                best_clear = max(best_clear, clear_frac)
                best_breach = min(best_breach, br / 100.0)
                save_record(best_streak, upd, clear_rate=clear_frac, breach_rate=br / 100.0)
            if best >= eval_rounds - 1 and eval_rounds < a.target_days:
                eval_rounds = min(a.target_days, int(np.ceil(eval_rounds * 1.5)))
            json.dump({
                "update": upd,
                "best_streak": int(max(best_streak, 0)),
                "obs_dim": obs_dim,
                "rollout_clear_pct": round(clear_frac * 100, 2),
                "rollout_breach_pct": round(br, 2),
                "instances": a.instances,
                "max_day_retries": max_day_retries,
                "retries_active": retries_active,
            }, open(prog, "w"), indent=2)

    print("\nGPU done | streak=%d | clear_batch=%.1f%% | obs_dim=%d | champion=best_sigon.pt"
          % (int(max(best_streak, 0)), max(best_clear, 0) * 100, obs_dim), flush=True)


if __name__ == "__main__":
    main()
