"""Bot 1.5 GPU Edition — multi-symbol SIGON trainer (Monty 2026-07-25).

WHAT: one brain, thousands of parallel instances. Each instance = one day × symbol
× random/focus goal. Learns to clear target without floor breach.
Champion = highest consistency (clear↑, breach=0). Serial saves include obs_dim.

USAGE (Colab L4):
  python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600
  python scripts/gpu_train.py --csv data/XAUUSD_curriculum_2026.csv --instances 64 --max-updates 3 --device cpu

CHANGE LOG:
- 2026-07-25  Jarvis hot-reload + status; 3-fail instance restart; auto best_sigon
- 2026-07-25  multi-symbol pool, 3 day-retries/instance, day_board JSON, signal_accuracy stub,
  best_sigon champion, sampling via auto_ranges — WHY: SIGON Colab path.
- 2026-07-20  created — WHY: 8,000 random-X instances, streak record auto-save.
# NEXT EDITOR: append date + WHY; keep this line.
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

from core.configs import path as rpath, training_cfg, policy_hidden, decide_every as cfg_decide, load as load_cfg  # noqa: E402
from training.policy import Brain  # noqa: E402
from training.fastsim import FastSim, SELF_DIM  # noqa: E402
from training.gpu_rollout import rollout, ppo_update  # noqa: E402
from training.gpu_data import build_day_tensors, load_multi_symbol_pool  # noqa: E402
from evaluation.consistency import auto_ranges  # noqa: E402
from training.day_board import write_day_board  # noqa: E402
from training.signal_accuracy import write_placeholder_accuracy  # noqa: E402
from training.jarvis import write_status, apply_inbox_to_sim  # noqa: E402


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")


def _sigon_cfg() -> dict:
    try:
        return load_cfg("sigon_train") or {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=1440.0)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--csv", default=None, help="Single CSV (legacy). Prefer --csv-dir for 4 symbols.")
    ap.add_argument("--csv-dir", default=None, help="Folder of M1 CSVs (XAUUSD, EURUSD, GBPUSD, US30).")
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
    ap.add_argument("--warm", default="")  # empty = fresh SIGON brain when dims differ
    ap.add_argument("--ckpt", default="gpu_live")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-day-retries", type=int, default=None)
    a = ap.parse_args()

    sc = _sigon_cfg()
    if a.instances is None:
        a.instances = int(sc.get("instances_default") or 8000)
    max_day_retries = int(a.max_day_retries if a.max_day_retries is not None else sc.get("max_day_retries") or 3)

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    _r = auto_ranges()
    if a.target_lo is None: a.target_lo = _r["tgt_lo"]
    if a.target_hi is None: a.target_hi = _r["tgt_hi"]
    if a.risk_lo is None: a.risk_lo = _r["risk_lo"]
    if a.risk_hi is None: a.risk_hi = _r["risk_hi"]
    if a.focus_frac is None: a.focus_frac = _r["focus_frac"]
    if a.focus_target is None: a.focus_target = _r["focus_target"]
    if a.focus_risk is None: a.focus_risk = _r["focus_risk"]

    # ---- data: multi-symbol preferred ----
    symbol_names = None
    sym_list = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    csv_dir = a.csv_dir
    if csv_dir is None and a.csv is None:
        # default: try data dir multi-symbol, else single curriculum
        cand = rpath("data")
        if os.path.isdir(cand):
            csv_dir = cand
    if csv_dir and os.path.isdir(csv_dir):
        print("MULTI-SYMBOL pool from %s | symbols=%s" % (csv_dir, sym_list), flush=True)
        print("CACHE note: if you flipped signal slots, delete artifacts/gpu_cache_*.npz and artifacts/symbol_cache/* once.", flush=True)
        if load_multi_symbol_pool is None:
            raise SystemExit('Missing load_multi_symbol_pool — copy training/gpu_data.py from SIGON_BUNDLE')
        do, dp, dl, dates, cols, symbol_names = load_multi_symbol_pool(
            csv_dir, symbols=sym_list, verbose=True)
    else:
        src = a.csv or rpath("data", "XAUUSD_curriculum_2026.csv")
        tag = os.path.splitext(os.path.basename(src))[0]
        cache = rpath("artifacts", "gpu_cache_%s.npz" % tag)
        print("SINGLE CSV %s (cache %s)" % (src, cache), flush=True)
        do, dp, dl, dates, cols = build_day_tensors(src, cache_path=cache, verbose=True)
        symbol_names = ["XAUUSD"] * int(do.shape[0])

    D = int(do.shape[0])
    market_cols = int(do.shape[2])
    obs_dim = 10 * (market_cols + SELF_DIM)
    print("BOT 1.5 GPU | device=%s | instances=%d | days-pool=%d | market_cols=%d | obs_dim=%d | day_retries=%d"
          % (dev, a.instances, D, market_cols, obs_dim, max_day_retries), flush=True)
    print("sampling: focus_frac=%.2f @ %.1f/%.1f | range tgt[%.1f,%.1f] risk[%.1f,%.1f]"
          % (a.focus_frac, a.focus_target, a.focus_risk, a.target_lo, a.target_hi, a.risk_lo, a.risk_hi), flush=True)

    sim = FastSim(do, dp, dl, cols, device=dev, K=a.K)
    brain = Brain(obs_dim).to(dev)
    loaded = "fresh_sigon"
    # optional warm only if matching obs_dim
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
                print("skip warm %s (obs_dim %s != %d)" % (name, od, obs_dim), flush=True)
                continue
            brain.load_state_dict(blob["model"])
            loaded = name
            print("warm-start %s (obs_dim=%d)" % (name, obs_dim), flush=True)
            break
        except Exception as e:
            print("warm skip %s: %s" % (name, e), flush=True)
    if loaded == "fresh_sigon":
        print("NEW Brain(%d) — SIGON lineage (do not expect PROVEN 1820 weights)" % obs_dim, flush=True)

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
            r = rollout(brain, sim, di, tg, rk, greedy=True, collect=False, decide_every=a.decide_every)
            hit = r["goal_hit"].bool() & ~r["breached"].bool()
            # longest run of True in this batch (simple)
            h = hit.cpu().numpy().astype(np.int32)
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

    def save_record(streak_count, eval_rounds, upd, clear_rate=None, breach_rate=None):
        payload = {
            "model": brain.state_dict(),
            "obs_dim": obs_dim,
            "market_cols": market_cols,
            "symbols": sorted(set(symbol_names)) if symbol_names else [],
            "streak": int(streak_count),
            "clear_rate": clear_rate,
            "breach_rate": breach_rate,
            "update": upd,
            "eval_rounds": eval_rounds,
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
        # champion record JSON
        rec = {
            "clear_rate": clear_rate,
            "breach_rate": breach_rate,
            "row": int(streak_count),
            "obs_dim": obs_dim,
            "symbols": payload["symbols"],
            "path": "artifacts/checkpoints/best_sigon.pt",
            "updated_at": payload["saved_at"],
            "serial": serial,
        }
        rp = rpath("artifacts", "checkpoints", "best_sigon_record.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print("   *** RECORD saved best_sigon + history/%s" % frozen, flush=True)

    # Monty: per-instance day retries (fail → retry same day up to max_day_retries)
    retry_left = torch.zeros(a.instances, dtype=torch.long, device=dev)
    sticky_di = torch.randint(0, D, (a.instances,), device=dev)

    write_placeholder_accuracy(500)
    t0 = time.time()
    best_streak = -1.0
    eval_rounds = a.eval_rounds
    evals_since_record = 0
    upd = 0

    while time.time() - t0 < a.minutes * 60:
        if a.max_updates and upd >= a.max_updates:
            break

        tg, rk = rand_x(a.instances)
        # sticky days while retries remain after a fail
        di = sticky_di.clone()
        fresh = retry_left <= 0
        if fresh.any():
            di[fresh] = torch.randint(0, D, (int(fresh.sum().item()),), device=dev)
            sticky_di[fresh] = di[fresh]
            retry_left[fresh] = max_day_retries

        stored = rollout(brain, sim, di, tg, rk, greedy=False, collect=True, decide_every=a.decide_every)
        stats = ppo_update(brain, opt, stored, sim.days_obs, gamma=gamma, lam=lam,
                           clip=clip, epochs=a.epochs, ent_coef=ent, env_mb=a.env_mb)
        upd += 1
        res = stored["results"]
        hit = res["goal_hit"].bool() & ~res["breached"].bool()
        failed = ~hit
        # consume a retry on fail; on success clear retries so next update draws new day
        retry_left = torch.where(failed, retry_left - 1, torch.zeros_like(retry_left))
        retry_left = torch.clamp(retry_left, min=0)

        gh = float(res["goal_hit"].float().mean().item()) * 100
        br = float(res["breached"].float().mean().item()) * 100
        mp = float(res["day_pnl"].mean().item())
        clear_frac = float(hit.float().mean().item())
        retries_active = int((retry_left > 0).sum().item())
        print("upd %4d | %.0fs | pnl %+.2f%% | hit %.1f%% | breach %.1f%% | clear_batch %.1f%% | entropy %.2f | retries_active %d"
              % (upd, time.time() - t0, mp, gh, br, clear_frac * 100, stats.get("entropy", 0.0),
                 retries_active), flush=True)

        # live day board for HUD (sample of this batch)
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
            extra={"update": upd, "instances": a.instances, "retries_active": retries_active},
        )

        # Jarvis hot channel (every update; cheap file read)
        try:
            write_status(
                update=upd,
                clear_batch_pct=round(clear_frac * 100, 2),
                breach_pct=round(br, 2),
                retries_active=retries_active,
                obs_dim=obs_dim,
                best_streak=int(max(best_streak, 0)),
                instances=a.instances,
                champion="artifacts/checkpoints/best_sigon.pt",
            )
            jlogs = apply_inbox_to_sim(sim)
            for line in jlogs:
                print("   [JARVIS] %s" % line, flush=True)
        except Exception as e:
            if upd % 50 == 0:
                print("   [JARVIS] channel skip: %s" % e, flush=True)

        if upd % a.eval_every == 0:
            best = eval_streak(eval_rounds)
            print("   eval: streak=%d (depth %d) | batch_clear=%.1f%% breach=%.1f%%"
                  % (int(best), eval_rounds, clear_frac * 100, br), flush=True)
            # champion: higher clear with breach near 0; also streak records
            accept = (br <= 0.5 and clear_frac >= best_clear - 1e-9 and clear_frac > 0) or (best > best_streak and br <= 1.0)
            if best > best_streak or (clear_frac > best_clear and br <= best_breach + 0.5):
                best_streak = max(best_streak, best)
                best_clear = max(best_clear, clear_frac)
                best_breach = min(best_breach, br / 100.0)
                save_record(best_streak, eval_rounds, upd, clear_rate=clear_frac, breach_rate=br / 100.0)
                evals_since_record = 0
            else:
                evals_since_record += 1
            if best >= eval_rounds - 1 and eval_rounds < a.target_days:
                eval_rounds = min(a.target_days, int(np.ceil(eval_rounds * 1.5)))
                print("   raising eval depth to %d" % eval_rounds, flush=True)
            json.dump({
                "update": upd, "best_streak": int(max(best_streak, 0)), "eval_rounds": eval_rounds,
                "last_eval": int(best), "evals_since_record": evals_since_record,
                "rollout_clear_pct": round(clear_frac * 100, 2), "rollout_breach_pct": round(br, 2),
                "obs_dim": obs_dim, "instances": a.instances,
                "target_range": [a.target_lo, a.target_hi], "risk_range": [a.risk_lo, a.risk_hi],
                "focus": [a.focus_frac, a.focus_target, a.focus_risk],
                "max_day_retries": max_day_retries,
                "symbols": sorted(set(symbol_names)) if symbol_names else [],
            }, open(prog, "w"), indent=2)
            if best_streak >= a.target_days:
                print("\n*** FINISH LINE: %d cleared days in a row." % int(best_streak), flush=True)
                break

    print("\nGPU chunk done | best_streak=%d | best_clear_batch=%.1f%% | obs_dim=%d"
          % (int(max(best_streak, 0)), best_clear * 100, obs_dim), flush=True)


if __name__ == "__main__":
    main()
