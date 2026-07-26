"""Bot 1.5 GPU Edition — multi-symbol SIGON trainer (Monty 2026-07-25).

WHAT: one brain, thousands of parallel instances. Each instance = one day × symbol
× random/focus goal. Learns to clear target without floor breach.
Champion = new all-time-high consistency STREAK with breach 0%.

USAGE (Colab L4):
  python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600 --entropy-coef 0.03 --warm best_sigon

OOM fallback (try in order): --instances 6000 → 4000 → 2000

HARD RULES:
  - Signals ON (~6820 obs). NEVER warm PROVEN 1820 into this brain.
  - Lock champion only on new ATH streak with breach 0%.
  - Filenames include streak count: best_sigon_streak05.pt + history/SIGON_streak05_...
  - best_sigon.pt always points at latest locked champion.
  - Per instance: fail day → up to max_day_retries (3); still fail → streak restarts day 1.

CHANGE LOG:
- 2026-07-25  streak-named locks (best_sigon_streakXX), per-instance streak climb,
  --entropy-coef CLI, breach 0% gate on lock, OOM 6000/4000/2000 docs — WHY: SIGON Colab.
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
    ap = argparse.ArgumentParser(
        description="SIGON multi-symbol GPU trainer. OOM? try --instances 6000 then 4000 then 2000."
    )
    ap.add_argument("--instances", type=int, default=None,
                    help="Parallel envs (default 8000 from configs/sigon_train.yaml). "
                         "OOM fallback: 6000 / 4000 / 2000.")
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
    ap.add_argument("--warm", default="",
                    help="Warm-start checkpoint name under artifacts/checkpoints/ "
                         "(e.g. best_sigon). Loads only if obs_dim matches. "
                         "Never loads PROVEN 1820 into SIGON ~6820.")
    ap.add_argument("--ckpt", default="gpu_live")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-day-retries", type=int, default=None,
                    help="Per-instance retries on a failed day before streak restarts (default 3).")
    ap.add_argument("--entropy-coef", type=float, default=None,
                    help="PPO entropy bonus (higher = more exploration). "
                         "Default from configs/training.yaml entropy_coef. "
                         "Warm-start still keeps weights; only exploration pressure changes.")
    a = ap.parse_args()

    sc = _sigon_cfg()
    if a.instances is None:
        a.instances = int(sc.get("instances_default") or 8000)
    max_day_retries = int(a.max_day_retries if a.max_day_retries is not None else sc.get("max_day_retries") or 3)
    oom_fb = sc.get("instances_oom_fallback") or [6000, 4000, 2000]

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
        cand = rpath("data")
        if os.path.isdir(cand):
            csv_dir = cand
    if csv_dir and os.path.isdir(csv_dir):
        print("MULTI-SYMBOL pool from %s | symbols=%s" % (csv_dir, sym_list), flush=True)
        print("CACHE note: if you flipped signal slots, delete artifacts/gpu_cache_*.npz "
              "and artifacts/symbol_cache/* once. Never delete .pt brains.", flush=True)
        print("OOM? lower --instances via fallbacks: %s" % oom_fb, flush=True)
        if load_multi_symbol_pool is None:
            raise SystemExit("Missing load_multi_symbol_pool — update training/gpu_data.py")
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
    # Sanity: pull indices must exist after __init__ (Colab critical path)
    if not hasattr(sim, "pull_buy_idx") or not hasattr(sim, "pull_sell_idx"):
        raise RuntimeError("FastSim missing pull_buy_idx/pull_sell_idx after __init__ — bug")
    brain = Brain(obs_dim).to(dev)
    loaded = "fresh_sigon"
    # optional warm only if matching obs_dim (never 1820 → 6820)
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
                print("skip warm %s (obs_dim %s != %d) — never load PROVEN 1820 into SIGON"
                      % (name, od, obs_dim), flush=True)
                continue
            brain.load_state_dict(blob["model"])
            loaded = name
            print("warm-start %s (obs_dim=%d) — weights kept; exploration via entropy-coef"
                  % (name, obs_dim), flush=True)
            break
        except Exception as e:
            print("warm skip %s: %s" % (name, e), flush=True)
    if loaded == "fresh_sigon":
        print("NEW Brain(%d) — SIGON lineage (do not expect PROVEN 1820 weights)" % obs_dim, flush=True)

    tc = training_cfg() or {}
    gamma = float(tc.get("gamma", 0.99))
    lam = float(tc.get("gae_lambda", 0.95))
    clip = float(tc.get("clip_eps", 0.2))
    ent = float(a.entropy_coef if a.entropy_coef is not None else tc.get("entropy_coef", 0.01))
    lr = float(tc.get("lr", 3e-4))
    opt = torch.optim.Adam(brain.parameters(), lr=lr)
    print("PPO entropy_coef=%.4f (higher → more exploration)" % ent, flush=True)

    def rand_x(n):
        """Every update: each instance gets random target/risk (40% focus, 60% range)."""
        tg = torch.empty(n, device=dev).uniform_(a.target_lo, a.target_hi)
        rk = torch.empty(n, device=dev).uniform_(a.risk_lo, a.risk_hi)
        if a.focus_frac > 0:
            m = torch.rand(n, device=dev) < a.focus_frac
            tg = torch.where(m, torch.full_like(tg, a.focus_target), tg)
            rk = torch.where(m, torch.full_like(rk, a.focus_risk), rk)
        return tg, rk

    def eval_batch_clear(rounds_hint):
        """Greedy batch clear/breach for progress reporting (not the lock judge)."""
        n = min(a.eval_envs, max(64, rounds_hint))
        di = torch.randint(0, D, (n,), device=dev)
        tg, rk = rand_x(n)
        r = rollout(brain, sim, di, tg, rk, greedy=True, collect=False, decide_every=a.decide_every)
        hit = r["goal_hit"].bool() & ~r["breached"].bool()
        clear = float(hit.float().mean().item())
        breach = float(r["breached"].float().mean().item())
        return clear, breach

    histdir = rpath("artifacts", "checkpoints", "history")
    os.makedirs(histdir, exist_ok=True)
    ckdir = rpath("artifacts", "checkpoints")
    os.makedirs(ckdir, exist_ok=True)
    prog = rpath("artifacts", "checkpoints", "gpu_progress.json")
    best_clear = -1.0
    best_breach = 1.0
    best_streak = 0  # all-time high locked streak count

    def save_record(streak_count, upd, clear_rate=None, breach_rate=None, reason="ath_streak"):
        """Lock champion. Filenames MUST include streak COUNT."""
        streak_n = int(streak_count)
        payload = {
            "model": brain.state_dict(),
            "obs_dim": obs_dim,
            "market_cols": market_cols,
            "symbols": sorted(set(symbol_names)) if symbol_names else [],
            "streak": streak_n,
            "clear_rate": clear_rate,
            "breach_rate": breach_rate,
            "update": upd,
            "saved_at": now(),
            "lineage": "SIGON",
            "reason": reason,
        }
        serial = hashlib.sha256(str(payload["saved_at"]).encode()).hexdigest()[:12]
        # history: SIGON_streak05_obs6820_SN-xxxx.pt
        frozen = "SIGON_streak%02d_obs%d_SN-%s.pt" % (streak_n, obs_dim, serial)
        # named champion: best_sigon_streak05.pt
        named = "best_sigon_streak%02d.pt" % streak_n
        paths = [
            rpath("artifacts", "checkpoints", a.ckpt + ".pt"),
            rpath("artifacts", "checkpoints", "gpu_best.pt"),
            rpath("artifacts", "checkpoints", "best_sigon.pt"),  # latest pointer
            rpath("artifacts", "checkpoints", named),
            os.path.join(histdir, frozen),
        ]
        for path in paths:
            tmp = path + ".tmp"
            torch.save(payload, tmp)
            os.replace(tmp, path)
        rec = {
            "clear_rate": clear_rate,
            "breach_rate": breach_rate,
            "streak": streak_n,
            "row": streak_n,  # HUD alias
            "obs_dim": obs_dim,
            "symbols": payload["symbols"],
            "path": "artifacts/checkpoints/best_sigon.pt",
            "named_path": "artifacts/checkpoints/%s" % named,
            "history_path": "artifacts/checkpoints/history/%s" % frozen,
            "updated_at": payload["saved_at"],
            "serial": serial,
            "reason": reason,
        }
        rp = rpath("artifacts", "checkpoints", "best_sigon_record.json")
        with open(rp, "w", encoding="utf-8") as f:
            json.dump(rec, f, indent=2)
        print("   *** LOCKED champion streak=%d | best_sigon.pt + %s + history/%s"
              % (streak_n, named, frozen), flush=True)
        return named

    # Per-instance: sticky day while retries remain; streak climb restarts on exhaust
    retry_left = torch.zeros(a.instances, dtype=torch.long, device=dev)
    sticky_di = torch.randint(0, D, (a.instances,), device=dev)
    instance_streak = torch.zeros(a.instances, dtype=torch.long, device=dev)

    write_placeholder_accuracy(500)
    t0 = time.time()
    eval_rounds = a.eval_rounds
    evals_since_record = 0
    upd = 0

    while time.time() - t0 < a.minutes * 60:
        if a.max_updates and upd >= a.max_updates:
            break

        # Every update: random target/risk per instance (sampling law from goals.yaml)
        tg, rk = rand_x(a.instances)
        # Sticky day while retries remain after a fail; fresh day when retry budget reset
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

        # --- per-instance streak climb ---
        # clear (target hit, no breach) → streak += 1
        instance_streak = torch.where(hit, instance_streak + 1, instance_streak)
        # fail → consume one retry; success → clear retries (next update may draw new day)
        retry_left = torch.where(failed, retry_left - 1, torch.zeros_like(retry_left))
        retry_left = torch.clamp(retry_left, min=0)
        # exhausted retries after fail → streak restarts at day 1 (new day next update)
        exhausted = failed & (retry_left == 0)
        instance_streak = torch.where(exhausted, torch.zeros_like(instance_streak), instance_streak)

        gh = float(res["goal_hit"].float().mean().item()) * 100
        br = float(res["breached"].float().mean().item()) * 100
        mp = float(res["day_pnl"].mean().item())
        clear_frac = float(hit.float().mean().item())
        retries_active = int((retry_left > 0).sum().item())
        max_inst = int(instance_streak.max().item())
        print("upd %4d | %.0fs | pnl %+.2f%% | hit %.1f%% | breach %.1f%% | clear_batch %.1f%% | "
              "entropy %.2f | retries_active %d | inst_streak_max %d"
              % (upd, time.time() - t0, mp, gh, br, clear_frac * 100, stats.get("entropy", 0.0),
                 retries_active, max_inst), flush=True)

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
            row=int(max(best_streak, max_inst)),
            obs_dim=obs_dim,
            extra={"update": upd, "instances": a.instances, "retries_active": retries_active,
                   "inst_streak_max": max_inst, "best_streak": best_streak},
        )

        # Jarvis hot channel (every update; cheap file read)
        try:
            write_status(
                update=upd,
                clear_batch_pct=round(clear_frac * 100, 2),
                breach_pct=round(br, 2),
                retries_active=retries_active,
                obs_dim=obs_dim,
                best_streak=int(best_streak),
                inst_streak_max=max_inst,
                instances=a.instances,
                champion="artifacts/checkpoints/best_sigon.pt",
                entropy_coef=ent,
            )
            jlogs = apply_inbox_to_sim(sim)
            for line in jlogs:
                print("   [JARVIS] %s" % line, flush=True)
        except Exception as e:
            if upd % 50 == 0:
                print("   [JARVIS] channel skip: %s" % e, flush=True)

        # --- LOCK on new all-time-high streak, only if batch breach is 0% ---
        # Instance clears already require no breach; gate batch breach for lock safety.
        if max_inst > best_streak and br <= 0.0:
            best_streak = max_inst
            best_clear = max(best_clear, clear_frac)
            best_breach = 0.0
            save_record(best_streak, upd, clear_rate=clear_frac, breach_rate=0.0,
                        reason="ath_instance_streak")
            evals_since_record = 0

        if upd % a.eval_every == 0:
            clear_e, breach_e = eval_batch_clear(eval_rounds)
            print("   eval: inst_streak_max=%d ATH=%d | greedy_clear=%.1f%% greedy_breach=%.1f%% | "
                  "batch_clear=%.1f%% batch_breach=%.1f%%"
                  % (max_inst, best_streak, clear_e * 100, breach_e * 100, clear_frac * 100, br),
                  flush=True)
            # Track best clear only when breach stays 0 (no lock without streak ATH)
            if breach_e <= 0.0 and clear_e > best_clear:
                best_clear = clear_e
                best_breach = 0.0
            elif br <= 0.0 and clear_frac > best_clear:
                best_clear = clear_frac
            else:
                evals_since_record += 1
            if best_streak >= eval_rounds - 1 and eval_rounds < a.target_days:
                eval_rounds = min(a.target_days, int(np.ceil(eval_rounds * 1.5)))
                print("   raising eval depth to %d" % eval_rounds, flush=True)
            json.dump({
                "update": upd,
                "best_streak": int(best_streak),
                "inst_streak_max": max_inst,
                "eval_rounds": eval_rounds,
                "evals_since_record": evals_since_record,
                "rollout_clear_pct": round(clear_frac * 100, 2),
                "rollout_breach_pct": round(br, 2),
                "eval_clear_pct": round(clear_e * 100, 2),
                "eval_breach_pct": round(breach_e * 100, 2),
                "obs_dim": obs_dim,
                "instances": a.instances,
                "entropy_coef": ent,
                "target_range": [a.target_lo, a.target_hi],
                "risk_range": [a.risk_lo, a.risk_hi],
                "focus": [a.focus_frac, a.focus_target, a.focus_risk],
                "max_day_retries": max_day_retries,
                "symbols": sorted(set(symbol_names)) if symbol_names else [],
                "champion_example": "artifacts/checkpoints/best_sigon_streak%02d.pt" % max(best_streak, 1),
            }, open(prog, "w"), indent=2)
            if best_streak >= a.target_days:
                print("\n*** FINISH LINE: %d cleared days in a row." % int(best_streak), flush=True)
                break

    print("\nGPU chunk done | best_streak=%d | best_clear_batch=%.1f%% | obs_dim=%d"
          % (int(best_streak), max(best_clear, 0.0) * 100, obs_dim), flush=True)
    if best_streak > 0:
        print("Champion pointer: artifacts/checkpoints/best_sigon.pt")
        print("Named lock example: artifacts/checkpoints/best_sigon_streak%02d.pt" % best_streak)


if __name__ == "__main__":
    main()
