"""Bot 1.5 GPU Edition — SIGON multi-symbol trainer.
USAGE: python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600 --entropy-coef 0.03 --warm best_sigon
CHANGE LOG:
- 2026-07-25  streak COUNT in best_sigon_streakXX.pt; entropy-coef CLI; 3-fail → streak day1
- 2026-07-25  multi-symbol, Jarvis, day_board, best_sigon
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, sys, time
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.configs import path as rpath, training_cfg, decide_every as cfg_decide, load as load_cfg
from training.policy import Brain
from training.fastsim import FastSim, SELF_DIM
from training.gpu_rollout import rollout, ppo_update
from training.gpu_data import build_day_tensors
try:
    from training.gpu_data import load_multi_symbol_pool
except Exception:
    load_multi_symbol_pool = None
from evaluation.consistency import auto_ranges
try:
    from training.day_board import write_day_board
except Exception:
    def write_day_board(*a, **k):
        return None
try:
    from training.signal_accuracy import write_placeholder_accuracy
except Exception:
    def write_placeholder_accuracy(n=500):
        return None
try:
    from training.jarvis import write_status, apply_inbox_to_sim
except Exception:
    def write_status(**k): return None
    def apply_inbox_to_sim(sim): return []

def now():
    return datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

def _sigon_cfg():
    try: return load_cfg("sigon_train") or {}
    except Exception: return {}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", type=int, default=None)
    ap.add_argument("--minutes", type=float, default=1440.0)
    ap.add_argument("--max-updates", type=int, default=0)
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv-dir", default=None)
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
    ap.add_argument("--warm", default="best_sigon")
    ap.add_argument("--ckpt", default="gpu_live")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-day-retries", type=int, default=None)
    ap.add_argument("--entropy-coef", type=float, default=None, help="higher = more exploration")
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

    symbol_names = None
    sym_list = [s.strip().upper() for s in a.symbols.split(",") if s.strip()]
    csv_dir = a.csv_dir or (rpath("data") if a.csv is None else None)
    if csv_dir and os.path.isdir(csv_dir) and load_multi_symbol_pool is not None:
        print("MULTI-SYMBOL pool", csv_dir, sym_list, flush=True)
        do, dp, dl, dates, cols, symbol_names = load_multi_symbol_pool(csv_dir, symbols=sym_list, verbose=True)
    else:
        src = a.csv or rpath("data", "XAUUSD_curriculum_2026.csv")
        tag = os.path.splitext(os.path.basename(src))[0]
        do, dp, dl, dates, cols = build_day_tensors(src, cache_path=rpath("artifacts", "gpu_cache_%s.npz" % tag), verbose=True)
        symbol_names = ["XAUUSD"] * int(do.shape[0])

    D = int(do.shape[0])
    market_cols = int(do.shape[2])
    obs_dim = 10 * (market_cols + SELF_DIM)
    print("BOT GPU | instances=%d | days=%d | obs_dim=%d | day_retries=%d" % (a.instances, D, obs_dim, max_day_retries), flush=True)

    sim = FastSim(do, dp, dl, cols, device=dev, K=a.K)
    brain = Brain(obs_dim).to(dev)
    loaded = "fresh"
    for name in (a.warm, "best_sigon", "gpu_best", a.ckpt):
        if not name: continue
        pth = rpath("artifacts", "checkpoints", "%s.pt" % name)
        if not os.path.exists(pth): continue
        try:
            blob = torch.load(pth, map_location=dev, weights_only=False)
            if int(blob.get("obs_dim", -1)) != obs_dim:
                print("skip warm %s dim mismatch" % name, flush=True); continue
            brain.load_state_dict(blob["model"]); loaded = name
            print("warm-start", name, flush=True); break
        except Exception as e:
            print("warm skip", name, e, flush=True)
    if loaded == "fresh":
        print("NEW Brain(%d)" % obs_dim, flush=True)

    tc = training_cfg() or {}
    gamma = float(tc.get("gamma", 0.99)); lam = float(tc.get("gae_lambda", 0.95))
    clip = float(tc.get("clip_eps", 0.2))
    ent = float(a.entropy_coef if a.entropy_coef is not None else tc.get("entropy_coef", 0.01))
    print("exploration entropy_coef=%.4f" % ent, flush=True)
    opt = torch.optim.Adam(brain.parameters(), lr=float(tc.get("lr", 3e-4)))

    def rand_x(n):
        # Always new target/risk draw (40% focus 2.5/3.5, else random ranges) — less same day+same goals
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
            h = (r["goal_hit"].bool() & ~r["breached"].bool()).cpu().numpy().astype(np.int32)
            run = cur = 0
            for v in h:
                cur = cur + 1 if v else 0
                run = max(run, cur)
            best = max(best, run)
        return float(best)

    histdir = rpath("artifacts", "checkpoints", "history")
    os.makedirs(histdir, exist_ok=True)
    prog = rpath("artifacts", "checkpoints", "gpu_progress.json")
    best_clear, best_breach, best_streak = -1.0, 1.0, -1.0

    def save_record(streak_count, eval_rounds, upd, clear_rate=None, breach_rate=None):
        payload = {"model": brain.state_dict(), "obs_dim": obs_dim, "market_cols": market_cols,
                   "symbols": sorted(set(symbol_names)) if symbol_names else [],
                   "streak": int(streak_count), "clear_rate": clear_rate, "breach_rate": breach_rate,
                   "update": upd, "saved_at": now(), "lineage": "SIGON"}
        serial = hashlib.sha256(str(payload["saved_at"]).encode()).hexdigest()[:12]
        row_n = int(streak_count)
        frozen = "SIGON_streak%02d_obs%d_SN-%s.pt" % (row_n, obs_dim, serial)
        named_best = "best_sigon_streak%02d.pt" % row_n
        for path in (
            rpath("artifacts", "checkpoints", a.ckpt + ".pt"),
            rpath("artifacts", "checkpoints", "gpu_best.pt"),
            rpath("artifacts", "checkpoints", "best_sigon.pt"),
            rpath("artifacts", "checkpoints", named_best),
            os.path.join(histdir, frozen),
        ):
            tmp = path + ".tmp"; torch.save(payload, tmp); os.replace(tmp, path)
        with open(rpath("artifacts", "checkpoints", "best_sigon_record.json"), "w") as f:
            json.dump({"row": row_n, "clear_rate": clear_rate, "breach_rate": breach_rate,
                       "obs_dim": obs_dim, "file": named_best, "updated_at": payload["saved_at"]}, f, indent=2)
        print("   *** RECORD locked streak=%d -> best_sigon.pt + %s + history/%s" % (row_n, named_best, frozen), flush=True)

    retry_left = torch.zeros(a.instances, dtype=torch.long, device=dev)
    sticky_di = torch.randint(0, D, (a.instances,), device=dev)
    write_placeholder_accuracy(500)
    t0 = time.time(); eval_rounds = a.eval_rounds; upd = 0

    while time.time() - t0 < a.minutes * 60:
        if a.max_updates and upd >= a.max_updates: break
        tg, rk = rand_x(a.instances)  # always new goals each update
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
        retry_left = torch.where(failed, retry_left - 1, torch.zeros_like(retry_left))
        exhausted = failed & (retry_left < 0)
        retry_left = torch.clamp(retry_left, min=0)
        if int(exhausted.sum().item()) > 0:
            n_ex = int(exhausted.sum().item())
            sticky_di[exhausted] = torch.randint(0, D, (n_ex,), device=dev)
            if upd % 50 == 0:
                print("   %d instances: 3-fail -> streak climb restarts (new day)" % n_ex, flush=True)

        gh = float(res["goal_hit"].float().mean().item()) * 100
        br = float(res["breached"].float().mean().item()) * 100
        mp = float(res["day_pnl"].mean().item())
        clear_frac = float(hit.float().mean().item())
        print("upd %4d | %.0fs | pnl %+.2f%% | hit %.1f%% | breach %.1f%% | clear_batch %.1f%% | entropy %.2f | retries_active %d"
              % (upd, time.time() - t0, mp, gh, br, clear_frac * 100, stats.get("entropy", 0.0),
                 int((retry_left > 0).sum().item())), flush=True)

        write_status(update=upd, clear_batch=clear_frac, breach_pct=br/100.0, row=int(max(best_streak,0)),
                     obs_dim=obs_dim, instances=a.instances)
        if upd % max(1, a.eval_every // 3) == 0:
            for line in apply_inbox_to_sim(sim):
                print("   [JARVIS]", line, flush=True)

        if upd % a.eval_every == 0:
            best = eval_streak(eval_rounds)
            if best > best_streak or (clear_frac > best_clear and br <= 0.5):
                best_streak = max(best_streak, best)
                best_clear = max(best_clear, clear_frac)
                best_breach = min(best_breach, br/100.0)
                save_record(best_streak, eval_rounds, upd, clear_rate=clear_frac, breach_rate=br/100.0)
            if best >= eval_rounds - 1 and eval_rounds < a.target_days:
                eval_rounds = min(a.target_days, int(np.ceil(eval_rounds * 1.5)))
            json.dump({"update": upd, "best_streak": int(max(best_streak,0)), "obs_dim": obs_dim,
                       "clear_pct": round(clear_frac*100,2), "breach_pct": round(br,2),
                       "instances": a.instances}, open(prog,"w"), indent=2)

    print("\nGPU done | best_streak=%d | clear=%.1f%% | obs_dim=%d" % (int(max(best_streak,0)), best_clear*100, obs_dim), flush=True)

if __name__ == "__main__":
    main()
