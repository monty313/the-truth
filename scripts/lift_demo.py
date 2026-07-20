"""THE LIFT DEMO — prove the GPU-edition pipeline can LEARN PROFIT, not just be safe.
Ladder: Phase 1 = master ONE rich day (2026-01-30, 14% range) to >= +3.0% with no breach.
        Phase 2 = extend to a 5-day rich pool; count cleared days 0 -> N.
Self-correcting: bumps exploration if stuck; fresh-restarts if dead. Every save gets a
SERIAL NUMBER (sha256[:12]) so no result can ever be duplicated/confused (Monty's rule).
"""
import os, sys, time, json, hashlib, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from core.configs import path as rpath
from training.gpu_data import build_day_tensors
from training.fastsim import FastSim, SELF_DIM
from training.gpu_rollout import rollout, ppo_update
from training.policy import Brain
from inference.loader import load_brain

torch.manual_seed(0); torch.set_num_threads(2)
TGT, RK, DE = 3.0, 3.5, 5
DAY1 = 8                       # 2026-01-30 (14.2% range)
POOL = [8, 9, 7, 44, 42]       # all >= 7.5% range days

csv = rpath("data", "XAUUSD_curriculum_2026.csv")
do, dp, dl, dates, cols = build_day_tensors(csv, cache_path=rpath("artifacts","gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)
obs_dim = 10 * (len(cols) + SELF_DIM)
sim = FastSim(do, dp, dl, cols, device="cpu", K=24)

seed_name = "lift_best" if (("--p2" in sys.argv) and
    os.path.exists(rpath("artifacts","checkpoints","lift_best.pt"))) else "PROVEN_2x_2026-07-19"
brain, _ = load_brain(seed_name)
print("seed brain:", seed_name, flush=True)
opt = torch.optim.Adam(brain.parameters(), lr=3e-4)
ENT = 0.04

def greedy_on(days):
    di = torch.tensor(days); n = len(days)
    r = rollout(brain, sim, di, torch.full((n,), TGT), torch.full((n,), RK),
                greedy=True, collect=False, decide_every=DE)
    pnl = r["day_pnl"].float(); clr = ((pnl >= TGT) & ~r["breached"]).int()
    return pnl.tolist(), clr.tolist(), sim.trades_used.float().tolist()

def save_serial(tag, extra):
    payload = {"model": brain.state_dict(), "obs_dim": obs_dim, "reward_state": {}, "seed": 0}
    tmp = rpath("artifacts","checkpoints","_lift_tmp.pt"); torch.save(payload, tmp)
    sha = hashlib.sha256(open(tmp,"rb").read()).hexdigest()[:12]     # THE SERIAL NUMBER
    stamp = datetime.datetime.now().strftime("%H%M%S")
    name = "lift_%s_%s_SN-%s.pt" % (tag, stamp, sha)
    hist = rpath("artifacts","checkpoints","history"); os.makedirs(hist, exist_ok=True)
    os.replace(tmp, os.path.join(hist, name))
    torch.save(payload, rpath("artifacts","checkpoints","lift_best.pt"))
    print("   SAVED %s  (serial %s)  %s" % (name, sha, extra), flush=True)
    return name

def prog(d):
    json.dump(d, open("/tmp/lift_progress.json","w"), indent=1)

def train_phase(name, days, n_env, max_upd, max_min, success_fn, eval_every=5):
    global ENT, brain, opt
    t0 = time.time(); best = -99; stuck = 0; ok_streak = 0; restarted = False
    pool = torch.tensor(days)
    for u in range(1, max_upd + 1):
        if time.time() - t0 > max_min * 60: print("   (time cap)", flush=True); break
        di = pool[torch.randint(len(days), (n_env,))]
        st_r = rollout(brain, sim, di, torch.full((n_env,), TGT), torch.full((n_env,), RK),
                       greedy=False, collect=True, decide_every=DE)
        stats = ppo_update(brain, opt, st_r, sim.days_obs, gamma=0.999, lam=0.95, clip=0.2,
                           epochs=2, ent_coef=ENT, env_mb=n_env)
        res = st_r["results"]
        if u % eval_every == 0:
            gp, gc, gt = greedy_on(days)
            score = sum(gc) + max(gp) / 100.0
            print("%s u%3d %4ds | sample: pnl %+5.2f%% hit %4.1f%% brch %4.1f%% | greedy: %s clr %d/%d | ent %.2f"
                  % (name, u, time.time()-t0, res["day_pnl"].mean(),
                     res["goal_hit"].float().mean()*100, res["breached"].float().mean()*100,
                     " ".join("%+.1f" % p for p in gp), sum(gc), len(days), stats["entropy"]), flush=True)
            prog({"phase": name, "upd": u, "greedy_pnl": gp, "cleared": sum(gc), "of": len(days), "ent": ENT})
            if score > best + 1e-6:
                best = score; stuck = 0
                if sum(gc) > 0: save_serial(name, "cleared %d/%d, pnls %s" % (sum(gc), len(days), ["%+.2f" % p for p in gp]))
            else:
                stuck += 1
            if success_fn(gp, gc):
                ok_streak += 1
                if ok_streak >= 2: print("   PHASE %s SUCCESS" % name, flush=True); return True
            else: ok_streak = 0
            # self-correction ladder
            if stuck == 12 and ENT < 0.08:
                ENT = 0.08; print("   (stuck -> exploration up: ent 0.08)", flush=True)
            if stuck == 22 and not restarted:
                restarted = True; ENT = 0.06
                brain = Brain(obs_dim, hidden=128); opt = torch.optim.Adam(brain.parameters(), lr=3e-4)
                print("   (dead -> FRESH RESTART)", flush=True)
    return False

print("baseline (PROVEN, greedy):", flush=True)
gp, gc, gt = greedy_on(POOL)
print("  pool days %s -> pnls %s | cleared %d/5" % ([str(dates[i]) for i in POOL], ["%+.2f" % p for p in gp], sum(gc)), flush=True)

ok1 = True
if "--p2" not in sys.argv:
    print("\nPHASE 1 — master %s (target +%.1f%%, floor %.1f%%)" % (dates[DAY1], TGT, RK), flush=True)
    ok1 = train_phase("P1", [DAY1], 128, 400, 42, lambda gp, gc: gc[0] == 1)
print("\nPHASE 2 — the 5-day pool", flush=True)
ok2 = train_phase("P2", POOL, 160, 400, 55, lambda gp, gc: sum(gc) >= 4, eval_every=10)
gp, gc, gt = greedy_on(POOL)
print("\nFINAL: cleared %d/5 | pnls %s | phase1 %s phase2 %s" % (sum(gc), ["%+.2f" % p for p in gp], ok1, ok2), flush=True)
print("LIFT DEMO DONE", flush=True)
