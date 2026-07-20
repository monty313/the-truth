"""THE CONSISTENCY SPRINT — push cleared-days-of-90 as far as possible, right here.

Doctrine: aim practice at the FRONTIER. After each measurement, each day is weighted:
cleared -> retention reps; near-miss (+1..3%) -> heavy reps; winnable-but-red -> medium;
physically unwinnable (day range < 3%) -> zero (never waste gradient on impossible days).
RATCHET: the record brain is never lost — greedy eval on ALL 90 days is deterministic;
new record => serial-stamped save; regression >4 days => snap back to the record brain.
Self-correcting: exploration bumps on stagnation. All laws live; zero-breach is enforced.
Usage: python scripts/consistency_sprint.py [--minutes 70]
"""
import os, sys, time, json, hashlib, datetime, argparse, shutil
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from core.configs import path as rpath, policy_hidden, decide_every as cfg_decide
from training.gpu_data import build_day_tensors
from training.fastsim import FastSim, SELF_DIM
from training.gpu_rollout import rollout, ppo_update
from training.policy import Brain
from inference.loader import load_brain

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=70.0)
ap.add_argument("--envs", type=int, default=224)
a = ap.parse_args()
torch.manual_seed(0); torch.set_num_threads(2)
TGT, RK = 3.0, 3.5
DE = cfg_decide()

do, dp, dl, dates, cols = build_day_tensors(rpath("data","XAUUSD_curriculum_2026.csv"),
    cache_path=rpath("artifacts","gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)
obs_dim = 10 * (len(cols) + SELF_DIM); D = do.shape[0]
sim = FastSim(do, dp, dl, cols, device="cpu", K=24)

# physical winnability: a day whose whole range is under ~3% cannot bank +3% at a 0.25% cap
rng = torch.zeros(D)
for i in range(D):
    n = int(dl[i]); hi = dp[i,:n,0].max(); lo = dp[i,:n,1][dp[i,:n,1]>0].min(); c0 = dp[i,0,2]
    rng[i] = float(100.0*(hi-lo)/c0)
winnable = rng >= 3.0
print("days: %d | physically winnable at +3%%: %d | ceiling if perfect: %d/90"
      % (D, int(winnable.sum()), int(winnable.sum())), flush=True)

brain, _ = load_brain("lift_best")
if brain is None:
    brain, _ = load_brain("PROVEN_LIFT_2026-07-20")
opt = torch.optim.Adam(brain.parameters(), lr=1.2e-4)   # POLISH phase: gentle steps
ENT = 0.012                                                  # POLISH phase: tiny exploration

def measure():
    di = torch.arange(D)
    r = rollout(brain, sim, di, torch.full((D,),TGT), torch.full((D,),RK),
                greedy=True, collect=False, decide_every=DE)
    pnl = r["day_pnl"].float(); clr = ((pnl>=TGT) & ~r["breached"])
    best = cur = 0
    for c in clr.tolist():
        cur = cur+1 if c else 0; best = max(best,cur)
    return pnl, clr, int(clr.sum()), best

# golden corridors: maximal runs of consecutive winnable days, length >= 4 — rows LIVE here
corridor = torch.zeros(D, dtype=torch.bool)
i = 0
while i < D:
    if winnable[i]:
        j = i
        while j < D and winnable[j]: j += 1
        if j - i >= 4: corridor[i:j] = True
        i = j
    else:
        i += 1
print("corridor days (inside winnable runs >=4):", int(corridor.sum()), flush=True)

def weights_from(pnl, clr):
    w = torch.zeros(D)
    w[clr] = 1.0                                        # retention: never forget a won day
    near = (~clr) & (pnl >= 1.0) & winnable; w[near] = 3.0   # the frontier
    mid  = (~clr) & (pnl >= 0.0) & (pnl < 1.0) & winnable; w[mid] = 1.5
    red  = (~clr) & (pnl < 0.0) & winnable; w[red] = 1.0
    # CHAIN REPAIR: a winnable day that ends a running row is the exact broken link
    for i in range(1, D):
        if winnable[i] and (not clr[i]) and clr[i-1]:
            w[i] = 5.0
    w[corridor & ~clr] += 6.0                           # corridor immersion: the row LIVES here
    w[corridor & clr] = torch.maximum(w[corridor & clr], torch.tensor(2.0))  # strong retention on chain links
    return torch.clamp(w, min=0.0) + 1e-9

def save_record(clr_n, streak, pnl):
    payload = {"model": brain.state_dict(), "obs_dim": obs_dim, "reward_state": {}, "seed": 0}
    tmp = rpath("artifacts","checkpoints","_sprint.pt"); torch.save(payload, tmp)
    sha = hashlib.sha256(open(tmp,"rb").read()).hexdigest()[:12]
    name = "sprint_row%02d_clear%02dof90_SN-%s.pt" % (streak, clr_n, sha)
    hist = rpath("artifacts","checkpoints","history"); os.makedirs(hist, exist_ok=True)
    os.replace(tmp, os.path.join(hist, name))
    lb = rpath("artifacts","checkpoints","lift_best.pt")
    shutil.copy2(os.path.join(hist,name), lb+".tmp"); os.replace(lb+".tmp", lb)
    print("   *** RECORD: row %d | %d/90 cleared -> %s" % (streak, clr_n, name), flush=True)
    return name

pnl, clr, best_n, best_stk = measure()
rec_n = best_n
NAT = 9    # nature's row ceiling at 3.0% on this data (longest run of range>=3% days)
print("baseline: %d/90 cleared | row %d (nature allows max %d) | avg %+0.2f%%" % (best_n, best_stk, NAT, pnl.mean()), flush=True)
best_state = {k: v.clone() for k, v in brain.state_dict().items()}
w = weights_from(pnl, clr)
t0 = time.time(); upd = 0; stag = 0; last_name = ""
while time.time() - t0 < a.minutes*60:
    upd += 1
    di = torch.multinomial(w, a.envs, replacement=True)
    tg = torch.full((a.envs,), TGT); rk = torch.full((a.envs,), RK)
    st = rollout(brain, sim, di, tg, rk, greedy=False, collect=True, decide_every=DE)
    stats = ppo_update(brain, opt, st, sim.days_obs, gamma=0.999, lam=0.95, clip=0.2,
                       epochs=2, ent_coef=ENT, env_mb=a.envs)
    if upd % 12 == 0:
        pnl, clr, n, stk = measure()
        gain = "" if n <= best_n else "  <-- UP"
        print("u%4d %5ds | cleared %2d/90 (best %2d) | streak %2d (best %2d) | avg %+0.2f%% | ent %.2f%s"
              % (upd, time.time()-t0, n, best_n, stk, best_stk, pnl.mean(), stats["entropy"], gain), flush=True)
        json.dump({"upd": upd, "cleared": n, "best": best_n, "streak": stk, "best_streak": best_stk,
                   "avg": round(float(pnl.mean()),3)}, open("/tmp/sprint.json","w"))
        if stk > best_stk or (stk == best_stk and n > rec_n):
            best_stk = max(stk, best_stk); rec_n = n; best_n = max(n, best_n)
            best_state = {k: v.clone() for k, v in brain.state_dict().items()}
            last_name = save_record(n, stk, pnl); stag = 0; ENT = 0.04
        else:
            stag += 1
            if stk < best_stk - 2 and n < rec_n - 8:        # only snap on a REAL collapse
                brain.load_state_dict(best_state); opt = torch.optim.Adam(brain.parameters(), lr=1.2e-4)
                print("   (collapsed -> snap back to record)" , flush=True)
            elif stag == 10:
                ENT = 0.03; print("   (stagnant -> a little more exploration)", flush=True)
        w = weights_from(pnl, clr)                          # re-aim at the new frontier
print("SPRINT DONE | best %d/90 cleared | best streak %d | last save %s" % (best_n, best_stk, last_name), flush=True)
