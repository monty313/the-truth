"""Can a FRESH (un-saturated) brain learn to trade toward 3%? Warm-starting from the
collapsed brain gives a dead gradient; this tests a random-init brain instead."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from core.configs import path as rpath, training_cfg
from training.gpu_data import build_day_tensors
from training.fastsim import FastSim, SELF_DIM
from training.gpu_rollout import rollout, ppo_update
from training.policy import Brain

torch.manual_seed(0)
csv = rpath("data", "XAUUSD_curriculum_2026.csv")
do, dp, dl, dates, cols = build_day_tensors(csv, cache_path=rpath("artifacts","gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)
obs_dim = 10*(len(cols)+SELF_DIM); D = int(do.shape[0])
sim = FastSim(do, dp, dl, cols, device="cpu", K=12)
tc = training_cfg(); p = tc["ppo"]
brain = Brain(obs_dim, hidden=128)                     # FRESH — high entropy, live gradient
opt = torch.optim.Adam(brain.parameters(), lr=3e-4)
print("FRESH brain | %d days | rewards: net_profit=%.0f did_nothing=%.0f death=%.0f | entropy=%.2f"
      % (D, sim.w["w_net_profit"], sim.w["w_did_nothing"], sim.w["w_death_penalty"], float(p["entropy_coef"])))
print("target 3.0%% / risk 3.5%% | watch: does ploss move and hit-target climb off 0?")
t0=time.time()
for u in range(1, 41):
    di = torch.randint(0, D, (256,)); tg = torch.full((256,),3.0); rk = torch.full((256,),3.5)
    stored = rollout(brain, sim, di, tg, rk, greedy=False, collect=True, decide_every=5)
    st = ppo_update(brain, opt, stored, sim.days_obs, gamma=0.999, lam=0.95, clip=0.2,
                    epochs=1, ent_coef=float(p["entropy_coef"]), env_mb=256)
    r = stored["results"]
    if u%2==0 or u<=4:
        print("upd %3d | %3ds | mean pnl %+.2f%% | hit-3%% %4.1f%% | breach %4.1f%% | ploss %+.4f"
              % (u, time.time()-t0, r["day_pnl"].mean()*100 if r["day_pnl"].mean()<1 else r["day_pnl"].mean(),
                 r["goal_hit"].float().mean()*100, r["breached"].float().mean()*100, st["policy_loss"]), flush=True)
