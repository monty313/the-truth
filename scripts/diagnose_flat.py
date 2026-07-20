"""WHY is the bot flat? Measure ACTIVITY (trades/day) + results for every key brain
at native vs new settings. The per-trade risk cap is 0.25%, so hitting 3%/day NEEDS
many trades: a brain doing ~0-2 trades/day is 'lid on' (timid), not unlucky.
Also replays the sacred +6.53% day (2026-01-29) in the twin at native settings."""
import os, sys, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import torch
from core.configs import path as rpath
from training.gpu_data import build_day_tensors
from training.fastsim import FastSim, SELF_DIM
from training.gpu_rollout import rollout
from training.policy import Brain
from inference.loader import load_brain

torch.manual_seed(0)

def load(name, obs_dim):
    if name == "fresh":
        from core.configs import policy_hidden
        return Brain(obs_dim, hidden=policy_hidden())
    b, _ = load_brain(name)
    if b is None:
        print("(%s not on this machine — skipping its rows)" % name, flush=True)
    return b

def bench(tag, brain, sim, D, tg, rk, de):
    if brain is None:                      # brain not present on this machine (fresh clone)
        return None, None
    t0 = time.time()
    di = torch.arange(D)
    r = rollout(brain, sim, di, torch.full((D,), tg), torch.full((D,), rk),
                greedy=True, collect=False, decide_every=de)
    pnl = r["day_pnl"].float()
    tr = sim.trades_used.float()
    print("%-34s | clr %4.0f%% | brch %3.0f%% | avg %+6.2f%% | grn %3.0f%% | trades/day %5.1f (max %3.0f) | best %+5.2f%% | %3.0fs"
          % (tag, r["goal_hit"].float().mean()*100, r["breached"].float().mean()*100,
             pnl.mean(), (pnl > 0).float().mean()*100, tr.mean(), tr.max(), pnl.max(), time.time()-t0), flush=True)
    return r, tr

# ---------- curriculum (90 days) ----------
csv = rpath("data", "XAUUSD_curriculum_2026.csv")
do, dp, dl, dates, cols = build_day_tensors(csv, cache_path=rpath("artifacts","gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)
obs_dim = 10 * (len(cols) + SELF_DIM); D = do.shape[0]
sim = FastSim(do, dp, dl, cols, device="cpu", K=24)
print("curriculum: %d days, %s .. %s | per-trade risk cap 0.25%% -> 3%%/day NEEDS activity" % (D, dates[0], dates[-1]), flush=True)
print("-" * 128, flush=True)

proven = load("PROVEN_2x_2026-07-19", obs_dim)
gbest  = load("gpu_best", obs_dim)
fresh  = load("fresh", obs_dim)

bench("PROVEN @2.5/4.0 de=1 (native)", proven, sim, D, 2.5, 4.0, 1)
bench("PROVEN @2.5/4.0 de=5",          proven, sim, D, 2.5, 4.0, 5)
bench("PROVEN @3.0/3.5 de=1",          proven, sim, D, 3.0, 3.5, 1)
bench("gpu_best @3.0/3.5 de=1",        gbest,  sim, D, 3.0, 3.5, 1)
bench("gpu_best @3.0/3.5 de=5",        gbest,  sim, D, 3.0, 3.5, 5)
bench("FRESH @3.0/3.5 de=1 (greedy)",  fresh,  sim, D, 3.0, 3.5, 1)

# ---------- the sacred day: 2026-01-29 at native settings ----------
tgt_day = [i for i, d in enumerate(dates) if "2026-01-29" in str(d)]
if tgt_day and proven is not None:
    i = tgt_day[0]
    r = rollout(proven, sim, torch.tensor([i]), torch.tensor([2.5]), torch.tensor([4.0]),
                greedy=True, collect=False, decide_every=1)
    print("-" * 128, flush=True)
    print("SACRED DAY 2026-01-29 (proof said +6.53%%): twin says %+0.2f%% | trades %d | breached %s"
          % (r["day_pnl"].item(), int(sim.trades_used.item()), bool(r["breached"].item())), flush=True)
else:
    print("2026-01-29 NOT in curriculum dates", flush=True)

# ---------- drill data (PROVEN's home turf) ----------
do2, dp2, dl2, dates2, cols2 = build_day_tensors(rpath("data","XAUUSD_M1_drill.csv"),
    cache_path=rpath("artifacts","gpu_cache_XAUUSD_M1_drill.npz"), verbose=False)
sim2 = FastSim(do2, dp2, dl2, cols2, device="cpu", K=24)
D2 = do2.shape[0]
print("-" * 128, flush=True)
print("drill (home turf): %d days, %s .. %s" % (D2, dates2[0], dates2[-1]), flush=True)
bench("PROVEN @2.5/4.0 de=1 on DRILL", proven, sim2, D2, 2.5, 4.0, 1)
td = [i for i, d in enumerate(dates2) if "2026-01-29" in str(d)]
if td and proven is not None:
    r = rollout(proven, sim2, torch.tensor([td[0]]), torch.tensor([2.5]), torch.tensor([4.0]),
                greedy=True, collect=False, decide_every=1)
    print("SACRED DAY on drill data: %+0.2f%% | trades %d" % (r["day_pnl"].item(), int(sim2.trades_used.item())), flush=True)
print("DIAGNOSIS DONE", flush=True)
