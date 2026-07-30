"""Score a brain at YOUR target% and risk% (runtime inputs — no retrain).

Usage: python scripts/prove_it.py [brain_name] [target%] [risk%]
Examples:
  python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
  python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 2.5 2.5

If you changed include_signal_agent_slots in features.yaml, delete:
  artifacts/gpu_cache_XAUUSD_curriculum_2026.npz
before running.
"""
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(ROOT, 'src')); sys.path.insert(0, ROOT)
import torch
from core.configs import path as rpath, decide_every as cfg_decide
DE = cfg_decide()
from training.gpu_data import build_day_tensors
from training.fastsim import FastSim, SELF_DIM
from evaluation.consistency import evaluate
from training.gpu_rollout import rollout
from training.meta_tuner import day_after_day_streak
from inference.loader import load_brain

name = sys.argv[1] if len(sys.argv) > 1 else "PROVEN_2x_2026-07-19"
TGT = float(sys.argv[2]) if len(sys.argv) > 2 else 3.0
RISK = float(sys.argv[3]) if len(sys.argv) > 3 else 3.5

csv = rpath("data", "raw", "XAUUSD_curriculum_2026.csv")
do, dp, dl, dates, cols = build_day_tensors(
    csv, cache_path=rpath("artifacts", "gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False
)
obs_dim = 10 * (len(cols) + SELF_DIM)
D = int(do.shape[0])
print("obs: %d market cols | frame_dim=%d | days=%d" % (len(cols), obs_dim, D))
sim = FastSim(do, dp, dl, cols, device="cpu", K=24)
brain, meta = load_brain(name)
if brain is None:
    print("could not load brain:", name)
    sys.exit(1)

rng = {
    "tgt_lo": TGT, "tgt_hi": TGT, "risk_lo": RISK, "risk_hi": RISK,
    "focus_target": TGT, "focus_risk": RISK, "focus_frac": 1.0,
}
g = torch.Generator(device="cpu").manual_seed(0)
res = evaluate(brain, sim, list(range(D)), n_episodes=D, focus_frac=1.0, decide_every=DE, gen=g, ranges=rng)
g2 = torch.Generator(device="cpu").manual_seed(0)
stk = day_after_day_streak(brain, sim, list(range(D)), gen=g2, focus_frac=1.0, decide_every=DE, ranges=rng)

di = torch.arange(D)
tg = torch.full((D,), TGT)
rk = torch.full((D,), RISK)
r = rollout(brain, sim, di, tg, rk, greedy=True, collect=False, decide_every=DE)
pnl = r["day_pnl"].float()

print("=" * 64)
print("BRAIN: %s   |   %d real trading days" % (name, D))
print("AT YOUR NUMBERS  ->  target %.1f%% / risk %.1f%%  (runtime inputs — no retrain)" % (TGT, RISK))
print("  Re-run with different target/risk anytime:  python scripts/prove_it.py %s <tgt> <risk>" % name)
print("-" * 64)
print("  cleared (hit target, NO breach):   %5.0f%% of days" % (res["consistency"] * 100))
print("  breached the risk floor:           %5.0f%% of days" % (res["breach_rate"] * 100))
print("  longest cleared streak in a row:   %5d days" % stk["longest_streak"])
print("  average day result:               %+6.2f%%" % pnl.mean().item())
print("  median day result:                %+6.2f%%" % pnl.median().item())
print("  green days (made money):           %5.0f%% of days" % ((pnl > 0).float().mean().item() * 100))
print("  best / worst day:                 %+6.2f%% / %+6.2f%%" % (pnl.max().item(), pnl.min().item()))
print("=" * 64)
