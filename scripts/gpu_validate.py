"""Validate the twin against the REAL sim + replay the proof (Bot 1.5).

5W+I: WHO Claude for Monty. WHAT runs the SAME brain, greedy, over the SAME days
on BOTH the real DaySim (via TradingEnv/bench) and FastSim, and prints them
side by side so we can SEE the gap before trusting the twin. Also spotlights the
proof day. WHEN 2026-07-20. WHERE container or Colab. WHY Monty's rule: the real
sim is the judge; the twin only earns training if it matches.
USAGE:  python scripts/gpu_validate.py --csv data/XAUUSD_M1_drill.csv --ckpt PROVEN_2x_2026-07-19

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  created — WHY: prove FastSim matches DaySim + replay the proof.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.configs import path as rpath                          # noqa: E402
from data_io.loader import read_mt5_m1, trading_days            # noqa: E402
from features.engine import build_features, obs_columns         # noqa: E402
from training.env import TradingEnv                             # noqa: E402
from training.ppo import PPO                                    # noqa: E402
from evaluation.champion import bench                           # noqa: E402
from training.fastsim import FastSim, SELF_DIM                  # noqa: E402
from training.gpu_rollout import rollout                        # noqa: E402

PHYS = ["high", "low", "close", "spread", "15min::atr14", "mask_buy_blocked", "mask_sell_blocked"]


def tensors_from_days(days, cols):
    D = len(days); Lmax = max(len(g) for _, g in days)
    do = np.zeros((D, Lmax, len(cols)), np.float32)
    dp = np.zeros((D, Lmax, len(PHYS)), np.float32)
    dl = np.zeros(D, np.int64)
    for i, (_, g) in enumerate(days):
        n = len(g)
        do[i, :n] = np.nan_to_num(g[cols].to_numpy(np.float32), nan=0.0, posinf=5.0, neginf=-5.0)
        dp[i, :n] = np.nan_to_num(g[PHYS].to_numpy(np.float32), nan=0.0)
        dl[i] = n
    return do, dp, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=rpath("data", "XAUUSD_M1_drill.csv"))
    ap.add_argument("--ckpt", default="PROVEN_2x_2026-07-19")
    ap.add_argument("--goal", type=float, default=2.5)
    ap.add_argument("--floor", type=float, default=4.0)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    print("building features once (shared by both sims)...", flush=True)
    m1 = read_mt5_m1(a.csv)
    F = build_features(m1)
    cols = obs_columns(F)
    days = trading_days(F)
    dates = [d for d, _ in days]
    print("days: %d | cols: %d | obs_dim: %d" % (len(days), len(cols), 10 * (len(cols) + SELF_DIM)), flush=True)

    # ---- REAL sim (the judge) ----
    env = TradingEnv(days, a.goal, a.floor)
    ppo = PPO(env)
    ok = ppo.load(a.ckpt)
    print("loaded %s: %s (obs_dim match => no shape mismatch)" % (a.ckpt, ok), flush=True)
    real = bench(ppo, len(days))["day_details"]
    real_pnl = np.array([d.get("pnl_pct", 0.0) for d in real], np.float32)
    real_br = np.array([1 if d.get("breached") else 0 for d in real])

    # ---- FastSim twin (same brain weights) ----
    do, dp, dl = tensors_from_days(days, cols)
    sim = FastSim(do, dp, dl, cols, device=a.device)
    N = len(days)
    di = torch.arange(N, device=a.device)
    g = torch.full((N,), a.goal, device=a.device)
    fl = torch.full((N,), a.floor, device=a.device)
    res = rollout(ppo.brain.to(a.device), sim, di, g, fl, greedy=True, collect=False)
    fast_pnl = res["day_pnl"].cpu().numpy()
    fast_br = res["breached"].cpu().numpy().astype(int)

    # ---- compare ----
    print("\n date        real%    twin%    gap")
    for i in range(N):
        star = "  <-- proof" if str(dates[i]) == "2026-01-29" else ""
        print(" %-10s %+7.2f %+7.2f  %+6.2f%s"
              % (dates[i], real_pnl[i], fast_pnl[i], fast_pnl[i] - real_pnl[i], star))
    gap = np.abs(fast_pnl - real_pnl)
    print("\nMEAN abs gap: %.3f%% | MAX abs gap: %.3f%%" % (gap.mean(), gap.max()))
    print("breach agreement: %d/%d days match" % (int((real_br == fast_br).sum()), N))
    print("real breaches: %d | twin breaches: %d" % (int(real_br.sum()), int(fast_br.sum())))
    j = [i for i in range(N) if str(dates[i]) == "2026-01-29"]
    if j:
        print("PROOF DAY 2026-01-29 -> real %+.2f%%, twin %+.2f%%" % (real_pnl[j[0]], fast_pnl[j[0]]))


if __name__ == "__main__":
    main()
