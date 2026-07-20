"""Momentum One — the SELF-TUNER entry point (Monty's spec, 2026-07-20).

WHAT (plain words): this is the bot that learns to learn. It doesn't just practice —
it also TUNES ITS OWN rewards, penalties and learning settings toward one thing:
CONSISTENCY (clearing your target without breaching, day after day in a row). Every
"generation" it tries a batch of small tweaks, trains each a little, and KEEPS a tweak
ONLY if it clearly clears more days than the champion on FRESH days. The champion is
never lost; on a plateau it explores wider. It saves a new best (with the passed-day
count in the name) and resumes exactly where it left off after any Colab restart.

The ONLY two numbers you ever set are daily target% and risk% (configs/goals.yaml).
Everything else here is derived or self-tuned.

USAGE (Colab L4):  python scripts/self_tune.py
Tiny CPU smoke:    python scripts/self_tune.py --device cpu --smoke

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  created — WHY: Phase 1 entry for the self-tuner run() loop (two-stage
  ratchet, CRN, resume). Thin wrapper: build sim from data, 3-way day split, call run().
  (Distinct from the legacy scripts/meta_train.py CPU any-X trainer.)
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import argparse
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.configs import path as rpath, training_cfg          # noqa: E402
from training.fastsim import FastSim, SELF_DIM                # noqa: E402
from training.gpu_data import build_day_tensors               # noqa: E402
from training.meta_tuner import run, hold_out_audit           # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=rpath("data", "XAUUSD_curriculum_2026.csv"))
    ap.add_argument("--minutes", type=float, default=1440.0)   # run until Colab stops it
    ap.add_argument("--device", default="auto")
    ap.add_argument("--K", type=int, default=24)               # position slots (match GPU trainer)
    ap.add_argument("--decide-every", type=int, default=5)
    ap.add_argument("--instances", type=int, default=2048)     # per-probe env count
    ap.add_argument("--n-updates", type=int, default=4)        # PPO updates per probe (short train)
    ap.add_argument("--smoke", action="store_true")            # tiny, fast end-to-end sanity run
    a = ap.parse_args()

    dev = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    seed = int(training_cfg().get("seed", 20260718))
    torch.manual_seed(seed); np.random.seed(seed)

    tag = os.path.splitext(os.path.basename(a.csv))[0]
    do, dp, dl, dates, cols = build_day_tensors(a.csv, cache_path=rpath("artifacts", "gpu_cache_%s.npz" % tag))
    obs_dim = 10 * (len(cols) + SELF_DIM)
    D = do.shape[0]

    K, n_updates, instances, minutes = a.K, a.n_updates, a.instances, a.minutes
    candidates, n_eval = None, None                            # None -> from configs/training.yaml self_tuner
    if a.smoke:                                                # tiny end-to-end proof (fast on CPU)
        do, dp, dl = do[:16], dp[:16], dl[:16]; D = 16
        K, n_updates, instances, minutes = 4, 1, 32, 0.2
        candidates, n_eval = 3, 24                             # shrink the per-generation cost drivers

    work, audit = hold_out_audit(D)          # audit is PERMANENTLY held out; select rotates inside the work pool
    print("=" * 70, flush=True)
    print("MOMENTUM ONE — SELF-TUNER | device=%s | days=%d (work %d / audit %d; select rotates each gen)"
          % (dev, D, len(work), len(audit)), flush=True)
    print("obs_dim=%d (same shape as PROVEN) | tuning rewards+penalties+lr toward CONSISTENCY"
          % obs_dim, flush=True)
    print("north star: clear the target without breaching, day after day in a row", flush=True)
    print("=" * 70, flush=True)

    sim = FastSim(do, dp, dl, cols, device=dev, K=K)
    out = run(sim, obs_dim, work, audit, minutes=minutes, n_updates=n_updates,
              instances=instances, decide_every=a.decide_every, base_seed=seed,
              candidates=candidates, n_eval=n_eval)
    print("done:", out, flush=True)


if __name__ == "__main__":
    main()
