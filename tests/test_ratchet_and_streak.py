"""Pins the 2026-07-20 review fixes forever.

5W+I: WHO Claude for Monty (full-repo review). WHAT two regression tests:
(1) _day_end_reward must NEVER move streak/record for envs that are not finalizing
    (the cross-batch contamination that could counterfeit 'days in a row' records);
(2) THE LIFT ITSELF as a test — the committed frozen brain PROVEN_LIFT_2026-07-20
    must bank >= +3.0% on 2026-01-30 at 3.0/3.5 with no breach under the corrected
    ratchet law (lock = goal + flat_cost). If the sim laws, the brain loader, or the
    ratchet drift, this fails and names the day it broke.
WHY: the streak metric is Monty's headline; the lift is the profitability proof.
INTERCONNECTED: training/fastsim, training/gpu_rollout, inference/loader,
data/XAUUSD_curriculum_2026.csv (tracked), artifacts/checkpoints/PROVEN_LIFT (tracked).

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  created — WHY: review found the streak bug + asked that the ratchet
  law fix and the lift be pinned by tests so the two engines can't drift apart.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.configs import path as rpath                       # noqa: E402
from training.gpu_data import build_day_tensors              # noqa: E402
from training.fastsim import FastSim                         # noqa: E402
from training.gpu_rollout import rollout                     # noqa: E402


def _sim(n_days=4):
    do, dp, dl, dates, cols = build_day_tensors(
        rpath("data", "XAUUSD_M1_drill.csv"),
        cache_path=rpath("artifacts", "gpu_cache_XAUUSD_M1_drill.npz"), verbose=False)
    return FastSim(do, dp, dl, cols, device="cpu", K=8), dates


def test_streak_and_record_only_move_for_finalizing_envs():
    sim, _ = _sim()
    sim.reset(torch.tensor([0, 1, 2]), torch.tensor([2.5, 2.5, 2.5]),
              torch.tensor([4.0, 4.0, 4.0]))
    # env1 carries a 10-day streak and a record; env0 finalizes NOW, env1/env2 do not.
    sim.streak = torch.tensor([0.0, 10.0, 0.0])
    sim.record = torch.tensor([0.0, 5.0, 0.0])
    sim.balance = sim.balance.clone()
    sim.balance[2] += sim.eq0 * 0.05                 # env2 mid-day at +5% (would look like a 'win')
    just = torch.tensor([True, False, False])
    sim._day_end_reward(just)
    assert sim.streak[1].item() == 10.0, "non-finalizing env's streak must not move"
    assert sim.streak[2].item() == 0.0, "non-finalizing env must not gain streak"
    assert sim.record[2].item() == 0.0, "non-finalizing env must not absorb a record"
    # and the finalizing env's own bookkeeping did happen (flat day -> streak reset stays 0)
    assert sim.streak[0].item() == 0.0


def test_the_lift_is_pinned_proven_brain_banks_the_target():
    from inference.loader import load_brain
    brain, _ = load_brain("PROVEN_LIFT_2026-07-20")
    assert brain is not None, "the committed frozen lift brain must load"
    do, dp, dl, dates, cols = build_day_tensors(
        rpath("data", "XAUUSD_curriculum_2026.csv"),
        cache_path=rpath("artifacts", "gpu_cache_XAUUSD_curriculum_2026.npz"), verbose=False)
    sim = FastSim(do, dp, dl, cols, device="cpu", K=24)
    day = [i for i, d in enumerate(dates) if "2026-01-30" in str(d)]
    assert day, "2026-01-30 must exist in the curriculum data"
    r = rollout(brain, sim, torch.tensor(day[:1]), torch.tensor([3.0]), torch.tensor([3.5]),
                greedy=True, collect=False, decide_every=5)
    pnl = float(r["day_pnl"].item())
    assert not bool(r["breached"].item()), "the lift day must not breach"
    assert pnl >= 3.0, ("the frozen lift brain banked %.2f%% (<3.0) on its mastered day — "
                        "a sim law, loader, or ratchet drifted" % pnl)
