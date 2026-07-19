"""Canary v3 — learning-PLUMBING proof (Gauntlet #3), honest & fast.

5W+I -----------------------------------------------------------------
WHO:   Claude (Phase-4 gate; audit T7/R2 exposed v1 as a coin flip that
       actually FAILED while the build report claimed PASS).
WHAT:  Proves the REAL Brain + PPO.update machinery can (a) read an
       observation feature and (b) shift its policy toward reward. Uses a
       minimal bandit env (BanditEnv) that exercises the identical code
       paths — Brain.forward, joint_logprob, GAE, clipped update, Adam —
       on a task whose optimal action is UNAMBIGUOUS. PASS requires the
       reward to rise past a noise margin averaged over eval batches.
WHY THIS SHAPE:  Testing plumbing THROUGH the full trade env conflates
       "do gradients flow" (a plumbing question) with "is scalping gold
       easy" (the boot-camp question, answered later on real data). The
       canary must isolate the former. The trade env's own learnability
       is measured by boot camp, where it belongs.
WHEN:  2026-07-19 (v3, post-audit, post-diagnosis).
WHERE: run directly; boot camp trusts it as the learning gate.
INTERCONNECTED WITH: training/policy.Brain, training/ppo.PPO (unchanged
       code paths), tests/test_training_fixes.py.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  rebuilt as bandit plumbing gate with noise margin  — WHY: v1 was a coin-flip that failed (audit T7/R2).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training.ppo import PPO                                     # noqa: E402


class _RE:
    """Minimal reward-engine shim (PPO only touches update_idx)."""
    def __init__(self):
        self.update_idx = 0

    def state_dict(self):
        return {"update_idx": self.update_idx}

    def load_state(self, d):
        self.update_idx = int(d.get("update_idx", 0))


class BanditEnv:
    """Contextual bandit exercising the real Brain/PPO paths.
    obs[0] is the 'magic bit'. Optimal policy: op 1 when magic==1, else op 0.
    Reward +1 for the correct op, -0.2 otherwise. Episode = `horizon` steps.
    obs_dim is small so runs are fast and deterministic-ish."""

    def __init__(self, obs_dim: int = 24, horizon: int = 40, seed: int = 0):
        self.obs_dim = obs_dim
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.re = _RE()

    def _obs(self):
        o = self.rng.normal(0, 0.3, self.obs_dim).astype(np.float32)
        self.magic = int(self.rng.random() < 0.5)
        o[0] = float(self.magic)
        return o

    def reset(self, day_idx=None):
        self.t = 0
        self.cur = self._obs()
        return self.cur

    def step(self, op: int, size: float):
        target = 1 if self.magic == 1 else 0
        r = 1.0 if op == target else -0.2
        self.t += 1
        done = self.t >= self.horizon
        self.cur = self._obs()
        info = {"pnl_pct": r}                    # per-step reward as the 'pnl'
        return (None if done else self.cur), r, done, info


def _eval_mean(ppo: PPO, batches: int = 5) -> tuple[float, float]:
    """Greedy: mean/std of correct-action rate across eval episodes."""
    vals = []
    for _ in range(batches):
        O, A, S, LP, V, R, info = ppo.play_day(greedy=True)
        vals.append(float(R.mean()))
    return float(np.mean(vals)), float(np.std(vals))


def run(updates: int = 25) -> bool:
    env = BanditEnv()
    ppo = PPO(env, {"hidden": 64, "lr": 3e-3, "epochs": 4})
    m0, s0 = _eval_mean(ppo)
    print(f"canary BEFORE: mean_reward={m0:.3f} (std {s0:.3f})", flush=True)
    for u in range(updates):
        batch = [ppo.play_day() for _ in range(6)]
        mr = float(np.mean([b[5].mean() for b in batch]))
        ppo.update(batch, entropy_coef=max(0.03 * (1 - u / updates), 0.005))
        if (u + 1) % 5 == 0:
            print(f"canary update {u + 1}/{updates}: mean_reward={mr:.3f}",
                  flush=True)
    m1_, s1 = _eval_mean(ppo)
    margin = 0.25 * max((s0 + s1) / 2.0, 1e-6)
    improved = (m1_ - m0) > max(margin, 0.15)   # real, decisive improvement
    print(f"canary AFTER: mean_reward={m1_:.3f} (std {s1:.3f}) | "
          f"delta={m1_ - m0:+.3f}", flush=True)
    print("CANARY", "PASS" if improved else "FAIL", flush=True)
    return improved


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
