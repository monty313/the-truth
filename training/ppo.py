"""PPO trainer v2 — joint op+size learning, config-driven, stateful.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0006/0007; audit round-2 fixes).
WHAT:  Day-episode rollouts -> GAE -> clipped PPO over the recurrent
       Brain with JOINT (op + Beta-size) log-probs (audit T1). Reads
       hyperparameters from configs/training.yaml through core.configs
       (audit R11 — training.yaml was decorative). Checkpoints carry the
       RewardEngine state so anti-gravity wheels dissolve across capped
       runs (audit T8) and streak/record memory survives resume.
WHEN:  2026-07-19 (v2, post-audit).
WHERE: scripts/train_bootcamp.py + training/canary.py drive it;
       inference/loader.py restores.
WHY:   A learning loop that silently can't learn one of its action
       dimensions is a dummy in disguise — Monty's instruction forbids it.
INTERCONNECTED WITH: env, policy (joint_logprob), rewards.state_dict,
       experiments/tracker, telemetry (span 'ppo_update' — in STAGES),
       artifacts/checkpoints/.
----------------------------------------------------------------------
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as Fnn

from core.configs import path as rpath, training_cfg
from telemetry import tracer
from training.policy import Brain

CKPT = rpath("artifacts", "checkpoints")
os.makedirs(CKPT, exist_ok=True)


class PPO:
    def __init__(self, env, cfg: dict | None = None, seed: int | None = None):
        tc = training_cfg()
        p = dict(tc.get("ppo", {}))
        p.update(cfg or {})
        hidden = int((cfg or {}).get("hidden", tc.get("policy", {}).get("hidden", 128)))
        self.seed = int(seed if seed is not None else tc.get("seed", 20260718))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        self.env = env
        env.reset(0)
        self.brain = Brain(env.obs_dim, hidden)
        self.opt = torch.optim.Adam(self.brain.parameters(),
                                    lr=float(p.get("lr", 3e-4)))
        self.gamma = float(p.get("gamma", 0.999))
        self.lam = float(p.get("lam", 0.95))
        self.clip = float(p.get("clip", 0.2))
        self.epochs = int(p.get("epochs", 4))
        self.entropy0 = float(p.get("entropy_coef", 0.01))

    # ---------- rollout ----------
    def play_day(self, day_idx: int | None = None, greedy: bool = False):
        env = self.env
        obs = env.reset(day_idx)
        h = None
        O, A, S, LP, V, R = [], [], [], [], [], []
        info = {}
        while True:
            t_obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            op, size, lp, v, h = self.brain.act(t_obs, h, greedy=greedy)
            nobs, r, done, info = env.step(op, size)
            O.append(obs); A.append(op); S.append(size); LP.append(lp)
            V.append(v); R.append(r)
            if done:
                break
            obs = nobs
        return (np.array(O, np.float32), np.array(A), np.array(S, np.float32),
                np.array(LP, np.float32), np.array(V, np.float32),
                np.array(R, np.float32), info)

    # ---------- learning ----------
    def update(self, days_batch: list, entropy_coef: float | None = None):
        ec = self.entropy0 if entropy_coef is None else entropy_coef
        self.env.re.update_idx += 1
        seqs = []
        for O, A, S, LP, V, R, info in days_batch:
            adv = np.zeros_like(R)
            last = 0.0
            for t in range(len(R) - 1, -1, -1):
                nv = V[t + 1] if t + 1 < len(V) else 0.0     # terminal V = 0
                delta = R[t] + self.gamma * nv - V[t]
                last = delta + self.gamma * self.lam * last
                adv[t] = last
            seqs.append((O, A, S, LP, adv, adv + V))
        all_adv = np.concatenate([s[4] for s in seqs])
        mu, sd = all_adv.mean(), all_adv.std() + 1e-8

        stats = {}
        with tracer.span("ppo_update", days=len(seqs),
                         steps=int(sum(len(s[1]) for s in seqs))):
            for _ in range(self.epochs):
                for O, A, S, LP, adv, ret in seqs:          # one day = one seq
                    obs = torch.tensor(O).unsqueeze(0)      # (1, T, D)
                    op_d, sz_d, value, _ = self.brain(obs)
                    ops = torch.tensor(A)
                    sizes = torch.tensor(S)
                    lp_new = self.brain.joint_logprob(
                        op_d, sz_d, ops.unsqueeze(0), sizes.unsqueeze(0))[0]
                    ratio = torch.exp(lp_new - torch.tensor(LP))
                    advt = torch.tensor((adv - mu) / sd)
                    pl = -torch.min(
                        ratio * advt,
                        torch.clamp(ratio, 1 - self.clip, 1 + self.clip) * advt
                    ).mean()
                    vl = Fnn.mse_loss(value[0], torch.tensor(ret))
                    ent = (op_d.entropy().mean() + 0.1 * sz_d.entropy().mean())
                    loss = pl + 0.5 * vl - ec * ent
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
                    self.opt.step()
            stats = {"policy_loss": float(pl.item()),
                     "value_loss": float(vl.item()),
                     "entropy": float(ent.item())}
        return stats

    # ---------- persistence (audit T8/R6) ----------
    def save(self, name: str) -> str:
        p = os.path.join(CKPT, f"{name}.pt")
        torch.save({"model": self.brain.state_dict(),
                    "obs_dim": self.env.obs_dim,
                    "reward_state": self.env.re.state_dict(),
                    "seed": self.seed}, p)
        return p

    def load(self, name: str) -> bool:
        p = os.path.join(CKPT, f"{name}.pt")
        if not os.path.exists(p):
            return False
        d = torch.load(p, weights_only=False)
        if d.get("obs_dim") != self.env.obs_dim:
            return False
        try:
            self.brain.load_state_dict(d["model"])
        except RuntimeError:
            return False            # architecture drift (e.g. v1 checkpoint)
        self.env.re.load_state(d.get("reward_state", {}))
        return True
