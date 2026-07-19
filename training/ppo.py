"""PPO trainer — rollout day-lives, learn, checkpoint, report.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0006/0007; configs/training.yaml).
WHAT:  Collects day episodes from TradingEnv, computes GAE, runs clipped
       PPO updates over the recurrent Brain (sequence minibatches =
       whole days, keeping the GRU honest). Emits run cards, checkpoints,
       and a per-day journal for reports/trophy case.
WHEN:  2026-07-19 overnight build.
WHERE: scripts/train_bootcamp.py + training/canary.py drive it.
WHY:   The learning loop, kept small and inspectable — every update is
       a span; every day a journal row; nothing hidden.
INTERCONNECTED WITH: env, policy, rewards (update_idx decay), tracker,
       telemetry, artifacts/checkpoints/.
----------------------------------------------------------------------
"""
from __future__ import annotations
import os
import numpy as np
import torch
import torch.nn.functional as Fnn

from telemetry import tracer
from training.policy import Brain

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(ROOT, "artifacts", "checkpoints")
os.makedirs(CKPT, exist_ok=True)


class PPO:
    def __init__(self, env, cfg: dict, seed: int = 20260718):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env = env
        env.reset(0)                       # resolve obs columns
        self.brain = Brain(env.obs_dim, cfg.get("hidden", 128))
        self.opt = torch.optim.Adam(self.brain.parameters(), lr=cfg.get("lr", 3e-4))
        self.gamma = cfg.get("gamma", 0.999)
        self.lam = cfg.get("lam", 0.95)
        self.clip = cfg.get("clip", 0.2)
        self.epochs = cfg.get("epochs", 4)
        self.entropy0 = cfg.get("entropy_coef", 0.01)
        self.journal: list[dict] = []

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
    def update(self, days_batch: list, entropy_coef: float):
        self.env.re.update_idx += 1
        seqs = []
        for O, A, S, LP, V, R, info in days_batch:
            adv = np.zeros_like(R)
            last = 0.0
            for t in range(len(R) - 1, -1, -1):
                nv = V[t + 1] if t + 1 < len(V) else 0.0
                delta = R[t] + self.gamma * nv - V[t]
                last = delta + self.gamma * self.lam * last
                adv[t] = last
            ret = adv + V
            seqs.append((O, A, S, LP, adv, ret))
        all_adv = np.concatenate([s[4] for s in seqs])
        mu, sd = all_adv.mean(), all_adv.std() + 1e-8

        stats = {}
        with tracer.span("checkpoint", stage="ppo_update",
                         days=len(seqs), steps=int(sum(len(s[1]) for s in seqs))):
            for _ in range(self.epochs):
                for O, A, S, LP, adv, ret in seqs:      # one day = one sequence
                    obs = torch.tensor(O).unsqueeze(0)               # (1,T,D)
                    logits, size, value, _ = self.brain(obs)
                    logits, size, value = logits[0], size[0], value[0]
                    dist = torch.distributions.Categorical(logits=logits)
                    a = torch.tensor(A)
                    lp_new = dist.log_prob(a)
                    ratio = torch.exp(lp_new - torch.tensor(LP))
                    advt = torch.tensor((adv - mu) / sd)
                    pl = -torch.min(ratio * advt,
                                    torch.clamp(ratio, 1 - self.clip,
                                                1 + self.clip) * advt).mean()
                    vl = Fnn.mse_loss(value, torch.tensor(ret))
                    szl = Fnn.mse_loss(size.squeeze(-1), torch.tensor(S))
                    ent = dist.entropy().mean()
                    loss = pl + 0.5 * vl + 0.1 * szl - entropy_coef * ent
                    self.opt.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.brain.parameters(), 0.5)
                    self.opt.step()
            stats = {"policy_loss": float(pl.item()), "value_loss": float(vl.item()),
                     "entropy": float(ent.item())}
        return stats

    def save(self, name: str) -> str:
        path = os.path.join(CKPT, f"{name}.pt")
        torch.save({"model": self.brain.state_dict(),
                    "obs_dim": self.env.obs_dim}, path)
        return path
