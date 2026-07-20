"""Batched rollout + PPO for the GPU Edition (Bot 1.5).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (Bot 1.5 GPU Edition, 2026-07-20).
WHAT:  Drives the UNCHANGED training/policy.Brain through FastSim for N envs
       at once: batched act, batched GAE, clipped PPO. The PPO update never
       stores the giant obs tensor — it RECONSTRUCTS each env-minibatch's obs
       from the saved self-state history + the shared day_obs (so 8,000 envs
       fit in an L4). Same math as training/ppo.py (gamma/lam/clip/epochs from
       configs/training.yaml), just vectorized.
WHY:   One brain, thousands of markets, no shape change.
WHEN:  2026-07-20.
INTERCONNECTED WITH: training/fastsim.FastSim, training/policy.Brain,
       scripts/gpu_train, scripts/gpu_validate.
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):
- 2026-07-20  created — WHY: batched rollout/PPO with obs-reconstruction so the
  8,000-env update fits in GPU memory; brain untouched.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import torch

FRAME = 10
SELF_DIM = 12


@torch.no_grad()
def batched_act(brain, obs, h, greedy: bool):
    op_dist, size_dist, value, h2 = brain(obs, h)
    if greedy:
        op = torch.argmax(op_dist.logits, dim=-1)
        size = size_dist.mean.clamp(0.05, 1.0)
    else:
        op = op_dist.sample()
        size = size_dist.sample().clamp(0.05, 1.0)
    logp = brain.joint_logprob(op_dist, size_dist, op, size)
    return op, size, logp, value, h2


@torch.no_grad()
def rollout(brain, sim, day_idx, goal, floor, greedy=False, collect=True,
            streak_in=None, record_in=None, decide_every=1):
    """One batched day-episode. The brain DECIDES once per `decide_every` bars and
    HOLDS in between — that shrinks the sequence length, so both the rollout and the
    PPO update get faster (the safety Shell still runs on every bar inside the sim).
    collect=True stores one transition per decision for PPO."""
    obs = sim.reset(day_idx, goal, floor, streak_in, record_in)
    h = None
    hold = torch.zeros(sim.N, dtype=torch.long, device=sim.dev)   # op 0 = hold
    ops, sizes, logps, vals, rews, alives = [], [], [], [], [], []
    step = 0
    while True:
        alive = (~sim.dead & ~sim.finalized).float()          # real-transition mask
        op, size, logp, value, h = batched_act(brain, obs, h, greedy)
        r_acc = torch.zeros(sim.N, device=sim.dev)
        done = None
        for k in range(decide_every):                          # act once, then hold
            obs, r, done = sim.step(op if k == 0 else hold, size)
            r_acc = r_acc + r
            step += 1
            if bool(done.all()) or step >= sim.Lmax - 1:
                break
        if collect:
            ops.append(op); sizes.append(size); logps.append(logp)
            vals.append(value); rews.append(r_acc); alives.append(alive)
        if bool(done.all()) or step >= sim.Lmax - 1:
            break
    if not collect:
        return sim.results()
    st = lambda L: torch.stack(L, 0)
    return {"op": st(ops).long(), "size": st(sizes), "logp": st(logps),
            "value": st(vals), "reward": st(rews), "alive": st(alives),
            "self_hist": sim.self_hist.clone(), "day_idx": sim.day_idx.clone(),
            "results": sim.results(), "T": step, "decide_every": decide_every}


def compute_gae(reward, value, alive, gamma, lam):
    T, N = reward.shape
    adv = torch.zeros_like(reward)
    last = torch.zeros(N, device=reward.device)
    zero = torch.zeros(N, device=reward.device)
    for t in range(T - 1, -1, -1):
        nonterm = alive[t + 1] if t + 1 < T else zero
        nv = value[t + 1] * nonterm if t + 1 < T else zero
        delta = reward[t] + gamma * nv - value[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last * alive[t]
    returns = adv + value * alive
    return adv, returns


def _reconstruct(day_idx_mb, self_hist_mb, days_obs, T, decide_every=1):
    """Rebuild (B,T,obs_dim) obs from saved self-state history + shared day_obs,
    frame-stacked EXACTLY like FastSim._build_obs. Decision i sits at bar
    i*decide_every, and its 10 frames are the consecutive bars ending there."""
    B = day_idx_mb.shape[0]
    dev = days_obs.device
    C = days_obs.shape[2]
    Lmax = days_obs.shape[1]
    pos = torch.arange(T, device=dev) * decide_every
    out = torch.empty(B, T, FRAME * (C + SELF_DIM), device=dev)
    for k in range(FRAME):
        idx = torch.clamp(pos + (k - (FRAME - 1)), min=0, max=Lmax - 1)   # (T,)
        m = days_obs[day_idx_mb[:, None], idx[None, :], :]         # (B,T,C)
        sh = self_hist_mb[:, idx, :]                               # (B,T,12)
        out[:, :, k * (C + SELF_DIM):(k + 1) * (C + SELF_DIM)] = torch.cat([m, sh], dim=2)
    return out


def ppo_update(brain, opt, stored, days_obs, gamma=0.999, lam=0.95, clip=0.2,
               epochs=2, ent_coef=0.01, env_mb=512):
    op = stored["op"]; size = stored["size"]; logp_old = stored["logp"]
    value = stored["value"]; reward = stored["reward"]; alive = stored["alive"]
    self_hist = stored["self_hist"]; day_idx = stored["day_idx"]
    de = int(stored.get("decide_every", 1))
    T, N = reward.shape
    adv, returns = compute_gae(reward, value, alive, gamma, lam)
    m = alive > 0.5
    mu = adv[m].mean(); sd = adv[m].std() + 1e-8
    adv_n = (adv - mu) / sd
    pl = vl = ent = torch.tensor(0.0)
    for _ in range(epochs):
        perm = torch.randperm(N, device=reward.device)
        for s in range(0, N, env_mb):
            eb = perm[s:s + env_mb]
            obs_seq = _reconstruct(day_idx[eb], self_hist[eb], days_obs, T, de)   # (B,T,D)
            op_d, sz_d, val, _ = brain(obs_seq)
            a = alive[:, eb].transpose(0, 1)                                  # (B,T)
            denom = a.sum().clamp(min=1.0)
            lp = brain.joint_logprob(op_d, sz_d,
                                     op[:, eb].transpose(0, 1),
                                     size[:, eb].transpose(0, 1))
            ratio = torch.exp((lp - logp_old[:, eb].transpose(0, 1)).clamp(-20, 20))
            A = adv_n[:, eb].transpose(0, 1)
            pl = -(torch.min(ratio * A, torch.clamp(ratio, 1 - clip, 1 + clip) * A) * a).sum() / denom
            R = returns[:, eb].transpose(0, 1)
            vl = (((val - R) ** 2) * a).sum() / denom
            ent = ((op_d.entropy() + 0.1 * sz_d.entropy()) * a).sum() / denom
            loss = pl + 0.5 * vl - ent_coef * ent
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(brain.parameters(), 0.5)
            opt.step()
    return {"policy_loss": float(pl.item()), "value_loss": float(vl.item()),
            "entropy": float(ent.item())}
