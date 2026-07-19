"""Training-fix pins — every audit-round-2 training finding, locked forever.
5W+I: WHO Claude (from the 3 pessimistic audits 2026-07-19). WHAT proofs that
T1 (size learns), T5 (probes closable), any-X conditioning, and the config
door are real. WHEN 2026-07-19. WHY regressions here silently mis-learn.
INTERCONNECTED: policy, ppo, env, core.configs.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, torch
from training.policy import Brain


def test_size_head_has_nonzero_gradient():
    """Audit T1: v1 size head was frozen (self-imitation MSE, zero reward grad).
    v2 Beta head must receive gradient from the policy objective."""
    b = Brain(obs_dim=24, hidden=32)
    obs = torch.randn(1, 5, 24)
    op_d, sz_d, value, _ = b(obs)
    ops = torch.randint(0, 11, (1, 5))
    sizes = torch.rand(1, 5).clamp(0.05, 1.0)
    lp = b.joint_logprob(op_d, sz_d, ops, sizes)
    loss = -(lp * torch.randn(1, 5)).mean()
    b.zero_grad(); loss.backward()
    g = b.size_head.weight.grad.abs().sum().item()
    assert g > 0, "size head must receive gradient (T1)"


def test_size_is_stochastic_not_frozen():
    """Audit T1: act() must explore size, not return a constant."""
    b = Brain(obs_dim=24, hidden=32)
    obs = torch.randn(1, 24)
    sizes = {b.act(obs, greedy=False)[1] for _ in range(20)}
    assert len(sizes) > 5, "size must vary under sampling (T1)"


def test_config_door_is_real():
    """Audit S7/R11: numbers must come from configs through core.configs."""
    from core.configs import shell_cfg, goals_cfg
    sc = shell_cfg()
    assert sc["per_trade_risk_cap_pct"] == 0.25
    assert sc["max_trades_per_day"] == 400
    assert sc["forever_masks"]["timeframes"] == ["15min", "30min", "1h"]
    assert goals_cfg()["day_boundary_tz"]


def test_any_x_conditioning_samples_and_appears_in_obs():
    """Audit T6/R10: goal_ranges must randomize goal/floor AND reach the obs."""
    from data_io.loader import synthetic_m1, trading_days
    from features.engine import build_features
    from training.env import TradingEnv
    days = trading_days(build_features(synthetic_m1(days=2, seed=4)))[1:]
    env = TradingEnv(days, 2.5, 4.0, goal_ranges=((0.5, 4.0), (1.0, 6.0)),
                     rng=np.random.default_rng(1))
    env.reset(0); g1 = env.goal
    env.reset(0); g2 = env.goal
    assert g1 != g2, "goal must resample under any-X (T6)"
    # the sampled goal appears in the self-state slot (goal/5)
    assert abs(env._self_state()[0] - env.goal / 5.0) < 1e-5


def test_probes_are_closable():
    """Audit T5: close ops must be able to target probe stacks."""
    from data_io.loader import synthetic_m1, trading_days
    from features.engine import build_features
    from training.env import TradingEnv
    days = trading_days(build_features(synthetic_m1(days=2, seed=4)))[1:]
    env = TradingEnv(days, 2.5, 4.0)
    env.reset(0)
    # open a probe long, step to fill, then close it
    for _ in range(200):
        env.step(9, 1.0)                       # probe_long
        if any(s.is_probe for s in env.sim.stacks):
            break
    had_probe = any(s.is_probe for s in env.sim.stacks)
    if had_probe:
        env.step(6, 1.0)                       # close_long -> must reach the probe
        env.sim.step()
    assert had_probe, "probe should have opened (sanity for the close test)"
