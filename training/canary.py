"""Canary — proves the learning plumbing can learn AT ALL (Gauntlet #3).
5W+I: WHO Claude (Phase-4 gate). WHAT plants an obvious pattern (a synthetic
feature that predicts the next bars), trains the real Brain+PPO briefly, and
demands reward improvement. WHEN 2026-07-19. WHY if the canary can't learn a
gift-wrapped edge, boot camp would waste days. INTERCONNECTED: env, ppo.
"""
import numpy as np, pandas as pd, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data_io.loader import synthetic_m1, trading_days
from features.engine import build_features
from training.env import TradingEnv
from training.ppo import PPO


def planted_days(n_days=4, seed=3):
    """Synthetic days where set1::S1_buy_event is planted RIGHT BEFORE up-moves:
    the signal IS the answer key."""
    m1 = synthetic_m1(days=n_days + 1, seed=seed)
    rng = np.random.default_rng(seed)
    close = m1["close"].to_numpy().copy()
    plant = np.zeros(len(m1), np.float32)
    i = 200
    while i < len(m1) - 60:
        if rng.random() < 0.35:
            plant[i] = 1.0
            close[i + 2: i + 30] += np.linspace(0, 18.0, 28)  # a real move
            i += 60
        else:
            i += 15
    m1["close"] = close
    m1["high"] = np.maximum(m1["high"].to_numpy(), close + 0.4)
    m1["low"] = np.minimum(m1["low"].to_numpy(), close - 0.4)
    F = build_features(m1)
    F["set1::S1_buy"] = plant
    F["set1::S1_buy_event"] = plant
    return trading_days(F)[1:]


def run(updates=6):
    days = planted_days()
    env = TradingEnv(days, goal=2.5, floor=4.0)
    ppo = PPO(env, {"hidden": 64, "lr": 1e-3, "epochs": 3})
    first, last = None, None
    for u in range(updates):
        batch = [ppo.play_day(i % len(days)) for i in range(len(days))]
        mean_r = float(np.mean([b[5].sum() for b in batch]))
        mean_pnl = float(np.mean([b[6].get("pnl_pct", 0) for b in batch]))
        if first is None:
            first = mean_r
        last = mean_r
        ppo.update(batch, entropy_coef=0.02)
        print(f"canary update {u}: mean_day_reward={mean_r:.3f} mean_pnl={mean_pnl:.3f}%")
    improved = last > first
    print("CANARY", "PASS" if improved else "FAIL",
          f"(first {first:.3f} -> last {last:.3f})")
    return improved


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
