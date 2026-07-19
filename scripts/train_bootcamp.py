"""Boot camp driver — Phase 5. Trains on ONE symbol/week until the bar.
5W+I: WHO Claude runs, Monty approves tuning between runs. WHAT loads gold
week (real if present, synthetic else — labeled), trains PPO with run cap,
evaluates the bar (+2x goal EVERY day, zero breaches), reports. WHEN
2026-07-19. WHY ADR-0006 stage 0. INTERCONNECTED: env/ppo/rewards/tracker.
USAGE: python scripts/train_bootcamp.py [--updates N] [--smoke]
"""
import sys, os, json, argparse, glob, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from data_io.loader import read_mt5_m1, synthetic_m1, trading_days
from features.engine import build_features
from training.env import TradingEnv
from training.ppo import PPO
from training.rewards import RewardEngine
from experiments.tracker import Run
from telemetry.logging_setup import setup

log = setup("bootcamp")
ap = argparse.ArgumentParser()
ap.add_argument("--updates", type=int, default=40)
ap.add_argument("--smoke", action="store_true", help="short plumbing proof")
a = ap.parse_args()

GOAL, FLOOR, BAR = 2.5, 4.0, 5.0
real = sorted(glob.glob("../data/XAUUSD_M1_*.csv") + glob.glob("data/XAUUSD_M1_*.csv"))
if real:
    src = "REAL_XAUUSD"; m1 = read_mt5_m1(real[0]).last("21D")
else:
    src = "SYNTHETIC (ADR-0010 — rerun when zip lands)"; m1 = synthetic_m1(days=9, seed=11)
F = build_features(m1)
days = trading_days(F)[1:]
week = days[:5]                                  # the training week
log.info("source=%s week_days=%d", src, len(week))

run = Run("bootcamp_smoke" if a.smoke else "bootcamp",
          symbols=["XAUUSD" if real else "SYNTH"], timeframes=["4set-matrix"],
          data_window=f"{week[0][0]}..{week[-1][0]}", seed=20260718,
          assumptions={"fills": "paranoid", "source": src, "bar": f"+{BAR}%/day"})
re_engine = RewardEngine()
env = TradingEnv(week, GOAL, FLOOR, reward_engine=re_engine)
ppo = PPO(env, {"hidden": 128, "lr": 3e-4, "epochs": 2 if a.smoke else 4})
updates = 3 if a.smoke else a.updates
t0 = time.time()
best = -1e9
for u in range(updates):
    batch = [ppo.play_day(i) for i in range(len(week))]
    mean_r = float(np.mean([b[5].sum() for b in batch]))
    pnls = [b[6].get("pnl_pct", 0.0) for b in batch]
    goals = sum(1 for b in batch if b[6].get("goal_hit"))
    breaches = sum(1 for b in batch if b[6].get("breached"))
    ent = max(0.05 * (1 - u / max(1, updates)), 0.01)   # explorative week-1 decay
    stats = ppo.update(batch, entropy_coef=ent)
    run.log(**{f"u{u}_mean_reward": mean_r, f"u{u}_mean_pnl": float(np.mean(pnls)),
               f"u{u}_goal_days": goals, f"u{u}_breaches": breaches})
    log.info("update %d/%d reward=%.3f pnl=%s goals=%d/%d breaches=%d ent=%.3f",
             u + 1, updates, mean_r, [round(p, 2) for p in pnls], goals,
             len(week), breaches, ent)
    if mean_r > best:
        best = mean_r
        ppo.save("champion_candidate")
    if time.time() - t0 > 3 * 3600:
        log.info("run cap (3h) reached — Monty's review rule"); break

# --- greedy evaluation vs THE BAR ---
evals = [ppo.play_day(i, greedy=True)[6] for i in range(len(week))]
perfect = all(e.get("pnl_pct", 0) >= BAR and not e.get("breached") for e in evals)
summary = {"source": src, "bar": f"+{BAR}% every day, zero -{FLOOR}% touches",
           "eval_days": [{k: round(v, 3) if isinstance(v, float) else v
                          for k, v in e.items()} for e in evals],
           "PERFECT": perfect}
json.dump(summary, open("artifacts/bootcamp_eval.json", "w"), indent=2)
run.artifact("artifacts/bootcamp_eval.json")
run.finish(f"{'SMOKE ' if a.smoke else ''}bootcamp on {src}: perfect={perfect}",
           model_version="champion_candidate")
print(json.dumps(summary, indent=2)[:1200])
