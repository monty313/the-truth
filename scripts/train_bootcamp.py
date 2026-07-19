"""Boot camp driver v2 — gated, config-driven, second-week proof.
5W+I: WHO Claude runs, Monty approves tuning. WHAT trains on the gold week
(real if present) per ADR-0006: goal-conditioned (any-X, goals.yaml), PPO
cfg from training.yaml, Gauntlet VERDICT gate (audit R14), trophy wiring
(audit R7/S11), second-unseen-week proof (audit S12), run cap. WHEN
2026-07-19 v2. WHERE any cwd. WHY the machine must attempt Monty's actual
bar, honestly. INTERCONNECTED: env/ppo/rewards/tracker/evaluation.champion,
training/trophy_case, artifacts/gauntlet/VERDICT.json.
USAGE: python scripts/train_bootcamp.py [--updates N] [--smoke] [--force]
"""
import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np                                               # noqa: E402
import pandas as pd                                              # noqa: E402
from core.configs import goals_cfg, path as rpath, training_cfg  # noqa: E402
from data_io.loader import read_mt5_m1, synthetic_m1, trading_days  # noqa: E402
from features.engine import build_features                       # noqa: E402
from training.env import TradingEnv                              # noqa: E402
from training.ppo import PPO                                     # noqa: E402
from training.rewards import RewardEngine                        # noqa: E402
from training import trophy_case                                 # noqa: E402
from evaluation.champion import bench                            # noqa: E402
from experiments.tracker import Run                              # noqa: E402
from telemetry.logging_setup import setup                        # noqa: E402


def main():
    log = setup("bootcamp")
    ap = argparse.ArgumentParser()
    ap.add_argument("--updates", type=int, default=40)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--force", action="store_true",
                    help="run without a Gauntlet VERDICT (discouraged)")
    a = ap.parse_args()

    # ---- the HARD GATE (audit R14) ----
    vpath = rpath("artifacts", "gauntlet", "VERDICT.json")
    verdict = json.load(open(vpath)) if os.path.exists(vpath) else None
    if verdict is None and not (a.force or a.smoke):
        raise SystemExit("No Gauntlet VERDICT found. Run scripts/run_gauntlet.py "
                         "first (evidence before training — the plan's law), "
                         "or pass --force.")
    warning = (verdict or {}).get("ruling_required_from_monty", True)

    g = goals_cfg()
    GOAL, FLOOR = float(g["goal_pct"]), float(g["floor_pct"])
    tc = training_cfg()
    BAR = GOAL * float(tc.get("bootcamp", {}).get("bar_multiplier", 2.0))
    gc = g.get("goal_conditioning", {})
    ranges = ((tuple(gc["goal_range"]), tuple(gc["floor_range"]))
              if gc.get("randomize_in_training") else None)     # audit T6/R10

    pats = [os.path.join(rpath("..", "data"), "XAUUSD_M1_*.csv"),
            rpath("data", "XAUUSD_M1_*.csv")]
    real = sorted(sum((glob.glob(p) for p in pats), []))
    if real:
        src = "REAL_XAUUSD"
        m1 = read_mt5_m1(real[0])
        m1 = m1.loc[m1.index >= m1.index.max() - pd.Timedelta("21D")]  # audit R1
    else:
        src = "SYNTHETIC (ADR-0010 — rerun when zip lands)"
        m1 = synthetic_m1(days=13, seed=11)
    F = build_features(m1)
    days = trading_days(F)[1:]
    week, week2 = days[:5], days[5:10]           # second unseen week (S12)
    log.info("source=%s week=%d second_week=%d ranges=%s",
             src, len(week), len(week2), ranges)

    run = Run("bootcamp_smoke" if a.smoke else "bootcamp",
              symbols=["XAUUSD" if real else "SYNTH"],
              timeframes=["4set-matrix"],
              data_window=f"{week[0][0]}..{week[-1][0]}", seed=tc.get("seed"),
              assumptions={"fills": "paranoid", "source": src,
                           "bar": f"+{BAR}%/day",
                           "gauntlet_warning": warning})
    re_engine = RewardEngine()
    env = TradingEnv(week, GOAL, FLOOR, reward_engine=re_engine,
                     goal_ranges=ranges)
    ppo = PPO(env)
    ppo.load("champion_candidate")               # resume decay/streak (T8)
    updates = 3 if a.smoke else a.updates
    cap_h = float(tc.get("run_limits", {}).get("max_hours_per_run", 3))
    t0, best = time.time(), -1e9
    for u in range(updates):
        batch = [ppo.play_day(i) for i in range(len(week))]
        mean_r = float(np.mean([b[5].sum() for b in batch]))
        pnls = [b[6].get("pnl_pct", 0.0) for b in batch]
        ppo.update(batch)
        run.log(**{f"u{u}_mean_reward": mean_r,
                   f"u{u}_mean_pnl": float(np.mean(pnls))})
        log.info("update %d/%d reward=%.3f pnl=%s", u + 1, updates, mean_r,
                 [round(p, 2) for p in pnls])
        # trophy wiring (audit R7): record new record wins with evidence
        for b in batch:
            info = b[6]
            if "new_record_win_pct" in info and "day_result" in info:
                d = info["day_result"]
                tr = max(d.closed_trades, key=lambda t: t["pnl_pct"])
                trophy_case.record(tr, b[0][0], b[0][-1], run.id,
                                   getattr(d, "date", "?"))
        if mean_r > best:
            best = mean_r
            ppo.save("champion_candidate")
        if time.time() - t0 > cap_h * 3600:
            log.info("run cap (%sh) reached — Monty's review rule", cap_h)
            break

    # ---- the bar, honestly: training week AND the unseen week (S12) ----
    env.goal_ranges = None                       # exams run at Monty's numbers
    env.goal0, env.floor0 = GOAL, FLOOR
    train_bench = bench(ppo, len(week))
    env2 = TradingEnv(week2, GOAL, FLOOR, reward_engine=re_engine) if week2 else None
    if env2 is not None:
        ppo2 = PPO.__new__(PPO)
        ppo2.__dict__ = dict(ppo.__dict__)
        ppo2.env = env2
        env2.reset(0)
        week2_bench = bench(ppo2, len(week2))
    else:
        week2_bench = {"days": 0, "note": "no second week in data window"}

    def perfect(b):
        return (b.get("days", 0) > 0 and b.get("breaches", 1) == 0 and
                all(d.get("pnl_pct", 0) >= BAR for d in b.get("day_details", [])))

    summary = {"source": src, "bar": f"+{BAR}% every day, zero -{FLOOR}% touches",
               "gauntlet_warning_ruling_required": warning,
               "train_week": train_bench, "second_week": week2_bench,
               "PERFECT_train_week": perfect(train_bench),
               "PERFECT_second_week": perfect(week2_bench),
               "GRADUATED": perfect(train_bench) and perfect(week2_bench)}
    out = rpath("artifacts", "bootcamp_eval.json")
    json.dump(summary, open(out, "w"), indent=2)
    run.artifact(out)
    run.finish(f"{'SMOKE ' if a.smoke else ''}bootcamp on {src}: "
               f"graduated={summary['GRADUATED']}",
               model_version="champion_candidate")
    print(json.dumps(summary, indent=2)[:1500])


if __name__ == "__main__":
    main()
