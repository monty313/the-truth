"""Run the bridge (dry-run by default) — one-physics live loop.
5W+I: WHO Claude for Monty. WHAT loads features + the frozen champion, runs
the Bridge day-by-day writing HUD state; dry-run paper-trades anywhere, demo/
live needs Windows+MT5 and Monty's gates. WHEN 2026-07-19 v2 (audit R5: the
loop didn't exist). WHERE any cwd (ROOT-anchored). WHY the champion must
actually meet a price stream through the Shell. INTERCONNECTED: bridge, env,
inference/loader, features/engine, data_io/loader, HUD.
USAGE: python scripts/run_live.py [--days N]
"""
import argparse, glob, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import pandas as pd                                              # noqa: E402
from core.configs import goals_cfg, load as cfg, path as rpath   # noqa: E402
from data_io.loader import read_mt5_m1, synthetic_m1, trading_days  # noqa: E402
from features.engine import build_features                       # noqa: E402
from training.env import TradingEnv                              # noqa: E402
from execution_bridge.mt5_bridge import Bridge                   # noqa: E402
from inference.loader import load_brain                          # noqa: E402
from telemetry.logging_setup import setup                        # noqa: E402


def main():
    log = setup("live")
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1)
    a = ap.parse_args()

    ec = cfg("execution")
    g = goals_cfg()
    pats = [os.path.join(rpath("..", "data"), "XAUUSD_M1_*.csv"),
            rpath("data", "XAUUSD_M1_*.csv")]
    real = sorted(sum((glob.glob(p) for p in pats), []))
    if real:
        m1 = read_mt5_m1(real[0])
        m1 = m1.loc[m1.index >= m1.index.max() - pd.Timedelta("5D")]
    else:
        m1 = synthetic_m1(days=a.days + 2, seed=21)
    days = trading_days(build_features(m1))[1:1 + a.days]

    env = TradingEnv(days, float(g["goal_pct"]), float(g["floor_pct"]))
    brain, meta = load_brain("champion_candidate")
    log.info("bridge mode=%s | champion=%s | days=%d",
             ec.get("mode"), "loaded" if brain else "NONE (observe-only)", len(days))
    bridge = Bridge(ec, env, brain)
    if not bridge.connect():
        raise SystemExit("MT5 connect failed — check configs/execution.yaml")
    for i in range(len(days)):
        info = bridge.run_day(i, throttle=float(ec.get("dry_run_bars_per_sec", 0)) and
                              1.0 / ec["dry_run_bars_per_sec"] or 0.0)
        log.info("day %s done: pnl=%.3f%% goal_hit=%s breached=%s trades=%s",
                 days[i][0], info.get("pnl_pct", 0), info.get("goal_hit"),
                 info.get("breached"), info.get("trades"))
    log.info("HUD state at artifacts/hud_state.json — start scripts/run_hud.py to view")


if __name__ == "__main__":
    main()
