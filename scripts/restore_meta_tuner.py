#!/usr/bin/env python3
"""One-shot restore of training/meta_tuner.py with Phase-4 unlocks.
Run from repo root: python scripts/restore_meta_tuner.py
"""
from __future__ import annotations
import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/monty313/the-truth/main/training/meta_tuner.py"
OUT = Path("training/meta_tuner.py")

def main():
    print("Fetching base meta_tuner from main...")
    text = urllib.request.urlopen(URL, timeout=60).read().decode()
    old_bounds = '''    "w_no_drawdown_close": ( 0.0,   1.0),
    "lr":                  (1e-5,  3e-3),
    "entropy_coef":        ( 0.0,   0.1),
}'''
    new_bounds = '''    "w_no_drawdown_close": ( 0.0,   1.0),
    "w_pullback_with_htf": ( 0.0,   1.0),  # bread-and-butter; unlocked 2026-07-24
    "lr":                  (1e-5,  3e-3),
    "entropy_coef":        ( 0.0,   0.1),
}'''
    if old_bounds not in text:
        raise SystemExit("bounds block not found — main file changed shape")
    text = text.replace(old_bounds, new_bounds, 1)
    old_fb = '''_FALLBACK = {"w_death_penalty": -10.0, "w_did_nothing": -6.0, "w_idleness_hunger": -0.002,
             "w_day_goal_hit": 2.0, "w_streak_per_day": 0.15, "w_trade_consistency": 0.10,
             "w_net_profit": 6.0, "w_no_drawdown_close": 0.02, "lr": 3e-4, "entropy_coef": 0.01}'''
    new_fb = '''_FALLBACK = {"w_death_penalty": -10.0, "w_did_nothing": -6.0, "w_idleness_hunger": -0.002,
             "w_day_goal_hit": 2.0, "w_streak_per_day": 0.15, "w_trade_consistency": 0.10,
             "w_net_profit": 6.0, "w_no_drawdown_close": 0.02, "w_pullback_with_htf": 0.02,
             "lr": 3e-4, "entropy_coef": 0.01}'''
    if old_fb not in text:
        raise SystemExit("fallback block not found")
    text = text.replace(old_fb, new_fb, 1)
    needle = "CHANGE LOG (newest first — APPEND on every edit with date + WHY; keep this line):\n"
    insert = needle + "- 2026-07-24  unlock w_pullback_with_htf in BOUNDS — WHY: bread-and-butter must be tunable via gated self-tuner (Phase 4 / Standing Laws).\n"
    if "unlock w_pullback_with_htf" not in text:
        text = text.replace(needle, insert, 1)
    old_ws = '''        for name in ("lift_best", "PROVEN_LIFT_2026-07-20", "gpu_best",
                     "PROVEN_2x_2026-07-19", "best_trading"):'''
    new_ws = '''        for name in ("PROVEN_SPRINT_row04_clear24_2026-07-20", "lift_best",
                     "PROVEN_LIFT_2026-07-20", "gpu_best",
                     "PROVEN_2x_2026-07-19", "best_trading"):'''
    if old_ws in text:
        text = text.replace(old_ws, new_ws, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)
    assert "w_pullback_with_htf" in OUT.read_text()
    print(f"Wrote {OUT} ({OUT.stat().st_size} bytes) with w_pullback_with_htf unlocked.")

if __name__ == "__main__":
    main()
