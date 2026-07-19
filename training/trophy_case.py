"""Trophy case — record-win evidence ladder (critic-round decision).
5W+I: WHO Monty's idea, encoded by Claude. WHAT when a closed win beats the
run's record, save market state before entry + at close to a growing JSONL —
(a) evidence for Monty, (b) observations for later training. WHEN 2026-07-19.
WHY 'the RL model produces evidence'. INTERCONNECTED: rewards (record bonus,
won-day gated), artifacts/trophy_case.jsonl, HUD (displays ladder).
"""
import json, os, time
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = os.path.join(ROOT, "artifacts", "trophy_case.jsonl")

def record(trade: dict, obs_before, obs_at_close, run_id: str, day: str):
    rec = {"ts": time.time(), "run_id": run_id, "day": day,
           "pnl_pct": trade["pnl_pct"], "bars": trade["bars"],
           "adds": trade["adds"], "why": trade["why"],
           "obs_before": [round(float(x), 4) for x in obs_before[:64]],
           "obs_at_close": [round(float(x), 4) for x in obs_at_close[:64]]}
    with open(PATH, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec
