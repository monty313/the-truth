# THE TARGET — the benchmark the bot is learning toward (saved 2026-07-19)

This is the **standard**. It is not a trained bot — it's the **oracle**: a
best-case trader that respects the −4% floor, run on your **real XAUUSD** data.
It proves there is genuine edge in your gold for the bot to reach.

## The proof (real XAUUSD, 21 trading days, Apr 27 – May 26 2026)
- **+4.63% per day on average**
- **0 floor breaches** — never once touched −4% in 21 days
- Beat your **+2.5% goal on 14 of 21 days (two out of three)**
- Reached **2× (+5%) on 9 of 21 days**
- A dumb baseline strategy bled slowly: **−0.14%/day**

Source files (kept in this repo): `artifacts/gauntlet/oracle_days.csv`,
`artifacts/gauntlet/evidence_report.json`, `artifacts/gauntlet/VERDICT.json`.

## Why it's the target
The oracle above used **fixed tiny position sizing** — it is a *lower bound*, not
the ceiling. The bot can size its own lots, so its real ceiling is **higher**.
Proof: under a profit-led reward the bot already made **+6.38% in one day** on a
day this oracle capped at ~2.6%.

## The rule for saved checkpoints
Every checkpoint we keep must be **more consistent** than the last — more days at
2×, then more days at goal, then more positive days, then steadier — and **never**
with a floor breach. `best_trading.pt` always holds the most consistent policy found.
