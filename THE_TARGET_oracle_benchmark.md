# THE TARGET — the benchmark the bot must beat (locked 2026-07-19)

This is the **standard**. It is not a trained bot — it is the **oracle**: a
best-case trader that respects the −4% floor, run on your **real XAUUSD** data.
It proves genuine edge exists, and it is the bar every saved checkpoint must beat.

---

## BAR 1 — the north star (real XAUUSD, 21 days, Apr 27 – May 26 2026)
The "there IS edge" proof, on a strong stretch of gold:
- **+4.63% per day on average**
- **0 floor breaches** in 21 days
- beat the **+2.5% goal on 14 of 21 days (two out of three)**
- reached **2× (+5%) on 9 of 21 days**
- a dumb baseline bled slowly: **−0.14%/day**

## BAR 2 — the fair bar for the DRILL WEEK (Jan 27 – Feb 2 2026)
This is the week the bot is actually training on, so this is the honest,
apples-to-apples bar to beat. Oracle best-case, per day:

| Day | Oracle best-case |
|-----|------------------|
| Jan 27 | +2.80% |
| Jan 28 | +2.82% |
| Jan 29 | +2.65% |
| Jan 30 | +2.42% |
| Feb 2  | +3.39% |

- average **+2.82%/day**, **0 breaches**, beat goal on **4 of 5 days**, 2× on **0 of 5**.

Note: this oracle uses **fixed tiny sizing — it's a LOWER bound.** The bot can
size its own lots, so it can BEAT this per day. Proof: it already made **+6.38%
on Jan 29**, where this oracle capped at +2.65%.

---

## THE MISSION (what "better than the benchmark" means)
On the drill week the bot must:
1. **Match the safety:** 0 floor breaches — non-negotiable.
2. **Beat the consistency:** all **5 of 5** days beat the goal (oracle got 4/5).
3. **Exceed it where it can:** push days to **2× (+5%)** by sizing up its edge.

## THE RULE FOR SAVED CHECKPOINTS
`best_trading.pt` holds the most consistent policy ever found and is **never**
replaced by a worse or breaching one. Rank order: more days ≥2×, then more days
≥goal, then more positive days, then steadier (lower spread). Seeded from the
existing best at every startup so it can only ever climb.
