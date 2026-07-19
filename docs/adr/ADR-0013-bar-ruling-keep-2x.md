# ADR-0013 — The 2× boot-camp bar: Monty rules KEEP

Status: Accepted (Monty, 2026-07-19)
Relates to: ADR-0001 (goal/floor are Monty's X-inputs), ADR-0006 (boot camp),
ADR-0010 (synthetic placeholder, now replaced by real data), Gauntlet VERDICT
gate (audit R14).

## Context
The boot-camp bar is "**+2× the goal every single day, zero floor touches**"
(GOAL 2.5% → BAR 5.0%). The Gauntlet is the feasibility gate that must pass
before training is trusted. It runs an **oracle** (perfect-hindsight upper bound,
still floor-respecting) and a naive **baseline**, then asks: can even the ceiling
clear the bar *every* day?

First **real-data** Gauntlet — REAL_XAUUSD, 2026-04-27 .. 05-26, 21 trading days:

- oracle: **+4.63%/day avg**, min +2.08%, max +9.62%, **0 floor breaches**
- oracle cleared the +5% (2×) bar on **only 9 of 21 days**
- baseline (naive): −0.14%/day
- data audit: 28,951 M1 rows, 0 bad candles, spread median 4 pts

So "2× every day" is **not physically present** in this 30-day window — 12 days
lacked the range. The Gauntlet set `ruling_required_from_monty = true`: the bar
bends only by Monty's hand.

## Decision
**Monty ruled KEEP.** The 2×-goal-every-day bar stands as the aspirational
north-star. The bot trains toward it; on low-range days it cannot reach it and
every report **honestly flags** those days. The ruling is recorded
machine-readably in `goals.yaml:bar_ruling` (`decision: keep`,
`acknowledged_warning: true`); boot camp reads it and stamps "Monty ruled KEEP"
instead of an open question.

## Consequences
- `bar_multiplier` stays 2.0 (training.yaml). **No change to training behavior** —
  only the report stamp changes from "ruling required" to "Monty ruled KEEP".
- GRADUATION (a perfect training week AND an unseen week, both at the 2× bar)
  remains the hardest possible target by design. Monty accepts it may rarely be
  met on real gold; "how close to the ceiling" is the real signal.
- **Open follow-up (not yet run):** 30 days is one market regime. A multi-window
  Gauntlet across 2020–2026 would show whether 2×/day is reachable in
  higher-volatility regimes. Recorded here so it is not lost.

# NEXT EDITOR: if Monty re-rules the bar, append a NEW ADR that supersedes this
# one and update goals.yaml:bar_ruling — never change the ruling silently.
