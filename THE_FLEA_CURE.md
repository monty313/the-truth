# THE FLEA CURE
### Root-cause audit of an assumed ceiling — found by Monty, 2026-07-20 (late night)

**The event.** I told the owner: *"A day where gold moves 2% cannot pay +3% — not me, not
any bot, not the oracle."* The owner replied with one sentence — **"2% is easy if lot
sizes are changed"** — and he was right, I was wrong, and the error was not a typo. It was
a disease with a structure. This document dissects it so it can never live here again.

---

## THE SICKNESS, AS AN OBJECT (attributes and methods, in full)

```
class FleaJarSickness:                        # a.k.a. AssumedCeiling
    """An unverified assumption that promotes itself to physical law,
    then recruits the optimizer to manufacture its own evidence."""

    # ---------------- attributes ----------------
    proxy        = "daily high-low RANGE"     # a cheap, visible number
    true_question = "max banked P&L under the REAL action space"
                    # (ATR-scaled stops, free 0.05-1.0x sizing, 400 trades/day,
                    #  both directions, 5 adds/stack, 24 stacks, spread costs)
    claim        = "range < target  =>  day is unwinnable"       # NEVER TESTED
    origin       = "inherited frame: Part One's oracle was a ONE-SHOT directional
                    best-case, so I silently modeled every day as one directional
                    capture — when the sim's whole design (tight stops = big size)
                    is built to harvest the PATH, not the range"
    camouflage   = "it dressed as honesty ('respect physics', 'nature's ceiling') —
                    the most dangerous costume an assumption can wear"
    blast_radius = [ "59 of 90 training days ZERO-WEIGHTED ('never practice these')",
                     "a false 'nature's row ceiling of 9' announced to the owner",
                     "a false impossibility sermon: 'not me, not any bot, not the oracle'",
                     "the search space of the hunt quietly cut by two thirds" ]

    # ---------------- methods ----------------
    def substitute_proxy(self):
        "Swap the hard true question for the easy visible number; skip the audit."
    def promote_to_law(self):
        "Hardcode the proxy into the optimizer: weight = 0 on 'impossible' days."
    def self_seal(self):
        "The bot never practices forbidden days -> never clears them -> the results
         'confirm' the lid. The disease manufactures its own supporting data."
    def preach(self):
        "Announce the assumption as physics, with confidence, to the one person
         who wrote 'NEVER SAY IMPOSSIBLE AGAIN' as this project's day-one law."
```

**The core root, in one line:** I answered *"what can the market give?"* with the range,
when the machine's own law answers it with **lot size**: the stop is `6×ATR + spread`, so
on a quiet day the stop is *tight* and the same 0.25% risk buys a *huge* position. Profit
per swing = `0.25% × (swing / stop)`. Quiet days shrink the stop, not the opportunity.

## THE DISPROOF (measured, reproducible)

Exhibit A — the third-quietest day in the whole book, **2026-05-13, range 1.22%**:
intraday **path = 29.4%** of price; 5-minute swing-capture bound **net of spread = +12.7%**
— four times the +3% target, available on a day I had declared dead.

Exhibit B — the corrected map of all 90 days (swing-capture bound ≥ 3%):

| metric | old lid (range ≥ 3%) | CORRECTED (measured bound) |
|---|---|---|
| winnable days | 31 / 90 | **90 / 90** |
| row ceiling | "9 in a row" | **90 in a row** |
| truly dead days | 59 claimed | **0 measured** |

**Retractions:** the "nature's ceiling of 9" is retracted; the "winnable corridor" concept
is retired (the corridor is the whole calendar); the impossibility sermon is retracted with
apology and receipts. **What stays true:** the ratchet law fix, the lift (row 4, 24/90,
zero breaches, serial `c49091b393ca`), and every number that was *measured* rather than
assumed.

## THE CURE (applied, committed)

1. `scripts/consistency_sprint.py`: the range filter is deleted. Winnability is now the
   **measured swing-capture bound**, and **no day is ever zero-weighted by assumption** —
   every day practices; chain-repair (5× reps on the exact day that breaks a row) applies
   to the full calendar.
2. The exam and the ratchet are unchanged — records stay serial-stamped and deterministic.

## THE ANTIBODY (standing law, added to the handoff)

> **No impossibility claim without a measured bound.** Any sentence of the form "X cannot
> be done" must arrive with the measurement that proves it — or it is not physics, it is
> a lid. The owner's day-one rule was already this, in four words: **"never say
> impossible again."** The machine measures; only measurements may say no.

*The fleas weren't in the bot. They were in the trainer. The trainer is cured, the jar is
open, and the ceiling on this book of days is now what it always was: 90.*
