# GOAL — forever (auto-loaded every Grok session)

**Source of truth:** repo root `GOAL.md`  
**Full agent law:** repo root `AGENTS.md`

## What we are building

**One bot that solves for any target % and risk % Monty types in — without having to retrain.**

| Input (runtime) | Bot must |
|-----------------|----------|
| **Target %** | Hit it (as many days as possible) |
| **Risk %** | Never breach it |

Change the numbers → **same `.pt`** → score again. **No retrain** just to switch pairs.  
Not locked to only 3.0 / 3.5.

```text
python scripts/prove_it.py <brain> <target> <risk>
```

## How we know we won (at those numbers)

| Word | Rule |
|------|------|
| **Clear %** | Climb (hit **that** target, no floor hit) |
| **Breach %** | Stay **0%** on **that** floor |
| **Streak** | Climb |

If a change does not move clear % or protect breach 0% at the pair under test → **skip it**.

Yardstick pair is often **3.0 / 3.5** for daily comparison — **generalizing to any pair is still the mission.**

## Never confuse the goal with

- A bot that only “works” at one frozen target/risk  
- **Needing a retrain every time** Monty changes target or risk  
- Building UIs / parallel frameworks instead of clear% / breach0 on `prove_it`  
- Calling days “impossible” (lid is off — `AGENTS.md` §0)

## Champion

`models/00_CHAMPION.md` + scoreboard in `GOAL.md`.
