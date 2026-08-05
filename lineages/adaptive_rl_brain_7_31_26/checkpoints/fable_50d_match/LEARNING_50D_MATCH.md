# LEARNING — Fable 50-day Mark match loop

**Goal:** policy day outcomes match Mark full-day plans on frozen 50 calendar days.  
**Window:** first 50 loadable days `2026-01-20` → `2026-03-30`, seed=42, soft_bias=false.  
**Code:** `fable_50d_mark_match_loop.py`

## Frozen baseline (measured)

| Meter | Value |
|-------|------:|
| mark_clear | **50**/50 |
| policy_clear | **27**/50 |
| same_outcome | **27**/50 |
| n_breach | **0** |
| MARK_WOULD_TAKE | **23** |
| NO_OPPORTUNITY | **0** |

Interpretation: Mark soul plans clear **every** day in window. All 23 policy misses are learnable (not dead markets).

## Loop recipe (Fable)

1. Miss-class from Mark plan vs policy award  
2. Mark full-day labels + DAgger on MWT days  
3. Reward sample weights (thrash/misread dials)  
4. BC + KL anchor to best embryo  
5. Re-score same recipe → **keep only if not worse** (clear/same not down, breach 0)

## Cycle log (live)

| Cycle | same | policy | mwt | breach | decision |
|------:|-----:|-------:|----:|-------:|----------|
| 0 baseline | 27 | 27 | 23 | 0 | freeze |
| 1 | **30** | **30** | **20** | 0 | **KEEP** (dir_match 0.86) |
| 2 | 28 | 28 | 21 | **1** | **REJECT** (breach; hold-rate too low) |
| 3 | stalled on label collect | | | | **KILLED** |
| sprint | (entry-focus from best=30) | | | | ran |
| one_day #1 | 28 | 28 | 22 | 0 | **REJECT** focus 2026-03-27 |
| one_day #2 | **33** | **33** | **17** | 0 | **KEEP** focus 2026-02-25 |
| one_day #3 | 31 | 31 | 19 | 0 | **REJECT** focus 2026-02-20 |
| one_day #4 | 33 | 33 | 17 | 0 | **REJECT** focus 2026-02-05 (dir high, hold low, no same gain) |
| one_day #5 | 33 | 33 | 17 | 0 | **REJECT** focus 2026-03-11 |
| one_day #6 | (running…) | | | | focus 2026-01-29 |

**Best so far:** same_outcome **33**/50 · policy_clear **33** · mwt **17** · breach **0**  
(↑ from baseline 27; Mark still 50/50)

### Fable5-as-MARK-HERE (KAG)
Army agent `fable5_mark_here_kag` now indexes army + the-truth + pt5 and writes to first Mark:
`FABLE5_MARK_HERE_BRIEF__latest.md` · ARMY `outputs/army/FABLE5_TO_FIRST_MARK__consistency.md`

### Process note
Cycle 3 of full loop froze on DAgger label collect (CPU flat). Switched to `fable_50d_sprint.py`: entry-focused BC on remaining MWT + Mark plan dirs on award days, keep/reject, dual final score.

### Lessons so far

1. **Directional oversample is mandatory.** First attempt (HOLD-heavy) → dir_match ~0.09, collapse. After 6× dir copies + DAgger filter → dir_match ~0.86 and +3 days.
2. **Too little HOLD → breach.** Cycle 2 pred_hold_rate ~0.05 → breach 1. Keep/reject correctly restored best.
3. **Need balance:** enough Mark entries to convert MWT days, enough HOLD to not thrash into floor.
4. Gate `flat_undefined` fix remains in place (not the remaining gap).

## Target

`same_outcome == 50` or policy clears all Mark-clear days (all 50) with breach 0, without `policy_clear` below baseline 27.
