# LEARNING â€” Fable 50-day Mark match loop

**Goal:** policy day outcomes match Mark full-day plans on frozen 50 calendar days.  
**Window:** first 50 loadable days `2026-01-20` â†’ `2026-03-30`, seed=42, soft_bias=false.  
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
5. Re-score same recipe â†’ **keep only if not worse** (clear/same not down, breach 0)

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
| one_day #6 | 33 | 33 | 17 | 0 | **REJECT** focus 2026-01-29 (prior run; incomplete) |
| squad-od #1 | 29 | 29 | 21 | 0 | **REJECT** focus 2026-01-27 (pack drop; keep/reject OK) |
| squad-od #2 | 33 | 33 | 17 | 0 | **REJECT** focus 2026-02-13 (no convert; pack held) |
| squad-od #3 | 31 | 31 | 19 | 0 | **REJECT** focus 2026-02-20 (converted day, pack 33â†’31) |
| squad-od #4 | (runningâ€¦) | | | | focus 2026-02-05 Â· **PACK-repair** code armed |

**Best so far:** same_outcome **33**/50 Â· policy_clear **33** Â· mwt **17** Â· breach **0**  
(â†‘ from baseline 27; Mark still 50/50)

### Fable5-as-MARK-HERE (KAG)
Army agent `fable5_mark_here_kag` now indexes army + the-truth + pt5 and writes to first Mark:
`FABLE5_MARK_HERE_BRIEF__latest.md` Â· ARMY `outputs/army/FABLE5_TO_FIRST_MARK__consistency.md`

### Process note
Cycle 3 of full loop froze on DAgger label collect (CPU flat). Switched to `fable_50d_sprint.py`: entry-focused BC on remaining MWT + Mark plan dirs on award days, keep/reject, dual final score.

### Lessons so far

1. **Directional oversample is mandatory.** First attempt (HOLD-heavy) â†’ dir_match ~0.09, collapse. After 6Ã— dir copies + DAgger filter â†’ dir_match ~0.86 and +3 days.
2. **Too little HOLD â†’ breach.** Cycle 2 pred_hold_rate ~0.05 â†’ breach 1. Keep/reject correctly restored best.
3. **Need balance:** enough Mark entries to convert MWT days, enough HOLD to not thrash into floor.
4. Gate `flat_undefined` fix remains in place (not the remaining gap).

## Target

`same_outcome == 50` or policy clears all Mark-clear days (all 50) with breach 0, without `policy_clear` below baseline 27.

| spine-shadow 0 | 33 | 33 | 17 | 0 | **BASELINE** (false_fire) |

| spine-shadow 1 | 33 | 33 | 17 | 0 | **REJECT** (wrong_size_or_timing) |

### Pack pipeline one-day (live, additive)
| Cycle | same | policy | mwt | breach | decision |
|------:|-----:|-------:|----:|-------:|----------|
| pack start | 33 | 33 | 17 | 0 | resume embryo |
| pack-od #1 2026-01-27 | 29 | 29 | 21 | 0 | **REJECT** (pack thrash; dir_match high) |
| pack-od #2 2026-02-13 | **35** | **35** | **15** | 0 | **KEEP** (collateral pack rise) |
| pack-od #3 2026-02-20 | 28 | 28 | 22 | 0 | convert+pack-drop; repair… (hold_rate 0.25 disease) |

**Best so far:** same **35**/50 · outside-box pathology triage + HOLD-floor armed for next chain.

| spine-shadow 2 | 33 | 33 | 17 | 0 | **REJECT** (wrong_size_or_timing) |

### Durable recreate (ARMY + KAG) — do not lose the mind

- **Playbook:** `ARMY/kag_mark_doctrine/PLAYBOOK_50D_MARK_MATCH.md`
- **Machine recipe:** `01_SYSTEM/outputs/army/teachers/pack_50d_bridge/RECREATE_50D__latest.json`
- **Snapshot 35:** `01_SYSTEM/outputs/army/teachers/pack_50d_bridge/backups/KEEP35__20260806/`
- **Pattern:** `pack_safe_judgment_v1` in army PATTERN_MEMORY / MEMORY.jsonl
- **Working:** dir oversample · HOLD floor · KEEP/REJECT · convert?KEEP · collateral KEEP · pathology triage
- **Feel:** pack still allows fire — not only this day.


| pack-od #4 2026-02-05 | 31 | 31 | 18 | **1** | **REJECT** (hold_rate 0.20; breach; dir_match 0.97 useless alone) |
| pack-od #5 2026-03-11 | … | … | … | … | running · best still 35 |

**Lesson R4:** high dir_match without HOLD floor ? **breach**. KEEP/REJECT saved the 35 embryo.

| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** (wrong_size_or_timing) |

| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** (wrong_size_or_timing) |

| spine-shadow 1 | 34 | 34 | 16 | 0 | **REJECT** (wrong_size_or_timing) |

| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** (wrong_size_or_timing) |

| spine-shadow 1 | 27 | 27 | 23 | 0 | **REJECT** (wrong_size_or_timing) |

| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** (wrong_size_or_timing) |

| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** (unknown_no_spine) |

| spine-shadow 1 | 31 | 31 | 19 | 0 | **REJECT** (unknown_no_spine) |

| spine-shadow 2 | 30 | 30 | 20 | 0 | **REJECT** (unknown_no_spine) |

| spine-shadow 3 | 31 | 31 | 19 | 0 | **REJECT** (unknown_no_spine) |

| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** (unknown_no_spine) |

| spine-shadow 1 | 31 | 31 | 19 | 0 | **REJECT** (unknown_no_spine) |

| spine-shadow 2 | 30 | 30 | 20 | 0 | **REJECT** (unknown_no_spine) |
