# Diagnosis — why Mark doesn’t win all 10 (with full chart + pt5)

**When:** 2026-08-05T00:45:05.119870+00:00

## Question
If Mark **sees the same 10 charts** and uses **pt5 principles**, he should know how to win. What’s the issue when he doesn’t?

## Issue classes

| Code | Meaning |
|------|---------|
| **A_physics_impossible_under_shell** | No single-entry plan banks target without hitting floor. Day may need multi-leg or be too tight for target/risk. |
| **B_principles_block_all_winning_entries** | A win exists only by trading **against** live HTF force (breaks pt5.1). |
| **C_teacher_missed_valid_principle_win** | At least one **force-aligned** entry would have won — codified Mark didn’t take it (timing/selectivity bug). |
| **D_teacher_already_won** | Baseline doctrine already cleared. |

## Tallies

| Issue | Count |
|-------|------:|
| `A_physics_impossible_under_shell` | **1** |
| `B_principles_block_all_winning_entries` | **1** |
| `C_teacher_missed_valid_principle_win` | **1** |
| `D_teacher_already_won` | **7** |
| **Total days** | **10** |

## Day-by-day

| Date | T/R | Base clear | Physics wins | Principle wins | Issue |
|------|----:|:----------:|-------------:|---------------:|-------|
| 2026-03-17 | 3.0/3.5 | n | 0 | 0 | `A_physics_impossible_under_shell` |
| 2026-03-18 | 2.0/3.0 | Y | 7 | 4 | `D_teacher_already_won` |
| 2026-03-19 | 2.0/3.0 | Y | 4 | 3 | `D_teacher_already_won` |
| 2026-03-20 | 2.5/3.5 | Y | 4 | 2 | `D_teacher_already_won` |
| 2026-03-23 | 2.0/2.5 | Y | 1 | 1 | `D_teacher_already_won` |
| 2026-03-24 | 2.0/3.5 | Y | 6 | 1 | `D_teacher_already_won` |
| 2026-03-25 | 2.5/3.5 | n | 3 | 0 | `B_principles_block_all_winning_entries` |
| 2026-03-26 | 1.5/2.0 | n | 8 | 2 | `C_teacher_missed_valid_principle_win` |
| 2026-03-27 | 1.0/2.0 | Y | 9 | 1 | `D_teacher_already_won` |
| 2026-03-30 | 1.5/2.5 | Y | 4 | 2 | `D_teacher_already_won` |

## Root-cause read (for the lab)

- Already won under doctrine: **7/10**
- Teacher missed a **legal** (pt5 force-aligned) win: **1/10** ← fix encoding/timing
- Win only by **violating** HTF permission: **1/10** ← principles vs greed conflict
- **No** single-entry win under shell risk: **1/10** ← target/risk too hard for that day path, or need multi-leg / different size rules

### What “Mark sees the chart” does NOT mean here
- It does **not** mean ignore pt5 and buy the perfect hindsight trade against HTF.
- It **does** mean: with full price path known offline, find the best plan **allowed by principles + risk shell**.

### Primary bug if C > 0
Doctrine is **too timid or mistimed**: a force-aligned single entry would bank, but the live teacher didn’t take it (or overtraded and gave it back).
### Primary wall if A
Even omniscient single-entry under this shell cannot bank that T/R. Need multi-entry skill, different stops, or accept no-award day.

## Reproduce
```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/diagnose_10d_why_mark_misses.py
```
