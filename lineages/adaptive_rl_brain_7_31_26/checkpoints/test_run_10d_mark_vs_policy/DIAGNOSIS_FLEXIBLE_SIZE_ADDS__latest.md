# Diagnosis v2 — Mark with flexible lots + adds (full chart)

**When:** 2026-08-05T00:54:32.350803+00:00

## Your point
Mark would **change lot sizes** and **add** into opportunities relative to **that day's goal**. Then it often **doesn't make sense** for him to fail a day if he already sees the chart.

## What we wrong-footed before
v1 diagnosis only allowed **one entry** + **fixed** risk_use_frac=0.35 / cap=0.25. That understates Mark.

## What we search now (offline, chart known)
- Size grid: risk_use_frac × per_trade_cap (small → aggressive)
- Single entry **or** entry + one same-side **add**
- Prefer **force-aligned** (pt5.1); only then try any-side if needed
- Floor / bank shell still on

## Tallies

| Issue | Count |
|-------|------:|
| `C_fixed_shell_missed_flexible_principle_win` | **3** |
| `D_already_won_fixed_shell` | **7** |

| Days winnable with **force + flexible size/add** (incl. baseline wins) | **10/10** |
| Days winnable if also allow side stretch | **10/10** |

## Day-by-day

| Date | T/R | Fixed clear | Flex+force win | Flex any win | Issue |
|------|----:|:-----------:|:--------------:|:------------:|-------|
| 2026-03-17 | 3.0/3.5 | n | Y | Y | `C_fixed_shell_missed_flexible_principle_win` |
| 2026-03-18 | 2.0/3.0 | Y | Y | Y | `D_already_won_fixed_shell` |
| 2026-03-19 | 2.0/3.0 | Y | Y | Y | `D_already_won_fixed_shell` |
| 2026-03-20 | 2.5/3.5 | Y | Y | Y | `D_already_won_fixed_shell` |
| 2026-03-23 | 2.0/2.5 | Y | Y | Y | `D_already_won_fixed_shell` |
| 2026-03-24 | 2.0/3.5 | Y | Y | Y | `D_already_won_fixed_shell` |
| 2026-03-25 | 2.5/3.5 | n | Y | Y | `C_fixed_shell_missed_flexible_principle_win` |
| 2026-03-26 | 1.5/2.0 | n | Y | Y | `C_fixed_shell_missed_flexible_principle_win` |
| 2026-03-27 | 1.0/2.0 | Y | Y | Y | `D_already_won_fixed_shell` |
| 2026-03-30 | 1.5/2.5 | Y | Y | Y | `D_already_won_fixed_shell` |

## Conclusion

**You were right for almost all days:** with flexible lots/adds under force, **10/10** are winnable. Prior “Mark failed” was mostly **rigid size + no adds + timid teacher**, not “Mark can’t see the chart.”

### What the policy/teacher must learn next
1. **Size relative to remaining distance to target** (not fixed 0.35 forever)
2. **Adds** on continuation with HTF force (not banned thrash reverse package)
3. **Stop thrashing** when one good sized entry would bank
4. Keep floor sacred while sizing up when heat allows

## Reproduce
```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/diagnose_mark_flexible_size_adds.py
```
