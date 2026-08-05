# Mark soul → policy (Fable 5 as translator)

**When:** 2026-08-05  
**Order:** Mark sees chart + sizes + adds → write day → BC policy → pure policy acts like Mark.

## What “soul” means here (trading)

Not chat personality. **Trading soul** = what Mark would do on that chart under that day’s **target% / risk%**:

1. **Force first** — five-law doctrine + MARK SETS LAW (all 4 stacks)
2. **Goal-relative size** — lots grow when lagging the day goal and heat allows; shrink near bank / floor
3. **Force-aligned adds** — same-side add when still lagging (not reverse thrash)
4. **Floor / bank sacred** — shell still kills on breach, banks at target
5. **Not** the banned IRAC package **trail + cushion + scale-in**

Evidence: `checkpoints/test_run_10d_mark_vs_policy/DIAGNOSIS_FLEXIBLE_SIZE_ADDS__latest.md`  
→ fixed shell teacher **7/10**; force + flexible size/add **10/10**.

## Code

| Piece | Where |
|-------|--------|
| Size dials | `equity_day.GoalEquityDay.mark_soul_size_dials` |
| Adds | `_try_add` + teacher `_mark_soul_want_add` |
| Default ON | `eyes_mode=mark_doctrine` → `mark_soul=True` |
| Claim path | `legacy_set2` → soul OFF (fixed dials, no adds) |
| BC | `train_mark_clone_bc.py` → `mark_clone_soul_v1.pt` + `mark_clone_doctrine_v1.pt` |

## Transfer loop

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# 1) Soul teacher + BC into policy weights
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 40 --max-train-days 50

# 2) 10-day Mark diary vs pure policy (same random pairs as before)
python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py --seed 7 --start-idx 40
```

## Definition of done (soul in policy)

| Meter | Pass |
|-------|------|
| Teacher Mark soul clear on 10d seed7/start40 | ≥ 9/10, breach 0 |
| BC dir_match | ≥ 0.85 |
| Pure policy same 10d | clear climb vs pre-soul embryo |
| PROVEN | untouched |

## Results (2026-08-05)

| Meter | Result |
|-------|--------|
| Soul **plan** teacher 10d seed7/start40 | **10/10 clear, 0 breach** |
| Fixed thrash walk (old) | 7/10 |
| Soul plan labels train (20d multi-pair) | **20/20 plan wins** |
| BC embryo `mark_clone_soul_v1` / `mark_clone_doctrine_v1` | saved |
| Policy pure greedy practice | 14/20 clear (70%), 0 breach |
| Policy pure greedy forward | 15/20 clear (75%), 2 breach |
| A/B hard 3.0/3.5 forward | policy **70%** clear vs base 36% (breach 1) |
| A/B soft 1.0/2.0 | policy 78% clear (breach 1) |
| BC dir_match (sparse plans) | ~0.69 (needs more plan days / epochs) |
| **10d seed7/start40 Mark soul** | **10/10 clear, 0 breach** |
| **10d same policy (soul shell)** | **8/10 clear, 0 breach** (was 5/10 pre-soul) |
| Policy misses on 10d | 3/26, 3/27 (thrash 14–18 entries; Mark plans were sparse 2-entry) |

**Soul is in the teacher (10/10).** Policy has real soul transfer (5→8/10, hard pair 36%→70%) but still thrashier than Mark’s sparse plans on soft days. Next: more plan-label days + thrash penalty, then re-award streak.

Fable 5 role: **encode Mark’s sizing/add soul into shell teacher + BC** — not invent a second trader.
