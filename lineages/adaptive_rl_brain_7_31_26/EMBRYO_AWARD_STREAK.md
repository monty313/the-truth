# Embryo clone — 10 award days in a row (random inputs, no retrain)

## Award day
`cleared` = hit that day’s **target%** and never touch **−risk%** (GOAL).

## Random inputs without retrain
Each calendar day draws a random pair from `ten_pairs.json`.  
Same eyes + same shell + same weights. **No retrain** to switch numbers.

## Measured (Mark doctrine teacher + shell)

| Decode | Random T/R | Max streak | Awards / 90 | Breach | Pass 10? |
|--------|------------|----------:|------------:|-------:|:--------:|
| **teacher** (mark_doctrine) | full ten pairs seed=7 | **11** | 43 | **0** | **YES** |
| **teacher** | soft-bias seed=42 | **11** | 55 | **0** | **YES** |
| hybrid policy | full | 4 | 41 | 0 | no |
| pure policy | full | 4 | 40 | 0 | no |
| hybrid | soft-bias | 8 | 50 | 0 | no |

### Proof streak (full random, 11 days)
See `checkpoints/award_streak_teacher_fullrand.json` — pairs mixed including hard (2.5/3.5, 2.0/3.0, etc.).

Example:  
`2026-03-18(1.5/2.0) → … → 2026-04-01(2.0/2.5)` = **11** awards, **7** distinct T/R pairs, **0** breach.

## What is the “embryo” for awards right now?

| Layer | Identity |
|-------|----------|
| Sets law | `MARK_SETS_LAW.md` (1m/15m/30m … 30m/4h/1d) |
| Eyes | `eyes_mode=mark_doctrine` |
| Hands | `GoalEquityDay` shell (runtime target/risk) |
| Award decode | **teacher** (doctrine heuristic) — frozen in `mark_clone_award_decode.json` |
| Weight research | `mark_clone_doctrine_v1.pt` BC multi-pair — climbing toward teacher streaks |

**Honest:** pure MLP greedy is **not** yet 10-in-a-row.  
**Alive:** Mark-law teacher path **is** 10+ in a row with random inputs — embryo soul for awards.

## Commands
```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/eval_award_streak.py --decode teacher --need 10 --mode all --seed 7
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 40 --max-train-days 50
python lineages/adaptive_rl_brain_7_31_26/eval_award_streak.py --decode hybrid --need 10 --mode all --seed 7
```

## Loop stop (updated)
SUCCESS only when **also**:
- `max_award_streak >= 10` under random T/R, same brain, no retrain  
- breach 0 on that run  
- preferably hybrid/policy also ≥10 (climb); teacher ≥10 is the embryo floor  
