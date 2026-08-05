# Multi-pair consistency (10 inputs)

**Simple win bar:** same brain, **10** (target%, risk%) pairs, each with **≥30 clear days** and **0% breach** on real XAUUSD days.

## Do this

```powershell
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all
```

## Files

| File | What |
|------|------|
| `ten_pairs.json` | The 10 pairs + seed + day split |
| `equity_day.py` | Goal-conditioned equity day (clear / breach) |
| `score_ten_pairs.py` | Score all pairs |
| `train_multi_pair.py` | Optional BC + dial helpers |
| `checkpoints/multi_pair_consistent_v1.pt` | Frozen weights + dials |
| Full story | `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md` |

## Decode

Uses **perception heuristic** + dials stored in the checkpoint (not pure RL argmax alone).  
Target% and risk% are **runtime inputs**. Same `.pt` for every pair.

## PROVEN

This lineage does **not** overwrite `models/PROVEN_*.pt`.
