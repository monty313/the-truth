# Progress saved — 2026-08-05

Canonical handoff (root): **`../../HANDOFF_2026-08-05.md`**  
Consistency scores: **`checkpoints/mark_consistency/CONSISTENCY__latest.md`**

## Embryo

| Field | Value |
|-------|--------|
| File | `checkpoints/mark_clone_full_obs_v1.pt` |
| Also | `checkpoints/mark_clone_latest.pt` |
| obs_dim | 168 |
| hidden | 128 |
| full_obs | true |
| mark_soul | size + force adds + thrash/danger caps |
| mark_align | force/capital gate (not thrash teacher slave) |
| proven_touched | **false** |

## Headline scores

- Mark soul plans 10d: **10/10** breach 0  
- Policy 10d: **8/10** breach 0 (3/27 fixed)  
- BC dir_match: **~0.84**  
- 50d forward streak: **max 11** · breach **0**  

## Resume

```powershell
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/mark_consistency_loop.py --epochs 40 --max-train-days 30 --streak-days 100
```
