# GSD RUN — Mark full clone (the-truth T3)

**When:** 2026-08-05  
**Goal:** ARMY `2026-08-05__goal__mark-full-clone-meta-rl-gsd.md`  
**PROVEN touched:** **false**

---

## Gate board

| Gate | Result | Evidence |
|------|--------|----------|
| **G1** Full eyes ckpt | **PASS** | `mark_clone_full_obs_v1.pt` · obs_dim=**168** · hidden=128 |
| **G2** Mark soul 10d | **PASS** | seed7/start40 · Mark plans **10/10** · breach 0 |
| **G3** Policy ≈ Mark | **PASS (soft)** | full-obs policy **9/10** · breach **0** · miss 3/27 only |
| **G4** Chart HITL pack | **SHIPPED** | `mark_chart_hitl/HITL__latest.md` — Mark sign-off **pending** (HITL human) |
| **G5** BC dir_match ≥0.85 | **FAIL** | dir_match **0.78** · overall match 0.52 (sparse HOLD labels) |
| **G6** Award streak ≥10 breach 0 | **PARTIAL** | max_streak **13** (≥10) · window breach **2** · awards 23/30 |
| **G7** Meta = Mark | **DOCUMENTED** | meta_role in ckpt; no PROVEN overwrite |

---

## Phase log

### A — Full-obs BC
- 20 practice days · multi-pair soul plans · 20/20 plan wins  
- samples=343 · BUY/SELL sparse  
- train dir_match=0.78  
- ckpt: `checkpoints/mark_clone_full_obs_v1.pt`

### B — 10d Mark vs policy (full-obs)
| | Clear | Breach |
|--|------:|-------:|
| Mark soul plans | **10/10** | 0 |
| Policy full-obs | **9/10** | 0 |
| Bar agree | 0.616 | |

Policy miss: **2026-03-27** 1.0/2.0 only.

### C — Thrash cap
- Runtime: mark_soul refuses open/reverse after 6 entries  
- Soft day 3/26: policy **2 entries · clear** (was thrash fail pre-cap)

### D — Chart HITL
- `checkpoints/mark_chart_hitl/HITL__latest.md`  
- **Action for Mark:** double-click `MARK HERE!.lnk` · open pack + chart · sign disagree bars

### E — Award streak (forward 30d · seed 42)
| Decode | max_streak | awards | breach | pass need=10 & breach0 |
|--------|----------:|-------:|-------:|:----------------------:|
| **Policy full-obs** | **13** | 23/30 | 2 | no (breach) |
| Online teacher | 8 | 24/30 | 1 | no |

Policy beat online teacher on streak length; need breach→0 for full G6.

### F — Meta
- Policy is Mark-clone attention over full eyes (168).  
- Meta may retune attention knobs only; shell laws locked.  
- Do **not** promote over PROVEN until G5+G6 green + Mark HITL.

---

## Products shipped

| File |
|------|
| `checkpoints/mark_clone_full_obs_v1.pt` |
| `checkpoints/mark_clone_latest.pt` |
| `checkpoints/GSD_PHASE_A__full_obs_bc.json` |
| `checkpoints/test_run_10d_mark_vs_policy/TEST_RUN_10D__latest.md` |
| `checkpoints/mark_chart_hitl/HITL__latest.md` |
| `checkpoints/award_streak_full_obs_policy.json` |
| `checkpoints/award_streak_teacher_online.json` |
| this file |

---

## Next GSD slice (not done)

1. Mark HITL sign-off on 3/27 + disagree bars  
2. More BC epochs / class weight for dir_match ≥0.85  
3. Tighten size when danger high → breach 0 on 30d streak  
4. Re-score G5+G6  

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --full-obs --epochs 50 --max-train-days 40 --hidden 128
python lineages/adaptive_rl_brain_7_31_26/eval_award_streak.py --decode policy --ckpt lineages/adaptive_rl_brain_7_31_26/checkpoints/mark_clone_full_obs_v1.pt --need 10 --max-days 40 --full-obs --mode forward
```
