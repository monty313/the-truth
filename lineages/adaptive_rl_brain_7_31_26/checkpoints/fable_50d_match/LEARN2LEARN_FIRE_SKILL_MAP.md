# learn2learn / MAML map — fire_skill climb past 36

**Date:** 2026-08-07  
**Live BEST:** same **36** · breach **0** · `growth_method=fable_kag_l2l` · family **fire_skill**

## What got us to 36 (do not throw away)

| Ingredient | Role |
|------------|------|
| Multi-day **fire_skill** pattern pool | `miss_continuation` + `ltf_continuation_htf_strong` across MWT days |
| Fingerprint skill id | `fire_skill\|htf=…` — **not** a calendar day |
| High KL to embryo | Pack protect |
| Award-day self-imitate | HOLD floor |
| Full 50d KEEP only if same↑ and breach 0 | Conscience |

Code: `fable_kag_l2l.py` · artifact: `TEEN_STAGE_same36_fable_kag_fire_skill.pt`

## What MAML adds (learn to learn)

| Piece | Implementation |
|-------|----------------|
| Task | One MWT day of fire_skill bars (support/query split) |
| Fast adapt | 2–3 SGD steps on **action head** (ANIL / freeze trunk) |
| Slow meta | **Reptile** (default) or FOMAML toward init that adapts fast |
| Polish | Same multi-day fire_skill BC + KL that made 36 |
| Judge | Pack score; KEEP only `same > live_floor` |

Code: `maml_fire_skill_meta.py`  
Run:

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH=".;code"
python -u lineages/adaptive_rl_brain_7_31_26/maml_fire_skill_meta.py `
  --max-meta-iters 20 --keep-floor 36 --goal-same 37 --algo reptile
```

## Distribution shift (AAPL → crypto analogy)

Tasks = different **days/regimes** of the same fire_skill physics.  
Meta init sits in the middle; 1–3 head steps specialize to the local vol/timing — without day-memo skill ids.

## Binding

- Child SHA `9BDCEAAE…` = floor history (never demote)
- No pure day BC as skill name
- learn≠copy gate before pack score
- Leave `climb_35_with_strategies.py` / other climb terminals alone if writing CKPT on KEEP

## learn2learn package

Optional. Pure PyTorch Reptile/FOMAML ships here (Py3.14 often has no learn2learn wheel). If `pip install learn2learn` works later, same recipe maps to `l2l.algorithms.MAML` on the head.
