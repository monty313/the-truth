# Spine Shadow session baseline (F0 freeze)

**Frozen at:** 2026-08-05 (goal implementer session)  
**Method:** Spine Shadow (Fable 5 Alternate)  
**Doctrine:** `01_SYSTEM/Fable 5 Alternate — Spine Shadow.md` + pt5 basic knowledge  

## Start meters (practice 50d, seed=42, soft_bias=false, pure_greedy, mark_align)

| Meter | Value |
|-------|------:|
| same_outcome | **33**/50 |
| policy_clear | **33** |
| mwt | **17** |
| breach | **0** |
| keep_floor | 33 |
| mark_clear (soul plans) | 50/50 |

**Embryo:** `checkpoints/mark_clone_full_obs_v1.pt`  
**Source of best:** `one_day_KEEP_2026-02-25`  
**PROVEN:** not loaded; fingerprint written at session start (scratch + re-check at end).

## Diagnosis (why not 50)

All gaps are **MARK_WOULD_TAKE** — Mark spine wins, policy fails execution  
(subclass historically: wrong_size_or_timing). Dense dir_match can still miss the day.

## Products this session ships

| Product | Path |
|---------|------|
| Day spine compiler | `compile_day_spine.py` |
| Oracle score | `spine_oracle_score.py` → `spine_oracle_score__latest.json` |
| Shadow train loop | `train_spine_shadow.py` |
| Held-out dual score | `score_spine_heldout.py` |
| Unit tests | `tests/lineages/test_spine_shadow.py` |
| Learning log | `SPINE_SHADOW_LEARNING.md` |
| Error cards | `SPINE_ERROR_CARD__latest.md` |

## House laws

- PROVEN never written  
- Agents = sensors only (clue_prior attention, not side owners)  
- Trail+cushion+scale-in package off  
- HTF force-gate wraps policy via `mark_align_policy=True`  
- No 3 parallel LLM teachers  
- No entry-reward crank as primary  

## Finish line

Held-out 50 calendar days **disjoint** from practice fit days; dual run same recipe;  
**same_outcome 50/50 · breach 0** twice. Practice climbs with KEEP/REJECT along the way.
