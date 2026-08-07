# Spine Shadow — KAG improvement notes

**Purpose:** durable reasoning for later KAG: why same rose/fell, which spine error remains, what surgical fix to try next.

## Baseline vs best (practice 50d · seed42 · soft_bias=false · pure_greedy · mark_align)

| Snapshot | same | policy | mwt | breach | source |
|----------|-----:|-------:|----:|-------:|--------|
| Frozen plan floor | 33 | 33 | 17 | 0 | one_day_KEEP_2026-02-25 |
| Session live start | 35 | 35 | 15 | 0 | pack_one_day_KEEP_2026-02-13 (peer pipeline) |
| Goal target | 50 | 50 | 0 | 0 | held-out dual run |

## Oracle spine gate (S1)

| Meter | Value |
|-------|------:|
| same_outcome (gold spine exec) | **50**/50 after fallback fix |
| breach | **0** |
| Pass (≥48, breach0) | **yes** |

**Compiler note:** day `2026-02-17` is `soul_online_fallback` (no sparse plan). First oracle run treated empty HOLD plan as gold → miss. Fix: re-exec online soul walk when `mark_source=soul_online_fallback` or `t1 is None`. After fix → 50/50.

## Dominant MWT spine error classes (at same=35)

| Class | Count (approx) | Meaning | Surgical fix |
|-------|---------------:|---------|--------------|
| wrong_size_or_timing | ~8 | fires but misses award (size/timing off spine) | plan path + spine fire/add weight near t1/t2; size dials from spine |
| false_fire | ~7 | too many entries vs spine | wait_loaded / HOLD-on-spine; cut dir_copy |
| false_hold | rare | never fires | boost fire/add, DAgger |

## What failed (KAG must not repeat)

1. **Pack-wide BC with pack-dominant class** — when pack said `false_fire`, train boosted wait for ALL MWT including wrong_size days → flat 33, dir ratio 0.18.  
2. **Equal weight all MWT** — dilutes the day that would convert.  
3. **Entry-reward crank / 3 teachers** — banned; already known to collapse hold → breach.  
4. **Train without oracle green** — forbidden; we waited for S1.

## What works (method)

1. Compile soul plan → DaySpine (sparse events) — lossless plan round-trip unit-tested.  
2. Oracle gold exec ≈ Mark (50/50).  
3. **Per-day error class** pick focus; rotate after fails.  
4. Heavy focus labels + award protect + pack-repair if focus converts but pack slips.  
5. KEEP only if breach=0, same ≥ live floor, not thrash.

## Products on disk

- `compile_day_spine.py` · `spine_oracle_score.py` · `train_spine_shadow.py` · `spine_one_day.py` · `score_spine_heldout.py`  
- `checkpoints/spines/SPINE_INDEX__latest.json` (50 spines)  
- `spine_oracle_score__latest.json`  
- `SPINE_ERROR_CARD__latest.md` · `SPINE_SHADOW_LEARNING.md`  
- tests: `tests/lineages/test_spine_shadow.py` (7 passed)

## Gap diagnosis (2026-08-06) — critical perspective

On 15 MWT days at pack same=35:

| Condition | Awards |
|-----------|-------:|
| Policy alone (mark_align pure greedy) | **0**/15 |
| Policy + Mark size dials locked | **1**/15 |
| Gold spine `run_plan` | **15**/15 |

**Implication: TIMING/PATH is primary, not size.**  
Offline plan-path BC can hit dir_match ~0.95 while online trajectory never hits t1/t2 (covariate shift).  
Surgical fix: **multi-iter DAgger** on policy states labeled by spine plan (`spine_dagger_climb.py`).

## Price data (do not look elsewhere)

All M1 under **the-truth/data/raw/** via `price_data.RAW_DIR`:

- `XAUUSD_curriculum_2026.csv` — practice 50d  
- `XAUUSD_M1_full.csv` — 2020-09→2026-05 (~1461 loadable days) for **100d holdout**  
- Also EURUSD/GBPUSD/US30 curricula  

Holdout construction (`score_forward_100d.py`): 40 post-fit future + 60 pre-fit past, **∩ fit = ∅**.

## Next (until forward 100 consistent)

1. `spine_dagger_climb.py` — multi-iter DAgger, KEEP/REJECT, log every cycle.  
2. Practice climb toward 40+ then 50 with breach 0.  
3. Dual-run `score_forward_100d.py` on 100 never-fit days from M1_full.  
4. Stubborn MWT → HITL spine only if chart ambiguous.  
5. Never promote PROVEN.
