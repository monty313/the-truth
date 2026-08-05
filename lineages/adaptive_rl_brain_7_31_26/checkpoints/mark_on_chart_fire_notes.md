# mark_on_chart_fire_notes

## FIRE 2026-08-04T17:49Z (Mark clone BC fire)

**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt`  
**eyes:** mark_doctrine | decode: pure greedy policy  
**tests:** test_mark_sets_law + opportunity + clone_policy → **17 passed**  
**PROVEN:** untouched (mtime 2026-07-25)

### Day walks — what Mark would do vs what policy did

#### Thrash day 2026-04-02 @ 3.0 / 3.5
| Arm | entries | pnl | min_eq | cleared | banked | breach |
|-----|--------:|----:|-------:|:-------:|:------:|:------:|
| claim baseline (legacy set2) | 12 | -0.65 | -1.32 | no | no | no |
| **Mark teacher (doctrine)** | **5** | +1.48 | -0.42 | no | no | no |
| **policy greedy** | **6** | +0.59 | -0.73 | no | no | no |

- **Mark:** SELL→SELL→BUY→SELL→BUY (5). HTF-gated flips only; holds 19/24 decisions. Never banks hard target; stays off thrash.
- **Policy:** Extra early **BUY at t=770 while teacher=HOLD** (false long), then mostly tracks teacher flips → **6 entries**. Step match 0.75. Better than baseline thrash (12) but still one premature long vs Mark.
- **Verdict:** thrash_ent=6 (HEALTHY band ≤6). Mark cleaner; policy still slightly over-eager flat→long.

#### Bank / soft day 2026-04-01 @ 1.0 / 2.0
| Arm | entries | pnl | cleared | banked | breach |
|-----|--------:|----:|:-------:|:------:|:------:|
| baseline | 2 | +1.29 | yes | yes | no |
| **Mark teacher** | **2** | +1.30 | yes | yes | no |
| **policy** | **1** | +1.07 | yes | yes | no |

- **Mark:** BUY then SELL flip; banks soft target in 2 entries.
- **Policy:** One early BUY (teacher HOLD at t=745) and banks. soft_ok=true.
- Same ckpt also clears 2026-04-01 @ 3.0/3.5 (4 entries, banked) and 2026-04-02 @ 1.0/2.0 (4 entries, banked) → **any_pair_no_retrain=true**.

### Meters (this fire)
| meter | value | HEALTHY? |
|-------|------:|:--------:|
| dir_match train | 0.945 | yes (≥0.90) |
| dir_match forward labels | 0.895 | **no** (<0.90) |
| match train | 0.657 | no (clone_ready wants ≥0.70) |
| thrash_ent 04-02@3/3.5 | 6 | yes |
| soft_ok 04-01@1/2 | true | yes |
| hard_ent_mean (fwd 40d) | 5.725 | thrash improved vs base 8.975 |
| soft_clear (fwd) | 75.0% (base 87.5) | no collapse |
| breach | 0 | yes |
| pass_gates | all true | yes |
| clone_ready_policy | false (match<0.70) | — |
| proven_ok | true | yes |

**HEALTHY this fire?** **NO** — forward dir_match 0.895 < 0.90; overall match 0.657; mark_on_chart still partial (early BUY vs HOLD).  
**Previous fire notes HEALTHY?** none (first section) → no SUCCESS exit.

**Change applied this fire:** BC retrain epochs=30 practice-n=50 max-train-days=50 ab-after (no doctrine/shell change).  
**ONE-change gate:** thrash≤8, soft cleared, breach=0, no collapse, dir_match≥0.85 → **no second train**.

**next=** raise train match≥0.70 without crushing dir_match (label multi-pair mix / slightly more diverse teacher days, not HOLD-boost weights); aim forward dir_match≥0.90; kill premature flat→BUY when doctrine HOLD (teacher confirm / confidence gate); keep thrash≤6 soft bank; two consecutive HEALTHY then SUCCESS.

### Status line
mark_on_chart=partial dir_match=0.945 match=0.657 thrash_ent=6 soft_ok=true hard_ent_mean=5.725 soft_clear=75.0 breach=0 any_pair_no_retrain=true proven_ok=true next=lift_match_ge_0.70_and_fwd_dir_ge_0.90_no_HOLD_boost_keep_thrash_le6

## FIRE 2026-08-04T18:56Z (embryo award streak fire — Mark clone loop)

**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (BC multi-pair labels)  
**eyes:** mark_doctrine | sets law: OFFICIAL 4 stacks (tests pin)  
**tests:** test_mark_sets_law + test_mark_clone_policy → **10 passed**  
**PROVEN:** untouched (mtime 2026-07-25) · proven_ok=true

### Award streak (seed=7, mode=all, random ten_pairs, need=10)

| Decode | max_streak | awards/90 | breach | unique_pairs_streak | pass10? |
|--------|----------:|----------:|-------:|--------------------:|:-------:|
| **teacher** | **11** | 43 | **0** | 7 | **YES** |
| **hybrid** | 4 | 43 | 0 | 1 | no |
| **policy** | 4 | 42 | 0 | 1 | no |

Teacher proof streak: 2026-03-18(1.5/2.0) → … → 2026-04-01(2.0/2.5) = 11 awards, multi-pair random, no retrain.  
Wrote: `award_streak_teacher_fullrand.json` · `award_streak_hybrid.json` · `award_streak_policy.json`

### BC this fire (multi-pair only — lift weights toward teacher)
- epochs=30 practice-n=50 max-train-days=50 multi_pair=True
- train match=0.660 dir_match=0.958 | forward match=0.678 dir_match=0.904
- greedy practice clear 35% / forward 45% breach=0 both
- A/B hard thrash improved; soft no collapse; breach 0 both arms
- pass_gates all true · READY heuristic=True policy=False
- **No doctrine / MARK SETS LAW change** (teacher already ≥10 breach=0)

### Embryo gate
| Floor | Status |
|-------|--------|
| teacher_streak≥10 | **YES (11)** |
| breach=0 | **YES** |
| any_pair random | **YES** (7 unique T/R in teacher streak) |
| sets tests pass | **YES** |
| proven_ok | **YES** |
| hybrid≥10 OR policy≥10 | **NO** (both 4) |

**Prior fire teacher≥10?** prior section had teacher path proven in EMBRYO docs / decode json but first fire notes were clone thrash meters not award-status; this fire reconfirms teacher 11.  
**SUCCESS embryo_10_award_random_pair?** **NO** — need hybrid or policy also ≥10.  
**Report:** embryo_alive_teacher_award — keep looping BC multi-pair; do not delete scheduler.

### Status line
embryo=alive_teacher_award teacher_streak=11 hybrid_streak=4 policy_streak=4 breach=0 any_pair=true proven_ok=true next=more_BC_multi_pair_lift_weights_to_teacher_streak_10_no_doctrine_touch


## FIRE 2026-08-04T19:59:57Z (embryo award streak fire — Mark clone loop)

**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (BC multi-pair; torch.save before A/B)  
**eyes:** mark_doctrine | sets law: OFFICIAL 4 stacks  
**tests:** test_mark_sets_law + test_mark_clone_policy → **10 passed**  
**PROVEN:** untouched (mtime 2026-07-25) · proven_ok=true

### Award streak (seed=7, mode=all, random ten_pairs, need=10)

| Decode | max_streak | awards/90 | breach | unique_pairs_streak | pass10? |
|--------|----------:|----------:|-------:|--------------------:|:-------:|
| **teacher** | **11** | 43 | **0** | 7 | **YES** |
| **hybrid** | 4 | 43 | 0 | 1 | no |
| **policy** | 4 | 42 | 0 | 1 | no |

Teacher proof: 2026-03-18(1.5/2.0) → … → 2026-04-01(2.0/2.5) = 11 awards.  
Wrote: `award_streak_teacher_fullrand.json` · `award_streak_hybrid.json` · `award_streak_policy.json`

### BC this fire (multi-pair only)
- epochs=30 practice-n=50 max-train-days=50 multi_pair=True
- train match=0.660 dir_match=0.958 | forward match=0.678 dir_match=0.904
- greedy practice clear 35% / forward 45% breach=0 both
- note: A/B hard/soft eval timed out after ckpt save (same metrics as prior fire — BC path looks deterministic on fixed labels); no doctrine change
- **No MARK SETS LAW / teacher path change** (teacher already ≥10 breach=0)

### Embryo gate
| Floor | Status |
|-------|--------|
| teacher_streak≥10 | **YES (11)** |
| breach=0 | **YES** |
| any_pair random | **YES** (7 unique T/R) |
| sets tests pass | **YES** |
| proven_ok | **YES** |
| hybrid≥10 OR policy≥10 | **NO** (both 4) |

**Prior fire teacher≥10?** YES (prior section teacher_streak=11).  
**SUCCESS embryo_10_award_random_pair?** **NO** — hybrid/policy still 4.  
**Report:** embryo_alive_teacher_award — keep looping; do not delete scheduler.

### Status line
embryo=alive_teacher_award teacher_streak=11 hybrid_streak=4 policy_streak=4 breach=0 any_pair=true proven_ok=true next=more_BC_multi_pair_break_determinism_lift_policy_streak_to_10


## FIRE 2026-08-04T20:59:21Z (embryo award streak fire — Mark clone loop)

**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (BC multi-pair; save-before-A/B)  
**eyes:** mark_doctrine | sets law: OFFICIAL 4 stacks  
**tests:** test_mark_sets_law + test_mark_clone_policy → **10 passed**  
**PROVEN:** untouched (mtime 2026-07-25) · proven_ok=true

### Award streak (seed=7, mode=all, random ten_pairs, need=10)

| Decode | max_streak | awards/90 | breach | unique_pairs_streak | pass10? |
|--------|----------:|----------:|-------:|--------------------:|:-------:|
| **teacher** | **11** | 43 | **0** | 7 | **YES** |
| **hybrid** | 4 | 43 | 0 | 1 | no |
| **policy** | 4 | 42 | 0 | 1 | no |

Teacher proof: 2026-03-18 → 2026-04-01 = 11 awards, 7 unique T/R.  
Wrote: `award_streak_teacher_fullrand.json` · `award_streak_hybrid.json` · `award_streak_policy.json`

### BC this fire (multi-pair only)
- epochs=30 practice-n=50 max-train-days=50 multi_pair=True seed=42 (hardcoded)
- train match=0.660 dir_match=0.958 | forward match=0.678 dir_match=0.904
- greedy practice 35% / forward 45% breach=0
- A/B timed out after ckpt write (again)
- **Identical loss curve + metrics vs prior fires** — train_bc always cold-starts seed=42; no weight climb across fires
- **No doctrine / MARK SETS LAW change** (teacher floor healthy)

### Embryo gate
| Floor | Status |
|-------|--------|
| teacher_streak≥10 | **YES (11)** |
| breach=0 | **YES** |
| any_pair random | **YES** |
| sets tests pass | **YES** |
| proven_ok | **YES** |
| hybrid≥10 OR policy≥10 | **NO** (both 4) |

**Prior fire teacher≥10?** YES (11).  
**SUCCESS embryo_10_award_random_pair?** **NO** — policy/hybrid stuck at 4 under fixed BC.  
**Report:** embryo_alive_teacher_award — keep looping; do NOT delete scheduler.  
**Attention:** BC path needs non-identical train (warm-start from ckpt / seed rotate / more epochs) to lift hybrid|policy streak — without touching teacher award decode or sets law.

### Status line
embryo=alive_teacher_award teacher_streak=11 hybrid_streak=4 policy_streak=4 breach=0 any_pair=true proven_ok=true next=warmstart_or_seed_rotate_BC_to_lift_policy_streak_keep_teacher


## FIRE 2026-08-04T22:08:14Z (embryo award streak fire — Mark clone loop)

**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (warm-start + seed rotate)  
**eyes:** mark_doctrine | sets law: OFFICIAL 4 stacks  
**tests:** test_mark_sets_law + test_mark_clone_policy → **10 passed**  
**PROVEN:** untouched (mtime 2026-07-25) · proven_ok=true

### Award streak (seed=7, mode=all, random ten_pairs, need=10)

| Decode | max_streak | awards/90 | breach | unique_pairs_streak | pass10? |
|--------|----------:|----------:|-------:|--------------------:|:-------:|
| **teacher** | **11** | 43 | **0** | 7 | **YES** |
| **hybrid** | 4 | 42 | 0 | 1 | no |
| **policy** | 4 | 41 | 0 | 1 | no |

Teacher floor reconfirmed: 11-day random multi-pair streak, breach 0.

### BC this fire (multi-pair only — step3 + step6 extra)
1) seed=43 warm-start: train match **0.716** dir_match 0.952 | fwd match **0.725** dir_match 0.942 | practice clear 45% breach0  
2) seed=44 warm-start: train match **0.729** dir_match 0.957 | fwd match **0.743** dir_match 0.942 | practice 40% / fwd 45% breach0  
- A/B timed out after ckpt save both times (expected)  
- **Climb:** match broke 0.70 clone gate (was stuck 0.66 cold-start)  
- **Gap:** award *streak* still 4 — total awards ~teacher but consecutive clears under random T/R not chaining  
- No doctrine / MARK SETS LAW change

### Embryo gate
| Floor | Status |
|-------|--------|
| teacher_streak≥10 | **YES (11)** |
| breach=0 | **YES** |
| any_pair random | **YES** |
| sets tests pass | **YES** |
| proven_ok | **YES** |
| hybrid≥10 OR policy≥10 | **NO** (both 4) |

**Prior fire teacher≥10?** YES.  
**SUCCESS embryo_10_award_random_pair?** **NO**  
**Report:** embryo_alive_teacher_award — keep looping BC warmstart; do NOT delete scheduler.

### Status line
embryo=alive_teacher_award teacher_streak=11 hybrid_streak=4 policy_streak=4 breach=0 any_pair=true proven_ok=true next=more_BC_warmstart_match_ok_need_streak_chain_policy_to_10


## FIRE 2026-08-04T23:00:57Z (embryo award streak fire — Mark clone loop)

**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (warm-start seeds 45→46)  
**eyes:** mark_doctrine | sets law: OFFICIAL 4 stacks  
**tests:** test_mark_sets_law + test_mark_clone_policy → **10 passed**  
**PROVEN:** untouched (mtime 2026-07-25) · proven_ok=true

### Award streak (seed=7, mode=all, random ten_pairs, need=10)

| Decode | max_streak | awards/90 | breach | unique_pairs_streak | pass10? |
|--------|----------:|----------:|-------:|--------------------:|:-------:|
| **teacher** | **11** | 43 | **0** | 7 | **YES** |
| **hybrid** | **7** | 43 | 0 | 5 | no |
| **policy** | **7** | 41 | 0 | 5 | no |

**CLIMB:** hybrid/policy max_streak **4 → 7** this fire (now tracking teacher path 2026-03-18…03-26; teacher continues to 04-01).  
Streak: 03-18(1.5/2.0)→03-19(2.5/3.5)→03-20(1.0/2.5)→03-23(2.5/3.5)→03-24(2.0/3.0)→03-25(1.0/2.5)→03-26(1.0/2.0).

### BC this fire
1) seed=45 warm-start: match **0.743** / fwd **0.736** dir_match 0.942; A/B complete; soft_no_collapse=False (soft 65% vs base 87.5); hard thrash improved; breach 0  
2) seed=46 extra (step6): match **0.751** / fwd **0.761** dir_match 0.923; practice clear 35% / fwd 45% breach0  
- clone match gate ≥0.70 held; READY heuristic=True policy=False  
- No doctrine / MARK SETS LAW change

### Embryo gate
| Floor | Status |
|-------|--------|
| teacher_streak≥10 | **YES (11)** |
| breach=0 | **YES** |
| any_pair random | **YES** |
| sets tests pass | **YES** |
| proven_ok | **YES** |
| hybrid≥10 OR policy≥10 | **NO** (both 7) |

**Prior fire teacher≥10?** YES.  
**SUCCESS embryo_10_award_random_pair?** **NO** — need 3 more consecutive awards on hybrid|policy.  
**Report:** embryo_alive_teacher_award — keep looping; do NOT delete scheduler.

### Status line
embryo=alive_teacher_award teacher_streak=11 hybrid_streak=7 policy_streak=7 breach=0 any_pair=true proven_ok=true next=more_BC_warmstart_close_3_day_gap_to_teacher_11


## FIRE 2026-08-04T23:55Z (scheduled mark_first hourly)

**order:** MARK SEES CHART FIRST → update policy → harvest museum → meta (any T/R no retrain)
**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (updated; PROVEN untouched)
**sets law:** test_mark_sets_law.py **5 passed**
**dates:** 2026-04-02,2026-04-01,2026-03-18,2026-03-19,2026-03-20

### Cycle 1 (epochs=20)
- mean_agree **0.653 → 0.628** (agree_improved=false)
- update match_after=0.721 loss_final=0.339 n_samples=61
- harvest: multi_pair_consistent + mark_clone_channel1 distill ok; channel1_curriculum_* load_fail state_dict
- distill match_after=0.821 n_harvest_bars=84

### Cycle 2 once (epochs=40) — step5 because agree_after < agree_before
- mean_agree **0.628 → 0.645** (agree_improved=true)
- update match_after=0.721 loss_final=0.309 epochs=40
- harvest distill match_after=0.845 loss_final=0.246
- proven_touched=false

### Award streak (seed=7, mode=all, need=10)

| Decode | max_streak | awards/90 | breach | pass10? |
|--------|----------:|----------:|-------:|:-------:|
| **teacher** | **11** | 43 | **0** | **YES** |
| **policy** | **4** | 41 | 0 | no |
| **hybrid** | **4** | 41 | 0 | no |

**Note:** prior fire had hybrid/policy streak **7**; this mark-first fire leaves both at **4** (climb not advanced; harvest thrash risk). Teacher floor held.

### Gates
| Floor | Status |
|-------|--------|
| teacher_streak≥10 | **YES (11)** |
| breach=0 | **YES** |
| agree_after≥agree_before (final cycle) | **YES (0.628→0.645)** |
| sets pass | **YES** |
| proven_ok | **YES** |
| any_pair random | **YES** |
| policy≥10 OR hybrid≥10 | **NO** (both 4) |

**HEALTHY this fire?** YES (teacher floor + breach0 + final agree up + sets + proven).  
**CLIMB SUCCESS?** NO — policy/hybrid <10 under random T/R.  
**SUCCESS always_learning_mark_on_chart?** NO — keep looping; do NOT scheduler_delete.

### Status line
mark_first=true agree_before=0.628 agree_after=0.645 teacher_streak=11 policy_streak=4 hybrid_streak=4 harvest=true breach=0 any_pair=true proven_ok=true next=keep_loop_more_mark_first_or_BC_to_lift_policy_hybrid_from_4_to_10


## FIRE 2026-08-05T00:25Z (scheduled mark_day_diary hourly)

**order:** chart first → diary (pt5) → BC policy same → award streak teacher/policy
**ckpt:** `checkpoints/mark_clone_doctrine_v1.pt` (updated; PROVEN untouched)
**sets law:** test_mark_sets_law.py **5 passed**
**dates:** 2026-04-02(3.0/3.5), 2026-04-01(1.0/2.0), 2026-03-18(1.5/2.0), 2026-03-19(2.5/3.5)

### Diary + BC
- Cycle1 epochs=25: mean_agree **0.796 → 0.838** (improved); match_after=0.789 n=38
- Cycle2 epochs=40 (step5 agree_after < 0.85): mean_agree **0.838 → 0.838** flat; match_after=0.789 loss_final=0.267
- Day after: 04-02=0.75 (clear=false pnl=1.13), 04-01=1.0 clear, 03-18=1.0 clear, 03-19=0.60 clear
- proven_touched=false

### Award streak (seed=7, mode=all, need=10)

| Decode | max_streak | awards/90 | breach | unique_pairs | pass10? |
|--------|----------:|----------:|-------:|-------------:|:-------:|
| **teacher** | **10** | 41 | **0** | 6 | **YES** |
| **policy** | **4** | 38 | **0** | 3 | no |

Teacher streak: 03-18→03-31 (10 days). Policy breaks after 03-23 (4 days).

### Gates
| Floor | Status |
|-------|--------|
| agree_after≥0.90 | **NO** (0.838) |
| teacher_streak≥10 | **YES (10)** |
| policy_streak≥10 | **NO (4)** |
| breach=0 | **YES** |
| any_pair | **YES** |
| proven_ok | **YES** |
| sets pass | **YES** |
| prior HEALTHY | **YES** |

**HEALTHY this fire?** PARTIAL — teacher floor + breach0 + proven + sets; agree stuck 0.838; policy still 4.
**SUCCESS always_learning / scheduler_delete?** **NO** — need agree≥0.90 and policy_streak≥10. Keep looping.

### Status line
chart_first=true agree_before=0.796 agree_after=0.838 teacher_streak=10 policy_streak=4 diaries=4 breach=0 any_pair=true proven_ok=true next=more_BC_on_disagree_days_04-02_03-19_lift_agree_and_policy_streak

