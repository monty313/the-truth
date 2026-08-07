# KAG Fix Plan — Mark × Student × Fable (evidence)

**Issue:** Premature Execution Drift / path-timing desync  
**Goal:** same > 35 with breach 0; then toward 50; forward holdout later  
**Sources:** KAG index + lab telemetry + live LLM

---

## Issue attributes (evidence)

| Attribute | Evidence |
|-----------|----------|
| Class: path/timing | Gap: gold MWT 15/15 vs policy 0/15; implication=TIMING/PATH is primary — spine event recall + DAgger |
| Early fire | HITL e.g. Mark HOLD pol SELL/BUY at open bars |
| Missed fire | HITL Mark SELL/BUY pol HOLD mid/late day |
| Pack death on teach | Live: convert focus then same 35→32; REJECT restores 35 |
| Ceiling exists | Oracle spine same=50 pass=True; Mark plans 50/50 |
| Not size-primary | size_lock only 1/15 extra |
| Prior climb works | 27→35 one-day+KL+KEEP (WHAT_WORKS/LEARNING) |

---

## KAG hits (top)

- score=40.625 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\LEARNING_50D_MATCH.md: 
- score=37.375 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\FABLE5_MARK_HERE_BRIEF__latest.md: 
- score=35.75 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\THINKING__spine_shadow_now.md: 
- score=35.75 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\WHAT_WORKS__GOAL.md: 
- score=35.1 path=C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM\data\knowledge\army\META_RL_SQUAD_BUS.jsonl: 
- score=34.125 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\MARK_HERE_ANSWER__why_rewards_alone.md: 
- score=30.875 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\MARK_HERE_STUDENT_DIALOGUE__latest.md: 
- score=28.6 path=C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM\data\knowledge\army\KAG_INDEX__mark_policy.json: 
- score=27.625 path=C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth\lineages\adaptive_rl_brain_7_31_26\checkpoints\fable_50d_match\SPINE_SHADOW_KAG_NOTES__latest.md: 
- score=26.0 path=C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM\data\knowledge\army\FABLE5_MARK_HERE_DIALOGUE.jsonl: 

---

## STUDENT plan (LLM + evidence)

I propose the following 6-step fix plan to correct premature execution drift, maintain the same=35 baseline, and ensure breach=0.

### Step 1: Enforce Spine Shadow Alignment on MWT Days
* **Action:** Activate `mark_aligned_decode.py` with `GoalEquityDay.mark_align_policy=True` during DAgger rollouts on Mark Would Take (MWT) days.
* **Evidence:** KAG hit `WHAT_WORKS__GOAL.md` states: *"Force-gate / mark_align: HTF side law outside the net (pt5); wraps policy proposals" -> `mark_aligned_decode.py` · `GoalEquityDay.mark_align_policy=True`*.
* **Why it works:** It forces policy proposals to conform to high-timeframe Mark side laws outside the net, preventing premature execution drift without opening new teachers.

### Step 2: Apply KEEP/REJECT Conscience Gate
* **Action:** Wrap the evaluation loop with `spine_safe_one_day.py` and enforce the pack-crater cutoff (`post_same < best - 3 -> REJECT restore immediately`).
* **Evidence:** KAG hit `WHAT_WORKS__GOAL.md` states: *"KEEP/REJECT conscience: Restores best embryo when pack falls or breach>0 — only reason best is still 35 not 23" -> `fable_50d_one_day.py` · `spine_*` loops*.
* **Why it works:** It guarantees that any regression below the `same=35` benchmark or any non-zero breach triggers an instant revert to the frozen baseline before state corruption compounds.

### Step 3: Maintain HOLD Mass Floor to Prevent Breach
* **Action:** Constrain the training objective to prevent `pred_hold_rate` from dropping below ~0.27 (the threshold identified in the historical crash from 35->30).
* **Evidence:** KAG hit `WHAT_WORKS__GOAL.md` states: *"Directional oversample + enough HOLD: Dir-heavy labels convert MWT; HOLD mass stops breach / pack death ... safe climb: keep train `pred_hold_rate` from collapsing (~0.27 cratered 35→30)"*.
* **Why it works:** Preserving sufficient HOLD mass stops premature trade entries in ambiguous zones, directly enforcing `breach = 0`.

### Step 4: Utilize Oracle Cache for Spine Execution
* **Action:** Route all evaluation and DAgger passes through `MARK_ORACLE_CACHE__50d.json` indexed by `date|T|R`.
* **Evidence:** KAG hit `WHAT_WORKS__GOAL.md` states: *"Oracle cache: Reuse Mark plans per `date|T|R` — no re-search every cycle" -> `MARK_ORACLE_CACHE__50d.json`*.
* **Why it works:** Eliminates search variance across iterations and ensures deterministic recreation of the exact Mark soul plans that achieved `same=35, breach=0`.

### Step 5: Execute One-Day Focus + High KL Anchor
* **Action:** Train using `fable_50d_one_day.py` applying Behavior Cloning (BC) combined with a high Kullback-Leibler (KL) anchor tied strictly to the `pack_one_day_KEEP_2026-02-13` checkpoint.
* **Evidence:** KAG hit `WHAT_WORKS__GOAL.md` states: *"One-day focus + award protect + high KL: Only method that raised same (27→30→33; peer pack to 35) without opening 3 teachers" -> `fable_50d_one_day.py` · BEST `pack_one_day_KEEP_2026-02-13`*.
* **Why it works:** The KL anchor prevents policy drift away from the proven parameter space while permitting localized gradient updates for MWT timing corrections.

### Step 6: Validate via Gold Day Spine Re-Execution
* **Action:** Run `compile_day_spine.py` and `spine_oracle_score.py` under the shell on the frozen 50-day window.
* **Evidence:** KAG hit `WHAT_WORKS__GOAL.md` states: *"Gold Day Spine exec: Re-exec of compiled plan under shell → same 50/50, breach 0 (after online-fallback fix)" -> `compile_day_spine.py` · `spine_oracle_score.py`*.
* **Why it works:** Confirms that the compiled execution path matches the verified Mark soul plans identically before committing any state to the live registry.

---

## MARK final plan (LLM + laws)

FINAL PLAN:

1. **Action:** Activate `mark_aligned_decode.py` with `GoalEquityDay.mark_align_policy=True` during DAgger rollouts on MWT days to enforce pt5 high-timeframe side law outside the net.
   **Evidence:** `WHAT_WORKS__GOAL.md` (Force-gate / mark_align: HTF side law outside the net (pt5); wraps policy proposals).
   **Success metric:** Policy proposals conform to Mark side laws with zero out-of-bounds structural violations.
   **FAIL=REJECT**

2. **Action:** Wrap the evaluation loop with `spine_safe_one_day.py` applying the pack-crater cutoff rule (`post_same < best - 3 -> REJECT restore immediately`).
   **Evidence:** `WHAT_WORKS__GOAL.md` (KEEP/REJECT conscience: Restores best embryo when pack falls or breach>0).
   **Success metric:** Instant revert triggered on any pack regression below threshold or positive breach count.
   **FAIL=REJECT**

3. **Action:** Constrain the training objective to maintain training `pred_hold_rate` at or above ~0.27 using directional oversampling on MWT days.
   **Evidence:** `WHAT_WORKS__GOAL.md` (Directional oversample + enough HOLD: HOLD mass stops breach / pack death; ~0.27 cratered 35→30).
   **Success metric:** `pred_hold_rate >= 0.27` across all active training batches.
   **FAIL=REJECT**

4. **Action:** Route all evaluation and DAgger passes through `MARK_ORACLE_CACHE__50d.json` indexed by `date|T|R`.
   **Evidence:** `WHAT_WORKS__GOAL.md` (Oracle cache: Reuse Mark plans per `date|T|R` — no re-search every cycle).
   **Success metric:** Zero search variance across iterations with deterministic retrieval of Mark soul plans.
   **FAIL=REJECT**

5. **Action:** Train using `fable_50d_one_day.py` combining Behavior Cloning with a high KL anchor tied strictly to `pack_one_day_KEEP_2026-02-13`.
   **Evidence:** `WHAT_WORKS__GOAL.md` (One-day focus + award protect + high KL: raised same 27→30→33; peer pack to 35).
   **Success metric:** `same_outcome` increases toward 35 while `breach == 0` and parameter drift remains bounded by the KL anchor.
   **FAIL=REJECT**

6. **Action:** Run `compile_day_spine.py` and `spine_oracle_score.py` under the shell on the frozen 50-day window.
   **Evidence:** `WHAT_WORKS__GOAL.md` (Gold Day Spine exec: Re-exec of compiled plan under shell → same 50/50, breach 0).
   **Success metric:** Verified match against compiled Mark soul execution path.
   **FAIL=REJECT**

---

## STUDENT risks

I am the-truth policy embryo. Confirming plan.

### Plan Confirmation
All six steps are confirmed under the mandatory pack evidence and functional mechanisms.

### Top 3 Risks and KEEP Mechanics

1. **Risk:** Catastrophic pack drop due to unanchored policy drift or sudden structural breaches during DAgger/training.
   * **How KEEP Catches It:** Step 2 executes `spine_safe_one_day.py` applying the pack-crater cutoff rule (`post_same < best - 3 -> REJECT restore immediately`). Any drop exceeding 3 points triggers an instant revert to the best embryo.

2. **Risk:** Loss of critical HOLD mass leading to structural breaches and zero-outcomes (cratering the pack from ~35 to 30).
   * **How KEEP Catches It:** Step 3 enforces directional oversampling to maintain `pred_hold_rate >= 0.27`. If the training objective violates this threshold or results in positive breach counts, the evaluation loop flags `breach > 0` and Step 2 enforces immediate restoration.

3. **Risk:** Uncontrolled parameter drift away from the proven baseline trajectory despite behavior cloning.
   * **How KEEP Catches It:** Step 5 applies `fable_50d_one_day.py` with a high KL anchor tied strictly to `pack_one_day_KEEP_2026-02-13`. If drift exceeds the anchor bounds or fails to raise `same_outcome` toward 35 while maintaining `breach == 0`, the evaluation framework rejects the iteration.

---

## FABLE verdict (will it work?)

**Fable 5 Evidence Judge Evaluation**

### Step-by-Step Evidence & Likelihood Breakdown

1. **Step 1:** `mark_aligned_decode.py` with `GoalEquityDay.mark_align_policy=True`
   * **Evidence Strength:** Medium (`WHAT_WORKS__GOAL.md` force-gate / mark_align)
   * **Likelihood helps same > 35:** Medium (Enforces structural bounds outside the net, mitigating breach risk during DAgger).

2. **Step 2:** `spine_safe_one_day.py` pack-crater cutoff rule (`post_same < best - 3`)
   * **Evidence Strength:** Strong (`WHAT_WORKS__GOAL.md` KEEP/REJECT conscience)
   * **Likelihood helps same > 35:** Strong (Directly addresses the observed 35→32 conversion drop by instantly reverting regressions).

3. **Step 3:** Directional oversampling for `pred_hold_rate >= 0.27`
   * **Evidence Strength:** Strong (`WHAT_WORKS__GOAL.md` HOLD mass stops breach / pack death)
   * **Likelihood helps same > 35:** Strong (Prevents pack death and maintains stability required to push beyond 35).

4. **Step 4:** `MARK_ORACLE_CACHE__50d.json` routing (`date|T|R`)
   * **Evidence Strength:** Medium (`WHAT_WORKS__GOAL.md` Oracle cache reuse)
   * **Likelihood helps same > 35:** Medium (Ensures deterministic execution and eliminates search variance across cycles).

5. **Step 5:** `fable_50d_one_day.py` with Behavior Cloning + high KL anchor tied to `pack_one_day_KEEP_2026-02-13`
   * **Evidence Strength:** Strong (`WHAT_WORKS__GOAL.md` One-day focus + award protect + high KL: 27→30→33; peer pack to 35)
   * **Likelihood helps same > 35:** Strong (Directly tied to the documented climb trajectory toward and past 35).

6. **Step 6:** `compile_day_spine.py` & `spine_oracle_score.py` execution on frozen 50-day window
   * **Evidence Strength:** Strong (`WHAT_WORKS__GOAL.md` Gold Day Spine exec: 50/50, breach 0)
   * **Likelihood helps same > 35:** Strong (Validates alignment against the 50/50 gold oracle benchmark).

---

### Overall Verdict

* **Conditional / Yes**
* **Conditions:** 
  1. Strict adherence to Step 2's `post_same < best - 3` cutoff to immediately trap and reverse any conversion regressions (such as the observed 35→32 drop).
  2. Maintenance of training `pred_hold_rate` at $\ge 0.27$ via Step 3 to safeguard against pack death.
  3. Preservation of the high KL anchor to `pack_one_day_KEEP_2026-02-13` during one-day training to drive `same_outcome` past 35 with zero breaches (`breach == 0`).

---

## Executable table (implementer)

| # | Action | Evidence solution works | Success | FAIL |
|---|--------|-------------------------|---------|------|
| 1 | Force-gate + mark_align stay on | pt5; HITL early wrong side; Mark orders | breach=0; gate on | remove gate → abort |
| 2 | One-day DAgger on policy path | Mark path labels; offline BC high dir still MWT | focus award or same↑ | same<34 crater → REJECT |
| 3 | KL≥0.55–0.72 + award protect | LEARNING hold collapse; 27→35 recipe | same≥35 after step | same↓ breach>0 → REJECT |
| 4 | Repair from BEST if convert-slip | overnight thrash-repair stuck 32 | same≥35 post-repair | else REJECT |
| 5 | HITL bars only for stubborn MWT | HITL exact disagrees | Mark-corrected BC only those bars | no Mark → skip invent |
| 6 | Dual 50d score on KEEP | house consistency | two scores agree; same≥best | mismatch → no KEEP |
| 7 | Forward-100 after climb | score_forward_100d design | holdout ∩ fit empty | practice-only ≠ done |

### Evidence chain (why believable)

1. Gold path wins → teachable ceiling  
2. Focus convert observed → weights can learn day  
3. Pack death measured → KEEP shape is the constraint  
4. 27→35 already proved one-day+KL+KEEP raises same  
5. HITL bars = surgical labels (smaller than full thrash BC)

### Falsifiers

- Many rounds: never convert AND never KEEP → need spine architecture not BC  
- Oracle <48 → fix shell first  
- Repeat breach on candidates → HOLD first  

---
*KAG session saved for implementer.*
