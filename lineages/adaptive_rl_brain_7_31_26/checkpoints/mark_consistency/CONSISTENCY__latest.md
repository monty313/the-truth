# Mark long-term consistency — soul = policy

**When:** 2026-08-05  
**Goal:** hundreds of days, random T/R, no retrain, breach 0, Mark sense  

---

## What Mark sense means (from HITL + pt5)

| Mark law | Fix in lab |
|----------|------------|
| Wait when force not ready | Force/capital **gate** on policy opens |
| Never open against HTF force | `mark_force_gate` blocks opposite |
| No revenge near floor | danger ≥ 0.45 → no new open |
| Soft days: few trades | thrash cap 4 if target ≤ 1.5 |
| Early thrash BUY while Mark HOLD | BC HOLD corrections (HITL auto) |
| Chart truth | `MARK HERE!.lnk` + `HITL__latest.md` |

**Not** slave every bar to noisy online teacher (that cut clear 9→7).  
**Yes** same laws + soul-plan labels + force gate.

---

## Scores (current embryo)

| Meter | Result |
|-------|--------|
| BC match / dir_match | **0.84 / 0.84** (was 0.52 / 0.78) |
| Mark soul plans 10d | **10/10** breach 0 |
| Policy 10d (force-gate) | **8/10** breach **0** |
| Fixed miss | **2026-03-27** now clears |
| Remaining miss | 3/24, 3/26 (soft undersize / wait) |
| **50d forward streak** | **max 11** · awards 33/50 · **breach 0** |
| Pass 10-streak & breach0 | **YES** on 50d window |

Curriculum forward depth ≈ 40–50 days after practice — extend data for true 100–200d.

---

## Loop (Mark + Fable forever)

```
1. MARK HERE looks at HITL / miss days on chart
2. Corrections → HOLD / side labels
3. mark_consistency_loop.py (soul plans + corrections + BC)
4. Score 10d + long streak
5. Repeat until 100+ day breach0 streak holds
```

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/mark_consistency_loop.py --epochs 40 --max-train-days 30 --streak-days 100
python lineages/adaptive_rl_brain_7_31_26/mark_chart_hitl.py --seed 7 --start-idx 40 --full-obs
# MARK HERE!.lnk → HITL__latest.md
```

Ckpt: `checkpoints/mark_clone_full_obs_v1.pt`  
PROVEN: **untouched**

---

## Streak gap autopsy (2026-08-05) — no-opp vs Mark-would-take

**Report:** `AUTOPSY_GAPS__latest.md` · dials: `STREAK_REWARD_DIALS__latest.json`  
**ARMY goal:** `2026-08-05__goal__streak-gap-autopsy-rewards-only.md`

| Class | Count | Meaning |
|-------|------:|---------|
| AWARD | 29 | Policy cleared |
| **MARK_WOULD_TAKE** | **10** | Mark soul plan clears; policy missed (**learnable**) |
| **NO_OPPORTUNITY** | **1** | Hard target, no force path (2026-05-22) — do not thrash |
| Learnable fraction | **0.91** | Almost all gaps are Mark-would-take |

All 10 learnable gaps subclass: `policy_wrong_size_or_timing` (policy lost while Mark plan hit target).

**Rewards-only dials after autopsy:** streak_award_base 5 · per_prior 1.5 · break −7.5 · mark_would_take EOD −10 · soul_side +3 / misread −4  
**Not touched:** shell · PROVEN · entry physics.

---

## Next for 100s of days

1. Train on **MARK_WOULD_TAKE** soul labels + streak reward pack (rewards only)  
2. Re-score streak; re-autopsy (Mark-would-take count should fall)  
3. Mark HITL on stubborn miss days  
4. More curriculum days / longer forward for true 100d window  
5. Keep **breach 0** while climbing award_pct  
