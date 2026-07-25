# Consistency Climb Plan — clear rate ↑, breach stays 0%

**Date:** 2026-07-25  
**Baseline brain:** `PROVEN_SPRINT_row04_clear24_2026-07-20`  
**Measure:** `python scripts/prove_it.py <brain> 3.0 3.5`  
**Hard gate:** breach **must stay 0%**. Any candidate that breaches is rejected.

---

## Current scoreboard

| Metric | Value | Meaning |
|--------|-------|---------|
| frame_dim | **1820** | PROVEN-compatible (signal slots OFF) |
| Clear rate | **21%** (≈19/90) | Hit target without floor breach |
| Breach rate | **0%** | Floor holds — do not trade this away |
| Longest clear streak | **2 days** | Chain repair is the growth edge |
| Avg day | **+0.17%** | Slightly green; median still red (−0.40%) |
| Green days | **46%** | Making money ≠ clearing 3% |
| Physics bound | **90/90 winnable** | Flea-jar: lid is off; problem is policy |

**Historical peak on this lineage:** ~27% clear, row 4 (pre TF-set realign).  
**Migration floor after SETS lock:** ~21% — climb from here, then beat 27%.

### Fresh IRAC (2026-07-25, 6 curriculum days)

| Day | pull flags | policy_hold on setup | high_miss pull |
|-----|------------|----------------------|----------------|
| 2026-01-20 | 0 | 0 | 0 |
| 2026-02-10 | 6 | 50 | 6 |
| 2026-03-03 | 4 | 76 | 4 |
| 2026-03-24 | 12 | 107 | 12 |
| 2026-04-15 | 16 | 103 | 16 |
| 2026-05-06 | 16 | 68 | 16 |
| **TOTAL** | | **404** | **54** |

**Class: Policy.** Setup visible, policy holds. High-miss pull = 54 (ghosts would have helped).  
Artifact: `artifacts/llm_curriculum/irac_PROVEN_SPRINT_row04_clear24_2026-07-20.json`

---

## Disease (do not misdiagnose)

| Class | Evidence | Wrong cure |
|-------|----------|------------|
| **Policy** (primary) | Setup visible (pull/cont under HTF trend) → policy **holds** | More weak ~55% signal stacks |
| Not Perception | Mind Probe shows flags; masks rarely veto | Rewriting indicators first |
| Not “impossible days” | Swing-capture bound clears target on all 90 | Zero-weighting quiet days |

**Cure order (doctrine):** rewards / practice → periods → logic last.

**Active incentive already in place:** `w_pullback_with_htf = 0.25` (was 0.02).  
Brain still hesitates → needs **frontier GPU practice under current rewards**, not only another YAML bump.

---

## North-star targets (phased)

| Phase | Clear % | Row (streak) | Breach | How we know |
|-------|---------|--------------|--------|-------------|
| **P0** Unblocked | 21% | 2 | 0% | ✅ prove_it 2026-07-25 |
| **P1** Recover peak | ≥27% | ≥4 | 0% | Match pre-realign PROVEN |
| **P2** Climb | ≥35% | ≥6 | 0% | Sprint + gated skill |
| **P3** Consistency | ≥50% | ≥10 | 0% | Meta-tuner multi-pair |
| **P4** Stretch | ≥70% | stretch | 0% | Only after P3 stable |

Do not skip phases by expanding obs or wiping weights.

---

## Workstream A — Diagnose (every climb cycle)

**Goal:** Confirm Policy class with fresh counts before changing anything.

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
python scripts\self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```

**Outputs to read:**
- `artifacts/llm_curriculum/irac_*.json` — `sum_policy_hold_on_setup`, class
- `artifacts/self_heal_epochs/epoch_*.json` — baseline clear/breach, proposal
- Mind Probe: policy_hold vs mask_veto vs no_ltf_setup

**Accept only if:** class remains Policy (or IRAC says otherwise with evidence).  
If Perception dominates, fix features/flags — not more did_nothing penalty.

---

## Workstream B — Frontier practice (main lever for clear %)

**Goal:** Teach the policy to **take** bread-and-butter when visible; repair day-after-day chains.

```powershell
# Short probe (CPU ok; GPU better for longer)
python scripts\consistency_sprint.py --minutes 60 --envs 64

# Host climb (preferred when GPU available)
python scripts\consistency_sprint.py --minutes 600 --envs 256
```

**Why this works:**
- Every day practices (no flea-jar zero-weight)
- Weights: cleared = retention; near-miss = heavy; chain-break day = 5×
- Ratchet: never lose record brain; snap back on clear regression

**After every sprint:**
```powershell
python scripts\prove_it.py <new_sprint_or_best_brain> 3.0 3.5
```
- Accept brain only if clear ≥ previous **and** breach = 0
- Copy winner to `artifacts/checkpoints/` as PROVEN-style name if it beats baseline
- Append win line to `doctrine/SUCCESS_LEDGER.md`

**Self-heal with train (gated skill + optional reward nudge):**
```powershell
python scripts\self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 `
  --days 12 --sprint-minutes 120 --auto-accept-skill --apply-reward-nudge
```
- Nudge only if IRAC Policy + high policy_hold; `w_pullback_with_htf` stays in [0.05, 0.50]
- Reject if clear falls or breach > 0

---

## Workstream C — Reward shaping (small steps, evidence only)

| Knob | Now | When to touch | Bound |
|------|-----|---------------|-------|
| `w_pullback_with_htf` | 0.25 | Still high policy_hold after practice | 0.05–0.50 (self_heal); meta up to 1.0 |
| `w_did_nothing` | −6.0 | Idle days dominate ghosts | meta_tuner bounds |
| `w_day_goal_hit` | 2.0 | Clears late / almost-clear | after sprint plateau |
| `w_streak_per_day` | 0.15 | Clear up but row stuck | after chain weights proven |
| `w_death_penalty` | −10.0 | **Do not weaken** while climbing clear | sacred |

**Rule:** one knob per gated cycle. Always prove_it before/after.

---

## Workstream D — Meta-tuner (any target/risk consistency)

**Only after** a warm-start brain holds breach 0 and clear ≥ P1.

```powershell
python scripts\meta_train.py --minutes 600
python scripts\prove_it.py <brain> 3.0 3.5
python scripts\prove_it.py <brain> 2.5 2.5
```

Target/risk stay **runtime inputs** — never retrain from scratch only to change them.

---

## Workstream E — Signal slots (later; requires NEW brain)

| Mode | `include_signal_agent_slots` | Obs | Use |
|------|------------------------------|-----|-----|
| **Now** | `false` | 1820 | All PROVEN_* scoring |
| **Later** | `true` | ~6820 | Fresh train; agreement 80–83 as engage cues |

**Do not** load PROVEN into expanded obs.  
When ON: warm-start architecture carefully or train new; gate on prove_it.

Agreement evidence (~70–81% @10 bars) is **real** — but it is observation fuel for a **new** brain, not a free upgrade for 1820-dim checkpoints.

---

## What we will NOT do

1. Full retrain from scratch to “fix” clear % (catastrophic forgetting risk).
2. Turn signal slots ON and force old checkpoints.
3. Accept any brain with breach > 0%.
4. Zero-weight days as “impossible.”
5. Delete PERFORMANCE_IS_POSSIBLE*, SUCCESS_LEDGER, flea-jar, PROVEN checkpoints.
6. Stack more low-precision indicators before fixing policy_hold.

---

## Recommended sequence (this week)

```
[Done] Restore engine + gate signals + prove_it baseline (21% / 0% / row 2)
  ↓
[1] Finish self_heal diagnose (--days 12) → record IRAC policy_hold counts
  ↓
[2] consistency_sprint --minutes 60–120 (smoke) → prove_it
  ↓
[3] If clear↑ breach0: keep brain; else ratchet snap
  ↓
[4] Longer sprint (hours on GPU) toward P1 ≥27% clear / row≥4
  ↓
[5] Optional: self_heal with --apply-reward-nudge only if policy_hold still huge
  ↓
[6] Meta-tuner once P1 held on 3.0/3.5 and spot-check 2.5/2.5
  ↓
[7] Plan expanded-obs train only after 1820-dim climb plateaus
```

---

## Scorecard template (paste after each prove_it)

```
date:
brain:
target/risk:
frame_dim:
clear%:
breach%:
row:
avg%/median%:
green%:
notes: (policy_hold count / reward change / sprint minutes)
decision: ACCEPT | REJECT | RATCHET_BACK
```

---

## One-line strategy

> **Practice the hesitation away under a floor-safe reward, measure only with prove_it, never trade breach for clear, climb 21→27→35→50 without expanding the jar until the 1820-dim policy actually takes the setups it already sees.**
