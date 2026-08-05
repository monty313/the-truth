# LEARNING LOG — 2x award streak goal (the-truth T3)

**Goal:** max_award_streak >= 2x baseline (8 -> 16), award_pct >= 72.5, breach 0  
**Date:** 2026-08-05  
**Track:** lineages/adaptive_rl_brain_7_31_26  

## Baseline (frozen)

| Meter | Value |
|-------|------:|
| max_award_streak | 8 |
| award_pct | 72.5 |
| n_award / n_days | 29 / 40 |
| n_breach | 0 |
| recipe | forward after practice_50, seed=42, full_obs mark_align, soft_bias=false |

## Autopsy lesson (before training)

Of 11 non-award gaps on an earlier 40d pack: **10 MARK_WOULD_TAKE**, 1 NO_OPPORTUNITY.  
Learnable fraction **0.91**. Subclass: policy_wrong_size_or_timing.

## What failed (training-only paths)

1. **Reward-weighted BC + REINFORCE** (loop_2x): PRE streak 11, POST stayed 11. Did not convert miss days.
2. **Heavy DAgger on miss days (oversample 15x, no KL):** awards collapsed 33->22, streak 11->3. Overfitting miss craft destroyed award-day behavior.
3. **Surgical Mark plan BC with KL anchor:** entry_hit on *plan path* reached ~0.92 but day still missed. Plan-path match != runtime path.
4. **Surgical + DAgger with KL:** POST streak 9 / awards 30 — regressed; rejected correctly.

## Root cause discovery (the real issue)

On miss day **2026-04-06** Mark plan entries: SELL@745, SELL@770 (clears under dynamic size).

Runtime policy at 745: **pol=S, rec=S, mark_plan=S** but **gated=H**.

Diagnosis at t=745:
- force_dir=0.0, m_conf=0.75, reg=**flat_undefined**, danger=0
- Gate rule was: `if "flat" in reg: return HOLD`
- Substring match on **flat_undefined** false-positive blocked a Mark-agreed SELL

So the bot already wanted Mark's action; the **force gate erased it**.  
BC/DAgger could not fix a post-policy hard block.

## Fix applied

File: mark_aligned_decode.py
1. Replace bare `"flat" in reg` with _is_dead_regime() that does NOT treat lat_undefined as sit-out.
2. When online Mark ecommended == policy directional: only capital danger may block.
3. Pass ecommended=mark_action into force gate.

Unit test: 	ests/test_streak_reward_path.py::test_force_gate_allows_mark_agreed_sell_in_flat_undefined

## Principle learned

| Wrong assumption | Truth |
|------------------|--------|
| Miss days = policy doesn't know Mark craft | Often policy knows; **decode gate** kills the trade |
| More BC always helps | Over-BC on miss days can destroy award days |
| Plan-path dir_match high => day clears | Runtime path + gate differ from expert path |
| "flat" substring safe | Regime names like flat_undefined are traps |

## Next measurements (this session)

- Dual score after gate fix (same recipe seed=42, 40d)
- If streak still <16: light reward-weighted BC *with* fixed gate (not heavy miss oversample)
- Log every cycle: pre/post streak, award_pct, miss list, breach

## Forbidden still held

- PROVEN untouched
- Shell heat/bank/breach floors not loosened
- No trail+cushion+scale-in package


## Gate-fix dual score (same recipe, restored embryo)

| Meter | Pre-fix (best) | Post-gate-fix |
|-------|----------------:|---------------:|
| max_award_streak | 11 | **12** |
| n_award / 40 | 33 | 31 |
| award_pct | 82.5 | 77.5 |
| n_breach | 0 | **0** |
| vs baseline 8 | 1.375x | 1.5x |
| gate_pass 2x | no | **no** (need 16) |

Longest streak dates post-fix: 2026-05-06 … 2026-05-21 (12 trading days).

### 2026-04-06 after gate fix (detailed)

- Mark plan: S@745, S@770 → clears +1.52%
- Gate now allows S@745 (was H under flat_undefined)
- Policy still **misses add @770** (pol=H, mark=S)
- Policy later thrashs / danger-blocks → pnl **-1.63%**, n_entries 3, still miss
- Lesson: gate fix necessary but not sufficient; need second-entry + anti-thrash via reward-weighted labels

### Miss list after gate fix

2026-04-06, 04-10, 04-13, 04-14, 04-15 (pnl+1.03 but target not hit), 04-22, 05-05, 05-22, 05-25

### Award_pct dip note

Allowing more Mark-agreed opens can **hurt some former award days** (33→31).  
2x gate also requires award_pct >= baseline 72.5 (still OK at 77.5).  
Training must lift miss days without collapsing awards (KL + award self-imitate).

## Cycle log (continuing)

