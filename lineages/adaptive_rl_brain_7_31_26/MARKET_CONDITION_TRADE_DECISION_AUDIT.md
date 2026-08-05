# Market Condition and Trade-Decision Audit

**Track:** multi-pair tutor — heuristic decode + equity shell only  
**Not:** Channel1 RL sandbox · PROVEN champion  
**Language rule:** standard indicator and market-structure terms only (no “gravity / orbit” language)  
**Source brief:** `todo_7_31_26.md` after **part 4000**  
**Date:** 2026-07-31  

**Proof anchors:**  
`checkpoints/ten_pair_score_all.json` · `ten_pair_score_forward.json` · `multi_pair_dials.json` · `equity_day.py` · `day_runner.py` · `perception/*` · `honest_gate/`  

**Shell (unchanged):** heat / refuse-open · floor-scaled sizing · every-bar stop & breach marks · bank at target · breach death · one signal flat and in-trade  

**Honesty:** historical dial search could use all days → multi-pair scores = **IN_SAMPLE_CLAIM** (not pure unseen). Eligible real days = **90** → **100-day conclusion NOT YET MEASURABLE**.

---

## 1. Executive summary (plain language)

1. **Breach is not the main problem** on measured windows: practice and forward both show **0% breach** with frozen dials and heuristic decode.  

2. **Hard targets (target% ≥ 2.5) clear less on forward than on practice.**  
   - Pair 2.5/3.5: practice **62%** clear (31/50) → forward **42.5%** (17/40), Δ **−19.5 pp**  
   - Pair 3.0/3.5: practice **56%** clear (28/50) → forward **30%** (12/40), Δ **−26 pp**  
   Soft targets (≤1.5) often **improve** slightly on forward (e.g. 1.0/2.0: 82% → 87.5%).  

3. **High entry counts on hard-target misses are an observed co-occurrence, not a proven cause.**  
   Forward 3.0/3.5 misses: mean **10.5** entries; **18/28** days had ≥10 entries. That can come from trend conflict, range, failed pullback, weak CCI/RSI, reverse churn, insufficient remaining movement, or target hardness — **we cannot tell from current score rows**.  

4. **Score artifacts only log:** `date, pnl_pct, min_eq_pct, cleared, breached, n_entries, banked`.  
   They do **not** log higher/lower trend, alignment, CCI/RSI state, pullback quality, setup type, session, range, spread, heat, reverses, or entry/exit reasons.  

5. **Many useful fields are already calculated** in perception / shell (pullback, scale conflict, velocity, confluence, trade tags, session phase, heat) but are **ignored by the heuristic claim path** and/or **not written to scores**.  

6. **Immediate goal:** one shared audit schema for practice and forward (decision bar + day), then answer the ten questions with sample sizes. **Do not** add new indicators, change the shell, or fit on forward until that log exists.

---

## 2. Audit schema (same columns for practice and forward)

Use these field groups on **every decision bar** and roll up to **every completed day**.  
Values must use only information available **at that bar** (no future bars).

### A. Trend and alignment

| Field | Allowed values / type | Source (existing or derived) |
|-------|----------------------|------------------------------|
| `htf_trend_dir` | bullish / bearish / neutral | Higher-TF stack direction (`perceive` higher) |
| `htf_trend_strength` | weak / normal / strong | Map from velocity (1 / 2 / 3 group agree) |
| `htf_slope` | rising / falling / flat | **Derive** from confirmation MA/channel slope or successive HTF closes (not stored today as a named field) |
| `ltf_trend_dir` | bullish / bearish / neutral | Lower/entry TF direction |
| `alignment` | aligned / conflicting / neutral | HTF vs LTF clear directions |
| `channel_position` | above / below / crossing / inside_ranging | Channel group flags (close vs SMA high/low shifted) |
| `channel_slope` | rising / falling / flat | **Derive** from channel lines over recent bars |

### B. Momentum and oscillator state

| Field | Allowed values / type | Source |
|-------|----------------------|--------|
| `cci_state` | above_zero / below_zero / crossing_zero / extended_high / extended_low / weakening / strengthening | CCI 30/100 vs ref (+ thresholds to define extended) |
| `rsi_state` | above_50 / below_50 / crossing_50 / extended_high / extended_low / weakening / strengthening | RSI 5/14 vs ref |
| `momentum_velocity` | strengthening / weakening / flat / conflicting | Set velocity + change vs prior decision bar |
| `indicator_agreement` | agree / disagree / unclear | Trend direction vs CCI/RSI vote |
| `divergence_flag` | **NOT AVAILABLE** | Not calculated in current code |

### C. Market condition and setup type

| Field | Allowed values / type | Source |
|-------|----------------------|--------|
| `setup_type` | trend_continuation / pullback_continuation / mean_reversion / range_consolidation / transition_uncertain / no_valid_setup | **Classify** from HTF/LTF/pullback/velocity (rules must be fixed in meaning version) |
| `pullback_state` | shallow_healthy / deep / failed / reversal_risk / n_a | Pullback flag + depth vs HTF (depth rules to pin) |
| `scale_conflict` | yes / no | Existing structure flag |
| `trade_tag_primary` | MINDLESS / WITH_VECTOR / QUALIFIED_MACRO / QUALIFIED_MICRO | `classify_trade` |
| `trade_tags_support` | list | Reasons / secondary flags |

### D. Opportunity and trading conditions

| Field | Allowed values / type | Source |
|-------|----------------------|--------|
| `target_remaining_pct` | float | `target% − equity%` (clip) |
| `risk_remaining_pct` | float | Distance from equity% to `−risk%` |
| `heat_ok` | bool | Residual heat allows open |
| `dist_to_floor_pct` | float | Same distance family as shell heat |
| `session_phase` | float [0,1] | Already in obs slot 31 |
| `time_remaining_frac` | float | `1 − session_phase` |
| `range_so_far_vs_typical` | float or low/normal/high | Day high−low so far vs practice median at same phase |
| `recent_move_used_pct` | float | Realized path / target so far |
| `remaining_opportunity_est` | float or low/med/high | Function of remaining target, time, recent range (**causal only**) |
| `spread_condition` | normal / wide | Bar spread vs day/practice median |
| `gap_flag` | bool | Open vs prior close |
| `thin_liquidity_flag` | bool | Low bar count / wide spread heuristic |
| `abnormal_range_flag` | bool | Range vs typical |

### E. Trade management and churn

| Field | Allowed values / type | Source |
|-------|----------------------|--------|
| `entry_number` | int | Count opens today |
| `reversal_number` | int | Count direction flips |
| `entry_direction` | buy / sell | Side |
| `holding_bars` | int | Bars in current trade |
| `flat_bars_so_far` | int | Bars flat |
| `consecutive_failed_entries` | int | Opens that stopped without progress (define fail) |
| `entry_reason` | short code | e.g. htf_bull / ltf_fallback / reverse_opposite |
| `exit_reason` | stop / bank / reverse / eod / breach | Shell events |
| `reversed_while_htf_unchanged` | bool | Reverse while `htf_trend_dir` same as prior entry |
| `entered_near_oscillator_neutral` | bool | CCI/RSI near zero/50 at entry |
| **Day final** | | |
| `cleared` / `breached` / `banked` | bool | Existing definitions |
| `pnl_pct` / `min_eq_pct` / `dist_to_target_pct` | float | Existing + `target − pnl` |

**Window fields (every score file):** `split` = practice | forward · `target_pct` · `risk_pct` · meaning_hash · dials_hash · decode · seed · honesty_label  

**Minimum sample rule:** any rate claim needs **n** stated; if n &lt; 8 for a cell → **INSUFFICIENT EVIDENCE** (do not call the setup “weak”).

---

## 3. Current senses table

| Input name | What it measures | Used by heuristic claim path | Logged in score artifact | Action |
|------------|------------------|------------------------------|---------------------------|--------|
| Higher-TF trend direction | Confirmation-stack direction | **Yes** (primary) | **No** | **Keep** · **log first** |
| Lower-TF trend direction | Entry TF fallback | **Yes** (if HTF neutral) | **No** | **Keep** · **log first** |
| HTF / LTF alignment | Conflict vs agree | **Partly** (implicit only) | **No** | **Log first** (no new indicator) |
| Official set dir / velocity / score ×4 | Multi-scale trend & agreement | **Partly** (collapsed to one higher) | **No** | **Keep** · **log first** |
| Sub-set dir / velocity / score ×5 | Lower-scale stacks | **Partly** (via lower) | **No** | **Keep** · **log first** |
| Pullback flag | Opposite LTF vs clear HTF | **No** | **No** | **Log first** · **test later** for attention |
| Scale conflict | Major vs minor clear opposite | **No** | **No** | **Log first** · **test later** |
| CCI 30/100 vs shifted SMA | Oscillator vs ref | **Partly** (inside confluence only) | **No** | **Keep** · **log first** (state labels) |
| RSI 5/14 vs shifted SMA | Oscillator vs ref | **Partly** (inside confluence only) | **No** | **Keep** · **log first** |
| Channel close vs SMA high/low | Price vs channel | **Partly** (inside confluence) | **No** | **Keep** · **log first** |
| Trade tags (MINDLESS / WITH_VECTOR / QUALIFIED_*) | Setup class | **No** (claim path) | **No** | **Log first** · **test later** |
| progress_to_goal | Equity vs target | **No** for direction | **No** | **Keep** (obs) · **log** as target_remaining |
| danger | Equity vs floor | **No** for direction | **No** | **Keep** · **log** as risk_remaining |
| session_phase | Fraction of day elapsed | **No** | **No** | **Log first** · **test later** |
| Heat / floor_scale / risk_use_frac | Risk budget before open | **Yes** (shell) | **No** | **Keep shell** · **log** |
| Equity% / min_eq / banked / breached | Outcome & path | Shell yes | **Yes** (partial) | **Keep** |
| n_entries | Activity count | Indirect | **Yes** | **Keep** · add reverses / holding |
| ATR-like stop distance | Vol proxy for size/stop | **Yes** (shell) | **No** | **Keep** · **log** range flags from OHLC |
| Bar spread | Cost / liquidity proxy | Costs path | **No** | **Log first** |
| Divergence (CCI/RSI) | Classic divergence | **No** | **No** | **NOT AVAILABLE** — do not invent until proven need |
| Channel1 MLP weights | Attention over 32 slots | **No** (claim = heuristic) | N/A | Separate track; do not confuse with claim |

---

## 4. Practice-versus-forward evidence table

**Settings:** same dials (`risk_use_frac=0.35`, `stop_atr_mult=2.0`, `per_trade_cap_pct=0.25`), decode **heuristic**, data `XAUUSD_curriculum_2026.csv`, 90 days.  
**Split:** practice = first 50 calendar days (2026-01-20→2026-03-30); forward = next 40 (2026-03-31→2026-05-26).  
**Source:** `ten_pair_score_all.json` day_rows split (matches forward JSON clears for pair 10).

### Soft / mid pairs (sample sizes full)

| Pair T/R | Practice clear% (n) | Forward clear% (n) | Δ clear pp | Breach P/F | Miss mean entries F |
|----------|--------------------:|-------------------:|-----------:|------------|--------------------:|
| 1.0/2.0 | 82.0% (50) | 87.5% (40) | +5.5 | 0/0 | (soft misses few) |
| 1.0/2.5 | 82.0% (50) | 87.5% (40) | +5.5 | 0/0 | |
| 1.5/2.0 | 70.0% (50) | 75.0% (40) | +5.0 | 0/0 | |
| 1.5/2.5 | 72.0% (50) | 85.0% (40) | +13.0 | 0/0 | |
| 1.5/3.0 | 72.0% (50) | 85.0% (40) | +13.0 | 0/0 | |
| 2.0/2.5 | 66.0% (50) | 67.5% (40) | +1.5 | 0/0 | |
| 2.0/3.0 | 66.0% (50) | 67.5% (40) | +1.5 | 0/0 | |
| 2.0/3.5 | 66.0% (50) | 67.5% (40) | +1.5 | 0/0 | |

### Hard targets (focus: target ≥ 2.5)

| Pair T/R | Practice clear | Forward clear | Δ | Breach | Miss n P/F | Miss mean entries P/F | Miss mean pnl F | Miss ≥10 entries F |
|----------|---------------:|--------------:|--:|--------|------------|----------------------:|----------------:|-------------------:|
| 2.5/3.5 | 31/50 (**62%**) | 17/40 (**42.5%**) | **−19.5** | 0/0 | 19/23 | 12.8 / 10.8 | +0.64 | 15/23 |
| 3.0/3.5 | 28/50 (**56%**) | 12/40 (**30%**) | **−26.0** | 0/0 | 22/28 | 12.4 / 10.5 | +0.82 | 18/28 |

**What this supports**

- Hard-target **clear rate** drops on forward; soft targets do not.  
- Forward hard misses often still **positive PnL** (not only floor-adjacent disasters).  
- High entry counts appear often on hard misses — **correlation only**.

**What this does not support (yet)**

- Cause = “thrashing” (must test conflict / range / pullback / CCI-RSI / reverse-while-HTF-stable / opportunity).  
- Any tag or CCI/RSI breakdown (fields not logged → **INSUFFICIENT EVIDENCE** for Q1–Q7 sensor answers).

---

## 5. Answers to the ten questions

Run **separately** for practice and forward, focus **target ≥ 2.5**.  
Where logs are missing, answer is **unknown** + smallest measurement.

| # | Question | Answer | Sample size / note |
|---|----------|--------|-------------------|
| **1** | On hard-target misses, how often did HTF and LTF directions **conflict**? | **Unknown.** Alignment not logged. | Need audit log on miss days: n_miss practice 19+22, forward 23+28 for hard pairs. |
| **2** | How often were misses in **range/consolidation or transition** vs **aligned trend**? | **Unknown.** `setup_type` not logged. | Same miss-day set; require setup_type fixed rules. |
| **3** | How often did **CCI/RSI and momentum agree** with entry direction vs conflict/weaken? | **Unknown.** Oscillator **state** not logged; only used inside confluence before collapse. | Log cci_state, rsi_state, momentum_velocity at each entry. |
| **4** | How often did **pullback continuation** become **reversal or range**? | **Unknown.** Pullback flag exists in code but is **not used** by heuristic and **not logged**. | Log pullback_state path over day on hard misses. |
| **5** | How often did **extra entries** come from **reversals while HTF direction unchanged**? | **Unknown** as a rate. We only know **n_entries** is often high on hard misses (e.g. forward 3.0: **18/28** misses had ≥10 entries). That does **not** prove reverse-while-HTF-stable. | Log `reversed_while_htf_unchanged` + reversal_number. |
| **6** | Does **remaining-opportunity** estimate predict a miss **before** repeated entries? | **Unknown.** Field not computed/logged. | On practice hard pairs only: compute estimate at first 1–2 entries; AUC or clear-rate split; n≥30 days preferred. |
| **7** | Which **setup type or trade tag** has largest clear-rate gap practice vs forward? | **Unknown.** Tags not in score JSON. | After logging: Δ clear% by tag, min n=8 per cell else INSUFFICIENT EVIDENCE. |
| **8** | Which inputs are **calculated but ignored** by the heuristic claim path? | **Yes — list:** pullback flag; scale conflict; velocity/confluence scores as decision gates; trade tags; session_phase; progress_to_goal/danger for direction; full official/sub vectors (collapsed to HTF then LTF fallback); Channel1 MLP. **Used:** HTF direction, LTF if HTF neutral, opposite-side reverse, shell heat/size/stop/bank/breach. | Code: `equity_day.recommended_action` + `day_runner.structure_action_at`. |
| **9** | Which inputs are **duplicate, ambiguous, or not useful**? | **Ambiguous:** single “higher” collapse loses which set/TF drove the vote. **Duplicate risk:** official + sub packs both encode multi-TF direction. **Not useful for claim today:** Channel1 weights (not claim decode); divergence (**NOT AVAILABLE**). **Usefulness of pullback/tags:** unknown until logged. | Do not remove until log + practice test. |
| **10** | What **single missing measurement** would reduce the most uncertainty? | **Per-day / per-decision audit row** with HTF/LTF direction, alignment, pullback, scale_conflict, trade_tag, entry/reversal counts, reverse-while-HTF-unchanged, target/risk remaining, and setup_type — **same schema practice and forward**. That single logging layer answers Q1–Q7. | Without it, all sensor causes stay unknown. |

---

## 6. Ranked wish list (max 8)

### 1. Shared practice/forward decision + day audit log  
| | |
|--|--|
| **Type** | Evaluation |
| **Exact blind spot** | Cannot attribute hard-target misses to trend conflict, range, CCI/RSI, pullback, or management |
| **Evidence** | Score fields only 7 columns; hard clear drop measured (n=40 forward days per hard pair) but causes unknown |
| **Existing inputs used** | `perceive()`, shell equity/heat, OHLC, spread |
| **New calculation or record needed** | Writer for schema A–E above; stamp meaning_hash |
| **What must remain unchanged** | Shell laws; dials; heuristic until after practice-only tests |
| **Practice-only test** | Export 50 practice days × hard pairs; verify columns filled; rates by alignment with n≥8 |
| **Forward success condition** | One frozen forward export under same schema; no dial change mid-window |
| **Failure condition** | Still only pnl/entries in artifacts |
| **Priority** | **P0** |

### 2. Official window score metrics (rates, streaks, near-floor, honesty_label)  
| | |
|--|--|
| **Type** | Evaluation |
| **Exact blind spot** | Absolute “≥30 clears” confuses 90-day vs 40-day windows; no streak/near-floor in multi-pair JSON |
| **Evidence** | Forward pass bar mixed with claim bar historically; honest_gate score_rules exist |
| **Existing inputs used** | day_rows cleared/breached |
| **New calculation or record needed** | clear_rate, max/end streak, near_floor_rate, honesty_label per window |
| **What must remain unchanged** | Shell |
| **Practice-only test** | Report practice and forward separately for all 10 pairs |
| **Forward success condition** | Hard-pair drop visible as **rate** with n stated |
| **Failure condition** | Single pooled number only |
| **Priority** | **P0** |

### 3. Management counters: entries, reversals, reverse-while-HTF-unchanged  
| | |
|--|--|
| **Type** | Evaluation (feeds Attention/Policy later) |
| **Exact blind spot** | High n_entries observed; cause among reverse churn vs many new signals unknown |
| **Evidence** | Hard forward misses often ≥10 entries (e.g. 18/28 on 3.0/3.5) — **outcome, not cause** |
| **Existing inputs used** | side flips, htf_trend_dir |
| **New calculation or record needed** | reversal_number, reversed_while_htf_unchanged, holding_bars |
| **What must remain unchanged** | Shell (counting ≠ trail/scale-in package) |
| **Practice-only test** | Clear rate on high reverse-while-HTF-stable vs low (min n=8 each) |
| **Forward success condition** | After practice-only attention rule (if any), hard clear↑, breach=0 |
| **Failure condition** | No association of reverse flags with miss → do not “fix thrash” by policy guess |
| **Priority** | **P0** |

### 4. Label and log existing CCI/RSI/channel **state** (not new oscillators)  
| | |
|--|--|
| **Type** | Perception (labels on **existing** series) |
| **Exact blind spot** | Oscillators computed but not stored as state at decision time |
| **Evidence** | live_indicators compute CCI/RSI/channel; heuristic never logs them |
| **Existing inputs used** | CCI 30/100, RSI 5/14, channel SMAs |
| **New calculation or record needed** | cci_state, rsi_state, channel_position enums only |
| **What must remain unchanged** | Indicator periods (meaning version); shell |
| **Practice-only test** | P(clear \| indicator_agreement) on hard targets, n≥8 |
| **Forward success condition** | Frozen report only after practice selection |
| **Failure condition** | States uncorrelated with clear → do not gate on them |
| **Priority** | **P1** |

### 5. Log and optionally attend to pullback + scale_conflict + velocity (existing flags)  
| | |
|--|--|
| **Type** | Attention |
| **Exact blind spot** | Flags calculated, ignored by claim heuristic |
| **Evidence** | structure flags in obs slots 27–28; recommended_action ignores them |
| **Existing inputs used** | pullback, scale_conflict, velocity |
| **New calculation or record needed** | Log first; practice-only attention rule later |
| **What must remain unchanged** | Shell; no forward fit |
| **Practice-only test** | Clear% by pullback×alignment cells |
| **Forward success condition** | Hard clear↑, breach=0 after freeze |
| **Failure condition** | Breach↑ or no lift |
| **Priority** | **P1** (after logging) |

### 6. Causal remaining-opportunity estimate (from equity, time, range so far)  
| | |
|--|--|
| **Type** | Perception |
| **Exact blind spot** | Hard targets often miss with **positive** PnL — may be insufficient remaining move vs target |
| **Evidence** | Forward 3.0 misses mean pnl +0.82 (n=28); soft targets clear well |
| **Existing inputs used** | equity%, target, session_phase, day range so far |
| **New calculation or record needed** | remaining_opportunity_est at bar (no future) |
| **What must remain unchanged** | Shell bank/breach |
| **Practice-only test** | Does low estimate before entry 3+ predict miss on hard pairs? |
| **Forward success condition** | If used for attention, hard clear↑ breach=0 |
| **Failure condition** | No predictive power on practice → do not ship into policy |
| **Priority** | **P1** |

### 7. Trade-tag clear-rate table (practice vs forward)  
| | |
|--|--|
| **Type** | Evaluation |
| **Exact blind spot** | Q7 unanswerable; tags computed in DayRunner path not scored |
| **Evidence** | TradeTag enum exists; absent from ten_pair day_rows |
| **Existing inputs used** | classify_trade / tag |
| **New calculation or record needed** | tag on day or dominant tag; P(clear), n, CI |
| **What must remain unchanged** | Shell; min n=8 rule |
| **Practice-only test** | Table for hard pairs practice |
| **Forward success condition** | Comparable forward table; largest Δ tag identified only if n≥8 both sides |
| **Failure condition** | All cells INSUFFICIENT EVIDENCE → need more days, not new indicators |
| **Priority** | **P1** |

### 8. Extend eligible real days to ≥100 under same meaning_version  
| | |
|--|--|
| **Type** | Evaluation |
| **Exact blind spot** | 100-day consistency not measurable on n=90 |
| **Evidence** | load_calendar_days → 90; honest_gate NOT_YET_MEASURABLE |
| **Existing inputs used** | M1 loader, min_bars=900 |
| **New calculation or record needed** | More **real** XAUUSD days; same meaning hash |
| **What must remain unchanged** | Shell; no synthetic pad |
| **Practice-only test** | Re-freeze split; leak test overlap=0 |
| **Forward success condition** | 100-day protocol can be defined |
| **Failure condition** | Synthetic days or meaning drift |
| **Priority** | **P2** for the bar (P0 for any “100-day pass” claim) |

**Not on the list:** new indicator families, meta-learning, extra markets, live account, shell unlock, trail/cushion/scale-in.

---

## 7. Immediate next step (explainability first)

1. Implement the **audit log** (schema A–E) on the multi-pair **heuristic + shell** path only.  
2. Score **practice** and **forward** with frozen dials; fill Q1–Q7 with **n** or **INSUFFICIENT EVIDENCE**.  
3. Only then allow **practice-only** attention tests that re-weight **existing** inputs.  
4. Score forward **once** after freeze. Breach must stay **0**.  

Do **not** change the shell. Do **not** fit on forward. Do **not** claim thrashing is the root cause until Q1–Q5 are measured.

---

## 8. Reproduce numbers used here

```powershell
$env:PYTHONPATH = ".;code"
# Frozen artifacts (no retrain):
#   lineages/adaptive_rl_brain_7_31_26/checkpoints/ten_pair_score_all.json
#   lineages/adaptive_rl_brain_7_31_26/checkpoints/ten_pair_score_forward.json
# Practice = first 50 dates in claim day_rows; forward = remaining 40.
```

---

*Multi-pair tutor note: I bank at your target and die on your floor. Right now the scores show I protect the floor and struggle more on hard targets later in the sample — but without this audit log I cannot honestly say whether the market was ranging, the pullback failed, CCI/RSI disagreed, or I reversed while the higher-timeframe trend never changed.*
