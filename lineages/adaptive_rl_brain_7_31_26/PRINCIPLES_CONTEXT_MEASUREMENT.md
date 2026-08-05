# Principles → measurable context (multi-pair tutor)

> **Prefer the look-first copy:** [`00_PRINCIPLES_CONTEXT_MEASUREMENT.md`](00_PRINCIPLES_CONTEXT_MEASUREMENT.md)  
> Plans copy: `references/plans/MULTI_PAIR_TUTOR_PRINCIPLES_CONTEXT_MEASUREMENT.md`

**Purpose:** Turn nine **interpretation principles** into labels, logs, and tests — **not** into fixed entry rules, new indicators, new rewards, shell changes, or automatic policy changes.

**Track:** multi-pair tutor (heuristic decode + equity shell) only.  
**Shell locked:** heat/refuse-open, floor-scaled sizing, every-bar stop & breach marks, bank at target, breach termination, one-signal flat and in-trade — **unchanged**.

**Language:** standard market terms only (higher/lower timeframe trend, pullback, range, consolidation, transition, CCI/RSI, channel, session, heat, target remaining).

**Honesty:** practice = fit / select attention later; forward = one frozen score. Min sample for any “trust” claim: **n ≥ 8** per cell else **INSUFFICIENT EVIDENCE**. Prior multi-pair claim may be **IN_SAMPLE_CLAIM** if dials saw all days.

**Related:** `MARKET_CONDITION_TRADE_DECISION_AUDIT.md` · `PRINCIPLES_OF_SUCCESS.md` (shell KEEP laws) · `honest_gate/`

---

## Shared rules for every principle

| Rule | Detail |
|------|--------|
| **Meaning** | Plain language of the principle |
| **Existing inputs** | What the bot already computes or can derive without a new indicator family |
| **State labels** | Fixed, versioned enums (stamp `meaning_hash` when rules freeze) |
| **Evidence before decisions** | What must be measured before any attention change |
| **Practice-only test** | Fit / select on practice only |
| **Forward validation** | Score once frozen; breach must stay 0 |
| **Does NOT mean** | Common misreads to forbid |

**Default readiness:** almost everything starts **log-only first**. Nothing here is “ready to influence the heuristic now” until its practice test passes and forward is scored once.

**Minimum log row (decision + day rollup):**  
`date, t, split, target_pct, risk_pct, htf_trend_dir, htf_trend_strength, ltf_trend_dir, alignment, market_condition, pullback_state, cci_state, rsi_state, momentum_velocity, channel_position, channel_slope, agreement_profile, session_phase, target_remaining_pct, risk_remaining_pct, heat_ok, spread_condition, entry_number, reversal_number, reversed_while_htf_unchanged, action, entry_reason / exit_reason / no_trade_reason, cleared, breached, banked, pnl_pct, min_eq_pct`  
(Divergence: **NOT AVAILABLE** until separately proven needed.)

---

## PRINCIPLE 1 — Context comes before the signal

### 1. Plain-language meaning
A CCI move, RSI move, channel touch, or lower-timeframe direction is **not** a complete trade story by itself. Its quality depends on **higher-timeframe trend direction and strength**, overall **market condition**, and whether lower-timeframe movement **agrees or conflicts** with that context.

### 2. Existing inputs that represent it
| Group | Existing source |
|-------|-----------------|
| Higher-TF trend direction | `perceive` / structure higher stack (confirmation TFs) |
| Higher-TF strength | Velocity (weak/medium/strong group agreement) |
| Lower-TF direction | Entry / lower TF direction |
| Alignment | Compare HTF vs LTF clear directions |
| LTF “signal” proxy | Heuristic BUY/SELL from HTF then LTF fallback; reverse on opposite |
| Oscillator/channel context | CCI/RSI/channel inside confluence (values exist; **state labels not logged**) |
| Pullback flag | Structure pullback (exists; not used by claim heuristic) |

### 3. Measurable state labels
| Label field | Values |
|-------------|--------|
| `htf_trend_dir` | bullish / bearish / neutral |
| `htf_trend_strength` | weak / normal / strong |
| `ltf_trend_dir` | bullish / bearish / neutral |
| `alignment` | aligned / conflicting / neutral |
| `ltf_signal_vs_htf` | with_trend / countertrend / none |
| `context_confidence` | high / medium / low *(derived: e.g. high = aligned + htf not neutral + strength ≥ normal)* |

Version these mapping rules in the meaning manifest when frozen.

### 4. Evidence needed before it affects decisions
- Clear rate and breach rate by `ltf_signal_vs_htf` × `htf_trend_dir` × target bucket  
- Separate **countertrend** outcomes from **with_trend** (do not merge)  
- Sample size per cell ≥ 8  

### 5. Practice-only test
On practice, hard pairs (target ≥ 2.5) and soft pairs separately:  
P(clear | with_trend) vs P(clear | countertrend) vs P(clear | htf neutral).  
Report n and confidence interval. **No rule change** in this step — measurement only.

### 6. Forward validation condition
After any later practice-only attention change (if ever): freeze; score forward once.  
**Pass:** clear rate not worse (or improved on target bucket under test) and **breach count = 0**.  
**Fail:** breach > 0 or large unexplained clear drop without degraded-condition report.

### 7. What this principle does NOT mean
- **Not** “never trade countertrend.”  
- **Not** an automatic filter to drop all conflicting LTF signals.  
- **Not** a new entry system; context **labels** signal quality for learning later.  
- **Not** permission to ignore the risk shell.

### Readiness
**Log-only first.** Needs more labeled decision rows. **Not ready** to influence the heuristic now.

---

## PRINCIPLE 2 — Trend, range, and transition are different conditions

### 1. Plain-language meaning
The same CCI/RSI/channel/LTF pattern can behave differently in a **trend**, a **range/consolidation**, or a **transition**. One standard for all conditions misleads measurement and later attention.

### 2. Existing inputs that represent it
| Condition idea | Inputs (existing / derived only) |
|----------------|----------------------------------|
| Trend | HTF/LTF mostly aligned; channel slope directional; momentum velocity not conflicting |
| Range / consolidation | Frequent direction flips over a window; channel flat or inside; CCI/RSI often near neutral / re-crossing |
| Transition | HTF strength weakening; alignment conflict; velocity weakening; channel slope flip vs prior bars |

No new indicator family — only **fixed rules** over current series and structure flags.

### 3. Measurable state labels
| Label | Values (versioned) |
|-------|---------------------|
| `market_condition` | trend / range_consolidation / transition / uncertain |

**Draft fixed definition (to freeze after practice calibration of thresholds, not forward fit):**

| Value | Working definition (must be coded deterministically) |
|-------|------------------------------------------------------|
| `trend` | `alignment == aligned` AND `htf_trend_dir != neutral` AND `htf_trend_strength ∈ {normal, strong}` AND `channel_slope` matches HTF direction (or not flat) |
| `range_consolidation` | (`alignment == conflicting` OR both dirs flip often over last K decisions) AND (`channel_slope == flat` OR `channel_position == inside_ranging`) AND oscillators often near neutral |
| `transition` | HTF strength weak or falling vs prior bar OR velocity weakening while alignment conflicts OR channel_slope changed vs prior decision |
| `uncertain` | None of the above cleanly |

Exact K and “often near neutral” thresholds: freeze in meaning_version after practice-only stability check (not optimized on forward).

### 4. Evidence needed before it affects decisions
- Clear rate by `market_condition` × target bucket on practice and (frozen) forward  
- Show that ranking of setup types **differs** by condition (interaction), not one global ranking  

### 5. Practice-only test
Build contingency: clear% for each `market_condition` on practice (all pairs and hard-only).  
If any cell n &lt; 8 → INSUFFICIENT EVIDENCE.  
Compare whether “aligned LTF signal” clear% differs trend vs range (interaction test).

### 6. Forward validation condition
Label distribution + clear% by condition on forward under frozen labeler.  
**Success:** labels apply without code change; breach = 0; document condition mix shift if clear drops.  
**Fail:** labeler non-deterministic or uses future bars.

### 7. What this principle does NOT mean
- **Not** “only trade trends.”  
- **Not** three separate bots or three shells.  
- **Not** a new indicator (e.g. ADX) unless later audit proves existing fields cannot separate conditions.  
- **Not** zero-weighting days as “impossible.”

### Readiness
**Log-only first** with versioned draft labels. **Needs more data** (labeled bars) before attention. **Not ready** to influence heuristic now.

---

## PRINCIPLE 3 — A pullback is not automatically a reversal

### 1. Plain-language meaning
A **pullback** is a temporary move against the prevailing higher-timeframe trend. A **reversal** is a change in that prevailing trend. Mixing them causes wrong management and wrong learning.

### 2. Existing inputs that represent it
| Input | Role |
|-------|------|
| Structure `pullback` flag | Higher clear + lower opposite clear |
| `htf_trend_dir` / strength | Prevailing trend still present? |
| `ltf_trend_dir` / velocity | Countertrend strength |
| `scale_conflict` | Multi-scale conflict |
| Channel position | Location of price vs structure |
| Later outcome (for audit only) | Did HTF flip after entry? — **for offline audit, not for live decision** |

### 3. Measurable state labels
| Label | Values |
|-------|--------|
| `pullback_state` | no_pullback / shallow_healthy / deep / failed / reversal_risk |

**Draft mapping (version later):**  
- `no_pullback`: structure pullback false  
- `shallow_healthy`: pullback true, HTF dir unchanged, LTF counter strength weak, scale_conflict false  
- `deep`: pullback true, larger adverse excursion or stronger LTF counter (threshold versioned)  
- `failed`: pullback was true, then continued against HTF with velocity strengthening counter  
- `reversal_risk`: HTF strength weak/transition + conflict rising  

**Critical:** current binary pullback flag is a **candidate input**, not ground truth. Offline audit compares flag vs subsequent HTF stability (without feeding future into the live label at decision time — use only info ≤ t for live; use t+h only in **offline evaluation reports**).

### 4. Evidence needed before it affects decisions
- Agreement of live `pullback_state` with offline “HTF still same N bars later” on practice  
- Clear rate by `pullback_state` at entry (n ≥ 8)  
- Do **not** assume binary pullback flag is correct  

### 5. Practice-only test
For entries with structure pullback true: rate of HTF direction still same after H decision bars (H fixed, e.g. 4–8); stratify by proposed `pullback_state`.  
Clear% by state on practice hard targets.

### 6. Forward validation condition
Same labeler frozen; forward clear% by `pullback_state`; breach = 0.  
If flag quality is poor offline, **do not** use it for attention — fix labeling rules on practice only.

### 7. What this principle does NOT mean
- **Not** “always fade the pullback” or “always buy the dip.”  
- **Not** treating every LTF opposite print as a reversal.  
- **Not** using future HTF flip inside the live decision label.  
- **Not** automatic reverse on deep pullback.

### Readiness
**Log-only first** + offline audit of the existing flag. **Needs more data.** **Not ready** to influence heuristic now.

---

## PRINCIPLE 4 — Momentum must confirm direction, not just appear

### 1. Plain-language meaning
CCI and RSI describe **momentum relative to equilibrium**. They are not stand-alone trade instructions. Confirmation means momentum state is **consistent with** the directional context (trend continuation, pullback reset, etc.), not merely that a line moved.

### 2. Existing inputs that represent it
| Input | Role |
|-------|------|
| CCI 30 / 100 vs shifted ref | Above/below/cross ref; extended via thresholds |
| RSI 5 / 14 vs shifted ref | Same vs 50 / ref |
| Velocity / confluence | Strengthening / weakening / conflict |
| HTF/LTF direction | Direction to confirm against |
| `market_condition` | Context for “extended = strength vs exhaustion” |

### 3. Measurable state labels
| Field | Values |
|-------|--------|
| `cci_state` | above_zero / below_zero / crossing_zero / extended_high / extended_low / weakening / strengthening |
| `rsi_state` | above_50 / below_50 / crossing_50 / extended_high / extended_low / weakening / strengthening |
| `momentum_velocity` | strengthening / weakening / flat / conflicting |
| `momentum_vs_direction` | confirms / conflicts / unclear |

**Divergence:** `divergence_flag = NOT_AVAILABLE` (not calculated today).

### 4. Evidence needed before it affects decisions
- Clear rate for `momentum_vs_direction` **within** each `market_condition` (interaction required)  
- Extended high/low performance **in trend vs range** separately  
- Never pool all conditions into one “RSI works” claim  

### 5. Practice-only test
On practice: clear% for confirms vs conflicts inside `market_condition == trend` and inside `range_consolidation` (min n=8 each).  
Hard targets reported separately.

### 6. Forward validation condition
Frozen state rules; same stratification on forward; breach = 0.  
If “confirms” only helps in trend on practice but is used globally later, expect forward failure — attention must stay condition-conditional if used at all.

### 7. What this principle does NOT mean
- **Not** “buy when RSI crosses 50” as a standalone rule.  
- **Not** “extended always means reverse.”  
- **Not** inventing divergence without a defined calculation and meaning version.  
- **Not** oscillator override of shell risk.

### Readiness
**Log-only first** (state labels on existing series). **Needs more data** for condition-conditional clear rates. **Not ready** to influence heuristic now.

---

## PRINCIPLE 5 — Agreement is stronger than a single input

### 1. Plain-language meaning
Confidence comes from **agreement among independent input groups** with different jobs (trend, momentum, structure location, trading conditions, risk capacity) — not from one loud line.

### 2. Existing inputs that represent it
| Group | Job | Existing pieces |
|-------|-----|-----------------|
| Trend | Directional context | HTF/LTF dir, strength, alignment |
| Oscillators | Momentum / reset / extension | CCI/RSI states |
| Channel / price | Location / structure | channel_position, channel_slope |
| Conditions | Tradability | session_phase, spread, range flags |
| Risk | Can the account act | heat_ok, risk_remaining, target_remaining |

### 3. Measurable state labels — **agreement profile** (not one opaque score)
Log a structured profile, e.g.:

```text
agreement_profile = {
  "trend_group": "bullish" | "bearish" | "mixed" | "unavailable",
  "momentum_group": "bullish" | "bearish" | "mixed" | "unavailable",
  "channel_group": "bullish" | "bearish" | "mixed" | "ranging" | "unavailable",
  "conditions_group": "normal" | "impaired" | "unavailable",
  "risk_group": "can_act" | "refuse" | "unavailable",
  "pairs_agree": ["trend-momentum", ...],   # list of pairwise agrees
  "pairs_conflict": ["trend-momentum", ...],
  "n_groups_available": int
}
```

**Do not** collapse to a single 0–1 score yet.

### 4. Evidence needed before it affects decisions
- Clear rate by number of agreeing groups and by which pairs conflict  
- Show multi-group agreement predicts clear better than any single group alone (practice)  

### 5. Practice-only test
Stratify entries by `len(pairs_agree)` and top conflict types; clear% and breach% with n ≥ 8.

### 6. Forward validation condition
Same profile logger frozen; forward tables; breach = 0.  
Any future attention weights fit only on practice using profile features — not on forward.

### 7. What this principle does NOT mean
- **Not** a secret super-score that replaces structure.  
- **Not** “all groups must agree or no trade” as a fixed rule.  
- **Not** majority vote reinvented as opaque AI.  
- **Not** risk_group overridable for harder targets.

### Readiness
**Log-only first.** Profile is the deliverable. **Not ready** to influence heuristic now.

---

## PRINCIPLE 6 — Do not confuse activity with opportunity

### 1. Plain-language meaning
Many entries do **not** prove opportunity. Repeated reversals can mean range noise, fading momentum, or unproductive churn. Activity must be classified using **context + outcome**, not entry count alone.

### 2. Existing inputs that represent it
| Input | Role |
|-------|------|
| `n_entries` | Already in score day_rows |
| Side flips / reverse actions | In shell path; **not fully logged** |
| Holding bars / flat bars | Derivable from run loop |
| `reversed_while_htf_unchanged` | HTF dir + reverse events |
| `entered_near_oscillator_neutral` | CCI/RSI near equilibrium at entry |
| Flat channel | channel_slope / position |
| Day outcome | cleared, pnl, dist to target |

### 3. Measurable state labels
| Field | Values |
|-------|--------|
| `entry_number`, `reversal_number` | ints |
| `holding_bars`, `flat_bars_so_far` | ints |
| `consecutive_failed_attempts` | int (define fail: stop with no progress toward target) |
| `reversed_while_htf_unchanged` | bool |
| `entered_near_oscillator_neutral` | bool |
| `day_activity_class` | productive / controlled_retry / unproductive_churn / unknown |

**Draft offline classification (for reporting, not live block):**  
- `productive`: cleared or pnl ≥ k×target with moderate entries  
- `controlled_retry`: not cleared but low reverses, HTF stable, condition trend  
- `unproductive_churn`: high reverses and/or reverse-while-HTF-unchanged and miss  
- Thresholds versioned on **practice** only after measurement — **no arbitrary entry cap now**

### 4. Evidence needed before it affects decisions
- Clear rate vs entry/reversal buckets on practice (hard and soft)  
- Where clear rate **starts falling** as reverses rise (find knee; do not invent cap first)  
- Share of reverses with HTF unchanged  

### 5. Practice-only test
Bin days by reversal_number and by reversed_while_htf_unchanged rate; clear% curve with n per bin.  
Mark bins with n &lt; 8 as INSUFFICIENT EVIDENCE.

### 6. Forward validation condition
Same counters; compare activity mix practice vs forward on hard targets.  
If forward has more unproductive_churn and lower clear with breach 0 → **attention/management research on practice**, not shell change.  
**Do not** impose entry caps until practice curve shows a clear damage region.

### 7. What this principle does NOT mean
- **Not** “high entries caused the miss” without context labels.  
- **Not** an immediate max-entries rule.  
- **Not** trail/scale-in or other banned shell packages.  
- **Not** punishing all retries in a trend pullback.

### Readiness
**Log-only first** (counters + offline day_activity_class). **Needs more data** before any cap or reverse dampener. **Not ready** to influence heuristic now.

---

## PRINCIPLE 7 — Target difficulty changes required opportunity, not the safety law

### 1. Plain-language meaning
A harder **target%** needs more net opportunity in the day. It must **never** justify weaker floor protection, more unsafe heat, or forced trades. Shell laws stay fixed for all targets.

### 2. Existing inputs that represent it
| Input | Role |
|-------|------|
| Runtime `target_pct`, `risk_pct` | Typed difficulty |
| Equity% → target_remaining, risk_remaining | Shell state |
| session_phase / time remaining | Clock |
| Range so far, recent move | Opportunity used |
| heat_ok / dist_to_floor | Risk capacity |
| spread_condition | Trading friction |
| `market_condition` | Context for opportunity quality |

### 3. Measurable state labels
| Field | Values / type |
|-------|----------------|
| `target_remaining_pct` | float |
| `risk_remaining_pct` | float |
| `time_remaining_frac` | float |
| `range_used_vs_typical` | low / normal / high |
| `remaining_opportunity_est` | low / medium / high or float *(causal function of above only)* |
| `miss_class_hypothesis` | not_enough_opportunity / misread_valid_opportunity / unknown |

`miss_class_hypothesis` is for **post-day offline** comparison (e.g. remaining_opportunity was low early vs high but still missed) — not an excuse to breach.

### 4. Evidence needed before it affects decisions
- On practice hard targets: does early `remaining_opportunity_est == low` predict miss **before** entry_number becomes large?  
- Separate days with high remaining_opportunity_est that still miss (misread) vs low estimate (insufficient opportunity)  
- Prove estimate is **not** used to raise heat or disable refuse-open  

### 5. Practice-only test
At decision bars with entry_number ≤ 2, compute estimate; compare later clear rate for low vs high estimate (n ≥ 8).  
Hard pairs only as primary table.

### 6. Forward validation condition
If estimate is ever used for **attention only** after practice proof: freeze; forward hard clear↑ or better thrash class mix; **breach = 0** always.  
**Automatic fail:** any use that increases risk_use beyond shell rules or disables heat refuse.

### 7. What this principle does NOT mean
- **Not** “hard target → take more risk.”  
- **Not** forcing trades when opportunity_est is low.  
- **Not** changing bank/breach definitions by target.  
- **Not** claiming day is impossible (lid-off: only measurement says no).

### Readiness
**Log-only first** for remaining_opportunity_est. **Needs more data** before attention. **Not ready** to influence heuristic now. Shell influence: **never**.

---

## PRINCIPLE 8 — Learn which inputs deserve trust only after honest testing

### 1. Plain-language meaning
Setup types and states are not equally reliable forever. Trust tables may guide **practice-only** attention updates. **Forward outcomes must not choose preferences.** Small samples are INSUFFICIENT EVIDENCE, not “weak forever.”

### 2. Existing inputs that represent it
| Input | Role |
|-------|------|
| Trade tags, market_condition, oscillator states, pullback_state, target bucket | Strata |
| cleared / breached | Outcomes |
| Practice / forward split | Honesty |
| meaning_hash / dials_hash | Freeze |

### 3. Measurable state labels
| Field | Values |
|-------|--------|
| `trust_table_row` | stratum keys + n, clear_rate, breach_rate, CI, status |
| `status` | ok / insufficient_evidence |
| `honesty_label` | PRACTICE_ONLY / IN_SAMPLE_CLAIM_CONTAMINATED |

Min n = **8** (or project standard); Wilson or simple CI required in report.

### 4. Evidence needed before it affects decisions
- Full trust tables on practice  
- Any attention/dial change search **only** on practice days  
- Leak test: forward dates ∉ search set  

### 5. Practice-only test
Build tables; pick at most one attention change if a stratum has large, stable clear gap and n adequate; re-score practice; keep shell fixed.

### 6. Forward validation condition
Score forward **once** after freeze.  
- Breach &gt; 0 → **REJECT** candidate  
- Breach = 0, clear down → identify degraded stratum; **return to practice**; do not fit forward; do not open shell  
- Insufficient cells stay unlabeled as weak  

### 7. What this principle does NOT mean
- **Not** online learning from today’s live P&amp;L into dials mid-window.  
- **Not** forward used as a fitness function.  
- **Not** “tag failed once → ban forever.”  
- **Not** PROVEN or Channel1 auto-merge.

### Readiness
**Log-only first** (trust tables). Attention changes only after tables + practice search exist. **Not ready** to influence heuristic now.

---

## PRINCIPLE 9 — The bot must explain its decision in context

### 1. Plain-language meaning
Every entry, exit, reversal, and no-trade must be explainable from **fields known at that time**. Explanations are factual and causal (no future bars, no post-hoc story after PnL is known).

### 2. Existing inputs that represent it
All fields from principles 1–7 that exist at bar t, plus shell outcomes as they fire (stop, bank, breach, reverse).

### 3. Measurable state labels / required explanation record
For each decision (and EOD):

| Required | Content |
|----------|---------|
| `market_condition` | trend / range_consolidation / transition / uncertain |
| `htf_trend_dir`, `ltf_trend_dir` | bullish / bearish / neutral |
| `alignment` | aligned / conflicting / neutral |
| `channel_position`, `channel_slope` | enums |
| `cci_state`, `rsi_state` | enums |
| `pullback_state` | enums |
| `agreement_profile` | structured (principle 5) |
| `session_phase`, range/spread flags | conditions |
| `target_remaining_pct`, `risk_remaining_pct`, `heat_ok` | risk context |
| `decision_kind` | entry / exit / reverse / no_trade |
| `reason_code` | short factual code from state at t (e.g. `entry_htf_bull_ltf_align`, `no_trade_heat_refuse`, `exit_stop`, `reverse_ltf_opposite`) |

`reason_code` must be a pure function of pre-decision state + chosen action — **not** of final day PnL.

### 4. Evidence needed before it affects decisions
- Spot-check: replay N days; every decision has complete explanation fields  
- No field uses t' &gt; t  
- Tutor day walk can read the same record  

### 5. Practice-only test
100% of decision rows on a practice sample have non-null required fields; schema matches forward export.

### 6. Forward validation condition
Same schema on forward; completeness rate 100%; still no future leakage.  
Explanation quality is a **gate for training**, not a trade rule.

### 7. What this principle does NOT mean
- **Not** marketing narratives or “I felt the market.”  
- **Not** rewriting reasons after the day clears or fails.  
- **Not** requiring human text models for the bot to trade.  
- **Not** a substitute for clear/breach metrics.

### Readiness
**Log-only first** — this is the **implementation backbone** for all other principles. **Not ready** to influence heuristic now; **required before** any principle influences decisions.

---

## Implementation order (ranked)

| Order | Principle | Why this order | Default mode |
|------:|-----------|----------------|--------------|
| **1** | **P9 Explain decision in context** | Without a factual log, no principle is measurable | Log-only |
| **2** | **P1 Context before signal** | Core HTF/LTF/alignment labels are the spine of the log | Log-only |
| **3** | **P6 Activity vs opportunity** | Counters already almost available; separates churn hypothesis from cause | Log-only |
| **4** | **P2 Trend / range / transition** | Needs P1 fields; versioned `market_condition` | Log-only |
| **5** | **P3 Pullback ≠ reversal** | Needs HTF/LTF + offline audit of existing flag | Log-only + audit |
| **6** | **P4 Momentum confirms direction** | State labels on existing CCI/RSI; interpret **inside** P2 condition | Log-only |
| **7** | **P5 Agreement profile** | Composes groups from P1–P4 + risk/conditions | Log-only |
| **8** | **P7 Target difficulty vs opportunity** | Needs session/range + target remaining; never touches shell | Log-only |
| **9** | **P8 Trust tables + honest testing** | Only after strata exist; practice-only fit; one forward score | Log / then optional attention |

**Never in this sequence:** new indicator family, new reward, automatic entry rule, shell unlock, forward fitting, entry caps before P6 curves, opaque single agreement score before P5 profile logs.

---

## Global readiness summary

| Principle | Ready to influence heuristic now? | Log-only first? | Needs more data? |
|-----------|-----------------------------------|-----------------|------------------|
| P1 Context before signal | **No** | **Yes** | Yes (labeled bars) |
| P2 Trend/range/transition | **No** | **Yes** | Yes |
| P3 Pullback ≠ reversal | **No** | **Yes** | Yes (+ flag audit) |
| P4 Momentum confirms | **No** | **Yes** | Yes |
| P5 Agreement profile | **No** | **Yes** | Yes |
| P6 Activity ≠ opportunity | **No** | **Yes** | Yes (reverse counters) |
| P7 Target vs opportunity | **No** | **Yes** | Yes |
| P8 Trust after honest test | **No** | **Yes** | Yes (then practice-only attention only if earned) |
| P9 Explain in context | **No** | **Yes** (mandatory) | Yes (complete schema) |

---

## What to build first (one sentence)

**Build the decision/day audit log that records P9’s explanation fields (including P1 alignment and P6 counters), freeze meaning versions for labels, fill practice and forward tables with sample sizes — only then may practice-only attention experiments begin; the risk shell stays locked the entire time.**

---

## Non-goals (explicit)

| Forbidden in this phase | Why |
|-------------------------|-----|
| New indicator families | Principles use existing series |
| New rewards / entry rules | Measurement before policy |
| Shell changes | Floor law independent of target difficulty |
| Forward-driven dial/attention fit | Contamination |
| Automatic “never countertrend” / entry caps | Principles require measurement first |
| Opaque single confidence score | P5 logs profile first |

---

*Multi-pair tutor: I still bank at your target and refuse heat at your floor. These principles teach me **how to name** what I already see — not how to invent a new entry recipe or open the floor.*
