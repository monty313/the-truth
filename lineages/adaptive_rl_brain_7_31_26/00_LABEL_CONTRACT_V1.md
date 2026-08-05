# 00 — Label Contract V1 (factual decision explanations)

**Principle this serves:** Principle 9 — factual decision explanations.  
**Scope:** Freeze **observation / explanation labels only**.  
**Not in scope:** entry/exit rules, rewards, indicators, attention weights, trade caps, shell changes, confidence scores.

| Pin | Value |
|-----|--------|
| **Contract version** | `label_contract_v1` |
| **Track** | multi-pair tutor (heuristic + equity shell) |
| **Shell** | locked (unchanged) |
| **Future bars in live labels** | **NO** |
| **Offline-only fields** | prefix `offline_` only; never feed heuristic |
| **Unclear default** | `unknown` or `unavailable` as specified per label |
| **Meaning impact** | Any change to a contract condition ⇒ new `label_contract` version + new `meaning_hash` stamp; re-run practice + forward logs |

**Canonical HTF / LTF sources (code-locked for V1):**

| Role | Source in code today |
|------|----------------------|
| **HTF direction / velocity** | Official Set **2** (“intraday”) confluence on confirmation TFs `30m` + `1h` — field `primary` / `higher` in `build_perception_at` (`day_runner.py`) |
| **LTF direction** | Entry direction on **`5m`** — field `lower` via `_entry_dir_from_ind(ind, "5m", ts)` |
| **Pullback / scale_conflict** | `structure_flags(higher, lower)` |
| **CCI / RSI / channel series** | `indicator_frame` + `snapshot_at` on a confirmation TF; dual flags via `dual_confirmation_flags` for Set 2’s two confirmation TFs |
| **Shell risk** | `GoalEquityDay` equity%, heat, banked, breached, side, n_entries |

**Direction encoding:** `Direction.BULL = +1` → label `bullish`; `BEAR = -1` → `bearish`; `NEUTRAL = 0` → `neutral`.

**Velocity encoding (locked in `confluence.velocity_strength`):**  
3 groups agree → `strong`; 2 → `medium`; 1 → `weak`; 0 or no clear majority → `none`.

**CCI extended thresholds (standard levels, fixed for V1):** `+100` / `−100` on **CCI30** (and require finite values).  
**RSI extended thresholds (standard levels, fixed for V1):** `≥70` / `≤30` on **RSI14** (finite).

**Prior-bar for “change” labels:** previous **decision bar** index only: `t_prev = t - decide_every` if `t_prev >= warmup`, else value = `unknown`. Never use a bar after `t`.

---

## Label contracts (one table per label)

### 1. `htf_trend_dir`

| Col | Content |
|-----|---------|
| **1. Label name** | `htf_trend_dir` |
| **2. Allowed values** | `bullish` · `bearish` · `neutral` · `unknown` |
| **3. Meaning** | `bullish`: higher-timeframe confirmation stack votes bull. `bearish`: votes bear. `neutral`: no clear majority. `unknown`: confluence not computable (missing bars / NaNs). |
| **4. Source fields** | `SetConfluence.direction` for Official Set 2 (`primary.direction` / `higher`) |
| **5. Deterministic assign** | `bullish` iff `direction == BULL`; `bearish` iff `BEAR`; `neutral` iff `NEUTRAL`; `unknown` if snapshot/confluence missing |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Changing which Official Set is “HTF” or confirmation TF pair ⇒ new contract version |
| **10. Status** | **ready to log** |

---

### 2. `htf_trend_strength`

| Col | Content |
|-----|---------|
| **1. Label name** | `htf_trend_strength` |
| **2. Allowed values** | `strong` · `medium` · `weak` · `none` · `unknown` |
| **3. Meaning** | How many of the three confirmation groups (CCI, RSI, channel) agree with HTF majority direction. |
| **4. Source fields** | `primary.velocity` (`VelocityStrength`) |
| **5. Deterministic assign** | `strong`←`STRONG` (3 agree); `medium`←`MEDIUM` (2); `weak`←`WEAK` (1); `none`←`NONE` (0 or no direction); `unknown` if confluence missing |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Changing group set or agreement count map ⇒ new version |
| **10. Status** | **ready to log** |

---

### 3. `ltf_trend_dir`

| Col | Content |
|-----|---------|
| **1. Label name** | `ltf_trend_dir` |
| **2. Allowed values** | `bullish` · `bearish` · `neutral` · `unknown` |
| **3. Meaning** | Lower / entry timeframe directional vote used as LTF in structure (5m entry direction path). |
| **4. Source fields** | `lower` from `_entry_dir_from_ind(ind, "5m", ts)` |
| **5. Deterministic assign** | Same mapping as `htf_trend_dir` on that `Direction` |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Changing LTF from `5m` to another TF ⇒ new version |
| **10. Status** | **ready to log** |

---

### 4. `alignment`

| Col | Content |
|-----|---------|
| **1. Label name** | `alignment` |
| **2. Allowed values** | `aligned` · `conflicting` · `neutral` · `unknown` |
| **3. Meaning** | Whether HTF and LTF clear directions agree, oppose, or cannot both be clear. |
| **4. Source fields** | `htf_trend_dir`, `ltf_trend_dir` (or raw `higher`, `lower`) |
| **5. Deterministic assign** | If either is `unknown` → `unknown`. Else if either is `neutral` → `neutral`. Else if equal → `aligned`. Else → `conflicting`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Redefining HTF/LTF sources ⇒ new version |
| **10. Status** | **ready to log** |

---

### 5. `channel_position`

| Col | Content |
|-----|---------|
| **1. Label name** | `channel_position` |
| **2. Allowed values** | `above` · `below` · `inside` · `unknown` |
| **3. Meaning** | Close vs channel bands (SMA(4) high/low, shift +2) on **both** HTF confirmation TFs (Set 2: 30m and 1h), same dual-AND rule as confluence. |
| **4. Source fields** | `dual_confirmation_flags` → `channel` `(both_above, both_below)`; underlying `ch_high_s2`, `ch_low_s2`, `close` on each conf TF |
| **5. Deterministic assign** | `above` iff dual channel both_above true; `below` iff both_below true; `inside` iff both flags false **and** on each conf TF with finite bands: `ch_low ≤ close ≤ ch_high`; if any required value non-finite → `unknown`. (If not both_above/below and not inside on both TFs → `unknown`.) |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | CHANNEL_N/SHIFT or dual vs single TF rule change ⇒ new version |
| **10. Status** | **ready to log** (needs derived dual + per-TF inside check; all from existing frames) |

---

### 6. `channel_slope`

| Col | Content |
|-----|---------|
| **1. Label name** | `channel_slope` |
| **2. Allowed values** | `rising` · `falling` · `flat` · `unknown` |
| **3. Meaning** | Direction of channel mid-line between previous decision bar and current decision bar on HTF confirmation TF **30m** (first Set-2 confirmation TF). |
| **4. Source fields** | `ch_high_s2`, `ch_low_s2` at asof(t) and asof(t_prev) on 30m `indicator_frame` |
| **5. Deterministic assign** | `mid = 0.5 * (ch_high_s2 + ch_low_s2)`. Require finite mid at t and t_prev. `delta = mid_t - mid_prev`. `flat` iff `abs(delta) <= 0.5 * POINT_SIZE` with `POINT_SIZE = 0.01` (same constant as `equity_day.POINT_SIZE`). `rising` iff `delta > that`. `falling` iff `delta < -that`. If t_prev missing → `unknown`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** (only t and earlier) |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Flat epsilon, TF choice, or mid definition change ⇒ new version |
| **10. Status** | **needs a derived calculation** (from existing channel series) |

---

### 7. `cci_state`

| Col | Content |
|-----|---------|
| **1. Label name** | `cci_state` |
| **2. Allowed values** | `above_zero` · `below_zero` · `mixed_zero` · `extended_high` · `extended_low` · `strengthening` · `weakening` · `unknown` |
| **3. Meaning** | CCI momentum state on **30m** confirmation snapshot (CCI30 & CCI100). Extended uses ±100. Strengthening/weakening = CCI30 change vs prior decision bar. |
| **4. Source fields** | `cci30`, `cci100`, `cci30_sma_s4`, `cci100_sma_s4` at t and t_prev |
| **5. Deterministic assign** | Priority order (first match wins): (1) if not finite cci30/cci100 → `unknown`. (2) if `cci30 >= 100` and `cci100 >= 100` → `extended_high`. (3) if `cci30 <= -100` and `cci100 <= -100` → `extended_low`. (4) else if finite cci30_prev: `strengthening` iff `cci30_t > cci30_prev`; `weakening` iff `cci30_t < cci30_prev`; if equal skip to (5). (5) if `cci30 > 0` and `cci100 > 0` → `above_zero`. (6) if `cci30 < 0` and `cci100 < 0` → `below_zero`. (7) else → `mixed_zero`. *Note:* ref-vs-SMA “both above/below” remains available as dual group flags for confluence; not a separate enum value here to avoid duplicate labels. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | ±100 thresholds, TF, or priority order change ⇒ new version |
| **10. Status** | **needs a derived calculation** (snapshots exist; enum packaging is new) |

---

### 8. `rsi_state`

| Col | Content |
|-----|---------|
| **1. Label name** | `rsi_state` |
| **2. Allowed values** | `above_50` · `below_50` · `mixed_50` · `extended_high` · `extended_low` · `strengthening` · `weakening` · `unknown` |
| **3. Meaning** | RSI momentum state on **30m** snapshot (RSI5 & RSI14). Extended: RSI14 ≥70 / ≤30. Strengthening/weakening: RSI14 vs prior decision bar. |
| **4. Source fields** | `rsi5`, `rsi14`, `rsi5_sma_s4`, `rsi14_sma_s4` at t and t_prev |
| **5. Deterministic assign** | Priority: (1) non-finite rsi14/rsi5 → `unknown`. (2) `rsi14 >= 70` → `extended_high`. (3) `rsi14 <= 30` → `extended_low`. (4) if finite rsi14_prev: strengthening if `rsi14_t > rsi14_prev`; weakening if `<`; equal → fall through. (5) `rsi14 > 50` and `rsi5 > 50` → `above_50`. (6) `rsi14 < 50` and `rsi5 < 50` → `below_50`. (7) else → `mixed_50`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | 70/30/50 levels or TF change ⇒ new version |
| **10. Status** | **needs a derived calculation** |

---

### 9. `momentum_velocity`

| Col | Content |
|-----|---------|
| **1. Label name** | `momentum_velocity` |
| **2. Allowed values** | `strong` · `medium` · `weak` · `none` · `strengthening` · `weakening` · `flat` · `unknown` |
| **3. Meaning** | **Level:** same as HTF velocity groups. **Change:** level rank vs prior decision bar. |
| **4. Source fields** | `primary.velocity` at t and t_prev |
| **5. Deterministic assign** | Rank map: `none=0`, `weak=1`, `medium=2`, `strong=3`. Emit **level** value always when known. Additionally emit **change** as separate field `momentum_velocity_change` in schema: `strengthening` if rank_t > rank_prev; `weakening` if rank_t < rank_prev; `flat` if equal; `unknown` if either unknown. **V1 logging rule:** store level in `momentum_velocity` as `strong|medium|weak|none|unknown` only; store change in `momentum_velocity_change` (see V1 schema). Do not overload one field with both. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Rank map change ⇒ new version |
| **10. Status** | **ready to log** (level) · **needs derived** (change vs t_prev) |

*Contract note:* User list said one field `momentumvelocity`. V1 **splits** level vs change for determinism; both are versioned under this contract.

---

### 10. `momentum_vs_direction`

| Col | Content |
|-----|---------|
| **1. Label name** | `momentum_vs_direction` |
| **2. Allowed values** | `confirms` · `conflicts` · `unclear` · `unknown` |
| **3. Meaning** | Whether oscillator group direction on HTF confirmation agrees with `htf_trend_dir`. |
| **4. Source fields** | Group votes on Set 2 confluence (`votes` for cci/rsi/channel); `htf_trend_dir` |
| **5. Deterministic assign** | Build `osc_dir = majority_direction` of the three group directions (same as confluence majority). If `htf_trend_dir` or osc_dir is neutral/unknown → `unclear` if both computable but either neutral; `unknown` if missing data. If both bullish or both bearish → `confirms`. If opposite clear sides → `conflicts`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Using LTF instead of HTF as reference dir ⇒ new version |
| **10. Status** | **ready to log** (from existing votes) |

---

### 11. `market_condition`

| Col | Content |
|-----|---------|
| **1. Label name** | `market_condition` |
| **2. Allowed values** | `trend` · `range_consolidation` · `transition` · `uncertain` · `unknown` |
| **3. Meaning** | Coarse regime from alignment, HTF strength, channel position/slope only (no new indicators). |
| **4. Source fields** | `alignment`, `htf_trend_dir`, `htf_trend_strength`, `channel_position`, `channel_slope` |
| **5. Deterministic assign** | Evaluate in order: (1) any required input `unknown` → `unknown`. (2) `trend` iff `alignment == aligned` AND `htf_trend_dir ∈ {bullish,bearish}` AND `htf_trend_strength ∈ {medium,strong}` AND `channel_slope ∈ {rising,falling}` AND sign-consistent: (bullish⇒rising or flat not required—**strict:** bullish⇒`channel_slope==rising` OR `channel_position==above`; bearish⇒`falling` OR `below`). (3) `range_consolidation` iff `channel_position == inside` AND `channel_slope == flat` AND `htf_trend_strength ∈ {weak,none}`. (4) `transition` iff `alignment == conflicting` AND `htf_trend_strength ∈ {weak,none}`. (5) else `uncertain`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` if inputs missing; else `uncertain` when rules miss |
| **9. Meaning-version impact** | Any change to rule order or predicates ⇒ new version |
| **10. Status** | **needs a derived calculation** (pure function of other V1 labels) |

---

### 12. `pullback_state`

| Col | Content |
|-----|---------|
| **1. Label name** | `pullback_state` |
| **2. Allowed values** | `no_pullback` · `pullback_active` · `pullback_with_scale_conflict` · `unknown` |
| **3. Meaning** | Structure pullback flag ± scale conflict. **Does not** claim “healthy/deep/failed” (those need depth/outcome). |
| **4. Source fields** | `structure.pullback`, `structure.scale_conflict` |
| **5. Deterministic assign** | If structure missing → `unknown`. Else if not pullback → `no_pullback`. Else if pullback and scale_conflict → `pullback_with_scale_conflict`. Else → `pullback_active`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Changing `is_pullback` definition ⇒ new version |
| **10. Status** | **ready to log** |

**Not in live V1 (would invent depth):** `shallow_healthy`, `deep`, `failed`, `reversal_risk` as distance-based states.  

**Offline only (never live):**  
`offline_pullback_resolved_as` ∈ {`continued_htf`, `htf_flipped`, `unclear`} using HTF dir at `t + H*decide_every` for fixed H (e.g. H=8) — **audit only**.

---

### 13. `scale_conflict`

| Col | Content |
|-----|---------|
| **1. Label name** | `scale_conflict` |
| **2. Allowed values** | `yes` · `no` · `unknown` |
| **3. Meaning** | Major vs minor clear directions opposite (`is_scale_conflict`). V1 uses same higher/lower pair as structure_flags default. |
| **4. Source fields** | `structure.scale_conflict` |
| **5. Deterministic assign** | `yes` iff True; `no` iff False; `unknown` if structure missing |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Changing major/minor pair ⇒ new version |
| **10. Status** | **ready to log** |

---

### 14. `agreement_profile`

| Col | Content |
|-----|---------|
| **1. Label name** | `agreement_profile` |
| **2. Allowed values** | Structured object (not a single score). Required keys below. |
| **3. Meaning** | Which input **groups** agree, conflict, or are unavailable. |
| **4. Source fields** | Labels/groups: trend (`htf_trend_dir`), momentum (`momentum_vs_direction` / osc majority), channel (`channel_position` mapped to bullish if above, bearish if below, ranging if inside), conditions (`spread_condition` when available else unavailable), risk (`heat_ok` → can_act / refuse) |
| **5. Deterministic assign** | ```json
{
  "trend_group": "bullish|bearish|neutral|unavailable",
  "momentum_group": "bullish|bearish|neutral|unavailable",
  "channel_group": "bullish|bearish|ranging|unavailable",
  "conditions_group": "normal|impaired|unavailable",
  "risk_group": "can_act|refuse|unavailable",
  "pairs_agree": ["trend-momentum", ...],
  "pairs_conflict": [...],
  "n_groups_available": 0-5
}
```
Group dir for momentum = osc majority mapped bullish/bearish/neutral. Pair agree if both available and same bullish/bearish; conflict if opposite bullish/bearish; skip if either neutral/unavailable. **No scalar confidence.** |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | group = `unavailable`; lists empty; `n_groups_available` counted |
| **9. Meaning-version impact** | Changing group membership or pair logic ⇒ new version |
| **10. Status** | **needs a derived calculation** (compose other labels; conditions may be unavailable in V1) |

---

### 15. `spread_condition`

| Col | Content |
|-----|---------|
| **1. Label name** | `spread_condition` |
| **2. Allowed values** | `normal` · `wide` · `unknown` |
| **3. Meaning** | Current bar spread vs **causal** baseline: median of spreads on M1 bars from day start through `t` inclusive, requiring at least 30 bars in that window. |
| **4. Source fields** | M1 spread column (or `spread_px` path used by `GoalEquityDay`) |
| **5. Deterministic assign** | Let `s = spread[t]`, `med = median(spread[0:t+1])`. If &lt;30 bars or non-finite → `unknown`. `wide` iff `s > 2.0 * med`; else `normal`. Constant **2.0** is fixed for V1. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Factor 2.0 or min bars 30 change ⇒ new version |
| **10. Status** | **needs a derived calculation** |

---

### 16. `thin_liquidity_flag`

| Col | Content |
|-----|---------|
| **1. Label name** | `thin_liquidity_flag` |
| **2. Allowed values** | — |
| **3. Meaning** | — |
| **4. Source fields** | Tick volume exists on some CSVs but **no locked, tested thin-liquidity rule** in this lineage |
| **5. Deterministic assign** | — |
| **6. Availability** | — |
| **7. Future bar** | — |
| **8. Default** | — |
| **9. Meaning-version impact** | — |
| **10. Status** | **NOT AVAILABLE** (do not invent) |

*Optional later offline research field only after a contract is written.*

---

### 17. `abnormal_range_flag`

| Col | Content |
|-----|---------|
| **1. Label name** | `abnormal_range_flag` |
| **2. Allowed values** | `yes` · `no` · `unknown` **only if** derived rule below is implemented; else treat whole label NOT AVAILABLE until coded under this contract |
| **3. Meaning** | Day range so far large vs this day’s own early distribution (causal). |
| **4. Source fields** | M1 high/low from day start through t |
| **5. Deterministic assign** | `range_so_far = max(high[0:t+1]) - min(low[0:t+1])`. Baseline: median of rolling 60-bar ranges ending at each bar from bar 60..t (if t&lt;120 → `unknown`). `yes` iff `range_so_far > 2.0 * baseline_median`; else `no`. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | 2.0 or window lengths change ⇒ new version |
| **10. Status** | **needs a derived calculation** (not coded today; **do not log until implemented to this exact rule**) |

Until implemented: status remains **not currently available** in outputs (omit field or write `unavailable` constant with contract note).

---

### 18. `remaining_opportunity_est`

| Col | Content |
|-----|---------|
| **1. Label name** | `remaining_opportunity_est` |
| **2. Allowed values** | `low` · `medium` · `high` · `unknown` |
| **3. Meaning** | Rough causal estimate of room left to approach target using time left and range so far — **not** a trade command. |
| **4. Source fields** | `target_pct`, `equity_pct` (or progress), `session_phase`, `range_so_far` (high-low day so far), `target_remaining = target_pct - equity_pct` |
| **5. Deterministic assign** | If any input non-finite → `unknown`. If `target_remaining <= 0` → `high` (already at/through target path). Else let `time_left = 1 - session_phase` (clip to [0,1]). Let `range_so_far_pct = 100 * range_so_far / close[t]` if close&gt;0. If `time_left < 0.15` AND `target_remaining > 0.5 * range_so_far_pct` → `low`. Else if `time_left >= 0.50` AND `target_remaining <= range_so_far_pct` → `high`. Else → `medium`. Constants **0.15, 0.50, 0.5** fixed for V1. |
| **6. Availability** | **before action** |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Any constant or formula change ⇒ new version |
| **10. Status** | **needs a derived calculation** (log only after coded; never drives shell) |

---

### 19. `entry_reason`

| Col | Content |
|-----|---------|
| **1. Label name** | `entry_reason` |
| **2. Allowed values** | See reason codes § below · `none` · `unknown` |
| **3. Meaning** | Why an open was attempted/accepted at this bar (factual). |
| **4. Source fields** | Pre-action state + selected action + whether `_try_open` would/did succeed (heat) |
| **5. Deterministic assign** | Pure function `entry_reason(state, action)` — see **Reason codes**. Only set when action is BUY/SELL and position was flat before action (or reverse counts as exit+entry with separate codes). |
| **6. Availability** | **after action** (needs selected action); inputs to the function are **before** action |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown`; `none` if not an entry bar |
| **9. Meaning-version impact** | New reason code set ⇒ new version |
| **10. Status** | **ready to log** (as pure function once state logged) |

---

### 20. `exit_reason`

| Col | Content |
|-----|---------|
| **1. Label name** | `exit_reason` |
| **2. Allowed values** | `stop` · `bank` · `reverse` · `eod_flatten` · `breach` · `none` · `unknown` |
| **3. Meaning** | Why position closed or flipped. |
| **4. Source fields** | Shell events: stop hit, banked, reverse action, EOD flatten, breached |
| **5. Deterministic assign** | Priority if multiple same bar: `breach` > `bank` > `stop` > `reverse` > `eod_flatten`. `none` if no exit. |
| **6. Availability** | **after action** / when mark fires (same bar) |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown` / `none` |
| **9. Meaning-version impact** | Priority order change ⇒ new version |
| **10. Status** | **ready to log** |

---

### 21. `no_trade_reason`

| Col | Content |
|-----|---------|
| **1. Label name** | `no_trade_reason` |
| **2. Allowed values** | See reason codes · `none` · `unknown` |
| **3. Meaning** | Why flat or HOLD produced no new open. |
| **4. Source fields** | banked, dead, heat, signal HOLD, already in position managing |
| **5. Deterministic assign** | Pure function when action is HOLD or open refused — see **Reason codes** |
| **6. Availability** | **after action** (refusal) / **before** if pre-check logs refuse |
| **7. Future bar** | **NO** |
| **8. Default if unclear** | `unknown`; `none` if entry/exit happened |
| **9. Meaning-version impact** | Code set change ⇒ new version |
| **10. Status** | **ready to log** |

---

### 22. `day_activity_class`

| Col | Content |
|-----|---------|
| **1. Label name** | `day_activity_class` |
| **2. Allowed values** | `cleared` · `miss_low_activity` · `miss_high_activity` · `breached` · `unknown` |
| **3. Meaning** | End-of-day activity bucket using **only day totals + outcome** (not a root-cause claim). |
| **4. Source fields** | `cleared`, `breached`, `n_entries`, `reversal_count` (must be logged) |
| **5. Deterministic assign** | If breached → `breached`. Else if cleared → `cleared`. Else if `n_entries + reversal_count <= 4` → `miss_low_activity`. Else if `n_entries + reversal_count >= 10` → `miss_high_activity`. Else → `unknown` (middle band not classified in V1). Thresholds **4** and **10** fixed for V1. |
| **6. Availability** | **end-of-day only** |
| **7. Future bar** | **NO** (uses full day that already completed; not used as a live decision label) |
| **8. Default if unclear** | `unknown` |
| **9. Meaning-version impact** | Thresholds 4/10 change ⇒ new version |
| **10. Status** | **needs a derived calculation** (needs `reversal_count` counter in day result) |

**Not V1 live:** `productive` / `unproductive_churn` narrative classes — those need context strata; keep offline later as `offline_day_activity_narrative` if desired.

---

## Offline-only fields (never live / never heuristic)

| Field | Purpose | Future bars? |
|-------|---------|--------------|
| `offline_pullback_resolved_as` | HTF same or flipped after H decision bars | YES (audit only) |
| `offline_clear` | Day clear flag mirrored for join | EOD outcome |
| `offline_any` | Must start with `offline_` | — |

---

## Reason codes (pure functions of state + selected action)

State snapshot **S** at bar t (all before trade effect). Action **A** ∈ {HOLD, BUY, SELL}. Position before **P** ∈ {flat, long, short}.

### `entry_reason` (when open or reverse-open occurs)

| Code | Condition |
|------|-----------|
| `entry_htf_bull` | A=BUY, P=flat, htf_trend_dir=bullish, heat_ok |
| `entry_htf_bear` | A=SELL, P=flat, htf_trend_dir=bearish, heat_ok |
| `entry_ltf_fallback_bull` | A=BUY, P=flat, htf_trend_dir=neutral, ltf_trend_dir=bullish, heat_ok |
| `entry_ltf_fallback_bear` | A=SELL, P=flat, htf_trend_dir=neutral, ltf_trend_dir=bearish, heat_ok |
| `entry_reverse_to_bull` | A=BUY, P=short, heat_ok |
| `entry_reverse_to_bear` | A=SELL, P=long, heat_ok |
| `entry_refused_heat` | A in {BUY,SELL}, would open, heat_ok=false |
| `entry_unknown` | open path but none of the above |

### `no_trade_reason` (when no new open)

| Code | Condition |
|------|-----------|
| `no_trade_banked` | banked |
| `no_trade_dead_or_breach` | dead or breached |
| `no_trade_signal_hold` | P=flat, A=HOLD, not banked/dead |
| `no_trade_manage_hold` | P≠flat, A=HOLD |
| `no_trade_heat_refuse` | A directional but heat refused |
| `no_trade_unknown` | else |

### `exit_reason`

| Code | Condition |
|------|-----------|
| `breach` · `bank` · `stop` · `reverse` · `eod_flatten` · `none` · `unknown` | As priority table above |

---

## Smallest Version 1 audit schema

### Required decision-bar fields

| Field | Notes |
|-------|--------|
| `label_contract_version` | `label_contract_v1` |
| `meaning_hash` | from frozen meaning manifest |
| `date`, `t`, `split` | practice \| forward |
| `target_pct`, `risk_pct` | runtime |
| `htf_trend_dir`, `htf_trend_strength` | |
| `ltf_trend_dir`, `alignment` | |
| `channel_position` | |
| `channel_slope` | derived |
| `cci_state`, `rsi_state` | derived |
| `momentum_velocity` | level |
| `momentum_velocity_change` | derived |
| `momentum_vs_direction` | |
| `market_condition` | derived from labels |
| `pullback_state`, `scale_conflict` | |
| `agreement_profile` | JSON object |
| `session_phase` | existing |
| `equity_pct`, `target_remaining_pct`, `risk_remaining_pct`, `heat_ok` | shell |
| `spread_condition` | derived |
| `remaining_opportunity_est` | derived |
| `action` | HOLD/BUY/SELL |
| `position_before` | flat/long/short |
| `entry_number`, `reversal_number` | counters |
| `reversed_while_htf_unchanged` | bool |
| `entry_reason`, `exit_reason`, `no_trade_reason` | codes |
| `future_bar_used` | always `false` for live rows |

### Required end-of-day fields

| Field | Notes |
|-------|--------|
| same ids + hashes | |
| `cleared`, `breached`, `banked` | GOAL definitions |
| `pnl_pct`, `min_eq_pct`, `dist_to_target_pct` | |
| `n_entries`, `reversal_count` | |
| `day_activity_class` | EOD only |
| `n_decision_rows` | completeness check |

### Optional — wait (do not require in V1)

| Field | Why wait |
|-------|----------|
| `thin_liquidity_flag` | NOT AVAILABLE |
| `abnormal_range_flag` | until exact rule coded |
| `shallow_healthy` / `deep` / `failed` pullback | needs depth/offline |
| `divergence_flag` | NOT AVAILABLE |
| any confidence score | forbidden |
| `offline_*` | separate files only |

---

## Replay tests (no policy update)

### A. Five-day practice replay

| Item | Spec |
|------|------|
| **Days** | First **5** practice calendar days from data contract (chronological, ≥900 bars) |
| **Pairs** | One fixed pair: target **1.0**, risk **2.0** (runtime only) |
| **Decode** | heuristic; frozen dials; shell locked |
| **Output** | `checkpoints/honest_gate/replay_practice_5d_v1.json` (or csv of decision rows + day rows) |

**Assert:**

1. **Schema completeness:** every required decision-bar and EOD field present on every row.  
2. **Columns:** set of decision columns == set defined in this contract V1.  
3. **No future-bar leakage:** `future_bar_used == false`; channel_slope / cci/rsi change use only `t_prev = t - decide_every`.  
4. **Deterministic re-run:** run twice → identical JSON bytes (or equal hashes of row payloads).  
5. **meaning_hash** present and equal on every row; equals frozen manifest hash.

### B. Five-day forward replay

| Item | Spec |
|------|------|
| **Days** | First **5** forward calendar days from data contract |
| **Pair / decode / dials** | Same as practice test |
| **Output** | `checkpoints/honest_gate/replay_forward_5d_v1.json` |

**Assert:** same 1–5 as practice, **and**:

6. **Same columns** as practice file (set equality of field names).  
7. **No training / dial search** in the test process.  
8. Forward dates ⊆ forward set; practice dates ⊆ practice set (leak test).

**Pass:** all asserts true.  
**Fail:** missing field, column mismatch, nondeterminism, missing meaning_hash, or any live label using bars &gt; t.

---

## Status summary

| Label | Status |
|-------|--------|
| htf_trend_dir | ready to log |
| htf_trend_strength | ready to log |
| ltf_trend_dir | ready to log |
| alignment | ready to log |
| channel_position | ready to log (derived from existing frames) |
| channel_slope | needs derived calculation |
| cci_state | needs derived calculation |
| rsi_state | needs derived calculation |
| momentum_velocity | ready to log (level) |
| momentum_velocity_change | needs derived calculation |
| momentum_vs_direction | ready to log |
| market_condition | needs derived calculation |
| pullback_state | ready to log (reduced V1 set) |
| scale_conflict | ready to log |
| agreement_profile | needs derived calculation |
| spread_condition | needs derived calculation |
| thin_liquidity_flag | **NOT AVAILABLE** |
| abnormal_range_flag | needs derived calculation (omit until coded) |
| remaining_opportunity_est | needs derived calculation |
| entry_reason | ready to log (pure function) |
| exit_reason | ready to log |
| no_trade_reason | ready to log |
| day_activity_class | needs derived calculation (EOD) |

---

## Explicit non-deliverables (this document)

- No heuristic change  
- No attention weights  
- No shell change  
- No new indicator family  
- No single confidence score  
- No trade caps  
- No claim of “consistent” or “ready to train policy”

**Approved deliverable:** this frozen, versioned Label Contract + V1 schema + replay test plan only.

---

## Next implementation step (after this file is accepted)

Implement a **logger only** that emits V1 schema for the 5+5 replay tests under `honest_gate/`, stamp `label_contract_v1` + `meaning_hash`, and keep the trading policy byte-identical.

---

*Contract id: `label_contract_v1` · Principle 9 · multi-pair tutor explanation layer.*
