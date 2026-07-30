# How to Make the LLM Create Better Strategies

> **Purpose:** Copy-paste prompt playbook so the diagnostic LLM (CMO) / any strategy LLM
> produces **Momentum One–compatible**, testable, multi-TF scalping strategies — not retail fluff.
>
> **Companions:** `doctrine/SYSTEM_DOCTRINE_CMO.md` · `PERFORMANCE_IS_POSSIBLE_PART4.md` ·
> `doctrine/SUCCESS_LEDGER.md` · Gravity sets A/B/C · signal slots + agreement evidence.
>
> **Rule:** Strategies are suggestions until scored (5/10/20 LTF bars) and only then considered
> for `configs/signal_slots.yaml`. Evidence only. Lid is off — nothing the bound allows is “impossible.”

---

## 0. Non-negotiables before you prompt

1. **Sets (always three TFs):**
   - **A:** 1m / 15m / 30m  
   - **B:** 5m / 1h / 4h  
   - **C:** 15m / 4h / 1d  
   First TF = execution (LTF). Last two = regime/bias (HTF).

2. **Entry on LTF only when HTF bias is clear** (both HTFs agree unless you explicitly design a 2-TF variant).

3. **Composition:** no lone indicator. Fast relative to slow; LTF relative to HTF. Prefer **shifts**, stacking, and self-banding over retail threshold crosses.

4. **Score before slots:** hit rate + bps at **5 / 10 / 20 bars on the lowest TF in the set**. Prefer setups that resolve in **~8–10 bars**.

5. **Portfolio > single hero:** high-precision (selective) + bread-and-butter (higher frequency) + stand-down + size by conviction + path management (partials, time stops).

6. **Do not require any one strategy to be 80% win rate.** Consistency comes from the **stack** (engage when dense, flat when dead, size when agreed).

7. **Daily target context:** design so a **2.5% day** is reachable via multiple path segments, not one perfect signal.

---

## 1. Master prompt block (paste as one message)

Use this block **exactly** (plus optional context from §2):

```text
Give me a complete portfolio of 4–5 multi-timeframe scalping strategies that use shifts and indicator combinations. Make them as high-quality as possible while still firing multiple times per day. For each one, write clear rules and full MT5 code with all parameters as inputs.

Focus on strategies where the entry is on the lower timeframe but only allowed when the higher timeframe shows clear directional bias. Prioritize setups that historically resolve within 8–10 bars.

Design the portfolio with explicit stand-down rules, conviction-based sizing, and path management (partials + time stops) so the system can target consistent daily gains without relying on any single strategy having 80% win rate.

Write the MT5 code so I can optimize the shifts, periods, and combination thresholds myself on XAUUSD (or my instrument) over long periods.

Show me how to structure the portfolio so that high-precision (selective) strategies and higher-frequency (bread-and-butter) strategies work together, with clear rules for when to size up or stay flat.
```

---

## 2. Context to prepend (Momentum One — keep short)

Paste **above** the master block when talking to the CMO / strategy LLM:

```text
Context (Momentum One):
- Instrument: XAUUSD (or stated symbol). Leverage mindset 1:100; floor is sacred; consistency > hero days.
- TF sets only: A=1m/15m/30m, B=5m/1h/4h, C=15m/4h/1d.
- Bread-and-butter: LTF pullback while both HTFs strongly trending.
- Untraditional sensors OK: stack same indicator (mass), shift (ghost baseline), band the oscillator (velocity) — not retail “RSI cross 30 = buy.”
- Evidence: agreement of independent families already prints ~70–81% on selected events (see PERFORMANCE_IS_POSSIBLE_PART4). Prefer independent axes that can vote/agree later.
- Output for each strategy: (1) plain-English rules, (2) long/short, (3) stand-down conditions, (4) which set A/B/C, (5) expected role: precision vs frequency, (6) MT5 EA/script with all periods, shifts, thresholds as input parameters, (7) suggested default inputs for XAUUSD.
- Do not claim a strategy is production-ready without a forward test plan at 5/10/20 LTF bars.
```

---

## 3. Required portfolio shape (what “good” looks like)

The LLM’s 4–5 strategies should map roughly to:

| Role | Job | Frequency | Precision target (events) |
|------|-----|-----------|---------------------------|
| **Precision gate** | 2+ independent conditions + dual HTF bias | Lower | High (agreement-class) |
| **Bread-and-butter pull** | LTF reclaim under firm HTF trend | Higher | Medium-high |
| **Continuation / thrust** | Unanimous alignment; allow size-up | Medium | Medium-high |
| **Stand-down module** | Explicit no-trade regimes (not optional flavor text) | — | Protects daily clear rate |
| **Optional 5th** | Structured reversal **or** session/liquidity filter | Low | Only with clear invalidation |

Every strategy must define:

- HTF bias rule (both HTFs)  
- LTF trigger (shift/stack/band stated)  
- Invalidation / stand-down  
- Hold horizon bias (~8–10 bars)  
- Partials + time stop (path management)  
- Size tier: flat / small / normal / high (conviction)

---

## 4. Follow-up prompts (use after the first draft)

### 4a — Force independence

```text
Rewrite the portfolio so no two strategies depend on the same single indicator family in the same way. I need independent failure modes so they can agree/vote later (like stoch-turn vs RSI extreme-turn vs structure band).
```

### 4b — Force optimizable MT5 inputs

```text
For every magic number (periods, shifts, level thresholds, ATR multiples), expose an input parameter with a sensible default. Group inputs: HTF periods/shifts, LTF periods/shifts, filters, risk/path (partial %, time-stop bars, size tiers).
```

### 4c — Force stand-down and sizing

```text
Add an explicit portfolio governor: when to stay flat, when only precision strategies may trade, when bread-and-butter may trade, when continuation may size up. Use HTF agreement depth + volatility state (e.g. ATR vs median). No vague language.
```

### 4d — Align to our scoreboard

```text
For each strategy, state the exact LTF bar definition for success (close beyond entry in trade direction at bar 8 and bar 10). List what would count as a failed signal. Do not invent historical win rates — give a test procedure only.
```

### 4e — Port to Python signal agent (optional)

```text
Also give a Python function shape compatible with a signal slot: input M1 OHLCV → resample to set TFs → return series in {+1, -1, 0} aligned to M1. Same rules as the MT5 version. No look-ahead.
```

---

## 5. Rejection checklist (throw the draft away if)

- Entry allowed without HTF bias  
- Only one TF used  
- “Overbought/oversold” as the whole logic  
- No shifts / no relative stack — pure retail cross  
- No stand-down rules  
- No path management (all-or-nothing hold)  
- Claims “80–90% win rate” with no test design  
- Hardcoded periods with no inputs  
- Strategies are near-duplicates (same indicator, same failure mode)

---

## 6. After the LLM answers — human / bot pipeline

1. Extract rules into a short spec (set, long, short, stand-down).  
2. Implement or port to `signals/*.py` style handler.  
3. Score on drill + curriculum: **5 / 10 / 20 LTF bars**, report **n, hit%, bps**.  
4. Keep only what beats or complements existing families (see PART4 agreement bar).  
5. Register slot only after evidence; wire `KIND_HANDLERS`.  
6. CMO may suggest reward/skill changes when policy_hold ignores strong slots — not new indicators first.

---

## 7. One-line instruction to the LLM

**Design a small portfolio of shifted, multi-TF, HTF-gated scalps with stand-down, conviction size, and 8–10 bar path management — optimizable MT5 inputs, independent enough to agree later, none required to be an 80% hero alone.**

---

*File location: repo root `how to make llm create better strategies.md`. Do not delete PERFORMANCE_IS_POSSIBLE* files. Update this playbook when a new prompt pattern reliably produces better scored agents.*
