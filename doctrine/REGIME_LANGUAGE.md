# Regime Language — the bot's vocabulary for strategy emergence

CHANGE LOG:
- 2026-07-24  created — WHY: Monty requires a shared language so the bot can form strategy combinations from observations and so every live/training decision documents *why* it entered or skipped under HTF trend.
# NEXT EDITOR: append dated WHY; keep this line.

---

## Purpose

Give the policy a **named language** over multi-timeframe Gravity observations so it can:

1. Compose strategies (pullback, continuation, reversal, …) from indicators — not isolated rules.
2. Document state on **every** training and live decision: what regime it saw, what LTF setup was present, and why it acted or held.

Constraint: language is derived from **existing** observation flags where possible (no obs-space break without explicit Monty approval). Emergence stays evidence-gated (SkillOpt-style).

---

## Multi-timeframe roles

| Role | Job |
|------|-----|
| **Higher timeframe (HTF)** | Dominant regime and directional bias (trend / range / transition). |
| **Mid timeframe** | Structure: accepting or rejecting the HTF bias; key levels. |
| **Lower timeframe (LTF)** | Execution, timing, risk control — entries and exits only. |

Composition law (Standing Law 2): no indicator and no TF works alone. LTF triggers are always relative to the same family of signals on larger periods / higher TFs.

---

## Target timeframe sets (Monty lock)

Exactly three sets; first TF = LTF (execution), last two = higher TFs (regime/bias):

| Set | LTF | HTF-A | HTF-B |
|-----|-----|-------|-------|
| A | 1m | 15m | 30m |
| B | 5m | 1h | 4h |
| C | 15m | 4h | 1d |

### Implementation residual (honest)

Code `features/engine.py` SETS today:

| Code | LTF | HTFs | vs target |
|------|-----|------|-----------|
| set1 | 1m | 15m, 30m | matches A |
| set2 | 5m | **30m, 1h** | B target is **1h, 4h** |
| set3 | 15m | **1h, 4h** | C target is **4h, 1d** |
| set4 | 30m | 4h, 1d | extra stack |

Aligning set2/set3 to B/C **changes the meaning of observation columns** and would invalidate frozen brains. Tracked as residual; do not silently change. Until aligned, regime language maps **onto current columns** and documents the mismatch.

---

## Regime vocabulary

### Primary structure regimes (HTF-led)

| Name | Meaning | Derived today from (approx.) |
|------|---------|------------------------------|
| **Trend** | HH/HL or LH/LL structure; directional bias | Sustained `cont_buy` or `cont_sell` (HTF S1_perm both sides aligned + LTF also perm) |
| **Range** | Oscillation between defined support/resistance | Neither side holds cont; alternating weak signals; no sustained perm |
| **Transition** | Trend↔range shift | `rev_buy` / `rev_sell` edges; cont side flip |

### Conditioning regimes

| Name | Meaning | Derived today from (approx.) |
|------|---------|------------------------------|
| **Volatility** | Compression / normal / expansion | `obs::*stretch`, ATR-relative features when present in row |
| **Liquidity** | Drive into highs/lows/imbalance before cont or rev | Residual — no first-class flag yet; do not invent |

### Execution setups (LTF relative to HTF)

| Name | Meaning | Code flag |
|------|---------|-----------|
| **Continuation** | LTF still aligned with HTF bias | `cont_buy` / `cont_sell` |
| **Pullback** | HTF bias intact; LTF dipped (bread-and-butter) | `pull_buy` / `pull_sell` |
| **Reversal** | Bias side flip | `rev_buy` / `rev_sell` |

Bread-and-butter (Standing Laws): **pullback on LTF while both HTFs remain strong-trend**.

---

## Strategy combination (emergence)

Legal compositions are not a closed list. Starting grammar:

```
HTF_regime × Mid_structure × LTF_setup → candidate strategy
```

Examples the bot may learn (evidence-gated):

- Trend × accepting × pullback → bread-and-butter entry with HTF bias
- Trend × accepting × continuation → add / hold with trend
- Trend × rejecting × pullback → skip or fade (policy must learn)
- Transition × * × reversal → reversal family
- Range × * × * → mean-revert family (if evidence supports)

The regime matrix is an **open starting set** — additional combinations may emerge if Ghost Trades + adopt_gate support them.

---

## Mandatory state documentation

**Whenever training or live**, each decision records at least:

1. `htf_regime` — trend_bull / trend_bear / range / transition / unknown  
2. `ltf_setup` — pullback / continuation / reversal / none  
3. `setup_side` — buy / sell / none  
4. `mask_veto` — buy_blocked / sell_blocked / none  
5. `chosen_op` + whether it **matches** the setup  
6. `skip_reason` — if HTF trend and (pull or cont) were present but policy held:

| skip_reason | Meaning |
|-------------|---------|
| `policy_hold` | Setup visible; op was hold — Policy candidate |
| `mask_veto` | Forever mask blocked the side |
| `no_ltf_setup` | HTF bias on; no pull/cont trigger on LTF |
| `no_htf_bias` | No clear HTF trend permission |
| `flat_ok` | No setup; hold is consistent |
| `acted` | Took an entry/exit op |

This is how we answer: *“HTF was trending — why no pullback/continuation entry?”*

Implementation: `telemetry/regime_language.py` + fields on Mind Probe `DecisionRecord`.

---

## IRAC use

When clear rate stalls under strong HTF days:

- **Perception** — setup flags never fire (language blind / set mismatch)
- **Policy** — flags fire, `skip_reason=policy_hold` dominates → reward shaping
- **Generalization** — works only on one regime label

Cure remains evidence-backed reward / skill-doc change, not weight reset.
