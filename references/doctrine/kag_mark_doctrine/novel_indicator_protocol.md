# Novel Indicator Protocol — Meaning Without Prior Training

**Goal:** The bot handles indicators it never saw or used in training by assigning **relational meaning**, then running the same decision chain. This is required for consistent forward testing when feature packs evolve.

---

## Core claim

An indicator’s **name** is not knowledge.  
Its **role in a relation** is knowledge.

```
UnknownName(period, tf, shift, params)
    → Role
    → Relation to other Roles on official sets
    → Topology
    → Act
```

If the chain cannot complete safely → **mask sensor** (untrusted), not invent opposite tide.

---

## Step 0 — Never panic

| Bad reflex | Mark reflex |
|------------|-------------|
| “Not in training → HOLD uncertain” | “What job would this sensor have if it were mine?” |
| “New line → ignore entire chart” | “Keep force from known mass; treat novel as optional velocity/confirm” |
| “Fit new weights overnight on forward” | “Role-map zero-shot; meta attention only under gate” |

---

## Step 1 — Slot the clock

Using [[Mark Sets Law]] slots only:

| If TF is… | Default role bias |
|-----------|-------------------|
| Set anchor (1m/5m/15m/30m first) | velocity / timing / breath |
| Set support (2nd or 3rd) | force / inertia / regime / mass |
| Unknown TF outside sets | **reject** until human adds set (do not invent) |

---

## Step 2 — Classify family shape

| Shape cues | Candidate roles |
|------------|-----------------|
| Oscillator bounded or zero-centered (RSI, CCI, Stoch, W%R, Momentum) | velocity (fast period) or inertia (slow period) |
| Midline / MA of price | equilibrium or force (slow/HTF) |
| High/Low band or envelope / BB rails | expansion + force (tunnel) |
| Width / ATR / stdev | regime_gate / volatility |
| +DI/−DI / ADX-like | regime strength / directional mass |
| Volume cumulative | volume_confirm (never sole tide) |
| Forward shift on structure | mass tunnel displacement (see forward-shift purpose) |

---

## Step 3 — Period relativity (dual clock)

If another instance of same family exists:

| Relative period | Role |
|-----------------|------|
| Shorter period | velocity |
| Longer period | inertia / force |
| Same period different TF | TF slot decides (HTF heavier) |

If alone: pair with **existing** force from another family on HTF before allowing fire. Lone novel oscillator never defines tide.

---

## Step 4 — Build relations (the meaning)

Compute only relative predicates:

- vs own baseline (MA of itself): above/below → with/against local kinetic
- vs price equilibrium: stretched / mean
- vs HTF G / force side: with_tide / against_tide
- vs efficiency/ADX gate: trusted / masked
- multi-set: agree / conflict

**Meaning of novel sensor** = these predicates in context, not its print value alone.

---

## Step 5 — Topology then act

Reuse known topologies:

| Relations | Topology | Act |
|-----------|----------|-----|
| Force fixed + novel velocity against + inertia intact | slingshot_load | wait_loaded |
| Force fixed + velocity re-aligns | slingshot_release | fire with tide |
| Fast+slow novel/known co-aligned expand | launch | ride / add rules |
| HTF force flips | collapse | kill / no new risk |
| Efficiency dead | chop mask | wait_no_trade |

---

## Step 6 — Confidence and masking

| Confidence | Behavior |
|------------|----------|
| High (shape+TF+period clear, force from known sensors) | full role in chain |
| Medium | role allowed for timing only; cannot move tide |
| Low | mask novel channel; trade only on known composition |
| Conflict with known force | known force wins; novel demoted |

Log every novel binding to KAG: `NovelBinding(sensor → role, conf, day)` for teacher review. Do not delete prior bindings; supersede with higher-conf evidence.

---

## Meta-RL training hooks (learn the protocol)

1. **Hold-out families in train:** drop one oscillator family from inputs some episodes; teacher still labels topology from remaining + synthetic role for held-out.
2. **Role-map aux head:** predict role distribution for each channel; loss vs teacher role.
3. **Permutation / rename aug:** randomly rename feature channels; labels are role/topology invariant.
4. **Forward shadow:** when live adds a new MT5 buffer, teacher runs this protocol; student learns online attention under gate — no full retrain required for meaning.

---

## Examples (feel)

### Never trained on Stochastic; knows RSI dual

- Stoch(5) on 5m → velocity (fast osc, LTF anchor)
- Stoch(20) on 1h → inertia/force bias (slow, HTF support)
- Same tension law as dual CCI: slow with, fast against → wait_loaded

### Never trained on DeMarker

- DeM on LTF → velocity candidate
- Requires HTF envelope/MA force still present
- Extreme DeM against intact G → load, not reverse

### Brand-new custom buffer “flow_z”

- Unknown shape → confidence low → mask for tide
- If correlates like oscillator (mean-revert series) teacher may bind velocity after evidence
- Until then: known mass + known velocity still run the day

---

## Forward-test acceptance

Novel protocol is working when:

- Adding a new allowed indicator does **not** require rewriting decision chain
- Daily goal consistency does not collapse on first week of new sensor
- Teacher and student agree on role≥ threshold before size-up uses novel channel
- No increase in thrash attributable to “unknown → random”

---

## One line

**Don’t memorize the soldier’s name — know whether they’re infantry or artillery, which hill they’re on, and whether they move with or against the army you already trust.**
