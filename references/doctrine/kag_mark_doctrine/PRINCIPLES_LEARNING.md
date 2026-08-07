# Principles and Learning (Not Copying)

Mark’s brain does not store a finite table of “if RSI then buy.”  
It stores **principles** and a **learning process** that re-derives action under new sensors, new days, and forward markets.

The meta-learning RL clone must do the same. KAG agents are the **teachers** of that process.

---

## Copy vs learn

| Mode | What it does | Failure in forward test |
|------|----------------|-------------------------|
| **Copy** | Memorizes indicator names, fixed thresholds, bar recipes from training days | Novel indicator or regime shift → freeze or thrash |
| **Learn (Mark)** | Learns roles, relations, decision chain, goal/heat judgment | Maps new sensors into old physics; still hits daily consistency |

**Law:** Training labels are **examples of principle application**, not the principle itself.  
Teacher agents must emit *why the topology is slingshot_load*, not only `action=HOLD`.

---

## What is learned (the transferable core)

1. **Decision chain habit** — tide → regime → breath/launch → act → finish (order is skill).
2. **Role assignment function** — any sensor → force | inertia | velocity | equilibrium | regime_gate | …
3. **Relational predicates** — with/against, intact/collapsed, inside/outside tunnel, multi-set agree.
4. **Topology → act map** — load/wait, release/fire, launch/ride, collapse/kill, chop/no-trade.
5. **Finish-line judgment** — size vs goal/floor/heat; still Mark at close.
6. **Generalization under distribution shift** — same principles on unseen days and unseen indicator instances.

What is **not** the soul: a single indicator’s absolute level, a single TF scream, agent vote spam.

---

## Zero-shot indicator law (never seen before)

If the bot meets indicator X never in training:

```
X is not “unknown → ignore forever”
X → infer Role from:
  - period relative to another instance of same family (fast vs slow)
  - timeframe slot in official set (anchor vs support)
  - shape (oscillator vs bound vs mid vs volatility)
  - relation to already-mapped force/velocity (with/against)
Then run decision chain with X only as that Role.
If role cannot be inferred with confidence → treat as untrusted sensor (mask), not reverse side.
```

Consistency target: long-term **daily goal reach rate**, not “I used this indicator in backtest.”

---

## Meta-learning job

Meta learns **how to attend and trust**, not a new soul:

| Meta may | Meta must not |
|----------|----------------|
| Raise attention to pullback-with-HTF topology | Overwrite PROVEN without order |
| Down-weight sensors that lie in chop | Rewrite Mark sets |
| Transfer role-mapper across families | Change BUY/SELL/HOLD meanings |
| Use teacher lessons as shaped rewards / BC | Fit forward window as train |
| Improve wait_loaded vs freeze discrimination | Reintroduce trail+cushion+scale-in silently |

---

## Teacher lesson object (canonical)

Every KAG teacher emission to the bot should look like:

```json
{
  "lesson_type": "principle_application",
  "not": "copy_answer",
  "set_id": 2,
  "sensors": [
    {"name": "CCI", "period": 100, "tf": "1h", "role": "inertia"},
    {"name": "CCI", "period": 30, "tf": "5m", "role": "velocity"},
    {"name": "NEW_OSC", "period": 14, "tf": "5m", "role": "velocity", "novel": true, "why_role": "fast oscillator on LTF anchor"}
  ],
  "relations": ["inertia_with_tide", "velocity_against", "G_fixed"],
  "topology": "slingshot_load",
  "act": "wait_loaded",
  "principle_ids": ["dual_period_tension", "ltf_never_votes_side", "wait_is_skill"],
  "goal_link": "preserve dry powder for resume; daily target still reachable",
  "forward_note": "same topology if NEW_OSC replaced by RSI or Stochastic"
}
```

The bot trains on **principle_ids + relations + topology**, with concrete sensors as **instances**.

---

## Forward testing preparation

Forward test is the exam for learning (not copying):

1. **Same mind** — train labels and live acts use identical action semantics.
2. **Novel days** — topologies recur; exact indicator paths may not.
3. **Novel sensors** — role mapper must fire without retraining from scratch.
4. **Consistency metric** — fraction of days goal hit without floor breach; not average PnL alone.
5. **Teacher on-line** — KAG agents may continue teaching in forward (HITL / shadow lessons); meta absorbs attention updates under gates.
6. **No thrash learning** — stop → instant reverse is not Mark learning; it is copy of noise.

---

## Feel sentence (clone internal monologue)

> “I don’t need to have trained on this line. I need to know if it’s mass or speed, on which clock, with or against the other clocks, and whether my day still allows fire. Then I wait or fire like I always do.”

That is learning. Copying would be: “I never saw NEW_OSC → HOLD uncertain.”
