# GOAL

> **Source of truth for the mission.**  
> Agents must never invent a different goal.  
> Auto-injected into every Grok session via `AGENTS.md` + `.grok/rules/00_goal.md`.  
> **We may update this file for clarity** — then update `AGENTS.md` GOAL section to match.

---

## One job only

Build and improve **one bot** that **solves for whatever target % and risk % Monty types in — without having to retrain**.

| You type | Bot must |
|----------|----------|
| **Target %** (daily profit goal) | Hit it on as many days as possible |
| **Risk %** (daily floor / max loss) | Never breach it |

**Without retrain** means: change the numbers → same brain → still tries to hit that target and respect that floor.  
**Not** “pick new target → full retrain.”  
**Not** a bot frozen to 3.0 / 3.5 only.

Examples (same `.pt`, different numbers):

```text
python scripts/prove_it.py <brain> 3.0 3.5
python scripts/prove_it.py <brain> 2.5 2.5
python scripts/prove_it.py <brain> 1.5 2.0
```

Or: **`USE/1_prove.bat`** (change the two numbers only — no retrain).

**How learning is split:** The environment computes meaning (normalized relationships in fixed order). The policy only learns attention (what to trust, how much, and when). The meta learns how to re-learn attention when the senses start lying.

---

## How we know we won (at YOUR numbers)

| Word | Meaning | Rule |
|------|---------|------|
| **Clear %** | Days that hit **your** target and never hit **your** floor | **Climb this** |
| **Breach %** | Days that hit **your** floor | **Must stay 0%** |
| **Streak** | Clears in a row at that pair | Climb this |

If a change does not raise clear % or protect breach 0% **at the target/risk under test** → **skip it**.

**Mindset:** the jar lid is **off**. Performance is possible.  
Low clear + 0 breach = hesitation / policy / training — not “market refused.”

AI: [AGENTS.md](AGENTS.md) (**GOAL** + §0) · [references/doctrine/00_LID_OFF_THE_JAR.md](references/doctrine/00_LID_OFF_THE_JAR.md)

---

## What “solves for any input without retrain” means in practice

1. **Goal and floor are runtime inputs** the bot **sees** each day (self-state / episode) — not baked into weights as one forever pair.  
2. **Switch numbers anytime** — same champion `.pt`; no retrain required just to use a new target or risk.  
3. **Training and meta** (when we practice) should cover a **range** of targets and risks so the brain **generalizes**.  
4. **Scoring** is always: `prove_it` **at the pair Monty cares about right now**.  
5. Climbing clear % at a yardstick pair (e.g. 3.0/3.5) is fine for daily work — **success is still: any pair Monty types, no retrain.**

---

## Current champion (yardstick brain)

| | |
|--|--|
| **Brain** | `PROVEN_SPRINT_row04_clear24_2026-07-20` |
| **Where** | `models/` → open **`00_CHAMPION.md`** first |
| **File** | `models/PROVEN_SPRINT_row04_clear24_2026-07-20.pt` |
| **Yardstick score** | ~**24% clear**, **0% breach** @ **3.0 / 3.5** (doc date 2026-07-30) |
| **Next** | Higher clear at yardstick **and** solid when target/risk change |

---

## Improve in this order

1. **Diagnose** — mind probe / IRAC  
2. **Dials / masks** — search, do not hardcode forever  
3. **Practice** — GPU / consistency sprint (vary goal/floor when training allows)  
4. **Prove** — only `prove_it` at **your** numbers counts  
5. **Keep or reject** — `models/` + success ledger  

---

## Easy finds (dyslexia map)

| I need… | Open |
|---------|------|
| What to do today | [DO_THIS.md](DO_THIS.md) or **`USE/`** |
| First file | [00_START_HERE.md](00_START_HERE.md) |
| Folder map | [MAP.md](MAP.md) |
| Active brain | [models/00_CHAMPION.md](models/00_CHAMPION.md) |
| Daily scripts only | [scripts/00_DAILY.md](scripts/00_DAILY.md) |
| Wins | [references/doctrine/SUCCESS_LEDGER.md](references/doctrine/SUCCESS_LEDGER.md) |
| Full history | [references/handoffs/HANDOFF.md](references/handoffs/HANDOFF.md) |
| AI rules | [AGENTS.md](AGENTS.md) |
| Lid-off law | [references/doctrine/00_LID_OFF_THE_JAR.md](references/doctrine/00_LID_OFF_THE_JAR.md) |

---

## Out of scope (unless Monty asks)

New UIs, random indicators, parallel frameworks, long essays without a `prove_it` gate,  
or a bot that only “works” for one frozen target/risk pair,  
or that needs a retrain every time Monty changes target or risk.
