# GOAL

**One job only.**

Hit **daily profit target %**  
without breaching **daily risk floor %**.

**Mindset:** the jar lid is **off**. Performance is possible.  
Clear % climbs under breach **0%**. Low clear is hesitation or training — not “market refused.”  
AI must load [AGENTS.md](AGENTS.md) §0 and [references/doctrine/00_LID_OFF_THE_JAR.md](references/doctrine/00_LID_OFF_THE_JAR.md).

---

## How we know we won

```text
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

**Or double-click:** `USE/1_prove.bat`

| Word | Meaning | Rule |
|------|---------|------|
| **Clear %** | Days that hit target and never breached | **Climb this** |
| **Breach %** | Days that hit the floor | **Must stay 0%** |
| **Streak** | Clears in a row | Climb this |

If a change does not move clear % or protect breach 0% → **skip it**.

---

## Current champion

| | |
|--|--|
| **Brain** | `PROVEN_SPRINT_row04_clear24_2026-07-20` |
| **Where** | `models/` → open **`00_CHAMPION.md`** first |
| **File** | `models/PROVEN_SPRINT_row04_clear24_2026-07-20.pt` |
| **Score** | ~**24% clear**, **0% breach** (2026-07-30) |
| **Next** | Self-heal + GPU practice → aim ≥27% clear |

Target and risk are **runtime inputs**.  
Do **not** retrain from scratch only to change 2.5 / 3.0 / 3.5.

---

## Improve in this order

1. **Diagnose** — mind probe / IRAC  
2. **Dials / masks** — search, do not hardcode forever  
3. **Practice** — GPU / consistency sprint  
4. **Prove** — only `prove_it` counts  
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
| AI rules + lid-off mind | [AGENTS.md](AGENTS.md) |
| Lid-off law (short) | [references/doctrine/00_LID_OFF_THE_JAR.md](references/doctrine/00_LID_OFF_THE_JAR.md) |
| Performance evidence | [references/performance/PERFORMANCE_IS_POSSIBLE.md](references/performance/PERFORMANCE_IS_POSSIBLE.md) |

---

## Out of scope (unless Monty asks)

New UIs, random indicators, parallel frameworks, long essays without a `prove_it` gate.
