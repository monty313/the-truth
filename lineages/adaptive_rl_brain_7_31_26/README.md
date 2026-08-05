# adaptive_rl_brain_7_31_26

**Parallel brain line.** Not PROVEN. Not the champion.

| Start here | |
|------------|--|
| **Owner** | [`00_START_HERE.md`](00_START_HERE.md) |
| **Snapshot** | [`STATUS.md`](STATUS.md) |
| **Next LLM** | [`HANDOFF.md`](HANDOFF.md) |
| **Phase 1 rules** | [`SPEC_PHASE1.md`](SPEC_PHASE1.md) |

---

## Hard rules

1. **Never touch PROVEN** (`models/PROVEN_*.pt`, living champion).  
2. **Flea-jar doctrine stays** — `references/doctrine/flea-jar/`.  
3. **Signal agents stay** — `signals/` (system-wide; this line uses Vector/set path first).  
4. New work lives **in this folder** (+ `tests/lineages/adaptive_rl_brain_7_31_26/`).

---

## Modules (what each file does)

| Path | Job |
|------|-----|
| `perception/types.py` | Enums / dataclasses |
| `perception/sets.py` | Official Sets 1–4, Sub A–E |
| `perception/confluence.py` | 3 groups → Direction + Velocity |
| `perception/structure.py` | Pullback + Scale-Conflict |
| `perception/classify.py` | Four tags + MINDLESS wall |
| `perception/live_indicators.py` | Real CCI/RSI/channel → votes |
| `perception/pipeline.py` | Live end-to-end assessment |
| `perception/observation.py` | Channel 1 vector (dim **32**) |
| `data/mtf.py` | M1 → multi-TF pack |
| `day_runner.py` | Day loop: obs, action, reward |
| `policy_stub.py` | Tiny policy (not PROVEN) |
| `rewards.py` | Dials + credit + MINDLESS −10 |
| `train_stub.py` | Smoke train / rollout |

---

## Tests

```text
tests/lineages/adaptive_rl_brain_7_31_26/
```

Run from repo root. Keep green before bigger changes.

---

## Status one-liner

Phase 1 + Phase 2 (through multi-TF + train stub) **built**.  
**Next:** first real training test **after** Monty approves organization.
