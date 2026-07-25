# HANDOFF — Momentum One / the-truth
**Date:** 2026-07-25  
**Owner:** Monty  
**Repo:** https://github.com/monty313/the-truth  
**Local path (Windows):** `C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth`

---

## What this project is

Self-healing RL scalper bot (Momentum One):

1. Hit a **daily profit target %** and never breach a **daily risk floor %** (both are **inputs you can change anytime** — no full retrain).
2. Trade via MT5 using multi-timeframe **Gravity Framework**.
3. Improve by **reward shaping + meta-tuner**, not by wiping neural weights.
4. LLM (Fable 5 / CMO) diagnoses failures from telemetry (Mind Probe, Ghost Trades) and proposes gated cures.

**North star:** consistency (clear the target without floor breach, day after day).

---

## Machine status (as of this handoff)

| Check | Status |
|--------|--------|
| Git clone of `the-truth` | Done |
| `training/meta_tuner.py` | On main; restore script OK |
| `python scripts/preflight_train.py` | **PASSED** |
| `prove_it` with PROVEN brain | **Crashed** until signal slots disabled (see below) |
| Target / floor as variables | Documented; self-state in obs; CLI on prove_it |

### Preflight last result

- SETS A/B/C locked  
- `w_pullback_with_htf=0.25`  
- goal/floor defaults 3.0 / 3.5 in yaml (override on CLI)  
- PROVEN_SPRINT checkpoint present  
- Warning only: `regime_language CODE_SETS may lag engine SETS` (non-blocking)

---

## CRITICAL BUG (must fix before prove_it works)

### Error

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (90x6820 and 1820x128)
```

### Cause

- **PROVEN_*** brains were trained at **obs frame dim ≈ 1820** (10 × (170 + 12 self)).
- Code later added **500 signal slots** → dim **≈ 6820**.
- Old brain cannot load the larger input layer.

### Fix (do on Windows now)

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
git pull origin main

# Wipe stale feature cache
del artifacts\gpu_cache_XAUUSD_curriculum_2026.npz
```

In `features\engine.py`, find the block that always does:

```python
from signals.encode import append_signal_obs
append_signal_obs(F, new)
```

Replace with:

```python
# Signal slots OFF so PROVEN brains match obs size (1820)
pass
```

Or ensure `configs/features.yaml` has:

```yaml
include_signal_agent_slots: false
```

**and** that `features/engine.py` actually reads that flag.  
If remote `engine.py` looks incomplete, restore a full gated copy before scoring.

Then:

```powershell
python scripts\prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

Expect `frame_dim=1820` (not 6820), then clear % / breach % / streak.

**Turning signals ON later** requires a **new training run** (obs expansion). Do not force old checkpoints into the expanded obs.

---

## Non-negotiable design rules

1. **Target % and risk % are RUNTIME INPUTS**  
   - `python scripts/prove_it.py <brain> <target> <risk>`  
   - Brain sees goal/floor in obs self-state.  
   - **Never** retrain from scratch only because target/risk changed.

2. **No catastrophic forgetting** — evolve via reward shaping + meta_tuner + policy practice.

3. **Floor breach = 0% is sacred** on measured runs.

4. **Never delete** PERFORMANCE_IS_POSSIBLE*, SUCCESS_LEDGER, flea-jar, proven checkpoints.

5. **Flea-jar mindset** — nothing is impossible; make it consistent.

---

## Gravity / TF sets (locked)

| Set | LTF | HTFs |
|-----|-----|------|
| A | 1m | 15m, 30m |
| B | 5m | 1h, 4h |
| C | 15m | 4h, 1d |

**Bread-and-butter:** LTF pullback while both HTFs strongly trending.

---

## Disease still open

**Policy hold:** setup visible → policy stands down.  
Cure: reward pressure + GPU practice + self-heal — not more ~55% signal stacks.

---

## Signal agents

500 slots in `configs/signal_slots.yaml`. Values +1 / -1 / 0.  
Agreement 80–83 tested ~70–81% (see PERFORMANCE_IS_POSSIBLE_PART4.md).  
Currently **OFF** for PROVEN dim compatibility.

---

## Key commands

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
git pull origin main
python scripts\restore_meta_tuner.py
python scripts\preflight_train.py
python scripts\prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
python scripts\prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 2.5 2.5
# optional later:
python scripts\consistency_sprint.py --minutes 60 --envs 64
```

---

## Next work (priority)

1. Unblock prove_it (signals off + cache delete) → record clear/breach/streak.  
2. Confirm breach 0.  
3. Self-heal on policy_hold.  
4. Short GPU sprint.  
5. Meta-tuner after baseline exists.  
6. Later: new brain if signal slots ON.

---

## One-line summary

> Preflight PASSED; prove_it fails 6820 vs 1820 until signal slots gated off + cache deleted; target/risk are runtime variables; disease is policy_hold; agreement signals exist but offline for dim compatibility.

*Update this file when prove_it baseline numbers are recorded.*
