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
| `prove_it` with PROVEN brain | **PASS** 2026-07-25 — see baseline below |
| Target / floor as variables | Documented; self-state in obs; CLI on prove_it |

### Preflight last result

- SETS A/B/C locked  
- `w_pullback_with_htf=0.25`  
- goal/floor defaults 3.0 / 3.5 in yaml (override on CLI)  
- PROVEN_SPRINT checkpoint present  
- Warning only: `regime_language CODE_SETS may lag engine SETS` (non-blocking)

---

## CRITICAL BUG — FIXED 2026-07-25

### Was

```text
RuntimeError: mat1 and mat2 shapes cannot be multiplied (90x6820 and 1820x128)
```

Also: remote `d6313e9` accidentally gutted `features/engine.py` (lost forever masks + full S1_perm/trig states) → `KeyError: mask_buy_blocked`.

### Fix applied locally

1. Restored full engine (masks, cont/pull/rev, S*_perm/trig) from pre-rewrite logic.
2. Gated 500 signal slots via `configs/features.yaml` `include_signal_agent_slots: false` + engine reads the flag.
3. Deleted stale `artifacts/gpu_cache_XAUUSD_curriculum_2026.npz` and rebuilt.

### Baseline (PROVEN_SPRINT_row04_clear24_2026-07-20 @ 3.0% / 3.5%)

```text
obs: 170 market cols | frame_dim=1820 | days=90
cleared (hit target, NO breach):      21% of days
breached the risk floor:               0% of days
longest cleared streak in a row:       2 days
average day result:                +0.17%
median day result:                 -0.40%
green days (made money):              46% of days
best / worst day:                  +6.46% /  -3.47%
```

**Breach 0% holds.** Clear rate 21%, streak 2 — disease still open is **policy_hold** (setup visible → policy stands down).

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

## Consistency climb

See **[CONSISTENCY_PLAN.md](CONSISTENCY_PLAN.md)** — phases 21→27→35→50 clear, breach gate, policy_hold cure, sprint/meta sequence.

## Next work (priority)

1. ~~Unblock prove_it~~ **DONE** — baseline recorded.  
2. ~~Confirm breach 0~~ **DONE** (0% @ 3.0/3.5).  
3. Self-heal on policy_hold (diagnose running / next).  
4. Short then long consistency_sprint.  
5. Meta-tuner after P1 (≥27% clear).  
6. Later: new brain if signal slots ON.  
7. **Commit + push** restored `features/engine.py`, HANDOFF, CONSISTENCY_PLAN.

---

## One-line summary

> prove_it PASS @ 3.0/3.5: frame_dim=1820, clear 21%, breach 0%, streak 2, avg +0.17%; signals gated off; engine restored; next = self-heal policy_hold + GPU sprint.

*Baseline recorded 2026-07-25.*
