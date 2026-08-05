# HANDOFF — adaptive_rl_brain_7_31_26

For the **next LLM** and for Monty. Plain language. Do not re-ask Phase 1 basics.

---

## 1. What this is

**Parallel lineage** named `adaptive_rl_brain_7_31_26`.

Goal: a self-improving scalper brain that:

- Reads Official Sets 1–4 and Sub-Sets A–E
- Votes Direction/Velocity with three confluence groups
- Tags trades: MINDLESS / WITH_VECTOR / QUALIFIED_MACRO / QUALIFIED_MICRO
- Learns with searchable reward dials
- Stays **off** the PROVEN path until explicitly promoted

### Absolute rule
**Never overwrite PROVEN checkpoints, champion docs, or promote without Monty’s order.**

---

## 2. What was done 2026-07-31 (sleep mission A–C)

### Phase A — stabilize
- Thrash limits in `day_runner.py` / `rewards.py`:
  - max open units = 3
  - reverse cooldown = 100 M1 bars
  - flip-flop penalty = −1.0
- Anti-hold stack confirmed (EOD −25, inactivity, correct-side bonus, MINDLESS wall)
- JSON reports under `checkpoints/`

### Phase B — curriculum
- `real_curriculum.py` picks real calendar days from `data/raw/`
- Doc: `CURRICULUM.md` (5 XAUUSD days from `XAUUSD_curriculum_2026.csv`)
- Multi-TF pack verified on those days

### Phase C — first serious train
- Script: `train_curriculum.py`
- Warm synthetic thrust → fine-tune real days
- Checkpoint: `checkpoints/channel1_curriculum_v1.pt`
- Report: `checkpoints/curriculum_train_report.json`

### Phase D
**Not run** for pure greedy RL. Health gate still fails (pure greedy all-hold on real).

### Multi-pair consistency (GOAL-style 10 inputs) — DONE 2026-07-31
- `ten_pairs.json` — 10 frozen (target%, risk%) pairs, seed 42
- `equity_day.py` — equity % clear/breach engine (bank at target, heat vs floor)
- `score_ten_pairs.py` — claim scorer (`--mode all`)
- Checkpoint: `checkpoints/multi_pair_consistent_v1.pt` (dials + BC policy; **decode=heuristic**)
- **Claim:** all 10 pairs ≥30 clear days, **0 breach** on 90 real XAUUSD days
- Process log: `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md`
- Short map: `MULTI_PAIR_README.md`
- IRAC keep: bar marks + floor-scale size + unified direction signal
- IRAC reject: trail/scale-in (reverted; 0/10)

---

## 3. Latest results (honest)

| Mode | Entries | Hold rate | Mindless | Mean reward | Notes |
|------|--------:|----------:|---------:|------------:|-------|
| Pure greedy real | 0 | 100% | 0% | −1.94 | **collapse** |
| Anti-hold greedy real | 2 | 95.8% | 35.4% | +0.02 | enters once/day |
| Stochastic real | 16 | 66.7% | 29.2% | −1.02 | thrash limits fire |
| Anti-hold greedy synthetic | 2 | 97.5% | 0% | +0.02 | sanity OK |

Thrash control works under sampling (scale blocks + cooldown blocks observed).  
EOD did-nothing fires when greedy freezes (2 × −25).

---

## 4. Map of this folder

```text
lineages/adaptive_rl_brain_7_31_26/
  00_START_HERE.md
  STATUS.md              ← owner snapshot
  HANDOFF.md             ← this file
  CURRICULUM.md          ← real day list
  todo_7_31_26.md        ← sleep mission plan
  rewards.py             ← dials + thrash constants + EOD
  day_runner.py          ← day loop + thrash + majority hooks
  practice_long.py       ← short synthetic sandbox practice
  train_curriculum.py    ← Phase C real train
  real_curriculum.py     ← day picker
  signal_majority.py     ← 92-agent panel
  price_data.py          ← data/raw loader
  perception/            ← eyes + tags
  data/mtf.py
  checkpoints/
    channel1_curriculum_v1.pt
    channel1_sandbox_latest.pt
    curriculum_train_report.json
    curriculum_days.json
    practice_long_report.json
```

Tests: `tests/lineages/adaptive_rl_brain_7_31_26/`  
(incl. `test_thrash.py`, `test_rewards.py`, `test_signal_majority.py`)

---

## 5. Locked numbers

- MINDLESS_PENALTY = −10  
- DID_NOTHING_EOD_PENALTY = −25  
- MAX_OPEN_UNITS = 3  
- REVERSE_COOLDOWN_BARS = 100  
- FLIP_FLOP_PENALTY = −1.0  
- Majority (when enabled): ≥10 active, ≥60% agree, exempt if ≥2 open  
- Policy: Channel1Policy dim 32 — not PROVEN  

---

## 6. What to do next (do not re-ask)

1. Read `STATUS.md` (numbers + owner next action)
2. Fix pure-greedy freeze on real data **or** train with anti-hold decode in-loop
3. Keep all work under this lineage + its tests
4. Do **not** run prove_it / do **not** touch `models/PROVEN*`
5. Optional later: turn majority on for eval only; expand curriculum days

---

## 7. Commands

```text
# thrash + rewards unit tests
python tests/lineages/adaptive_rl_brain_7_31_26/test_thrash.py
python tests/lineages/adaptive_rl_brain_7_31_26/test_rewards.py

# rebuild day list
python lineages/adaptive_rl_brain_7_31_26/real_curriculum.py

# Phase C train + report
python lineages/adaptive_rl_brain_7_31_26/train_curriculum.py

# short synthetic practice
python lineages/adaptive_rl_brain_7_31_26/practice_long.py
```

Use `PYTHONPATH=.;code` from repo root on Windows PowerShell:

```powershell
$env:PYTHONPATH = ".;code"
```

---

## 8. Doctrine (do not drop)

- Flea-jar: `references/doctrine/flea-jar/`
- Signal agents: `code/signals/`
- PROVEN: `models/` — read-only for this lineage

---

**End of handoff.** Next milestone: pure greedy (or accepted decode) with entries on real eval, thrash still limited, then optional longer train (Phase D) only if health gate passes.
