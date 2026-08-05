# DO THIS

**Mission:** [GOAL.md](GOAL.md)  
**House map:** [00_MAP_OF_THE_HOUSE.md](00_MAP_OF_THE_HOUSE.md)  
**Current handoff:** [HANDOFF_2026-08-05.md](HANDOFF_2026-08-05.md)  
**Buttons:** [USE/](USE/) ← easiest  
**Daily scripts list:** [scripts/00_DAILY.md](scripts/00_DAILY.md)

---

## Target & risk = dials (not retrain)

Change target/risk anytime. Measure with `prove_it`.  
Do **not** retrain only because the number changed.

---

## Path T3 — Mark soul (active 2026-08-05)

Soul first; agents frozen (`KEEP_AFTER_SOUL.md`).

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
# Full policy soul next: more BC days + thrash penalty
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 40 --max-train-days 40
python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py --seed 7 --start-idx 40
```

Details: `lineages/adaptive_rl_brain_7_31_26/00_TRACK_ORDER.md`

---

## Path A — PROVEN buttons (T1)

Open folder **`USE/`** and double-click:

1. **1_prove.bat** — score  
2. **2_preflight.bat** — ready check  
3. **3_self_heal.bat** — heal epoch  
4. **4_train.bat** — GPU train  

---

## Path B — type commands

### 1. Setup
```bash
git pull origin main
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
```

### 2. Score (baseline)
```bash
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

### 3. Self-heal
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```

### 4. Climb (GPU)
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <new_brain> 3.0 3.5
```

### 5. Meta-tuner
```bash
python scripts/meta_train.py --minutes 600
```

---

## Only score that counts

**prove_it → clear % + breach 0%** at the target/risk you pass in.
