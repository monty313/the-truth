# DO THIS

**Mission:** [GOAL.md](GOAL.md)  
**Buttons:** [USE/](USE/) ← easiest  
**Daily scripts list:** [scripts/00_DAILY.md](scripts/00_DAILY.md)

---

## Target & risk = dials (not retrain)

Change target/risk anytime. Measure with `prove_it`.  
Do **not** retrain only because the number changed.

---

## Path A — no typing (recommended)

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
