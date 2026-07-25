# DO THIS

**Fresh chapter:** [START_FROM_TODAY.md](START_FROM_TODAY.md)  
**Wins (never erase):** [doctrine/SUCCESS_LEDGER.md](doctrine/SUCCESS_LEDGER.md)

## 1. Setup
```bash
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
```

## 2. Baseline (know today’s line)
```bash
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

## 3. Self-heal epoch
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```

## 4. Climb (GPU)
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <new_brain> 3.0 3.5
```
Log every improvement in SUCCESS_LEDGER.

## 5. Meta-tuner
```bash
python scripts/meta_train.py --minutes 600
```

---
Score that counts: **prove_it** clear % + breach 0%.  
Past wins prove the ceiling is not real.
