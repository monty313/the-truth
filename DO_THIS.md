# DO THIS

Do these steps in order. Nothing else required for daily train.

## 1. Setup (first time / if something broke)
```bash
pip install -r requirements.txt
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
```
Preflight must say **PASSED**.

## 2. Train (GPU best)
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
```

## 3. Measure
```bash
python scripts/prove_it.py <brain_name> 3.0 3.5
```
Replace `<brain_name>` with the new file name printed by the sprint  
(or use `PROVEN_SPRINT_row04_clear24_2026-07-20`).

## Optional
```bash
# See what the bot is thinking
python scripts/give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 10

# Meta-tuner (adjusts rewards / penalties / lr)
python scripts/meta_train.py --minutes 600
```

---
More detail → `TRAINING.md`  
Folder map → `MAP.md`  
How to edit files → `UPDATE_RULES.md`
