# DO THIS

One path. Do the steps in order.

## Train
```bash
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <new_brain_name> 3.0 3.5
```

## Check the bot’s mind
```bash
python scripts/give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 10
```

## Meta-tuner (rewards / penalties / lr)
```bash
python scripts/meta_train.py --minutes 600
```

---

**Only number that counts:** `prove_it` clear % and breach %.

**More detail:** `TRAINING.md`  
**How to change files without making a mess:** `UPDATE_RULES.md`
