# DO THIS

Do these steps in order.

## 1. Setup
```bash
pip install -r requirements.txt
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
```
Preflight must say **PASSED**.

## 2. Self-heal epoch (self-correct + self-improve)
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 10
```
Adds trajectories + IRAC + prove_it gate. Skill stays pending until you accept.

### Accept skill when evidence is solid
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 10 --auto-accept-skill
```

### Full improve cycle (frontier train + gate + skill + optional reward nudge)
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 10 --sprint-minutes 120 --auto-accept-skill --apply-reward-nudge
```

## 3. Long GPU climb (when ready)
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <new_brain> 3.0 3.5
python scripts/self_heal_epoch.py <new_brain> 3.0 3.5 --auto-accept-skill
```

## 4. Meta-tuner (reward/penalty wrapper)
```bash
python scripts/meta_train.py --minutes 600
```

---
**Only score that counts:** `prove_it` clear % and breach %.  
**CMO doctrine:** `doctrine/SYSTEM_DOCTRINE_CMO.md`  
**Skill memory:** `doctrine/policy_skill.md`  
**Epoch logs:** `artifacts/self_heal_epochs/`
