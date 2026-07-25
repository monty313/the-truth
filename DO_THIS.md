# DO THIS

**Fresh chapter:** [START_FROM_TODAY.md](START_FROM_TODAY.md)  
**Wins (never erase):** [doctrine/SUCCESS_LEDGER.md](doctrine/SUCCESS_LEDGER.md)  
**Agreement evidence:** [PERFORMANCE_IS_POSSIBLE_PART4.md](PERFORMANCE_IS_POSSIBLE_PART4.md)

## Target & risk are variables (not retrain triggers)

You may change **daily target %** and **risk floor %** anytime:

| Mechanism | Role |
|-----------|------|
| `configs/goals.yaml` → `goal_pct` / `floor_pct` | Today's focus pair |
| `goal_conditioning.goal_range` / `floor_range` | Meta-training samples many pairs |
| Obs **self-state** (`goal`, `floor`, `dist_to_goal`, `dist_to_floor`) | Brain **sees** the active pair every step |
| `python scripts/prove_it.py <brain> <target> <risk>` | Measure **same brain** at any pair — **no retrain** |
| `training/meta_tuner.py` | Evolves rewards/hypers for consistency across pairs |

**Do not** retrain from scratch when you only change target/risk.

## 1. Setup
```bash
git pull origin main
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
```

## 2. Baseline at *your* numbers (example only)
```bash
# Example — replace 2.5 2.5 or 3.0 3.5 with whatever you will trade tomorrow
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 2.5 2.5
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```
Log clear % / breach / streak per pair. Breach must stay 0.

## 3. Self-heal epoch (still parameterized)
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```
Disease to kill: **policy_hold** when Gravity + pull/cont or `sig_080–083` is on.

## 4. Climb (GPU) — policy practice under current rewards
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <new_brain> <your_target> <your_risk>
```

## 5. Meta-tuner (any-X consistency)
```bash
python scripts/meta_train.py --minutes 600
```

---
Score that counts: **prove_it clear % + breach 0%** at the target/risk **you** pass in.  
Past wins prove the ceiling is not real.
