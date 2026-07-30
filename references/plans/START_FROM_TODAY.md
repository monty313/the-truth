# Start from today (2026-07-25)

New chapter. **Successes stay** (see `doctrine/SUCCESS_LEDGER.md`).  
Nothing historically winnable is “impossible.”

## 0. Remember the wins
Open **[doctrine/SUCCESS_LEDGER.md](doctrine/SUCCESS_LEDGER.md)** once.  
0% breach is already proven. Clear rate is the climb.

## 1. Machine setup
```bash
git pull origin main
python scripts/restore_meta_tuner.py   # until meta_tuner is always in the clone
python scripts/preflight_train.py      # must PASSED
```

## 2. Honest baseline (today’s starting line)
```bash
rm -f artifacts/gpu_cache_*.npz   # only if SETS/features changed
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```
Write the clear % / row / breach into SUCCESS_LEDGER if it moved.

## 3. Self-heal epoch (diagnose from evidence)
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```

## 4. Improve (GPU when you can)
```bash
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 \
  --days 12 --sprint-minutes 120 --auto-accept-skill --apply-reward-nudge
```
Longer climb:
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <new_brain> 3.0 3.5
```
Append every **breach=0** improvement to SUCCESS_LEDGER.

## Rules for this chapter
- Warm-start from PROVEN_* — no from-scratch wipe
- Gate: breach must stay **0%**
- Skill/reward changes only with IRAC counts + prove_it
- PERFORMANCE_IS_POSSIBLE* files are **never deleted**
- Target: beat **21%** clear under locked sets, then beat **27%**, then stretch the row

## Persona
`doctrine/SYSTEM_DOCTRINE_CMO.md` — CMO + Lead Quant, stack/shift/band, cure order rewards → periods → logic last.
