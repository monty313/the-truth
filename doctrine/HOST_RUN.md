# Host Run Playbook — Self-Heal Phase 4 Climb

WHO: Monty on GPU/Colab host with curriculum data + PROVEN brains.  
WHAT: measured consistency climb after merging `fable5/self-heal-plan`.  
WHEN: after PR #1 merge.  
WHY: sandbox cannot invent clear-rate numbers without data.

## Prerequisites

```bash
git fetch origin && git checkout fable5/self-heal-plan
# or main after merge
ls artifacts/checkpoints/PROVEN_SPRINT_row04_clear24_2026-07-20.pt
ls data/XAUUSD_curriculum_2026.csv
```

## 1. Verify MRI + IRAC (no training)

```bash
python tests/test_self_heal_mri.py
python scripts/mind_probe_day.py PROVEN_SPRINT_row04_clear24_2026-07-20 42 3.0 3.5
python scripts/diagnose_day.py PROVEN_SPRINT_row04_clear24_2026-07-20 42 3.0 3.5
python scripts/perception_scoreboard.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 90
```

## 2. Baseline score at your numbers

```bash
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

## 3. Self-tune climb (reward shaping only; gated adopt)

```bash
python scripts/meta_train.py
# or
python scripts/self_tune.py
```

Warm-start prefers `PROVEN_SPRINT_row04_clear24_2026-07-20`.  
`w_pullback_with_htf` is now in meta_tuner BOUNDS.

## 4. After any adopt — re-measure

```bash
python scripts/prove_it.py meta_best 3.0 3.5
# update doctrine/policy_skill.md CHANGE LOG with measured clear-rate / row delta
```

## Invariants

- No core weight retrain from scratch  
- Obs space frozen  
- Only rewards.yaml / skill-doc evolution that passes adopt_gate  
- Never claim impossible without measured bound  
