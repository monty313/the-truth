# Host Run — Give the LLM what it needs + climb to 80%

```bash
git checkout fable5/self-heal-plan
python scripts/restore_meta_tuner.py
python scripts/align_tf_sets.py

# Full diagnostic curriculum (cache + probe + ghosts + IRAC + prove_it)
python scripts/give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 15

# Frontier training (GPU hours — the missing piece)
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <sprint_record> 3.0 3.5
python scripts/give_llm_what_it_needs.py <sprint_record> 3.0 3.5 --days 15
```

Invariants: 0% breach, no from-scratch retrain, regime language on every decision.
Multi-symbol: drop M1 curricula for EURUSD/GBPUSD/US30 into data/ when ready.
