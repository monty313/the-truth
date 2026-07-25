# Training (detail)

**Daily path:** [DO_THIS.md](DO_THIS.md)

```bash
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <brain> 3.0 3.5
```

Meta-tuner (rewards/penalties/lr): `python scripts/meta_train.py --minutes 600`

Invariants: no from-scratch retrain · measure only with prove_it · regime SSOT = doctrine/LLM_REGIME_DEFINITIONS.yaml
