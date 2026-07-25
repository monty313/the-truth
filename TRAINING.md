# Training — start here

## 0. Preflight (required)
```bash
python scripts/restore_meta_tuner.py   # if meta_tuner missing
python scripts/align_tf_sets.py        # no-op if already aligned
python scripts/preflight_train.py      # must PASS
```

## 1. Diagnostic pack (optional but recommended)
```bash
python scripts/give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```

## 2. Climb consistency (GPU)
```bash
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <sprint_record_brain> 3.0 3.5
```

## 3. Meta-tuner (reward/penalty + hyperparams wrapper)
```bash
python scripts/meta_train.py --minutes 600
```

## Invariants
- No core weight retrain from scratch (warm-start only)
- Observation columns: freeze meaning after SETS lock; rebuild cache if SETS change
- Zero-breach target; clear rate measured only via `prove_it`
- Regime SSOT: `doctrine/LLM_REGIME_DEFINITIONS.yaml`
- Standing laws: `doctrine/STANDING_LAWS.md`

## Layout (where things live)
| Path | What |
|------|------|
| `configs/` | goals, rewards, timeframes, training knobs |
| `features/` | Gravity SETS + indicators |
| `training/` | Brain, PPO, fastsim, **meta_tuner** |
| `telemetry/` | Mind probe, ghosts, regime language |
| `scripts/` | All entry points |
| `doctrine/` | Laws, regime defs, HOST_RUN |
| `artifacts/checkpoints/` | PROVEN + sprint brains |
| `data/` | M1 curricula |
| `docs/history/` | Old handoffs / performance writeups |
