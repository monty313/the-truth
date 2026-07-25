# How to change this repo without making a mess

## Before you edit
Ask: *which folder owns this?* (see MAP.md)

## Rules
1. **One fact in one file** — do not copy the same paragraph everywhere
2. **DO_THIS.md stays short** — no essays there
3. **Numbers → `configs/`** only
4. **Commands you run → `scripts/`** only
5. **Code libraries → `training/`, `features/`, `telemetry/`** — not scripts
6. **Laws / regimes → `doctrine/`**
7. **Old long notes → `docs/history/`**
8. **Every code change:** add a line at the top CHANGE LOG: `date — what — WHY`
9. **After changes:** `python scripts/preflight_train.py` must PASS
10. **Do not** create `file_v2.py` or `final_final.md` — edit the real file

## Quick “where does this go?”

| I want to… | Edit this |
|------------|-----------|
| Change daily target or risk | `configs/goals.yaml` |
| Change a reward / penalty | `configs/rewards.yaml` |
| Change timeframes | `configs/timeframes.yaml` **and** `features/engine.py` SETS |
| Add an indicator | `features/` + register in `doctrine/LLM_REGIME_DEFINITIONS.yaml` |
| Add a command | new file in `scripts/` + one line in `scripts/README.md` |
| Change how regimes are named | `doctrine/LLM_REGIME_DEFINITIONS.yaml` only |

## Forbidden
- Second “start here” that replaces DO_THIS.md
- Leaving PLACEHOLDER in `training/meta_tuner.py`
- Changing SETS without deleting `artifacts/gpu_cache_*.npz` and running `prove_it` again
