# How to update this repo (keep it clean)

Read this before adding or changing files.

## Rules
1. **One job → one place.** Do not copy the same text into three docs.
2. **Start pages stay short.** `DO_THIS.md` stays under ~30 lines. Long notes go in `docs/` or `doctrine/`.
3. **Every code file keeps a CHANGE LOG** at the top (date + WHY). Append; do not delete old lines.
4. **Configs hold numbers.** No magic constants buried in scripts if they belong in `configs/`.
5. **Scripts are entry points only.** Library code lives in `training/`, `features/`, `telemetry/`, etc.
6. **Do not add a new top-level folder** unless nothing existing fits.
7. **Old writeups** go in `docs/history/`, not the repo root.
8. **Regime / indicator language:** edit only `doctrine/LLM_REGIME_DEFINITIONS.yaml` (append to registries).

## Where to put a change

| Kind of change | Put it here |
|----------------|-------------|
| Daily target / risk | `configs/goals.yaml` |
| Reward or penalty weight | `configs/rewards.yaml` |
| Timeframes / SETS | `configs/timeframes.yaml` + `features/engine.py` SETS (keep in sync) |
| New obs indicator | feature code + register in `doctrine/LLM_REGIME_DEFINITIONS.yaml` |
| Train / diagnose command | `scripts/` + one line in `scripts/README.md` |
| Law / regime meaning | `doctrine/` (not root) |
| Long explanation | `docs/` or `docs/history/` |
| Checkpoint / cache | `artifacts/` only |

## After you change something
```bash
python scripts/preflight_train.py
```
If it fails, fix that before training.

## Naming
- Scripts: `verb_noun.py` (`prove_it`, `restore_meta_tuner`)
- Do not create `meta_train2.py` / `final_final.py` — update the real file
- New brain files: `artifacts/checkpoints/` with a clear name

## Forbidden (creates mess)
- Second “start here” doc that duplicates `DO_THIS.md`
- Pasting the same IRAC into five markdown files
- Leaving `PLACEHOLDER` or empty critical modules on the branch
- Changing SETS without rebuilding feature cache and re-proving the brain
