# features/

Indicators and Gravity SETS.

## How to update
- SETS must match `configs/timeframes.yaml` and Monty lock (1m/15m/30m; 5m/1h/4h; 15m/4h/1d).
- After SETS change: delete `artifacts/gpu_cache_*.npz`, re-prove the brain.
- New indicator → implement here, list in `configs/features.yaml`, register in `doctrine/LLM_REGIME_DEFINITIONS.yaml`.
- Append CHANGE LOG in `engine.py` (date + WHY).
