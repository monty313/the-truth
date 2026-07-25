# signals/ — 500 suggestion slots in the observation

| Value | Meaning |
|-------|---------|
| +1 | buy suggestion |
| -1 | sell suggestion |
| 0 | empty slot or flat |

- RL **does not have to** follow these.
- Only slots listed under `filled:` in `configs/signal_slots.yaml` compute a signal.
- All other slots stay **0**.

## Add a strategy
1. Implement a `kind` in `signals/encode.py` KIND_HANDLERS (or reuse pull_set1 / cont_set1 / …).
2. Add one entry under `filled:` with a free index 0..499.
3. Rebuild feature cache (`rm artifacts/gpu_cache_*.npz`) and re-train / warm-start (obs dim grew by 500).

## Example
Slots 0–2 are filled as examples (set1/set2 pull & cont). Slots 3–499 = 0 until you file them.
