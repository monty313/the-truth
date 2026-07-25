# signals/ — 500 suggestion slots in observation

| Value | Meaning |
|-------|---------|
| +1 | buy suggestion |
| -1 | sell suggestion |
| 0 | empty / flat / disabled |

RL **does not have to** follow these.

## Slot plan

| Range | Content |
|-------|---------|
| 0–9 | Momentum One natives (pull / cont / rev) |
| 10–27 | Camillion alpha pack (proxied) |
| 28–499 | Free — file new strategies here |

## Manage

```bash
python scripts/manage_signal_slots.py summary
python scripts/manage_signal_slots.py list
python scripts/manage_signal_slots.py list --family camillion
python scripts/manage_signal_slots.py next-free
python scripts/manage_signal_slots.py kinds
```

## Add a strategy

1. Add `kind` handler in `signals/encode.py` if needed  
2. Pick free index: `python scripts/manage_signal_slots.py next-free`  
3. Add under `filled:` in `configs/signal_slots.yaml`  
4. `rm artifacts/gpu_cache_*.npz` and retrain  

## Fidelity

- **native** — exact MO Gravity flags  
- **proxy** — Camillion idea mapped onto MO pull/cont/strength  
- **weak_proxy / stub** — incomplete until indicators ported (ORB, ADX)  
