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
| 28–83 | Filled packs (sma/rsi/stoch/agree/…) |
| 84 | DVMR champion sticky (1h+1d, best research return) |
| 85 | DVMR 30m+4h v2 sticky (runner-up) |
| 86 | DVMR champion pulse (1h+1d entry bars only) |
| 87 | MV best quality 30m+4h long thr35 hm25 (~+100% US30) |
| 88 | MV strong#3 same entries as 87 (BT SL1.5 TP3.5) |
| 89 | MV strong#4 30m+4h long thr25 hm12 (~+81% US30) |
| 90 | BB/RSI-SMA Set A (5m + 30m/1h) |
| 91 | BB/RSI-SMA Set B (15m + 1h/4h) |
| 92 | BB/RSI-SMA Set C (30m + 4h/1d) |
| 93–499 | Free — file new strategies here |

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
