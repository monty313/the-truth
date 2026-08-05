# Colab SIGON Run

NEW brain (~6820 when signals ON). Do not load PROVEN 1820.

## Cache (plain English)
Delete once after signals/symbols change:
`artifacts/gpu_cache_*.npz` and `artifacts/symbol_cache/*`
Never delete `.pt` brains when clearing caches.

## Train
```bash
python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600
```
- 40% days @ 2.5%/3.5%; 60% random 2.5–70.5% / 1–4%
- 3 day-retries per parallel instance
- Multi-symbol: XAUUSD EURUSD GBPUSD US30
- Live board: `artifacts/llm_curriculum/day_board.json`
- Champion: `artifacts/checkpoints/best_sigon.pt`
- HUD: `hud/iron_man_sigon.html`

## Spreads
XAUUSD & US30: razor-thin, typically no txn fees. EURUSD & GBPUSD: normal FX.
