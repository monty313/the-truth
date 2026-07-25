# COLAB / Local GPU — SIGON Run Book

**SIGON** = Signal-ON multi-symbol GPU train → `best_sigon.pt`  
**Gate:** `prove_it` clear↑ and **breach = 0%**. Never load PROVEN 1820 weights into expanded obs.

---

## 1. Preflight (Windows or Colab)

```bash
cd /path/to/the-truth   # or C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
git pull origin main
```

### Enable signal slots (required for SIGON lineage)

Edit `configs/features.yaml`:

```yaml
include_signal_agent_slots: true
```

### Wipe stale caches

```powershell
# Windows
del artifacts\gpu_cache_*.npz
if (Test-Path artifacts\symbol_cache) { Remove-Item -Recurse -Force artifacts\symbol_cache }
```

```bash
# Linux / Colab
rm -f artifacts/gpu_cache_*.npz
rm -rf artifacts/symbol_cache
```

### Data

Put M1 CSVs under `data/`:

| File pattern | Required? |
|--------------|-----------|
| `XAUUSD_curriculum_*.csv` | **Yes** |
| `EURUSD*.csv` | Optional (skipped if missing) |
| `GBPUSD*.csv` | Optional |
| `US30*.csv` | Optional |

---

## 2. Train

```bash
# Full SIGON (multi-symbol pool, 3 day-retries, day_board, best_sigon)
python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600

# OOM on Colab
python scripts/gpu_train.py --csv-dir data --instances 4000 --minutes 600

# Single CSV fallback
python scripts/gpu_train.py --csv data/XAUUSD_curriculum_2026.csv --instances 2000 --minutes 120
```

Defaults from `configs/sigon_train.yaml`: `max_day_retries: 3`, `instances_default: 8000`.

Sampling target/risk via `evaluation.consistency.auto_ranges()` ← `configs/goals.yaml`.

---

## 3. While training

| Signal | Where |
|--------|--------|
| Logs | `clear %`, `breach %`, `retries` on each update |
| Day board JSON | `artifacts/llm_curriculum/day_board.json` |
| Progress | `artifacts/checkpoints/gpu_progress.json` |
| Champion | `artifacts/checkpoints/best_sigon.pt` |
| History | `artifacts/checkpoints/history/SIGON_row*_obs*_SN-*.pt` |
| HUD | open `hud/iron_man_sigon.html` (local HTTP server recommended) |

```bash
# optional: serve HUD + board
python -m http.server 8765
# browser: http://localhost:8765/hud/iron_man_sigon.html
```

---

## 4. Signal accuracy (offline)

```bash
python scripts/score_signal_accuracy.py --csv data/XAUUSD_curriculum_2026.csv --slots 80,81,82,83
# → artifacts/signal_accuracy/latest.json + agree_80_83.json
```

---

## 5. Measure (after train)

```bash
# Only works if include_signal_agent_slots matches the brain (true for SIGON)
python scripts/prove_it.py best_sigon 3.0 3.5
```

**ACCEPT** if clear ≥ previous best and breach == 0.  
**REJECT** if breach > 0.

For old PROVEN brains: set `include_signal_agent_slots: false`, delete caches, then:

```bash
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

---

## 6. Hard rules

1. SIGON = **new** `Brain(obs_dim≈6820)` — never force PROVEN 1820 state_dict.  
2. Breach 0% on prove_it or discard.  
3. Delete caches when flipping the signal flag.  
4. Never delete PERFORMANCE* / SUCCESS_LEDGER / flea-jar / PROVEN_*.  

See also: `HANDOFF_CONSISTENCY_TO_COLAB_SIGNALS.md`.
