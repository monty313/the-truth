# COLAB SIGON — short runbook

## Cache (plain English)

A **cache** is a saved shortcut of features so we do not rebuild indicators every run.

**After signals ON or new symbols**, delete **once**:

```bash
rm -f artifacts/gpu_cache_*.npz
rm -rf artifacts/symbol_cache
```

```powershell
del artifacts\gpu_cache_*.npz
if (Test-Path artifacts\symbol_cache) { Remove-Item -Recurse -Force artifacts\symbol_cache }
```

**Never delete** `.pt` brain checkpoints when clearing caches.

---

## Config already set for SIGON

- `configs/features.yaml` → `include_signal_agent_slots: true`
- `configs/goals.yaml` → 40% focus @ **2.5% / 3.5%**; random ranges **[2.5, 70.5]** / **[1.0, 4.0]**
- Sampling via `evaluation.consistency.auto_ranges()`

---

## Data

Put M1 CSVs in `data/`:

- `XAUUSD_*.csv` (required)
- `EURUSD_*.csv`, `GBPUSD_*.csv`, `US30_*.csv` (optional; skipped if missing)

**Spread:** XAUUSD & US30 typically razor-thin (often no txn fees). EUR/GBP normal FX. Brain sees `obs::spread_rel`.

---

## One command (Colab L4)

```bash
python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600
```

OOM: `--instances 4000` then `2000` then `1024`.

---

## Outputs

| What | Path |
|------|------|
| Champion | `artifacts/checkpoints/best_sigon.pt` |
| History | `artifacts/checkpoints/history/SIGON_row*_obs*_SN-*.pt` |
| Day board | `artifacts/llm_curriculum/day_board.json` |
| Progress | `artifacts/checkpoints/gpu_progress.json` |
| HUD | `hud/iron_man_sigon.html` (serve repo root; refresh 15s) |
| CMO inbox | `doctrine/cmo_inbox/MONTY_THINK.md` |

**obs_dim:** with signals ON expect **~6820** (`10 × (~670 + 12)`). Never load PROVEN_* 1820 into this brain.

---

## Gate

```bash
python scripts/prove_it.py best_sigon 2.5 3.5
```

ACCEPT only if clear improves and **breach = 0%**.
