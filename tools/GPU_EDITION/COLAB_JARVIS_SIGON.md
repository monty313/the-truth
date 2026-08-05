# Advanced notes (optional)

**You do not need this file to train.**

Main path:

1. Open: https://colab.research.google.com/github/monty313/the-truth/blob/main/GPU_EDITION/Momentum_One_RunAll.ipynb  
2. Follow **STEP 1** then **STEP 2** in that notebook.

---

## All 4 symbols (what STEP 2 runs)

```bash
python scripts/gpu_train.py --csv-dir data --symbols XAUUSD,EURUSD,GBPUSD,US30 --instances 4000 --minutes 600 --entropy-coef 0.03 --warm best_sigon
```

| Flag | Simple meaning |
|------|----------------|
| `--symbols ...` | All 4 markets |
| `--instances 4000` | How many parallel practice games (try 2000 if out of memory) |
| `--minutes 600` | How long to keep training |
| `--warm best_sigon` | Continue from last good SIGON brain if dims match |

---

## First run is slow (normal)

Each big CSV must build features **once** (often 10–40 min per symbol).  
**Do not press Stop** during `building features`.  
After that, caches make restarts much faster.

Training has started when you see:

```text
upd    1 | ...
```

---

## Price data on Drive

Folder: **Camillion_data**  
Needs CSVs that include names: XAUUSD, EURUSD, GBPUSD, US30.

---

## Jarvis (optional talk channel)

Only after STEP 2 is stopped or finished:

```bash
python scripts/jarvis_talk.py status
python scripts/jarvis_talk.py board
```

---

## Hard rules (short)

- Signals ON → new ~6820 brain (not PROVEN 1820)
- Lock champion only on new high streak with breach 0%
- Clear feature caches only — never delete `.pt` brains
