# Colab L4 — SIGON + Jarvis (Monty one-path)

**Signals ON** = new brain lineage (**obs_dim ~6820**).  
**Never** load PROVEN_* 1820 weights into this brain.

---

## Cell 0 — GPU

**Runtime → Change runtime type → Hardware accelerator → L4 GPU (or T4) → Save**

---

## Cell 1 — Drive + repo + price data

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/monty313/the-truth.git 2>/dev/null || true
%cd /content/the-truth
!git pull origin main
!pip -q install pyyaml >/dev/null 2>&1
```

### Copy M1 CSVs (XAUUSD, EURUSD, GBPUSD, US30)

Drive folder ID: `1gwTD6535FilsKqRUsi7vi6WlT0vgdhui`  
(URL: https://drive.google.com/drive/folders/1gwTD6535FilsKqRUsi7vi6WlT0vgdhui)

```python
import os, shutil, glob

candidates = [
    "/content/drive/MyDrive/Camillion_data",
    "/content/drive/MyDrive/the-truth-data",
    "/content/drive/MyDrive/MOMENTUM_ONE/02_PRICE_DATA",
    "/content/drive/MyDrive/1gwTD6535FilsKqRUsi7vi6WlT0vgdhui",
]
src = None
for c in candidates:
    if os.path.isdir(c) and glob.glob(c + "/**/*.csv", recursive=True):
        src = c
        break
if src is None:
    for root, dirs, files in os.walk("/content/drive/MyDrive"):
        names = " ".join(files).upper()
        if "XAUUSD" in names and any(x in names for x in ("EURUSD", "US30", "GBPUSD")):
            if any(f.lower().endswith(".csv") for f in files):
                src = root
                break

os.makedirs("data", exist_ok=True)
if src:
    for f in glob.glob(src + "/**/*.csv", recursive=True):
        base = os.path.basename(f).upper()
        if any(s in base for s in ("XAUUSD", "EURUSD", "GBPUSD", "US30")):
            dst = os.path.join("data", os.path.basename(f))
            shutil.copy2(f, dst)
            print("copied", dst)
else:
    print("WARNING: Drive folder not found — upload CSVs to data/ manually")

print("data CSVs:", os.listdir("data"))
```

### Clear feature caches once (after signals ON)

```python
!rm -f artifacts/gpu_cache_*.npz
!rm -rf artifacts/symbol_cache
# NEVER delete artifacts/checkpoints/*.pt brains when clearing caches
# NEVER delete PERFORMANCE*, SUCCESS_LEDGER, flea-jar, PROVEN_*.pt
```

**Config on main:** `include_signal_agent_slots: true` → expect **obs_dim ~6820**.

---

## Cell 2 — TRAIN (leave running)

### FAST START first (Monty — use this if you see `building features...` then `^C`)

Full multi-year CSVs (names like `EURUSD_M1_20210113...2026.csv`) take **hours** to build features once. Colab often stops mid-build (`^C`).  
**Start with gold curriculum only:**

```bash
!ls -lh data/*.csv
!python scripts/gpu_train.py --csv data/XAUUSD_curriculum_2026.csv --instances 4000 --minutes 600 --entropy-coef 0.03 --warm best_sigon
```

You are training when you see `upd 1`, `upd 2`, …  
If curriculum is missing: put `XAUUSD_curriculum_2026.csv` (and other `*_curriculum*.csv`) in Drive folder **Camillion_data**, re-run Cell 1.

### Full multi-symbol (after FAST works)

Prefer **curriculum** files in `data/` (not 2020–2026 full dumps). Code auto-picks curriculum when present.

```bash
!python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600 --entropy-coef 0.03 --warm best_sigon
```

| Flag | Meaning |
|------|---------|
| `--instances 8000` | Parallel envs (default). **OOM?** try `6000` then `4000` then `2000` |
| `--minutes 600` | Run length (10 hours) |
| `--entropy-coef 0.03` | Exploration (higher = more random actions). Warm-start **keeps** weights |
| `--warm best_sigon` | Resume SIGON champion if dims match; skips PROVEN 1820 safely |

| Output | Path |
|--------|------|
| Latest champion pointer | `artifacts/checkpoints/best_sigon.pt` |
| Streak-named lock | `artifacts/checkpoints/best_sigon_streak05.pt` (example streak=5) |
| History freeze | `artifacts/checkpoints/history/SIGON_streak05_obs6820_SN-xxxx.pt` |
| Day board | `artifacts/llm_curriculum/day_board.json` |
| Jarvis status | `artifacts/jarvis/status.json` |
| Progress | `artifacts/checkpoints/gpu_progress.json` |

Optional Drive backup:

```python
!cp -f artifacts/checkpoints/best_sigon*.pt /content/drive/MyDrive/ 2>/dev/null || true
```

---

## Cell 3 — JARVIS (after Interrupt, or after train ends)

Colab runs **one cell at a time**. While Cell 2 is running you cannot run Jarvis in another cell on free Colab unless you stop (Interrupt) train first — or open a second runtime. Easiest: **Interrupt → Jarvis → re-run train with `--warm best_sigon`**.

```bash
!python scripts/jarvis_talk.py status
!python scripts/jarvis_talk.py board
```

Hot updates (applied on next train updates via inbox — no code restart):

```bash
!python scripts/jarvis_talk.py "SET w_pullback_with_htf=0.35"
!python scripts/jarvis_talk.py "RELOAD_REWARDS"
!python scripts/jarvis_talk.py "NOTE dual HTF trend — take LTF pulls"
!python scripts/jarvis_talk.py outbox
```

Then resume train:

```bash
!python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600 --entropy-coef 0.03 --warm best_sigon
```

---

## Rules (hard)

| Rule | Behavior |
|------|----------|
| New ATH streak + breach 0% | LOCK `best_sigon_streakXX.pt` + `best_sigon.pt` + history freeze |
| Signals ON | NEW ~6820 lineage only |
| Fail same day 3× | That **instance** streak restarts at day 1; whole train continues |
| Target / risk | Runtime inputs in goals.yaml (40% @ 2.5/3.5; 60% random ranges) |
| Gold / US30 | Thin spread / typically no fees — still model `obs::spread_rel` |
| Caches | Clear `gpu_cache_*.npz` + `symbol_cache/*` once after signals ON — never delete `.pt` |

### Sampling law (`configs/goals.yaml`)

- **40%** of practice → target **2.5%** / risk **3.5%**
- **60%** → target uniform **[2.5, 70.5]** / risk **[1.0, 4.0]**
- Every update re-rolls target/risk per instance (same day sticky only while retries remain)
