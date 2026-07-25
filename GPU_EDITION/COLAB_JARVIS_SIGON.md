# Colab L4 — SIGON + Jarvis (no full restart)

## Cell 0 — GPU

**Runtime → Change runtime type → Hardware accelerator → L4 GPU → Save**

---

## Cell 1 — Drive + repo + price data

```python
from google.colab import drive
drive.mount('/content/drive')

!git clone https://github.com/monty313/the-truth.git 2>/dev/null || true
%cd /content/the-truth
!git pull origin main
```

### Copy M1 CSVs from Monty’s Drive folder

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
    if os.path.isdir(c) and glob.glob(c + "/*.csv"):
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
```

**Config on main:** `include_signal_agent_slots: true` → expect **obs_dim ~6820**.  
Do **not** warm-start PROVEN_* 1820 weights into this brain.

---

## Cell 2 — TRAIN (leave running)

```bash
!python scripts/gpu_train.py --csv-dir data --instances 8000 --minutes 600
```

OOM → `--instances 4000` then `2000` then `1024`.

| Output | Path |
|--------|------|
| Champion | `artifacts/checkpoints/best_sigon.pt` |
| Day board | `artifacts/llm_curriculum/day_board.json` |
| Jarvis status | `artifacts/jarvis/status.json` |
| Progress | `artifacts/checkpoints/gpu_progress.json` |

Optional Drive backup:

```python
!cp -f artifacts/checkpoints/best_sigon.pt /content/drive/MyDrive/best_sigon.pt 2>/dev/null || true
```

---

## Cell 3 — JARVIS (new cell anytime while train runs)

```bash
!python scripts/jarvis_talk.py status
!python scripts/jarvis_talk.py board
```

Hot updates (**no process restart**):

```bash
!python scripts/jarvis_talk.py "SET w_pullback_with_htf=0.35"
!python scripts/jarvis_talk.py "RELOAD_REWARDS"
!python scripts/jarvis_talk.py "NOTE dual HTF trend — take LTF pulls"
!python scripts/jarvis_talk.py outbox
```

Train log shows `[JARVIS] ...` when inbox is applied.

---

## Rules

| Rule | Behavior |
|------|----------|
| Highest consistency | Auto-save `best_sigon.pt` |
| Signals ON | NEW ~6820 lineage |
| Fail same day 3× | That **instance** draws a **new** day; train continues |
| Jarvis SET / RELOAD_REWARDS | Hot mid-train via `artifacts/jarvis/inbox.md` |
| 40% focus | goals.yaml 2.5% / 3.5%; 60% random ranges |
| Gold / US30 | Thin spread / typically no fees — still model `obs::spread_rel` |
