"""Patch Momentum_One_RunAll.ipynb STEP 2: always best_sigon + 4000 instances."""
import json
from pathlib import Path

p = Path(__file__).resolve().parents[1] / "GPU_EDITION" / "Momentum_One_RunAll.ipynb"
nb = json.loads(p.read_text(encoding="utf-8"))

new_md = """# STEP 2 — Train (continue best brain only)

Press **▶** once. Leave it running.

### What this does (automatic)
1. Checks **signals ON**
2. Copies your best brain from **Google Drive** into Colab
3. Starts training **only if** that brain loaded (no new empty brain)
4. Uses **4000** parallel games (bigger practice)

### Good lines to see
- `restored from Drive: best_sigon.pt` or `Drive champion ready`
- **`warm-start best_sigon (obs_dim=6820)`**
- then `upd 1` `upd 2` ...

### Bad line (cell will STOP)
- `STOP: --require-warm` or `STOP: no champion`

### If red out-of-memory (OOM)
In the last line of the code cell, change `4000` to `2000` or `1000`, press ▶ again.
"""

new_code = r'''%cd /content/the-truth
!git pull origin main

import re, os, shutil, glob

# --- 1) Signals must be ON ---
t = open('configs/features.yaml', encoding='utf-8').read()
m = re.search(r'^include_signal_agent_slots:\s*(\w+)', t, re.M)
print('include_signal_agent_slots =', m.group(1) if m else 'MISSING')
if not m or m.group(1).lower() != 'true':
    raise SystemExit('STOP: signals OFF — old bot. Do not train.')

# --- 2) Always copy best brain from Drive (your saved streak) ---
ck = 'artifacts/checkpoints'
os.makedirs(ck, exist_ok=True)
drive = '/content/drive/MyDrive/momentum_sigon_champs'
if not os.path.isdir(drive):
    raise SystemExit(
        'STOP: Drive folder missing: ' + drive + '\n'
        '  Run STEP 1 (mount Drive) first. Or run STEP 4 after a good train to create it.'
    )
found = glob.glob(drive + '/best_sigon*.pt')
if not found:
    raise SystemExit(
        'STOP: no best_sigon files in Drive.\n'
        '  Expected: /content/drive/MyDrive/momentum_sigon_champs/best_sigon.pt'
    )
for fp in found:
    shutil.copy2(fp, os.path.join(ck, os.path.basename(fp)))
    print('restored from Drive:', os.path.basename(fp))
local = os.path.join(ck, 'best_sigon.pt')
if not os.path.isfile(local):
    raise SystemExit('STOP: best_sigon.pt still missing after Drive restore.')
print('Drive champion ready:', local)

# --- 3) Train from that brain only (refuse NEW empty brain) ---
# instances=4000  |  if OOM red error: change 4000 -> 2000 or 1000
!python scripts/gpu_train.py --csv-dir data --symbols XAUUSD,EURUSD,GBPUSD,US30 --instances 4000 --env-mb 32 --max-days-per-symbol 120 --minutes 600 --entropy-coef 0.03 --warm best_sigon --require-warm
'''

new_intro = """# Momentum One — SIGON (signals ON · all 4 symbols)

## This is the NEW bot (not the old one)

| | OLD bot | THIS notebook (SIGON) |
|--|---------|------------------------|
| Signal agents | **OFF** | **ON** |
| Brain size | ~1820 | **~6820** |
| Checkpoint | PROVEN_*.pt | **best_sigon*.pt** |
| Load PROVEN into this? | — | **NEVER** |

## Do this in order

1. **Runtime** → **Change runtime type** → **L4** or **T4** → **Save**
2. Run **STEP 1** (play button) — mounts Drive + setup
3. Run **STEP 2** (play button) — **always continues your best Drive brain**
4. Leave it until `upd 1` … (first time building data can take a while)

**STEP 2 will not start a new empty brain.** It copies from Drive first.

Open again later:  
https://colab.research.google.com/github/monty313/the-truth/blob/main/GPU_EDITION/Momentum_One_RunAll.ipynb
"""

n_md = n_code = n_intro = 0
for cell in nb["cells"]:
    src = "".join(cell.get("source", []))
    if cell.get("cell_type") == "markdown" and src.strip().startswith("# Momentum One — SIGON"):
        cell["source"] = [ln + "\n" for ln in new_intro.splitlines()]
        n_intro += 1
    if cell.get("cell_type") == "markdown" and "STEP 2" in src and "Train" in src:
        cell["source"] = [ln + "\n" for ln in new_md.splitlines()]
        n_md += 1
    if cell.get("cell_type") == "code" and "gpu_train.py" in src and "warm best_sigon" in src:
        cell["source"] = [ln + "\n" for ln in new_code.splitlines()]
        n_code += 1

print("intro", n_intro, "md", n_md, "code", n_code)
if n_code != 1 or n_md < 1:
    raise SystemExit("patch failed to find cells")

p.write_text(json.dumps(nb, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print("wrote", p)
