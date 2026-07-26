# START HERE — Momentum One (SIGON)

**You only need one file.**

## NEW bot vs OLD bot (do not mix)

| | OLD | THIS (SIGON) |
|--|-----|----------------|
| Signal agents | OFF | **ON** |
| Size | ~1820 | **~6820** |
| Files | PROVEN_*.pt | **best_sigon*.pt** |
| Mix them? | | **NO — never load PROVEN into SIGON** |

---

## Open this notebook in Colab

Click this link (or copy-paste into your browser):

**https://colab.research.google.com/github/monty313/the-truth/blob/main/GPU_EDITION/Momentum_One_RunAll.ipynb**

That is the **only** notebook to use.

---

## Then do 4 things (in order)

| # | What | How |
|---|------|-----|
| 1 | Turn on GPU | Top menu: **Runtime** → **Change runtime type** → pick **L4** or **T4** → **Save** |
| 2 | Run STEP 1 | Click the cell → press **▶** → click **Allow** for Google Drive |
| 3 | Run STEP 2 | Press **▶** — copies **best brain from Drive**, then trains (**4000** games). **Will not** start empty NEW brain |
| 4 | Wait | First time can take a while building data. **Do not press Stop.** Good start = **`warm-start best_sigon`** then **`upd 1`** |

---

## All 4 symbols

Training uses:

- **XAUUSD** (gold)
- **EURUSD**
- **GBPUSD**
- **US30**

Price files come from your Drive folder **Camillion_data**.

---

## What “good” looks like

**Still preparing (normal):**
```text
pool+ EURUSD ...
gpu_data: building features ...
```

**Really training:**
```text
upd    1 | ...
upd    2 | ...
```

---

## If memory runs out (red OOM)

In STEP 2 code, change `--instances 4000` to `2000` or `1000`, then press ▶ again.

---

## Do not use these (old)

| Folder / file | Why |
|---------------|-----|
| `GPU_EDITION/OLD/` | Old notebooks — ignore |
| Other `.ipynb` names | Not the main path |

---

## Optional later

- **STEP 3** in the notebook = status check (only after you stop train or train ends)
- Long notes: `COLAB_JARVIS_SIGON.md` (advanced — not required to train)
