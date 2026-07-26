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
| 3 | Run STEP 2 | Press **▶** — this **starts training all 4 symbols** |
| 4 | Wait | First time can take **1–3 hours** building data. **Do not press Stop.** Training has started when you see **`upd 1`** |

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

## If memory runs out

In STEP 2, change `4000` to `2000`, then press ▶ again.

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
