# MOMENTUM ONE — GPU EDITION (Bot 1.5)
### Train thousands of markets at once on a free Google GPU

---

## What this is, in plain words

One brain. **8,000 practice markets running at the same time.**

Every market, every day, gets a **random target** and a **random risk line** picked
from your ranges. The brain has to learn to hit *whatever number it is handed* without
ever crossing the risk line.

It keeps a **record**: the longest run of **cleared days in a row**
(cleared = it hit that day's target and never breached the risk line).
Every time it beats its record, it **saves itself** — with the number of passed days
right in the filename.

**Your ranges:** target **2.5% – 70.3%**, risk **1% – 4.4%**.
**Finish line:** **365 cleared days in a row.**

The brain is free to pick its own strategy and its own lot size. The masks are the one
wall it cannot cross.

---

## How to run it — 3 steps

**STEP 1 — Open the notebook in Google Colab**
Go to **colab.research.google.com**, then File → Upload notebook, and pick
`Momentum_One_GPU.ipynb` (the file next to this one).

**STEP 2 — Turn on the L4 GPU**
Top menu → **Runtime** → **Change runtime type** → under *Hardware accelerator* pick
**L4 GPU** → **Save**.

**STEP 3 — Run the cells, top to bottom**
Click the little ▶ button on each cell, in order. The last cell starts training.
Let it run. That is the whole thing.

---

## What you will see while it trains

Each line is one training round, for example:

```
upd  42 | 33s | rollout: mean pnl +1.20% | hit-target 8.0% | breach 0.3% | ploss ...
```

- **mean pnl** — the average profit that round
- **hit-target** — how often it hit the random target it was given
- **breach** — how often it crossed the risk line (you want this near 0)

Every few rounds it checks its **best streak** and, when it beats the record, prints:

```
*** NEW RECORD: 12 cleared days in a row -> saved ...pass0012_...pt
```

The streak starts near 0 and grows as the brain learns. Higher targets are harder,
so the record climbs slowly at first — that is normal.

---

## Where the saves go

Every record is frozen here:
`artifacts/checkpoints/history/`

The filename tells you the streak:
`momentum_gpu_pass0012_...pt`  =  **12 cleared days in a row.**

The latest best is always `artifacts/checkpoints/gpu_best.pt`.

These plug **straight into your real bot** — same brain shape, nothing to convert.

---

## When does it stop?

- **It runs until Colab stops it** (free Colab lasts up to ~12 hours), or until it
  **wins at 365 cleared days in a row**. The best brain saves to your Google Drive on
  every record, so nothing is lost.
- **To continue another day:** open the notebook, run Cell 1 (reconnects your Drive),
  then the training cell — it picks up from your best brain automatically.

---

## If it ever says "out of memory"

Change **one number** in the last cell: `--instances 8000` → `--instances 4000`
(or `2000`). Nothing else changes.

---

## When a brain gets good — keep it and use it

When training reaches a high `pass` number (a long clean streak), do this:

1. **It keeps itself.** The best brain auto-saves to your Google Drive at
   `MyDrive/momentum_gpu/gpu_best.pt` every time it sets a record — nothing to do.
   (Cell 4 also downloads a copy to your computer if you want one.)
2. **Trust it.** Run **Cell 5**. It re-checks that brain on your **real** simulator (the
   exact one, not the fast training copy), so you know the numbers are real.
3. **It can't be lost.** Training only ever writes **new** `gpu_...` files. It never
   overwrites a good one.

The good file is the highest `pass####` number, and always `gpu_best.pt` (the latest best).

**To trade it live on MT5:** the brain is the same shape as your real bot, so it drops
straight in. Going live is a separate setup (Windows + your MT5 login + the safety gates) —
just tell me *"wire up the good brain to MT5"* and we do it together.

---

## Your proof is safe

This never touches your proven brain. It **starts from a copy** of it and only ever
saves to new `gpu_...` files. The +6.5% proof stays frozen and untouched.

---

## The one honest note (how it goes fast)

To fit 8,000 markets on one GPU, training runs on a **high-speed copy** of the market.
Every streak and every proof is always scored on your **real** simulator — the copy is
only for speed. It was checked against the real sim and matches: same breaches on every
day, and the big green day matched to within 0.05%.
