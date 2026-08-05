# GOAL proof — same champion, multiple target/risk (no retrain)

**Date (UTC):** 2026-07-31  
**Brain (champion):** `PROVEN_SPRINT_row04_clear24_2026-07-20`  
**File:** `models/PROVEN_SPRINT_row04_clear24_2026-07-20.pt`  
**Entry:** `python scripts/prove_it.py <brain> <target%> <risk%>`  
**Data:** 90 real XAUUSD curriculum days (`data/raw/XAUUSD_curriculum_2026.csv`)  
**PROVEN overwrite:** **none** (same weights for all pairs; mtime unchanged)

This is the GOAL.md contract check: **change the two numbers, same brain, score clear% / breach%.**

---

## Results (real `scripts/prove_it.py` stdout)

| Target % | Risk % | Clear % (cleared days) | Breach % | Longest clear streak | Mean day PnL |
|---------:|-------:|-----------------------:|---------:|---------------------:|-------------:|
| **3.0** | **3.5** | **24%** | **0%** | 2 | +0.29% |
| **2.5** | **2.5** | **20%** | **0%** | 2 | +0.21% |
| **1.5** | **2.0** | **29%** | **0%** | 3 | +0.12% |

All three runs: exit code **0**, **breach = 0%**, no train step between runs.

---

## How to reproduce

From repo root:

```powershell
$env:PYTHONPATH = ".;code"
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 2.5 2.5
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 1.5 2.0
```

Or yardstick only: `USE/1_prove.bat`

---

## Wiring note

`scripts/prove_it.py` now puts `code/` on `sys.path` (FinRL-clean layout) so the entry works with or without a pre-set `PYTHONPATH`.

---

## What this does / does not claim

| Claims | Does not claim |
|--------|----------------|
| Same champion `.pt` scores at 3 distinct target/risk pairs without retrain | Infinite clear-% climb |
| Breach stayed **0%** on all measured pairs | New brain promoted over champion |
| Yardstick clear ~24% @ 3.0/3.5 matches champion doc | Lineage sandbox is champion |

**Next (outside this proof):** climb clear % at yardstick while keeping breach 0 — only promote if prove_it beats champion with breach still 0 (`models/00_CHAMPION.md` rule).
