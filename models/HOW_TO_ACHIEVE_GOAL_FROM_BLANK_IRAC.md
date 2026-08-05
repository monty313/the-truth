# HOW TO ACHIEVE GOAL FROM A BLANK POLICY (IRAC)

**Folder:** `models/` (champion home)  
**Mission:** one brain that takes **any** typed target%/risk% **without retrain**, measured only by `scripts/prove_it.py`.  
**Yardstick:** clear% climb, **breach% must stay 0**.  
**Working example (2026-07-31):** `PROVEN_SPRINT_row04_clear24_2026-07-20` — 24% clear / 0% breach @ 3.0/3.5; also 0% breach @ 2.5/2.5 and 1.5/2.0.

**Shell + multi-pair KEEP/REJECT (from ten-pair IRAC → GOAL):**  
→ [references/plans/GOAL_FROM_TEN_PAIR_IRAC.md](../references/plans/GOAL_FROM_TEN_PAIR_IRAC.md)

No fluff. Each block is **I**ssue → **R**ule → **A**pplication → **C**onclusion.  
**WORKS** / **FAILS** called out in Conclusions.

---

## 0) Definition of done (do not redefine)

### IRAC-0 — What “won” means
- **I:** Agent confuses “train loss down” or “entries exist” with the mission.
- **R:** GOAL.md: clear% = hit **your** target and never hit **your** floor; breach% = hit floor; score at **runtime** target/risk.
- **A:** Only `python scripts/prove_it.py <brain> <tgt> <risk>` (or `USE/1_prove.bat`) counts. Same `.pt` for every pair.
- **C:** **WORKS** = multi-pair prove_it, breach 0 on each, clear reported. **FAILS** = sandbox reward curves, synthetic-only greens, lineage “entries > 0” without prove_it.

**Done checklist (blank → keep):**
1. Policy sees goal + floor every day (self-state / FastSim).
2. Train over a **range** of (target, risk), not one frozen pair.
3. Shell/floor law intact (death / floor sacred).
4. `prove_it` @ 3.0/3.5 → breach 0, clear > 0.
5. Same brain `prove_it` @ ≥2 other pairs → breach 0 each.
6. Promote to `models/PROVEN_*` only if clear **beats** current champion and breach still 0.

---

## 1) Architecture you must not break

### IRAC-1 — Goal/floor are runtime inputs, not weight constants
- **I:** New policy only works at one baked pair (e.g. always 3.0/3.5).
- **R:** Env computes meaning; policy learns attention; goal/floor enter obs each day (`FastSim.reset(day, goal, floor)` + ranges in train/meta).
- **A:** `prove_it` passes `TGT`/`RISK` via CLI into `evaluate`/`rollout` ranges — **no retrain** when numbers change.
- **C:** **WORKS:** goal/floor in observation + train ranges. **FAILS:** hardcode target into reward only; train only one pair then claim “any pair.”

### IRAC-2 — One production stack, one score path
- **I:** Parallel toy brains look good; champion path broken or ignored.
- **R:** Production = `code/training/*` + `inference.loader` + `models/PROVEN_*.pt` + `scripts/prove_it.py`. Packages under `code/` (`PYTHONPATH=repo;repo/code`).
- **A:** Load via `load_brain(name)`. Data: `data/raw/XAUUSD_curriculum_2026.csv` (+ `artifacts/gpu_cache_XAUUSD_curriculum_2026.npz`).
- **C:** **WORKS:** fix path wiring, re-run real prove_it. **FAILS:** invent a second judge; score only a 32-dim lineage sandbox and call it champion.

### IRAC-3 — Floor is sacred
- **I:** Clear% rises while days hit risk floor.
- **R:** Breach% must stay **0**. Floor death penalty / shell caps stay on. No “accept breach for more clear.”
- **A:** Every adopt gate: prove_it breach == 0. Reject otherwise.
- **C:** **WORKS:** keep breach 0 even if clear is low. **FAILS:** promote any brain with breach > 0.

---

## 2) Rebuild recipe (blank policy → prove_it)

Do in order. Skip a step only if you already have a green prove_it on disk.

### Step A — Boot stack
```powershell
cd <repo>
$env:PYTHONPATH = ".;code"
python -c "from inference.loader import load_brain; print(load_brain('PROVEN_SPRINT_row04_clear24_2026-07-20')[0] is not None)"
```
- If import fails: put `code/` on path (see `scripts/prove_it.py` / `path_bootstrap.py`).

### Step B — Train (new weights)
Use the **production** train path (`scripts/gpu_train` / consistency sprint / host train — not lineage sandbox alone).

Minimum training contract for multi-pair:
| Must | Why |
|------|-----|
| Sample **varying** goal/floor each day or episode | Generalize without retrain |
| Keep shell + death/floor penalties | Breach 0 |
| Multi-TF / fixed obs order | Policy attention only |
| Checkpoints under a named file, not silent overwrite of PROVEN | Safe rollback |

Warm-start allowed from:  
`PROVEN_SPRINT_row04_clear24_2026-07-20.pt` / `PROVEN_LIFT_*` / `PROVEN_2x_*` (see SUCCESS_LEDGER).

### Step C — Score (mandatory)
```powershell
python scripts/prove_it.py <YOUR_BRAIN_NAME> 3.0 3.5
python scripts/prove_it.py <YOUR_BRAIN_NAME> 2.5 2.5
python scripts/prove_it.py <YOUR_BRAIN_NAME> 1.5 2.0
```
Record clear%, breach%, streak. **Breach must be 0 on all three.**

### Step D — Promote (only if better)
If clear @ 3.0/3.5 **>** champion clear and breach still 0 on all pairs:
1. Save `models/PROVEN_<clear_reason>_<date>.pt`
2. Update `models/00_CHAMPION.md`
3. Update GOAL.md scoreboard
4. Append SUCCESS_LEDGER

If not better: keep champion; do not overwrite PROVEN.

---

## 3) Disease IRACs (what to fix when stuck)

### IRAC-4 — Policy hold / hesitation (visible setup, no trade)
- **I:** Low clear, 0 breach, Mind Probe: `policy_hold` high on pull/cont under HTF.
- **R:** Lid is off — market allows the day; hesitation is **Policy** (incentives), not “impossible market.”
- **A:** Mind Probe / ghosts → count `policy_hold` on setup. Search dials (do not freeze human “cure” forever): raise `w_pullback_with_htf` (known good move 0.02→**0.25**), raise engage pressure (`w_did_nothing`), keep floor sacred. Re-prove.
- **C:** **WORKS:** dial search + prove_it gate. **FAILS:** add random indicators; declare market dead; train only to remove hold without prove_it.

### IRAC-5 — Wrong side under HTF
- **I:** Acts against firm HTF (wrong_side_under_bull high / side_bias_bull low).
- **R:** Bread-and-butter + with-trend: LTF timing with HTF side; reverse only on structure, not noise.
- **A:** Search ↑ `w_with_trend_close`, ↓ `w_against_trend_close` (meta; defaults 0). Mask if Shell. Re-prove.
- **C:** **WORKS:** evidence dials + prove_it. **FAILS:** hardcode one side forever; mask away all reverse without proof.

### IRAC-6 — All-hold collapse (toy / small policy)
- **I:** Argmax always HOLD; EOD zero entries; “mean reward” misleading.
- **R:** Must engage to clear target; did-nothing and setup-hold pressure exist for a reason. Still: **production** score is prove_it, not toy mean reward.
- **A:** Guide/BC toward recommended side when setup; EOD did-nothing wall; then **prove_it**. Lineage 32-dim sandbox can practice anti-hold but is **not** the champion path until it passes prove_it and beats PROVEN.
- **C:** **WORKS:** anti-hold shaping **on the production obs/policy**, then prove_it. **FAILS:** declare victory from synthetic curriculum entries only (lineage lesson 2026-07-31).

### IRAC-7 — Thrash / flip-flop
- **I:** Open/close every bar; clear dies; noise trades.
- **R:** Caps on adds, cooldown after reverse, floor still sacred.
- **A:** Shell max_adds + thrash limits; flip tax optional. Measure prove_it not just entry count.
- **C:** **WORKS:** hard action limits + score. **FAILS:** unlimited scale-in to farm bonuses.

### IRAC-8 — Perception vs Policy vs Shell
- **I:** Wrong class of cure.
- **R:**  
  - **Perception** = flags missing → fix features/TF.  
  - **Policy** = flags present, hold/wrong side → dials/train.  
  - **Shell** = mask_veto / risk heat → shell config, not only reward.
- **A:** Mind Probe skip_counts before changing weights.
- **C:** **WORKS:** class first, then one change, then prove_it. **FAILS:** change five systems at once.

---

## 4) Dials — search, don’t worship

Known useful (from skill + IRAC history). **Adopt only if prove_it clear not worse and breach 0.**

| Dial | Direction when | Do not |
|------|----------------|--------|
| `w_pullback_with_htf` | High policy_hold on pull | Freeze forever without re-measure |
| `w_did_nothing` | All-day flat free ride | Remove floor death to “fix” hold |
| `w_day_goal_hit` / streak | Low consistency | Optimize mean PnL only |
| `w_death_penalty` | Any breach temptation | Soften to buy clear% |
| `w_with_trend_close` / `w_against_trend_close` | WrongSide | Human-fixed final magnitude |

---

## 5) Explicit FAILS (do not repeat)

| Action | Why it fails the GOAL |
|--------|------------------------|
| Train only @ 3.0/3.5 then never re-score other pairs | Not “without retrain for any typed pair” |
| Overwrite PROVEN without higher clear + breach 0 | Loses champion; lid back on |
| Score with custom notebook metrics only | Not the contract |
| Lineage-only (Channel1 dim 32) promote without prove_it | Different obs/policy; not production |
| Breach > 0 “but higher clear” | Illegal under GOAL |
| Delete SUCCESS_LEDGER / flea-jar | Erases working cures |
| New UI / new framework instead of dial+prove | Out of scope |
| Change target by full retrain each time | Direct GOAL violation |

---

## 6) Explicit WORKS (repeat these)

| Action | Proof it works |
|--------|----------------|
| Goal/floor in env + CLI prove_it | Same PROVEN file: 3.0/3.5 → 24% clear 0% breach; 2.5/2.5 → 20% / 0%; 1.5/2.0 → 29% / 0% (2026-07-31) |
| IRAC → one dial class → prove_it gate | `w_pullback_with_htf` → 0.25 after policy_hold IRAC |
| Warm-start from PROVEN lineage | SUCCESS_LEDGER brains |
| Multi-day real XAUUSD curriculum (90 days) | prove_it default data |
| Breach 0 as hard gate | All kept PROVEN rows |

---

## 7) Minimal loop (tattoo this)

```text
Mind Probe / ghosts
  → IRAC (class: Perception | Policy | Shell)
    → one dial or train change
      → prove_it 3.0 3.5
      → prove_it other pairs (same .pt)
        → breach 0 and clear ≥ champion? promote : reject
```

---

## 8) Commands (copy)

```powershell
cd <repo>
$env:PYTHONPATH = ".;code"

# Yardstick
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5

# Without-retrain proof (same brain, new numbers)
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 2.5 2.5
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 1.5 2.0

# After YOUR new train
python scripts/prove_it.py <YOUR_NEW_NAME> 3.0 3.5
python scripts/prove_it.py <YOUR_NEW_NAME> 2.5 2.5
python scripts/prove_it.py <YOUR_NEW_NAME> 1.5 2.0
```

Reference proof writeup: `outputs/reports/GOAL_PROVE_MULTI_PAIR_2026-07-31.md`  
Champion pointer: `models/00_CHAMPION.md`  
Doctrine: `GOAL.md`, `references/doctrine/SUCCESS_LEDGER.md`, `references/doctrine/flea-jar/`

---

## 9) One-line memory

**Train attention on a range of goals/floors; keep the floor sacred; change only what IRAC classifies; adopt only when prove_it says clear up (or not down) and breach 0 on every pair you care about — same weights, new numbers, no retrain.**
