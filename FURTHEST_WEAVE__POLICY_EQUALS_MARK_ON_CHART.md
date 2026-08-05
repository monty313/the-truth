# FURTHEST WEAVE — Policy = Mark on chart

**Status:** STOPPED / CHECKPOINT — do not invent progress beyond this file  
**Date frozen:** 2026-08-04  
**Owner persona:** MARK HERE (Mark Montgomery Jr.) · Fable 5 translator  
**Repo root:** `C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth`  
**Lineage (only stack that is “policy = Mark”):** `lineages/adaptive_rl_brain_7_31_26/`  
**Goal doctrine:** `POLICY_EQUALS_MARK_ON_CHART.md`  
**PROVEN production brain:** **untouched** (`models/PROVEN_*.pt` — never load into Mark-obs)

---

## 0) One-sentence goal

> The bot’s action on a bar must be what **Mark would take** looking at the **same multi-TF chart** — not thrash, not freeze, not raw signal-agent spam. Shell (bank / heat / breach) stays locked.

---

## 1) How far we got (scoreboard vs definition of done)

| # | Meter (from `POLICY_EQUALS_MARK_ON_CHART.md`) | Pass? | Furthest measured state |
|---|-----------------------------------------------|-------|-------------------------|
| 1 | Chart sets = 4 official Mark stacks | **YES** | Code always scans all 4 (`perception/sets.py`) |
| 2 | Law 1: no LTF side against HTF force in teacher | **YES** | `mark_doctrine.decide_doctrine` force gate |
| 3 | Thrash day: entries ≤ 6 @ hard target | **PARTIAL → better** | Hard mean entries ~4.8 teacher vs ~9 claim baseline; thrash fixed directionally |
| 4 | Breach = 0 practice + forward | **YES** | All reported windows 0 breach |
| 5 | BC dir_match ≥ 0.85 | **YES (label match)** | train dir_match **0.95**, forward label dir_match **0.92** |
| 6 | BC step match (incl HOLD) ≥ 0.75 | **PARTIAL** | label match ~0.75–0.76; **day-walk step_match ~0.70** (not clone-ready) |
| 7 | Day walk reasons Mark would own | **PARTIAL** | Walks often “MARK WOULD OWN THIS PATH”; hard clear still low |
| 8 | PROVEN untouched | **YES** | `proven_touched: false` on all Mark reports |

### Headline numbers (forward holdout, eyes = `mark_doctrine`)

Source: `lineages/adaptive_rl_brain_7_31_26/checkpoints/FORWARD_MARK_POLICY_TEST.json`  
(also see `mark_clone_bc_report.json`, `mark_clone_policy_ab_hard_soft.json`)

| Pair (target/risk) | Teacher clear% | Teacher mean entries | BC policy clear% | BC step_match | Breach |
|--------------------|----------------|----------------------|------------------|---------------|--------|
| Soft **1/2** | **67.5%** | 3.45 | **70.0%** | ~0.70 | 0 |
| Mid **2/3** | 37.5% | 4.38 | 35.0% | ~0.71 | 0 |
| Hard **3/3.5** | **22.5%** | 4.75 | **20.0%** | ~0.71 | 0 |

**Baseline claim (legacy set-2 eyes) hard clear was ~30% with mean entries ~9** — Mark teacher trades fewer times; hard clear dropped while thrash died. That trade was intentional after chart-read diagnosis.

### BC checkpoint identity

| Field | Value |
|-------|--------|
| Weights file | `lineages/adaptive_rl_brain_7_31_26/checkpoints/mark_clone_doctrine_v1.pt` |
| Size / time (this machine) | ~12 KB · 2026-08-04 ~20:23 local |
| Architecture | `Channel1Policy` · `obs_dim=CHANNEL1_DIM` · **hidden=64** · 3 actions (HOLD/BUY/SELL) |
| Teacher | `eyes_mode="mark_doctrine"` + `mark_soul=True` during BC label collection |
| clone_ready_heuristic | true (teacher path) |
| clone_ready_policy | **false** (BC not full Mark yet) |

### Chart-DNA fix that reached furthest teacher (critical)

**Bug measured:** soft single-set scalp (`law1_soft_single_set_scalp`) appeared on ~30/31 hard-target misses.

**Mark fix applied in code (must keep to recreate this weave):**

1. **`perception/mark_doctrine.py`**  
   - If `target_pct >= 2.5` → force `allow_single_set_scalp = False`  
   - Hard targets only fire multi-set BULL/BEAR tide releases (slingshot), not flat single-set noise  
   - Soft targets (`< 2.5`) may still use soft single-set scalp for early bank  

2. **`equity_day.py` (ride greens)**  
   - When equity > **40% of daily target**, refuse reverse on soft_single / weak flip  
   - Mark soul: when green, reverse only on quality opposite (`slingshot_release` / tide)  
   - Soft_single reverse blocked when already working  

3. **Doctrine opportunity floor**  
   - `DEFAULT_OPP_MIN_SCORE = 1.2` (shared quality gate)

**After that fix:** hard thrash path cleaned; hard clear sat ~15–22.5% depending on window; soft stayed ~65–70%. Quiet clears exist (e.g. 2026-03-31 soft 1 entry bank; hard quiet 1-entry clears on select days).

---

## 2) The policy (what “policy = Mark on chart” means in this weave)

### 2.1 Control chain (instruction = the chart)

```
READ CHART (4 Mark sets) 
    → FORCE (HTF last two of each set)
    → REGIME (bull / bear / chop / flat)
    → VELOCITY (LTF first of each set)  breath vs aligned/launch
    → ENTRY only if side(force)==side(setup) AND heat/risk OK
    → In trade: ride green; reverse only quality flip
    → Regime shift / floor → flatten / HOLD
```

### 2.2 Official Mark sets (immutable law of chart)

| Set | LTF (velocity) | HTF (force) |
|----:|----------------|-------------|
| 1 | 1m | 15m, 30m |
| 2 | 5m | 30m, 1h |
| 3 | 15m | 1h, 4h |
| 4 | 30m | 4h, 1d |

Code: `perception/sets.py` · docs: `MARK_SETS_LAW.md` · `MARK_DOCTRINE_FIVE_LAWS.md`

### 2.3 Five laws (teacher soul)

| Law | Name | Code gate |
|-----|------|-----------|
| 1 | Dominant trends — HTF permission, LTF timing | force / slingshot release |
| 2 | Breath vs launch | `classify_set_play` · BREATHER vs ALIGNED |
| 3 | Regime survival | CHOP → no trade; FLAT hard-target → no soft scalp |
| 4 | Capital | shell bank/heat/breach **locked** in `equity_day` |
| 5 | Speed vs weight | force=HTF · velocity=LTF |

### 2.4 Dual policy objects (do not mix claims)

| Object | What it is | How you run it |
|--------|------------|----------------|
| **A. Teacher (Mark eyes)** | Pure doctrine + shell + soul size/adds | `GoalEquityDay(..., eyes_mode="mark_doctrine", mark_soul=True)` · `use_heuristic=True` |
| **B. BC policy weights** | MLP clone of teacher actions | Load `mark_clone_doctrine_v1.pt` · greedy argmax logits |
| **C. Production PROVEN** | Old Brain/meta path | **Out of scope for this weave** — do not overwrite |

**Truth hierarchy:** Teacher doctrine is the decode of “what Mark would do.” BC must catch up. Until step_match high **and** day walks match teacher, claim is **PARTIAL**.

### 2.5 Runtime pairs (same weights, no retrain for numbers)

`lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` — 10 (target%, risk%) pairs. Shell goal/floor are **inputs**, not retrain triggers.

---

## 3) Exact recreation instructions

### 3.0 Prerequisites

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
# Python 3.10+ with torch, numpy (see requirements.txt)
```

**Data required:**

- Primary curriculum: `data/raw/XAUUSD_curriculum_2026.csv` (preferred by `ten_pairs.json`)  
- Full M1 also present: `data/raw/XAUUSD_M1_full.csv`  
- Loader: `lineages/adaptive_rl_brain_7_31_26/price_data.py` + `equity_day.load_calendar_days`

**Do not touch:**

- `models/PROVEN_*.pt`  
- Production `sets_lock` under live PROVEN sessions without cache wipe + retrain  
- Trail + cushion + scale-in (banned IRAC package)

### 3.1 Source files that *are* this weave (must exist)

```
POLICY_EQUALS_MARK_ON_CHART.md                          # goal / dual-stack law
FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md          # THIS FILE
lineages/adaptive_rl_brain_7_31_26/
  MARK_DOCTRINE_FIVE_LAWS.md
  MARK_SETS_LAW.md
  MARK_CLONE_POLICY.md
  ten_pairs.json
  equity_day.py                    # shell + ride-green + mark_soul adds
  day_runner.py
  policy_stub.py                   # Channel1Policy
  train_mark_clone_bc.py           # BC trainer
  forward_mark_policy_test.py      # forward teacher vs BC
  compare_mark_clone_attention.py  # hard/soft A/B
  mark_chart_read_diagnosis.py     # chart DNA thrash diagnosis
  perception/
    mark_doctrine.py               # five-law teacher (incl soft-scalp kill @ ≥2.5)
    mark_sets_opportunity.py
    sets.py · observation.py · pipeline.py · types.py · …
  checkpoints/
    mark_clone_doctrine_v1.pt      # BEST BC WEIGHTS TO DATE
    mark_clone_latest.pt           # copy/symlink of latest
    mark_clone_bc_report.json
    FORWARD_MARK_POLICY_TEST.json
    mark_clone_policy_ab_hard_soft.json
```

### 3.2 Recreate from *existing* weights (no retrain)

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# 1) Forward: teacher doctrine vs BC policy on holdout
python lineages/adaptive_rl_brain_7_31_26/forward_mark_policy_test.py

# 2) Hard vs soft A/B (claim baseline vs Mark teacher vs BC)
python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 3.0 3.5 --mode forward --eyes-only --eyes-mode mark_doctrine

# 3) Chart-read DNA (why thrash / soft_single on misses)
python lineages/adaptive_rl_brain_7_31_26/mark_chart_read_diagnosis.py

# 4) Optional 10d Mark vs policy diaries
python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py
```

Expect reports under:

- `checkpoints/FORWARD_MARK_POLICY_TEST.json`  
- `checkpoints/mark_clone_policy_ab_hard_soft.json`  
- `checkpoints/test_run_10d_mark_vs_policy/`

### 3.3 Recreate BC weights from scratch (teacher → clone)

Doctrine teacher is **code**, not a weight. BC relearns teacher labels:

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# Default recipe that produced mark_clone_doctrine_v1.pt shape:
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py `
  --epochs 40 `
  --max-train-days 50 `
  --practice-n 50 `
  --hidden 64 `
  --lr 0.001

# After train, re-score forward:
python lineages/adaptive_rl_brain_7_31_26/forward_mark_policy_test.py
```

**What BC trains on (must match):**

| Setting | Value |
|---------|--------|
| eyes_mode | `mark_doctrine` |
| mark_soul | `True` (size + force-aligned adds in teacher path) |
| mark_clone | `False` during label collect (teacher is doctrine, not old policy) |
| multi_pair | True — random pair from `ten_pairs.json` per day |
| decide_every | 25 (default day loop) |
| Output ckpt | `checkpoints/mark_clone_doctrine_v1.pt` |
| Report | `checkpoints/mark_clone_bc_report.json` |

**Teacher label path in code:**  
`train_mark_clone_bc.collect_teacher_dataset` → `GoalEquityDay.recommended_action` → doctrine + soul gates.

### 3.4 Recreate the *teacher* behavior only (no neural net)

Teacher is fully deterministic from:

1. M1 bars for calendar day  
2. MTF perception → 4 official sets  
3. `decide_doctrine(..., target_pct=...)`  
4. `equity_day` shell steps (bank at target, death at floor, heat)

Minimal mental model:

```text
if target >= 2.5%:  no soft_single_set_scalp
if regime == CHOP:  HOLD
if regime == BULL and LTF aligned long + quality:  BUY
if regime == BEAR and LTF aligned short + quality: SELL
else: HOLD
in trade + eq > 0.4 * target: ride; reverse only quality opposite
```

### 3.5 Production stack (NOT part of furthest Mark weave)

| Key | Value |
|-----|--------|
| sets_lock default | `proven_legacy` in `configs/features.yaml` |
| Mark sets for *new* production trains only | `sets_lock: mark` + wipe `outputs/artifacts/gpu_cache_*.npz` |
| Meta may move | reward dials in `TREND_KNOBS` (pullback, with/against, quick_pull) |
| Meta may never | shell physics, trail+scale-in, fit on forward as train |

Do **not** claim production PROVEN = Mark on chart until a full retrain under Mark sets passes the same meters.

---

## 4) What is still missing (honest gap list)

1. **BC step_match day-walk ~0.70** — need ≥ ~0.85 and HOLD-heavy match to call clone ready.  
2. **Hard clear recovery without thrash** — soft-scalp kill fixed thrash; multi-set launch capture still weak (~15–22% hard). Need better multi-set entry timing / session gates, not thrash return.  
3. **Re-BC after soft-scalp kill** — latest doctrine changes may not be fully reflected in the weights if trained before last teacher edit; recommended next step was re-BC then re-forward.  
4. **Session / time-left gates** — not fully codified.  
5. **Human hand labels** — optional gold standard not bulked.  
6. **sets_lock: mark production Brain** — optional, Monty-gated, not done.  
7. **/loop overnight train** — blocked until explicit cadence is ordered.

---

## 5) Mark-as-agent method used to reach this weave

Order of operations that got us furthest (do not reverse casually):

1. **Read the chart instruction** — official 4 sets + shell goal/floor pair.  
2. **Apply five-law knowledge** — force → regime → velocity → entry.  
3. **Measure thrash DNA** (`mark_chart_read_diagnosis`) — soft_single on hard misses.  
4. **Strategic fix outside pure RL** — kill soft scalp @ hard target; ride greens in shell.  
5. **Re-score** — thrash ↓, soft still banks, hard quieter.  
6. **BC clone** — imitate teacher; breach 0; direction good; HOLD clone incomplete.  
7. **Stop + write this file** when asked for furthest weave recreation.

---

## 6) Quick “am I on the same weave?” self-check

Run after any machine restore:

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
Test-Path lineages/adaptive_rl_brain_7_31_26/checkpoints/mark_clone_doctrine_v1.pt
Select-String -Path lineages/adaptive_rl_brain_7_31_26/perception/mark_doctrine.py -Pattern "target_pct.*>= 2.5"
Select-String -Path lineages/adaptive_rl_brain_7_31_26/equity_day.py -Pattern "0.40 \* self.target"
python lineages/adaptive_rl_brain_7_31_26/forward_mark_policy_test.py
```

Pass if:

- ckpt exists  
- soft-scalp kill line present  
- ride-green 40% gate present  
- forward report: breach 0; soft clear roughly 65–70%; hard ~15–25%; step_match ~0.7  

Fail (different weave) if:

- only legacy set-2 eyes  
- soft_single still fires on hard targets  
- PROVEN weights loaded into Channel1  
- trail+scale-in reintroduced  

---

## 7) Next step when resuming (not executed — checkpoint only)

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 40 --max-train-days 50
python lineages/adaptive_rl_brain_7_31_26/forward_mark_policy_test.py
```

Then raise multi-set launch quality **without** re-enabling hard-target soft_single thrash.

---

## 8) File index for agents

| Need | Open first |
|------|------------|
| Goal law | `POLICY_EQUALS_MARK_ON_CHART.md` |
| This furthest checkpoint | `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md` |
| Five laws | `lineages/adaptive_rl_brain_7_31_26/MARK_DOCTRINE_FIVE_LAWS.md` |
| Teacher code | `.../perception/mark_doctrine.py` |
| Shell + ride green | `.../equity_day.py` |
| BC train | `.../train_mark_clone_bc.py` |
| Forward score | `.../forward_mark_policy_test.py` |
| Best weights | `.../checkpoints/mark_clone_doctrine_v1.pt` |
| Score snapshot | `.../checkpoints/FORWARD_MARK_POLICY_TEST.json` |

**END OF CHECKPOINT — furthest weave as of 2026-08-04.**
