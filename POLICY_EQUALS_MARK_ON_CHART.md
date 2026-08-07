# Policy = Mark on chart

**One sentence:** The bot’s action on a bar must be what Mark would take looking at the same multi-TF chart — not thrash, not freeze, not raw signal-agent spam.

**Furthest weave checkpoint (progress + recreate recipe):** `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md`

**Owner:** Mark Montgomery Jr. / MARK HERE  
**Fable role:** translator of doctrine → measurable policy / rewards / meta search  
**Date:** 2026-08-04  

---

## 0) Two brains — do not mix claims

| Stack | Path | Role vs Mark-on-chart |
|-------|------|------------------------|
| **A. Production RL + meta** | `code/training/*` · `Brain` · `meta_tuner` · `rewards.yaml` · PROVEN | Consistency at typed goal/floor. **Obs/sets must eventually match Mark lock for *new* trains.** PROVEN weights = frozen yardstick — do not overwrite. |
| **B. Mark clone lineage** | `lineages/adaptive_rl_brain_7_31_26/*` | Explicit **policy = Mark on chart** teacher (five laws + 4 sets) + Channel1 BC. Shell clear/breach. |

| Track | When to use |
|-------|-------------|
| **PROVEN / meta_tuner** | Climb clear% on production FastSim; warm-start PROVEN; meta only moves **reward/hparams**, never shell laws |
| **Mark doctrine lineage** | Day-walk “I would do the same”; multi-set slingshot; ENTJ scalping logic |

**Hard rule:** Never load PROVEN into Mark-obs or Mark-sets without retrain. Never promote lineage over PROVEN without Monty.

---

## 1) Mark-on-chart control chain (both stacks must serve this)

```
FORCE (HTF last two of each set) → REGIME → allowed side + m_regime
        ↓
VELOCITY (LTF first) → breath (wait) vs aligned/launch (fire)
        ↓
ENTRY only if side(force)==side(setup) AND heat/daily risk OK
        ↓
Regime shift → flatten / HOLD / m→0
```

### Official sets (LTF first · HTF last two) — **law of chart**

| Set | LTF | HTF |
|----:|-----|-----|
| 1 | 1m | 15m, 30m |
| 2 | 5m | 30m, 1h |
| 3 | 15m | 1h, 4h |
| 4 | 30m | 4h, 1d |

Config: `configs/timeframes.yaml` → `sets_mark`  
Engine: `features/engine.py` respects `features.sets_lock`  
Lineage: `perception/sets.py` (always Mark)

### Five laws → production rewards / meta knobs

| Law | Mark meaning | Production dial / signal | Lineage code |
|-----|--------------|--------------------------|--------------|
| 1 Dominant trends | HTF permission; LTF timing | firm cont/pull tags; masks | `mark_doctrine` force gate |
| 2 Breath vs launch | Fast against force = wait; both with force = fire | `w_pullback_with_htf`, `w_quick_pull_close` | breath vs aligned |
| 3 Regime | chop/flat → no trade / small m | masks + setup_skip; future regime tags | `Regime` enum |
| 4 Capital | never breach floor | death penalty, heat (shell), `w_did_nothing` | `equity_day` shell locked |
| 5 Speed vs weight | force≠velocity | HTF firm vs LTF pull/cont | HTF conf + LTF entry_dirs |

---

## 2) Meta-learning — what it may and may not do

### Meta may search (attention / personality weights)

From `meta_tuner.BOUNDS` (and self-heal dials):

- `w_pullback_with_htf` — pay slingshot resumes with HTF (Law 1–2)  
- `w_with_trend_close` / `w_against_trend_close` — force alignment (Law 1)  
- `w_quick_pull_close` — fast release after breath (Law 2, 5)  
- `w_setup_skip` — punish flat when Mark setup visible (Law 1 fire)  
- `w_did_nothing` / idleness — anti-hold without thrash mandate  
- `w_death_penalty` — capital (Law 4)  
- `lr`, `entropy_coef` — learnability only  

**Adaptive focus (meta “learn to learn”):** when wrong-side / side-bias hot → aggressive mutate **Mark trend knobs** (`TREND_KNOBS` includes pullback + with/against + quick_pull).

### Meta must never

| Forbidden | Why |
|-----------|-----|
| Overwrite PROVEN without order | Yardstick |
| Change shell physics (bank/heat/every-bar marks) via reward search alone | Law 4 is code law |
| Fit on forward / unseen as train set | Honesty |
| Reintroduce trail+cushion+scale-in | IRAC kill |
| Treat signal agents as the policy soul | Agents = sensors |
| Flip `sets_lock` under a live PROVEN session without cache wipe + retrain | Semantic obs shift |
| Adopt a candidate because **practice** clear% went up | Forward is the only champion judge (2026-08-05) |

### Meta forward-consistency law (2026-08-05)

| Step | Pool | Role |
|------|------|------|
| Probe / short PPO | **Practice** (seen) | Search only |
| Score champion | **Forward** (unseen) | Sole adopt judge |
| Practice score | Practice sample | Screen — reject if clear% collapses |
| Weak forward clear/streak | — | Force-search `CONSISTENCY_FORWARD_KNOBS` + Mark TREND_KNOBS |

**Adopt only if:** forward consistency improves (paired gate) **and** forward breach not worse **and** forward longest day-streak not shorter **and** side veto ok **and** practice screen ok.

Code: `code/training/meta_tuner.py` · Mark dials CLI: `lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py`

---

## 3) RL model — what “updated” means

### Production `Brain` (GRU, 11 ops, size Beta)

- Learns from **reward personality** + PPO.  
- For Mark-on-chart: tags `pullback` / `with_trend` / `against_trend` must be computed under **correct sets** when `sets_lock: mark`.  
- Warm-start PROVEN only when `sets_lock: proven_legacy` and frame_dim match.

### Lineage Channel1 Mark clone

- Teacher = `eyes_mode=mark_doctrine` (five laws) + **Mark soul**  
  (goal-relative size + force-aligned adds; full-chart plans via `mark_soul_plan.py`).  
- New weights: `checkpoints/mark_clone_doctrine_v1.pt` / `mark_clone_soul_v1.pt`.  
- Success: day walk reasons = FORCE/REGIME/VELOCITY Mark would accept;  
  soul plans clear hard random-pair days without thrash. See `MARK_SOUL_TRANSFER.md`.

---

## 4) Config dual-lock (PROVEN safe)

| `features.yaml` key | Value | Use |
|---------------------|-------|-----|
| `sets_lock` | `proven_legacy` | Default — PROVEN / meta warm-start safe |
| `sets_lock` | `mark` | **New Mark-on-chart trains only** — wipe GPU cache; do not load PROVEN |

After flip mark: delete `outputs/artifacts/gpu_cache_*.npz` (not .pt).

---

## 5) Commands

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# Mark clone (lineage) — policy = Mark on chart teacher + BC
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 40 --max-train-days 50
python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 3.0 3.5 --mode forward --eyes-only --eyes-mode mark_doctrine

# Production meta (PROVEN path) — only if sets_lock proven_legacy
# python scripts/meta_train.py   # existing host recipe
```

---

## 6) Definition of done (policy = Mark on chart)

| # | Meter | Pass |
|---|-------|------|
| 1 | Chart sets | 4 official stacks as table above |
| 2 | Law 1 | No LTF side against HTF force in teacher |
| 3 | Thrash day | Entries ≤ 6 on known thrash day @ hard target |
| 4 | Breach | 0 on practice + forward windows |
| 5 | dir_match BC | ≥ 0.85 (directional Mark) |
| 6 | match BC | ≥ 0.75 (incl HOLD) |
| 7 | Day walk | Reasons Mark would not reject |
| 8 | PROVEN | Untouched unless explicit order |

Until 1–7 hold: **not** “our clone.” Until then, keep teacher doctrine as decode of truth; policy BC catches up.
