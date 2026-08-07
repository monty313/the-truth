# WHAT WORKS — Policy = Mark on chart (Spine Shadow lane)

**Audience:** later LLM / KAG.  
**Goal only:** one frozen embryo matches Mark day awards under random T%/R%, **breach = 0**, on **data not used to fit weights**. Practice board must not regress while climbing.  
**Price data:** `the-truth/data/raw/` only (`price_data.RAW_DIR`).

---

## Child stage freeze (start here)

| | |
|--|--|
| **Stage** | **child** |
| **same** | **35**/50 · mwt 15 · breach 0 |
| **Embryo** | `mark_clone_full_obs_v1.pt` + backup `CHILD_STAGE_same35_mark_clone_full_obs.pt` |
| **Freeze file** | `CHILD_STAGE__same35__frozen.json` |
| **pt5** | `all llm's have to know this is the most basic knowledge pt5 - Copy.txt` |

Grow from this child only — do not reset to random weights. PROVEN untouched.

---

## Grow up like Mark (learn to learn for forward)

**You lead. He grows.** Process = principles + error classes + attention + conscience + forward exam.  
**Not** day-memo BC alone (that is childhood; dies on forward and thrash-packs).

| Stage | What he learns | Product |
|-------|----------------|---------|
| Child | Act copy | old BC only |
| Teen | Path classes + phase/event/size/clue | `learn_to_learn_path` + `mark_shadow_policy` |
| Adult | Principles under T/R; forward gate | `grow_up_mark_style.py` + principle student |

```powershell
python lineages/adaptive_rl_brain_7_31_26/grow_up_mark_style.py
```
Report: `GROW_UP_MARK_STYLE__latest.md`

---

## Learn-to-learn + full Spine Shadow heads

**Doctrine:** `Fable 5 Alternate — Spine Shadow.md` §2.2

| Product | Heads |
|---------|--------|
| `mark_shadow_policy.py` | **phase** · **event** · **size** · **clue_gate** · act |
| `train_spine_shadow_full.py` | multi-head loss + KL + learn≠copy + KEEP/REJECT |
| `learn_to_learn_path.py` | path error classes + pattern memory (meta boost) |

| Head | Train target |
|------|----------------|
| phase | before_first_fire / in_trade / breath_reload / done_bank / killed |
| event | wait_loaded / fire / add / hold_on_spine / bank / kill |
| size | micro…max when fire/add else none |
| clue_gate | soft mask over 168 obs dims (who to trust) |
| act | HOLD/BUY/SELL for shell (force-gate wraps) |

learn≠copy: high act + low phase/event match → REJECT.

```powershell
$env:PYTHONPATH=".;code"
python lineages/adaptive_rl_brain_7_31_26/train_spine_shadow_full.py --max-rounds 6
python lineages/adaptive_rl_brain_7_31_26/learn_to_learn_path.py --max-rounds 8
```

---

## Proven stack (keep using)

| Piece | Why it works | Where |
|-------|----------------|-------|
| **Mark soul plans** | Clear **50/50** on practice window — teacher mind is solved for that pack | `mark_soul_plan.py` · `execute_mark_soul_day` |
| **Gold Day Spine exec** | Re-exec of compiled plan under shell → **same 50/50, breach 0** (after online-fallback fix) | `compile_day_spine.py` · `spine_oracle_score.py` |
| **Force-gate / mark_align** | HTF side law outside the net (pt5); wraps policy proposals | `mark_aligned_decode.py` · `GoalEquityDay.mark_align_policy=True` |
| **KEEP/REJECT conscience** | Restores best embryo when pack falls or breach>0 — only reason best is still 35 not 23 | `fable_50d_one_day.py` · `spine_*` loops |
| **One-day focus + award protect + high KL** | Only method that **raised same** (27→30→33; peer pack to **35**) without opening 3 teachers | `fable_50d_one_day.py` · BEST `pack_one_day_KEEP_2026-02-13` |
| **Directional oversample + enough HOLD** | Dir-heavy labels convert MWT; HOLD mass stops breach / pack death | LEARNING_50D_MATCH; safe climb: keep train `pred_hold_rate` from collapsing (~0.27 cratered 35→30) |
| **Pack-crater cutoff** | If post same &lt; best−3 → REJECT restore immediately (no long repair) | `spine_safe_one_day.py` |
| **Oracle cache** | Reuse Mark plans per `date\|T\|R` — no re-search every cycle | `MARK_ORACLE_CACHE__50d.json` |
| **100d holdout construction** | Fit = practice 50; holdout = 40 post-fit future + 60 pre-fit past from `XAUUSD_M1_full.csv`, **∩ empty** | `score_forward_100d.py` · ~1461 loadable days on full M1 |

---

## Diagnosis that is useful (actionable)

On MWT days at same=35:

| Condition | Awards on those days |
|-----------|---------------------:|
| Policy pure greedy mark_align | 0/15 |
| Policy + Mark size dials locked | 1/15 |
| Gold spine `run_plan` | **15/15** |

**Works as conclusion:** ceiling is the **spine timing path**, not “no edge” and not mainly size.  
Train so the policy **hits the same fire windows under its own trajectory** (path / DAgger), while **freezing award days** (KL + award self-imitate).

---

## KEEP rules that work

Keep an embryo only if **all** hold:

1. `n_breach == 0`  
2. `same_outcome >= live best` (never below session floor)  
3. `policy_clear` not below frozen baseline floor  
4. Hold skill not collapsed (entries thrash → reject)  
5. Optional: focus day converted **and** pack not down  

If focus converts but pack drops hard → pack-repair once; if still down → **REJECT restore**. Do not save.

---

## Recipe that raised the board (copy this shape)

**Ship file that produced KEEPs:** `fable_50d_one_day.py` (not experimental rewrites).

```
best = score(practice_50, seed=42, soft_bias=false, pure_greedy, mark_align)
loop:
  focus = MWT sorted by worst pnl, rotate by round index
  labels =
      plan_labels focus ×6 (dir_copy=10, hold_copy=4)
    + dagger_labels focus ×4
    + award_self awards[:24] ×1.5 weight
    + plan_labels 2 other MWT light
  train_bc(epochs=35, lr=2.5e-4, kl_coef=0.55, warm=best, kl_anchor=best)
  post = score(full practice_50)
  if focus_ok and pack slipped:
      pack-repair FROM best_state (not thrash weights):
        heavy award_self from BEST policy + light focus plan_labels
        kl≈0.72; optional pass-2 more award freeze
  KEEP if breach=0 and (same>best or (focus_ok and same>=best and clear>=best))
  else restore best weights
```

**Pack-repair that works (method):** graft focus onto **BEST embryo**, do not fine-tune the collapsed pack.

**Head-only update:** `freeze_trunk=True` reduces pack damage (POST same 34 vs 32) but often fails to convert focus. Use for repair / polish, not sole convert lever.

**Mid climb mix (trying):** full net · kl≈0.72 · epochs≈20 · award weight↑ · DAgger×2 not ×4 — between “convert+crater” and “safe+no-convert”.

**Embryo path:** `checkpoints/mark_clone_full_obs_v1.pt` (never PROVEN).  
**Meters file:** `checkpoints/fable_50d_match/BEST__latest.json`.

**Overnight note:** re-tuned “safe” variants with higher HOLD still cratered 35→30 without KEEP. Prefer **running `fable_50d_one_day.py` as shipped** over re-deriving the mix.

---

## Spine Shadow products that work (S0–S1 done)

| Product | Status |
|---------|--------|
| Sparse Day Spine compile from soul plan | **Works** — round-trip unit tests pass |
| Oracle gold exec gate ≥48/50 breach0 | **Works** — 50/50 after fallback fix |
| Unit tests `tests/lineages/test_spine_shadow.py` | **7 passed** |
| Error class taxonomy (false_hold / false_fire / late / early / wrong_size) | **Works** for cards |

---

## Forward consistency design that works (honest)

- **Do not** claim 100 calendar forward from curriculum alone (only ~40 days after practice 50).  
- **Do** use `data/raw/XAUUSD_M1_full.csv` via `load_raw_m1` / `score_forward_100d.py`.  
- Dual score same frozen recipe; log `fit_day_set`, `day_set`, `intersection=[]`.  
- Pass only if dual runs match goal meters and breach 0 — never invent meters.

---

## House laws that work (non-negotiable)

- PROVEN never load/overwrite for write  
- Agents = sensors only  
- Trail+cushion+scale-in package off  
- HTF gates side; LTF times  
- One process / no 3 parallel LLM teachers as curriculum  

---

## Live board (update when KEEP)

| Snapshot | same | mwt | breach | source |
|----------|-----:|----:|-------:|--------|
| Plan floor | 33 | 17 | 0 | one_day_KEEP_2026-02-25 |
| Best on disk (start overnight) | **35** | 15 | 0 | pack_one_day_KEEP_2026-02-13 |
| Oracle spines | **50** | — | 0 | spine_oracle_score__latest |

---

## Pointers for next agent

1. Read this file + `BEST__latest.json` + `spine_oracle_score__latest.json`.  
2. Climb with **one-day / high-KL / award-protect** shape; spine events weight focus fires.  
3. Every KEEP: append one line to `SPINE_SHADOW_LEARNING.md` with meters + focus date.  
4. When practice ≥40 or stalled: run `score_forward_100d.py` for honest forward meters.  
5. If something collapses the pack, REJECT is a **win for the method** — restore and note the lever that protected the board.

## Conscience levers that work (observed overnight)

| Lever | What it does for the goal |
|-------|---------------------------|
| **REJECT + restore** | Keeps best same at 35 when a step would save a worse pack |
| **Pack-crater cutoff** | If post same < best−3, skip long pack-repair; restore immediately |
| **Oracle green gate** | Train only after gold spines ≈ Mark (50/50) — else fix compiler |
| **PROVEN fingerprint** | Yardstick unchanged — hash file on disk |

## Live KEEP log (append-only)

| Event | same | mwt | breach | note |
|-------|-----:|----:|-------:|------|
| session open | 35 | 15 | 0 | pack_one_day_KEEP_2026-02-13 |

*Only successes and durable method relative to the goal. Failures omitted except as “do not do.”*
| KEEP fable-kag | **36** | 14 | 0 | family=fire_skill multi-day pattern BC |
