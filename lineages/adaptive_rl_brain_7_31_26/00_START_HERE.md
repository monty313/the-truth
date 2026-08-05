# NEW BRAIN — start here (T3 Mark soul lineage)

**This folder = one experiment.**  
It is **not** the champion.

The champion (PROVEN) lives in `models/`.  
**Do not save over PROVEN from here.**

**Parent house map:** `../../00_MAP_OF_THE_HOUSE.md`  
**Work order in this folder:** `00_TRACK_ORDER.md`  
**Soul status:** `MARK_SOUL_TRANSFER.md`  
**Current root handoff:** `../../HANDOFF_2026-08-05.md`

---

## Right now (2026-08-05)

| | |
|--|--|
| **Track** | **T3** — Mark soul → policy |
| **Mark soul plans** | **10/10** clear (full chart, size + force adds) |
| **Policy** | **~8/10** — next: more BC days + thrash penalty |
| **Agents / 92 slots** | **Keep; wire after full policy soul** (`../../KEEP_AFTER_SOUL.md`) |

---

## What this brain does (simple)

1. Looks at **all 4 Mark TF sets** (LTF first · HTF last two).  
2. **pt5 / five-law doctrine** — force → regime → velocity → entry.  
3. **Mark soul** — size relative to day goal; force-aligned adds (not thrash reverse package).  
4. Small Channel1 brain (32 obs) BC’d to Mark plans.  
5. Shell: bank at target, die on floor (sacred).  
6. **Not** PROVEN; **not** “wire all agents first.”  

---

## Where things are

### Words for you

| File | What |
|------|------|
| **00_START_HERE.md** | This page |
| **00_TRACK_ORDER.md** | **Soul first → agents later** (locked order) |
| **MARK_SOUL_TRANSFER.md** | Soul definition + scores + commands |
| **00_7_31_26_HANDOFF.md** | Older multi-pair / three-track handoff (history) |
| **STATUS.md** | Older overnight numbers (history) |
| **HANDOFF.md** | Older AI helper notes |
| **CURRICULUM.md** | Real training days list |
| **todo_7_31_26.md** | Overnight mission plan (A–D) |
| **README.md** | List of code files |
| **SPEC_PHASE1.md** | Locked rules (long) |
| **PRINCIPLES_OF_SUCCESS.md** | Student–tutor principles of the winning multi-pair policy |
| **00_PRINCIPLES_CONTEXT_MEASUREMENT.md** | Nine context principles → labels, logs, tests (no new entry rules) |
| **00_LABEL_CONTRACT_V1.md** | Principle 9: frozen label contracts + V1 audit schema + replay tests |
| **MARKET_CONDITION_TRADE_DECISION_AUDIT.md** | Hard-target audit schema + practice/forward evidence |
| **agents/MULTI_PAIR_TUTOR_PERSONA.md** | Personified policy — system voice for chat |
| **tutor_day_walk.py** | First-person walk of a real day (heuristic + shell) |
| **MARK_SETS_LAW.md** | **Immutable 4-set TF law** (LTF first, HTF last two) |
| **MARK_ALWAYS_LEARNING.md** | Mark sees chart **first** → updates policy → harvests others |
| **mark_soul_plan.py** | Full-chart soul plans (size + force add) |
| **mark_first_learn_cycle.py** | Runnable always-learning cycle |
| **mark_day_diary.py** | Chart first → write day (pt5 + soul) → BC policy same |
| **MARK_CLONE_AS_POLICY_ISSUES.md** | **Why policy thrash / chart diagnosis + issue IDs** |
| **MARK_CLONE_POLICY.md** | Mark clone transplant into attention + meta-learn |
| **MARK_DOCTRINE_FIVE_LAWS.md** | **How Mark thinks** (FORCE→REGIME→VELOCITY→ENTRY→RISK) |
| **../../POLICY_EQUALS_MARK_ON_CHART.md** | **Dual stack map** — production RL/meta + this lineage |
| **perception/mark_doctrine.py** | Five-law teacher code |
| **train_mark_clone_bc.py** | New Channel1 brain BC to doctrine teacher |
| **compare_mark_clone_attention.py** | A/B claim eyes vs Mark attention gates |
| **UNSEEN_CONSISTENCY_RECIPE.md** | Dream build list + **all gaps (G01–G25)** to fulfil |
| **../../references/plans/GOAL_FROM_TEN_PAIR_IRAC.md** | GOAL keep/reject from multi-pair IRAC |
| **../../references/plans/TEN_PAIR_CONSISTENCY_IRAC.md** | Full 10-pair process log |

### Code (this folder only)

| File / folder | What |
|---------------|------|
| **perception/** | Eyes + tags |
| **day_runner.py** | One day loop + thrash + rewards |
| **rewards.py** | Points for good/bad |
| **practice_long.py** | Short synthetic practice |
| **train_curriculum.py** | First real-data train |
| **real_curriculum.py** | Pick real days from `data/raw/` |
| **checkpoints/** | Weights + JSON reports only |

---

## What is done (2026-07-31)

- Phase 1–2 eyes + day loop  
- Anti-hold sandbox (entries on synthetic)  
- Thrash limits  
- Real XAUUSD curriculum (5 days)  
- First curriculum train + report  

## Multi-pair win (done)

Same brain, **10** target/risk pairs, each **≥30 clear days**, **0% breach** on real XAUUSD.

```powershell
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all
```

Full simple story: **`references/plans/TEN_PAIR_CONSISTENCY_IRAC.md`**  
Short: **`MULTI_PAIR_README.md`**

## What is not done

- **Full policy soul** (policy ≥9/10 + low thrash on soft days)  
- Award streak under random pairs without retrain  
- Wire 92 agents / rich obs **after** soul (frozen by design)  
- Pure greedy Channel1 RL hold-freeze (T4 parked)  
- Not ready to promote over PROVEN  

---

## Safe commands

From repo root:

```powershell
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/train_curriculum.py
```

Or tests only:

```text
USE/6_new_brain_tests.bat
```

---

## Latest weights

- `checkpoints/channel1_curriculum_v1.pt`  
- `checkpoints/channel1_sandbox_latest.pt`  

Read **STATUS.md** for numbers and the exact next step.

---

## Lost?

1. Repo root → **`00_START_HERE.md`**  
2. This page again  
3. **STATUS.md** for “what happened while I slept”  
4. Only open **one** file at a time  
