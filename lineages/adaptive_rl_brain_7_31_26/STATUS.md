# STATUS — adaptive_rl_brain_7_31_26

**Date:** 2026-07-31 (CEST)  
**Owner sleep mission:** Phase A → C complete  
**Multi-pair goal:** **PASS** — 10 pairs × ≥30 clear days × 0% breach on all real curriculum days  
**PROVEN:** not touched  
**Re-verified:** unit tests + ten-pair score path  

If lost → open **`00_START_HERE.md`** in this folder.  
Multi-pair story → **`references/plans/TEN_PAIR_CONSISTENCY_IRAC.md`**  
GOAL keep/reject (shell + pairs) → **`references/plans/GOAL_FROM_TEN_PAIR_IRAC.md`**  
Student–tutor principles of winning policy → **`PRINCIPLES_OF_SUCCESS.md`**  
Unseen consistency recipe + gaps G01–G25 → **`UNSEEN_CONSISTENCY_RECIPE.md`**

---

## Completed while owner slept

| Step | Result |
|------|--------|
| **A1 Thrash control** | Max open units=3, reverse cooldown=100 M1 bars, flip-flop penalty=−1.0 in `day_runner` + `rewards` |
| **A2 Anti-hold stack** | EOD did-nothing −25, inactivity floor, correct-side bonus, MINDLESS −10 still on |
| **A3 Logging** | JSON reports under `checkpoints/` with tags, actions, entries, mean reward, mindless, `proven_touched: false` |
| **B1 Curriculum** | 5 real XAUUSD days from `data/raw/XAUUSD_curriculum_2026.csv` → see `CURRICULUM.md` |
| **B2 Multi-TF** | M1→pack verified on all curriculum days (indicator cache already in DayRunner) |
| **C1 Train** | Warm synthetic + fine-tune real → `channel1_curriculum_v1.pt` |
| **C2 Eval** | Before/after report written (see key numbers) |
| **C3 Docs** | This file + `HANDOFF.md` + `00_START_HERE.md` updated |
| **Phase D** | **Not started** — health gate failed (pure greedy still freezes on real) |

---

## Latest checkpoint / report

| Kind | Path |
|------|------|
| **Curriculum ckpt (v1)** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/channel1_curriculum_v1.pt` |
| **Hold-shape ckpt (v2)** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/channel1_curriculum_v2_hold_shape.pt` |
| **Second-best ckpt (v3)** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/channel1_curriculum_v3_second_best.pt` |
| **Latest pointer** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/channel1_sandbox_latest.pt` |
| **Train report** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/curriculum_train_report.json` |
| **Hold-shape report** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/hold_shape_finetune_report.json` |
| **Second-best report** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/second_best_regret_report.json` |
| **Day list** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/curriculum_days.json` |
| **Curriculum doc** | `lineages/adaptive_rl_brain_7_31_26/CURRICULUM.md` |

### Hold-shape finetune (2026-07-31, pure greedy raw argmax)

| Metric | Before (v1) | After (v2) |
|--------|------------:|-----------:|
| HOLD rate | 100% | **50%** |
| Entries / day | 0, 0 | **3, 3** |
| Mean logits H/B/S | +0.44 / −0.24 / −0.17 | **+0.22 / −0.27 / +0.07** |
| Mindless | 0% (no entries) | 25% |
| Match rec on flat+dir | 0% | 40% |

Shaping: `DIRECTIONAL_HOLD_PENALTY=-3.5`, `STRUCTURE_MATCH_ENTRY_BONUS=+2.0`. CE only on flat+directional (no CE→HOLD). PROVEN untouched.

### Second-best regret finetune (v3, pure greedy raw argmax)

| Metric | Before (v2) | After (v3) |
|--------|------------:|-----------:|
| HOLD rate | 50.0% | **43.8%** |
| Entries / day | 3, 3 | **6, 4** |
| Mean logits H/B/S | +0.22 / −0.27 / +0.07 | **+0.09 / −0.28 / +0.27** |
| 2nd-best profitable when HOLD | 1/3 (33%) | **1/2 (50%)** |
| Mindless | 25% | 29.2% |
| BUY vs SELL actions | 0 / 24 | **2 / 25** (still SELL-heavy) |

Rule: flat HOLD → CF 2nd-best entry over 25 M1 bars (−0.50 fee); if CF PnL > 0 → regret −3.0. No sampler ban.

---

## Key numbers (Phase C — real XAUUSD eval days 2026-04-21, 2026-05-06)

### Pure greedy AFTER (raw policy argmax)

| Metric | Value |
|--------|------:|
| Entries | **0** |
| Hold rate | **100%** |
| Mindless rate | 0% |
| Mean reward | **−1.94** |
| Days did-nothing | **2** (EOD −25 each) |

→ **Still collapses to all-hold under pure greedy on real bars.**

### Anti-hold greedy AFTER (rec side if flat + hold)

| Metric | Value |
|--------|------:|
| Entries | **2** (1 per eval day) |
| Hold rate | **95.8%** |
| Mindless rate | **35.4%** |
| Mean reward | **+0.016** |
| Days did-nothing | **0** |

### Stochastic AFTER (sampling)

| Metric | Value |
|--------|------:|
| Entries | **16** |
| Hold rate | 66.7% |
| Mindless rate | 29.2% |
| Mean reward | −1.02 |
| Thrash blocks | scale_blocks=5, cooldown_blocks=1, reverses=6 |

### Synthetic anti-hold greedy (sanity)

| Metric | Value |
|--------|------:|
| Entries | **2** / 2 days |
| Mindless | **0%** |
| Mean reward | **+0.019** |

---

## Behavior answers (owner questions)

| Question | Answer |
|----------|--------|
| Still collapses to hold? | **Yes on pure greedy / real data.** Anti-hold greedy and stochastic still enter. EOD −25 fires correctly when frozen. |
| Thrash under control? | **Mostly yes.** Scale-in capped at 3; reverse cooldown + flip tax fire under stochastic thrash. |
| PROVEN touched? | **No.** |
| Phase D ready? | **No** — do not start longer train until pure greedy (or robust decode) enters without freeze. |

---

## Blockers

1. **Pure greedy all-hold on real XAUUSD** after train (OOD / mindless wall makes entries costly).  
2. **Mindless rate ~30–35%** on real when it does enter (heuristic mindless (a)(b)(c) still rough without live Vector M).  
3. **Signal majority off** during train for speed (`USE_SIGNAL_MAJORITY=False`).  
4. **Small curriculum** (5 days only from available calendar slices).

---

## Exact next action for owner when you wake up

**Do this first (pick one):**

1. **Fix pure-greedy freeze (recommended)**  
   - Open `lineages/adaptive_rl_brain_7_31_26/train_curriculum.py`  
   - Strengthen BC so non-HOLD recommendations win under argmax on real obs  
   - Or accept **anti-hold greedy** as execution decode and train with that in the loop  
   - Re-run:  
     `python lineages/adaptive_rl_brain_7_31_26/train_curriculum.py`  
   - Goal: pure greedy entries > 0 and hold_rate &lt; 90% on real eval without all-hold.

2. **Or inspect evidence only**  
   - Read `checkpoints/curriculum_train_report.json`  
   - Read `CURRICULUM.md`  
   - Do **not** run prove_it / do **not** touch PROVEN.

3. **Or expand data**  
   - Pull more clean XAUUSD days into `data/raw/` and rebuild curriculum via `real_curriculum.py`.

**Do not:** promote over PROVEN, run `prove_it`, or write under `models/`.

---

## Decisions still locked

1. `MINDLESS_PENALTY = -10`  
2. Heuristic mindless (a)(b)(c) for now  
3. Own small policy dim 32 — never PROVEN 1820  
4. `DID_NOTHING_EOD_PENALTY = -25`  
5. Majority rule (when on): ≥10 active, ≥60% agree, idle unless ≥2 open  
6. Thrash: `MAX_OPEN_UNITS=3`, `REVERSE_COOLDOWN_BARS=100`, `FLIP_FLOP_PENALTY=-1`  
7. Checkpoints only under this lineage folder  

---

## Multi-pair claim (2026-07-31) — 10/10 PASS

| Meter | Result |
|-------|--------|
| Pairs | 10 distinct (target%, risk%) in `ten_pairs.json` |
| Days | 90 real XAUUSD from `data/raw/XAUUSD_curriculum_2026.csv` |
| Clear floor | each pair ≥30 clear days |
| Breach | **0%** on every pair |
| Same brain | dials + heuristic decode in `checkpoints/multi_pair_consistent_v1.pt` |
| Score | `python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all` |

| Pair | Target | Risk | Clear | Breach |
|-----:|-------:|-----:|------:|-------:|
| 1 | 1.0 | 2.0 | 76 | 0 |
| 2 | 1.0 | 2.5 | 76 | 0 |
| 3 | 1.5 | 2.0 | 65 | 0 |
| 4 | 1.5 | 2.5 | 70 | 0 |
| 5 | 1.5 | 3.0 | 70 | 0 |
| 6 | 2.0 | 2.5 | 60 | 0 |
| 7 | 2.0 | 3.0 | 60 | 0 |
| 8 | 2.0 | 3.5 | 60 | 0 |
| 9 | 2.5 | 3.5 | 48 | 0 |
| 10 | 3.0 | 3.5 | 40 | 0 |

**What worked:** bar-level stop/breach marks; floor-scale sizing; same direction signal flat and in-trade.  
**What failed (reverted):** trail stop + scale-in (collapsed pass rate to 0/10).

Docs: `MULTI_PAIR_README.md` · full IRAC `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md`

---

## One line

**Multi-pair 10×30×0-breach PASS on real XAUUSD; pure RL greedy still freezes — multi-pair win uses perception heuristic + equity shell; PROVEN untouched.**
