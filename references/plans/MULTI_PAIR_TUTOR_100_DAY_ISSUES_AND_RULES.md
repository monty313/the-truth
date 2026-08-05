# Multi-pair tutor — issues + rules for 100-day GOAL consistency

**What this is:** decision catalog for what the **/multi-pair-tutor** stack needs so **GOAL.md** can hold on **real price data**, with **random daily target%/risk%**, under a **100-day consistency** conclusion bar.

**What this is not:** a claim that 100/100 clears already exist. Measured claim is lower (see ISS-02). Lid stays off — low clear is a policy/shell/practice problem, not “impossible.”

**Mission (GOAL.md, verbatim terms):**

| Term | Rule |
|------|------|
| **target%** | Runtime daily profit goal Monty types |
| **risk%** | Runtime daily floor Monty types |
| **clear%** | Days that hit **target%** and never hit **−risk%** → **climb** |
| **breach%** | Days that hit **−risk%** → **must stay 0%** |
| **streak** | Clears in a row at that pair → climb |
| **no retrain** | Change the two numbers → same brain (ckpt + dials + decode) |

**Tutor stack identity (claim winner):** heuristic direction + equity shell — **not** Channel1 pure-greedy RL, **not** PROVEN champion unless comparing tracks.

| Piece | Path |
|-------|------|
| Engine | `lineages/adaptive_rl_brain_7_31_26/equity_day.py` |
| Scorer | `lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py` |
| Pairs | `lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` |
| Dials | `lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_dials.json` |
| Claim JSON | `lineages/adaptive_rl_brain_7_31_26/checkpoints/ten_pair_score_all.json` |
| Forward JSON | `lineages/adaptive_rl_brain_7_31_26/checkpoints/ten_pair_score_forward.json` |
| Persona | `lineages/adaptive_rl_brain_7_31_26/agents/MULTI_PAIR_TUTOR_PERSONA.md` |
| Gaps G01–G25 | `lineages/adaptive_rl_brain_7_31_26/UNSEEN_CONSISTENCY_RECIPE.md` |
| IRAC KEEP/REJECT | `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md`, `GOAL_FROM_TEN_PAIR_IRAC.md` |

---

## How to use this document

1. Read **measured baseline** (below).  
2. For each **ISS-##**, pick **one** rule letter (A/B/C…).  
3. Fill **§ Chosen rules → 100-day conclusion**.  
4. That filled block **is** the pass/fail protocol — not a hope.

---

## Measured baseline (evidence, not fantasy)

### Data window (real M1 XAUUSD)

| Fact | Value | Evidence |
|------|-------|----------|
| CSV | `data/raw/XAUUSD_curriculum_2026.csv` | `ten_pairs.json` `data_source` |
| Days with ≥900 bars | **90** | `load_calendar_days` → claim `n_days=90` |
| Span | 2026-01-20 → 2026-05-26 | `ten_pair_score_all.json` day_rows |
| Practice / forward split | first **50** / remaining **40** | `ten_pairs.json` + `split_practice_forward` |
| Thin days dropped | **0** (all 90 ≥900 bars) | loader count 2026-07-31 |

**Gap:** curriculum has **90** usable days, not 100. A strict 100-calendar-day bar needs more bars (`data/raw/XAUUSD_M1_full.csv` exists, larger) or a redefined window (ISS-03).

### Multi-pair claim (mode=all, same brain, no retrain per pair)

Source: `checkpoints/ten_pair_score_all.json` · `n_pass=10` · `all_pass=true` · decode heuristic · dials `risk_use_frac=0.35`, `stop_atr_mult=2.0`, `per_trade_cap_pct=0.25`.

| Pair | target% | risk% | Clear days / 90 | clear% | breach | Pass (≥30 clear & 0 breach) |
|-----:|--------:|------:|----------------:|-------:|-------:|:---------------------------:|
| 1 | 1.0 | 2.0 | 76 | 84.4% | 0 | YES |
| 2 | 1.0 | 2.5 | 76 | 84.4% | 0 | YES |
| 3 | 1.5 | 2.0 | 65 | 72.2% | 0 | YES |
| 4 | 1.5 | 2.5 | 70 | 77.8% | 0 | YES |
| 5 | 1.5 | 3.0 | 70 | 77.8% | 0 | YES |
| 6 | 2.0 | 2.5 | 60 | 66.7% | 0 | YES |
| 7 | 2.0 | 3.0 | 60 | 66.7% | 0 | YES |
| 8 | 2.0 | 3.5 | 60 | 66.7% | 0 | YES |
| 9 | 2.5 | 3.5 | 48 | 53.3% | 0 | YES |
| 10 | 3.0 | 3.5 | 40 | 44.4% | 0 | YES |

**Hard target fact:** pair 10 (yardstick-like 3.0/3.5) clears **40/90**, not 90/90. Max clear streak claim pair1 ≈22; pair10 ≈5 (computed from day_rows).

### Forward holdout (mode=forward, 40 days)

Source: `checkpoints/ten_pair_score_forward.json` · `n_pass=5` · `all_pass=false` · span 2026-03-31 → 2026-05-26.

| Pair | Clear / 40 | clear% | breach | Pass (≥30 clear on 40d) |
|-----:|-----------:|-------:|-------:|:-----------------------:|
| 1–2 | 35 | 87.5% | 0 | YES |
| 3 | 30 | 75.0% | 0 | YES |
| 4–5 | 34 | 85.0% | 0 | YES |
| 6–8 | 27 | 67.5% | 0 | **NO** (clear short) |
| 9 | 17 | 42.5% | 0 | **NO** |
| 10 | 12 | 30.0% | 0 | **NO** |

**Honesty fact:** breach still **0%** on forward for all 10 pairs. Failures are **clear shortfall** under the absolute ≥30-on-40 gate — not floor death.

### Champion track (different brain)

Source: `outputs/reports/GOAL_PROVE_MULTI_PAIR_2026-07-31.md` · `models/00_CHAMPION.md`.

| Brain | Pair | clear% | breach% |
|-------|------|-------:|--------:|
| PROVEN_SPRINT_row04… | 3.0 / 3.5 | ~24% | 0% |
| PROVEN (same .pt) | 2.5 / 2.5 | 20% | 0% |
| PROVEN (same .pt) | 1.5 / 2.0 | 29% | 0% |

**Do not conflate** multi-pair heuristic claim with PROVEN `prove_it` scores.

### Near-floor stress (claim path)

From claim day_rows: day **2026-02-03** hits min_eq ≈ **−1.998** (pair1 risk 2.0), ≈ **−2.99** (pair5 risk 3.0), ≈ **−3.475** (pair10 risk 3.5) — still **breached=false**. Floor margin can be thin; shell heat is doing work, not magic invulnerability.

### REJECT families (IRAC)

| ID | What | Measured result | Source |
|----|------|-----------------|--------|
| **R1** | Trail stop + big cushion + scale-in package | multi-pair **6/10 → 0/10** | TEN_PAIR IRAC-01, equity_day CHANGE LOG |
| **R2** | Stops only on decision bars | floor walk-through risk | IRAC-02 KEEP every-bar marks |
| **R3** | Weaker in-trade signal | high targets stuck ~18–23 clears | IRAC-03 |
| **R4** | Pure greedy RL alone as multi-pair solver | freeze / no claim win | STATUS.md, PRINCIPLES R4 |
| **R5** | Huge dial grids on full rebuilds | too slow | IRAC |
| **R6** | Promote on “entries up” without clear/breach | not GOAL | GOAL_FROM_TEN_PAIR |

### Dial-search leak (code fact)

`train_multi_pair.py` line comment: **“Search on ALL days for the claim bar”** when `--search-dials`. Forward is not a pure never-fitted window if dials were chosen on all days. Matches gap **G04** in UNSEEN_CONSISTENCY_RECIPE.

### Tutor walk partial-day risk

`tutor_day_walk.py` defaults `--max-decisions 40` then stops narration and marks remaining bars **without further opens**. EOD can differ from full `GoalEquityDay.run()` if bank/entries would have continued. Gap **G22**.

---

## Issue catalog (complete for decision)

Each issue: **why it blocks 100-day random-input GOAL**, **evidence**, **MC rules**.  
Choosing letters fills the conclusion protocol in the last section.

---

### ISS-01 — What “won” means (win metrics)

**Problem:** People swap clear/breach for PnL folklore, entry counts, or “felt consistent.” GOAL only scores **clear%**, **breach%**, **streak** at typed **target%/risk%**.

**Evidence:** GOAL.md “How we know we won”; score JSON fields `cleared`, `breached`, `clear_pct`, `breach_pct`; PRINCIPLES P1/P9; no streak field in score JSON yet (G09).

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Official meters only: **clear%**, **breach%**, **streak** at the day’s typed pair. PnL is diagnostic, never pass/fail. |
| **B** | Same as A, plus require **banked=true** on clear days (must bank at target, not luck EOD). |
| **C** | Allow “soft clear” if end equity ≥ target even after a floor touch → **FORBIDDEN** (redefines breach). |

**Conclusion impact:** Only A or B may feed a 100-day pass. C is invalid under GOAL.

---

### ISS-02 — “100 consistent days” is not the measured claim

**Problem:** Claim is **≥30 clear / 90 days** per frozen pair (≈33% floor), with pair10 at **44% clear**. “100 consistent days” must be defined as a **protocol**, not assumed achieved.

**Evidence:** `ten_pairs.json` `min_clear_days_per_pair=30`; claim pair clears 40–76/90; forward pair10 only 12/40.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | **100-day window** = 100 consecutive eligible calendar days (≥900 bars). **Pass** = over that window, for the protocol’s inputs: **breach% = 0** and **clear% ≥ C_min** (choose C_min in ISS-04). Not “100/100 clears” unless C_min=100%. |
| **B** | **Strict every-day clear:** clear on **100/100** days and breach 0. (Ambition bar; current evidence does not support it yet — lid off, treat as climb target.) |
| **C** | Keep legacy claim only: ≥30 clear on whatever days exist; ignore “100”. |

**Conclusion impact:** A makes 100 a **window length**. B makes 100 a **perfect score**. C refuses the user’s 100-day bar.

---

### ISS-03 — Day window & data coverage (real price data)

**Problem:** Usable claim set is **90** XAUUSD days in curriculum CSV. Cannot score a 100-day window without more days or a redefined “day.”

**Evidence:** 90 days via `load_calendar_days`; `XAUUSD_M1_full.csv` exists (~121MB) but is **not** the frozen claim source; other symbols EURUSD/GBPUSD/US30 exist but multi-pair claim is XAUUSD-only (G19).

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Extend eligible real days from **same meaning** pipeline until **≥100** days (prefer chronological continuation of XAUUSD M1, min_bars=900). 100-day score uses that set. |
| **B** | Use **rolling 100** once data allows; until then report **90-day protocol** as interim with explicit “n=90 &lt; 100” flag (not a pass of 100-day bar). |
| **C** | Pad with synthetic days → **REJECT** for GOAL real-data bar. |
| **D** | Multi-symbol 100 days mixed → only if meaning_version identical and per-symbol breach 0 reported separately. |

**Conclusion impact:** Without A or temporary B, the 100-day conclusion is **impossible to measure** (data shortage), not “market refused.”

---

### ISS-04 — Clear rate floor on the 100-day window

**Problem:** Harder targets clear less (claim: 1.0%→84%, 3.0%→44%). Random inputs will include hard pairs. Must pick a clear floor that still respects lid-off climb.

**Evidence:** claim clear% table above; forward worse on hard pairs.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | **breach% = 0** mandatory. **clear% ≥ 50%** on the 100-day random-input aggregate (and/or per-pair if multi-pair mode). Climb later. |
| **B** | **breach% = 0** and **clear% ≥ claim-style 33%** (≥30/90 equivalent rate: clear_days ≥ ceil(0.333×n_days)). |
| **C** | **breach% = 0** and **clear% ≥ 70%** (stretch; needs more practice/attention work). |
| **D** | **breach% = 0** and **clear% = 100%** (same as ISS-02 B). |

**Conclusion impact:** Sets the numeric clear gate for “consistent.”

---

### ISS-05 — Random daily target%/risk% protocol (not frozen ten pairs)

**Problem:** Claim uses **fixed 10 pairs** every day. GOAL wants **any** typed pair; user wants **random inputs each day**. No scorer samples a new (target, risk) per day today.

**Evidence:** `ten_pairs.json` fixed list; `score_ten_pairs.py` scores each pair on **all** days in the window, not one random pair per day; GOAL examples 3.0/3.5, 2.5/2.5, 1.5/2.0.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | **Per-day independent draw:** for each calendar day *d*, sample `(target_d, risk_d)` from a frozen distribution (seed S). Same brain runs day *d* once with those inputs. Aggregate clear/breach over 100 days. |
| **B** | **Per-day draw from the 10 frozen pairs** only (uniform or weighted). Still random day-to-day; still no retrain. |
| **C** | **Two-level:** (1) fixed ten-pair claim still must pass; (2) separate random-input 100-day score must pass. Both required. |
| **D** | Random once per **window** (one pair for all 100 days) — weak vs GOAL “type any numbers.” |

**Sampling bounds (if A or C):** must also choose ISS-06.

**Conclusion impact:** Defines what “random inputs each day” means in the final protocol.

---

### ISS-06 — Allowed (target%, risk%) support set

**Problem:** Unbounded random (e.g. target 20% risk 0.1%) invents unwinnable floors or fairy targets. Bounds must be explicit without declaring market impossible.

**Evidence:** frozen pairs live in target ∈ [1.0, 3.0], risk ∈ [2.0, 3.5]; heat/size scale with risk (`floor_scale` in `_try_open`); GOAL yardstick 3.0/3.5.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Support box: **target% ∈ [1.0, 3.5]**, **risk% ∈ [1.5, 4.0]**, require **risk% ≥ target% × 0.8** (or risk ≥ target) so floor is not tighter than a coin-flip bank. Discrete 0.5 steps. Seed fixed for repro. |
| **B** | Only the **10 frozen pairs** as support (ISS-05 B). |
| **C** | Continuous uniform in A’s box (not only 0.5 grid). |
| **D** | Include yardstick always as mandatory second score even if random misses it. |

**Conclusion impact:** Bounds the random law so pass/fail is reproducible.

---

### ISS-07 — Same brain / no retrain across pairs and days

**Problem:** Switching numbers must not retrain. Dials + decode + ckpt must be frozen for the whole 100-day score.

**Evidence:** GOAL “without retrain”; claim same `multi_pair_consistent_v1.pt` + dials; `multi_pair_dials.json` decode=heuristic.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Pin **one** triple for the whole score: `{ckpt path hash, dials JSON, decode=heuristic|policy}`. Report refuses if any field mutates mid-window. |
| **B** | Allow attention re-fit **only on practice** mid-program, then freeze before 100-day score (meta path G16–G18) — score window still frozen. |
| **C** | Retrain per target/risk → **FORBIDDEN** under GOAL. |

**Conclusion impact:** Encodes no-retrain in the conclusion protocol.

---

### ISS-08 — Claim bar vs forward bar confusion

**Problem:** Absolute **≥30 clears** on 90 days (claim) vs **≥30 clears** on 40 days (forward) are different rates (33% vs 75%). Forward “FAIL” misleads.

**Evidence:** G08 UNSEEN recipe; forward 5/10 fail; all breach 0.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Publish **two meters always:** (1) claim/all-window clear%+breach%; (2) forward/holdout clear%+breach%. Pass rules may differ by rate, not raw 30. |
| **B** | 100-day random protocol **is** the only official bar; ten-pair claim becomes regression only. |
| **C** | Require both: ten-pair claim **and** 100-day random, each with own clear floor. |

**Conclusion impact:** Prevents false “failed unseen” when rate still good.

---

### ISS-09 — Streak meter missing

**Problem:** GOAL lists **streak**; multi-pair score JSON has no max streak field.

**Evidence:** G09; GOAL.md streak row; prove_it report includes longest clear streak for champion.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Compute **max clear streak** and **end streak** on the 100-day sequence; report both; pass does not require streak min unless chosen in B. |
| **B** | Pass requires max streak ≥ **K** (e.g. K=5 or 10) **and** breach 0. |
| **C** | Streak diagnostic only; never gate. |

---

### ISS-10 — Shell laws must stay locked (KEEP physics)

**Problem:** Without locked shell, “improvements” destroy breach 0 (R1 history).

**Evidence:** equity_day heat/bank/every-bar/one-signal; IRAC KEEP A–F; G11 no SHELL_LOCKED flag yet.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | **SHELL_LOCKED** for 100-day score: bank at target; heat refuse; floor-scale size; every-bar marks; one signal flat+in-trade; breach death. Changes require explicit unlock + dual re-score. |
| **B** | Shell editable during search if multi-pair claim still 10/10 breach 0 (risky; R1 caution). |
| **C** | Replace shell with pure RL death only → **rejects claim history** (R4). |

**Minimum shell laws (A):** map to `GoalEquityDay._try_open`, `_mark_bar`, `recommended_action`, `_check_breach_and_bank`.

---

### ISS-11 — Banned rule families (REJECT memory)

**Problem:** R1 can return quietly; no machine ban-list (G12). day_runner still has scale-in for Channel1 path; equity_day claim path does not use trail package.

**Evidence:** IRAC-01 0/10; PRINCIPLES R1; G12 open.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Frozen ban list for multi-pair / 100-day path: **trail+cushion+scale-in package**, decision-only stops, dual weaker in-trade eyes, pure-greedy-only claim. Test asserts equity_day claim path has no trail package. |
| **B** | Ban only trail+scale package; allow separate experiments behind a flag that cannot write claim ckpt. |
| **C** | No ban list — human memory only (status quo; high reintroduce risk). |

---

### ISS-12 — Decode identity (heuristic vs policy vs PROVEN)

**Problem:** Tutor persona is heuristic+shell. Channel1 freezes. PROVEN is another stack.

**Evidence:** dials decode=heuristic; STATUS pure greedy 100% HOLD on real; multi-pair claim uses `use_heuristic=True`.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | 100-day multi-pair-tutor conclusion uses **decode=heuristic** (or policy only if it matches heuristic actions ≥ threshold on practice). Log decode in every report. |
| **B** | Allow policy decode if forward breach 0 and clear ≥ chosen floor. |
| **C** | Score PROVEN via `prove_it` as **separate** 100-day protocol (not multi-pair tutor body). |

---

### ISS-13 — Perception / meaning factory versioning

**Problem:** Silent indicator/structure edits change “eyes”; unseen looks harder when eyes moved (G01–G03).

**Evidence:** UNSEEN G01–G03 open; meaning from day_runner / perception/*; no meaning_manifest.json in checkpoints.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Pin **meaning_version** hash (indicator params + TF stack + tag order). Score fails if hash ≠ freeze. Any eye change forces practice+forward re-score before KEEP. |
| **B** | Document meaning in markdown only (weaker). |
| **C** | No pin — accept silent drift (**rejects honest 100-day science**). |

---

### ISS-14 — Train / dial leak hygiene

**Problem:** Dial search on **all** days contaminates forward honesty (G04–G05).

**Evidence:** `train_multi_pair.py` “Search on ALL days”; practice/forward split exists but search ignores it for dials.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Dial/attention search **only on practice** days. 100-day / forward / claim scores **after** freeze. Automated test: train day set ∩ score holdout = ∅. |
| **B** | Allow all-day dial search for claim climb, but label forward as **inspection only** (status quo honesty). |
| **C** | Nested CV over calendar blocks for dials (stronger, heavier). |

**Conclusion impact:** A is required if “unseen consistency” is part of the 100-day story.

---

### ISS-15 — Practice vs score split for the 100-day bar

**Problem:** If all 100 days were used to pick dials, “consistency” is in-sample.

**Evidence:** 50/40 split; claim uses all 90.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | **Fit window** (practice) ≠ **score window** (100-day or forward). Score window never in dial search. |
| **B** | Single window: all days score + fit (claim style) — allowed only if labeled **in-sample claim**, not unseen. |
| **C** | Walk-forward: re-freeze dials every N practice days; each test segment unused in its fit (advanced). |

---

### ISS-16 — Hard-pair / high-target clear shortfall

**Problem:** Random inputs will hit 2.5–3.5% targets where claim clear is 44–53% and forward can drop to 30%. Consistency bar must not pretend pair1 rates apply to pair10.

**Evidence:** claim pair9–10; forward pair9–10 FAIL absolute 30/40.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Report clear% **stratified** by target bucket (≤1.5, ≤2.5, ≤3.5). Pass requires breach 0 **global** + clear floor **per bucket** (bucket floors may be lower for hard targets). |
| **B** | Single global clear floor (ISS-04) only — hard pairs can drag fail (honest, stricter). |
| **C** | Exclude targets &gt;2.0% from random support until clear climbs (narrows GOAL “any pair”). |

---

### ISS-17 — Near-floor / heat fragility

**Problem:** Some non-breach days sit within a few tenths of the floor (e.g. 2026-02-03). Random tighter risk% may convert near-misses into breaches.

**Evidence:** min_eq near −risk on claim rows; heat uses `risk_use_frac=0.35` and floor_scale.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Keep heat/refuse + floor-scale as sacred; add diagnostic **near_breach_rate** (min_eq ≤ −0.85×risk) without making it a pass gate until breach&gt;0. |
| **B** | Pass requires near_breach_rate ≤ X% (e.g. 10%). |
| **C** | Loosen risk_use_frac for clear climb → only if breach stays 0 on full protocol. |

---

### ISS-18 — Single-symbol / single-season regime

**Problem:** All evidence is XAUUSD Jan–May 2026 curriculum. 100 days in one season is not all regimes (G19–G20).

**Evidence:** data_source; no stress day tags; EURUSD etc. unused by multi-pair score.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Phase-1 conclusion: **XAUUSD real M1**, ≥100 days when available, tagged stress days optional. Phase-2: second season/symbol with same meaning_version. |
| **B** | Require multi-symbol before any “GOAL consistent” label. |
| **C** | XAUUSD only forever for multi-pair-tutor bar (document limit). |

---

### ISS-19 — Scoring product loop incomplete

**Problem:** Gaps G07, G10, G24 — one-pair summary, KEEP/REJECT file, side-by-side windows not automated as a product gate.

**Evidence:** `score_ten_pairs.py --pair` exists in usage string; no `last_score_verdict.json`; streak missing.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | 100-day conclusion artifact must write one JSON: window dates, sampled pairs per day, clear/breach/streak, dials hash, meaning hash, KEEP/REJECT vs previous. |
| **B** | Human checklist only after score_ten_pairs / custom script. |
| **C** | Only markdown report. |

---

### ISS-20 — Tutor fidelity (walk vs full run)

**Problem:** Max-decisions truncation can mis-narrate EOD vs full run (G22). Tutor answers must not invent wins.

**Evidence:** `tutor_day_walk.py` max_decisions default 40; persona requires code-faithful answers.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Any day verdict used in teaching or proof must come from full `GoalEquityDay.run()` (or walk with max_decisions ≥ all decision bars). Narration may truncate; **verdict may not**. |
| **B** | Allow truncated walks if labeled “partial.” |
| **C** | Persona-only answers without code — **forbidden** for score claims. |

---

### ISS-21 — Champion track vs multi-pair-tutor track

**Problem:** Two win conditions. Promoting multi-pair claim over PROVEN without `prove_it` is out of process.

**Evidence:** GOAL_FROM_TEN_PAIR §5; models/00_CHAMPION.md; multi-pair README “does not overwrite PROVEN.”

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | 100-day multi-pair-tutor protocol is **lineage sandbox**. Champion promotion only if same ideas raise `prove_it` clear with breach 0 on PROVEN path. |
| **B** | Dual official: both multi-pair 100-day **and** prove_it 100-day required for “GOAL done.” |
| **C** | Replace prove_it with multi-pair score only → **rejects GOAL champion meter**. |

---

### ISS-22 — Channel1 RL sandbox isolation

**Problem:** HOLD-freeze / regret experiments (v1–v3) are not the claim winner; confusion burns sessions (G06, R4).

**Evidence:** STATUS Phase D blocked; PRINCIPLES A8.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Channel1 scores never write multi-pair claim JSON. Separate scoreboard. Tutor speaks as heuristic+shell unless user asks Channel1. |
| **B** | Merge only when policy decode beats heuristic clear with breach 0 on same 100-day protocol. |
| **C** | Ignore isolation (high confusion risk). |

---

### ISS-23 — Metaplasticity / “senses lying” loop (future, optional for v1 conclusion)

**Problem:** GOAL text splits env meaning / policy attention / meta. Lineage almost lacks meta (G14–G18).

**Evidence:** UNSEEN G14–G18 open; no regime_report.py.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended for v1)** | **Defer meta** for first 100-day conclusion. v1 = frozen eyes + frozen dials + honest score. Meta is phase-2 after G14–G15 exist. |
| **B** | Require meta permit system before any 100-day pass label. |
| **C** | Allow unrestricted re-tuning when clear drops (risk of leak + thrash). |

---

### ISS-24 — Reproducibility seed & double-run

**Problem:** Claim recipe requires re-run twice with matching counts. Random protocol needs a seed.

**Evidence:** TEN_PAIR recipe seed 42; ten_pairs.json seed 42; IRAC forward-test rule.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Fix **seed S** for pair sampling + any stochastic decode. Run 100-day score **twice**; clear/breach/streak must match. |
| **B** | Single run sufficient if fully deterministic heuristic. |
| **C** | Monte Carlo many seeds; pass if ≥95% of seeds meet clear/breach floors. |

---

### ISS-25 — Fees / bank honesty (pay-cliff)

**Problem:** Lid-off doctrine forbids unpaid wins (banked &lt; goal after fees). equity_day banks on equity% ≥ target after flatten.

**Evidence:** equity_day `_check_breach_and_bank`; AGENTS flea cures pay-cliff; claim day_rows often banked=true on clears.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Clear requires final equity% ≥ target **and** never breached; bank path must leave equity ≥ target after flatten (already coded). No counterfeit clear if only unrealized spike. |
| **B** | Allow clear on max equity during day even if EOD below target → weaker / not current engine. |

---

### ISS-26 — What the multi-pair-tutor skill itself must know

**Problem:** `/multi-pair-tutor` can drift from shell laws or invent R1 advice.

**Evidence:** `.grok/skills/multi-pair-tutor/SKILL.md`; MULTI_PAIR_TUTOR_PERSONA.md hard refusals.

| Option | Rule if chosen |
|--------|----------------|
| **A (Recommended)** | Tutor must load persona + PRINCIPLES + equity_day rules; refuse R1; use GOAL definitions; for dated walks run tutor_day_walk / full run; never claim 100/100 without score artifact. |
| **B** | Free-form coach mode allowed. |
| **C** | Tutor may propose shell changes live without IRAC — **reject** for claim path. |

---

## Crosswalk: issues ↔ G01–G25 / IRAC

| Issue | Related gaps / IRAC |
|-------|---------------------|
| ISS-01,04,09 | G08, G09, GOAL meters |
| ISS-02,03 | data window; claim 30/90 |
| ISS-05,06 | not in G-list (new: random protocol) |
| ISS-07,12 | H1–H6, decode |
| ISS-08,15 | G04, G05, G08 |
| ISS-10,11 | IRAC KEEP/REJECT, G11, G12 |
| ISS-13 | G01–G03 |
| ISS-14 | G04–G05 |
| ISS-16,17 | claim/forward hard pairs; near-floor |
| ISS-18 | G19–G20 |
| ISS-19 | G07, G10, G24 |
| ISS-20 | G22 |
| ISS-21 | G25, GOAL tracks |
| ISS-22 | G06, R4 |
| ISS-23 | G14–G18 |
| ISS-24 | IRAC double-run |
| ISS-25 | bank physics, flea pay-cliff |
| ISS-26 | skill/persona |

---

## Iterative analysis trail (rounds until closed)

| Round | Focus | Finding |
|-------|--------|---------|
| R1 | GOAL + skill + persona | Tutor = heuristic+shell; meters clear/breach/streak/no-retrain |
| R2 | Claim JSON all 10 pairs | 10/10 pass; clear 44–84%; breach 0; 90 days |
| R3 | Forward JSON | 5/10 pass absolute 30/40; breach still 0 all pairs |
| R4 | equity_day + dials | heat, every-bar, one signal, bank; dials frozen |
| R5 | train_multi_pair dial search | **all-day** search leak confirmed in source |
| R6 | IRAC + PRINCIPLES | R1–R6 REJECT; KEEP shell A–F |
| R7 | Data loader | exactly 90 eligible days; full CSV exists unused |
| R8 | Champion prove report | PROVEN ~24% @ 3.0/3.5; separate track |
| R9 | Gaps G01–G25 | map completeness; random-input + 100-day definition **new** issues ISS-02–06 |
| R10 | Near-floor + streaks | thin margin days; streak not in score JSON |
| R11 | Tutor walk | max-decisions skew G22 |
| R12 | Catalog close | ISS-01…26 cover plan completeness list |

---

## Multiple-choice summary sheet (print / fill)

| ISS | Topic | Your pick |
|-----|--------|-----------|
| 01 | Win meters | A / B / C |
| 02 | What “100 days” means | A / B / C |
| 03 | Data / window | A / B / C / D |
| 04 | Clear floor | A / B / C / D |
| 05 | Random protocol | A / B / C / D |
| 06 | Support set | A / B / C / D |
| 07 | No retrain pin | A / B / C |
| 08 | Claim vs forward | A / B / C |
| 09 | Streak | A / B / C |
| 10 | Shell lock | A / B / C |
| 11 | Ban list | A / B / C |
| 12 | Decode | A / B / C |
| 13 | Meaning version | A / B / C |
| 14 | Leak hygiene | A / B / C |
| 15 | Fit vs score split | A / B / C |
| 16 | Hard-pair strata | A / B / C |
| 17 | Near-floor | A / B / C |
| 18 | Symbol/season | A / B / C |
| 19 | Score artifact | A / B / C |
| 20 | Tutor verdict | A / B / C |
| 21 | Track split | A / B / C |
| 22 | Channel1 isolation | A / B / C |
| 23 | Meta defer | A / B / C |
| 24 | Seed / double-run | A / B / C |
| 25 | Bank honesty | A / B |
| 26 | Tutor skill law | A / B / C |

---

## Recommended default pack (if Monty accepts “Recommended” letters)

| ISS | Pick | One line |
|-----|------|----------|
| 01 | **A** | clear% / breach% / streak only |
| 02 | **A** | 100 = window length, not 100/100 clears |
| 03 | **A** | extend real XAUUSD to ≥100 days |
| 04 | **A** | breach 0 + clear ≥50% (climb later) |
| 05 | **A** or **C** | per-day random draw (+ keep ten-pair regression if C) |
| 06 | **A** | boxed discrete support + seed |
| 07 | **A** | pin ckpt+dials+decode |
| 08 | **A** | dual meters rates |
| 09 | **A** | report streak |
| 10 | **A** | SHELL_LOCKED |
| 11 | **A** | ban R1 package |
| 12 | **A** | heuristic decode for tutor bar |
| 13 | **A** | meaning hash gate |
| 14 | **A** | practice-only dial search |
| 15 | **A** | fit ≠ score window |
| 16 | **A** | stratified hard targets |
| 17 | **A** | near_breach diagnostic |
| 18 | **A** | XAUUSD phase-1 |
| 19 | **A** | one conclusion JSON |
| 20 | **A** | full run verdict |
| 21 | **A** | lineage vs prove_it split |
| 22 | **A** | Channel1 isolated |
| 23 | **A** | meta deferred v1 |
| 24 | **A** | seed + double-run |
| 25 | **A** | bank honesty |
| 26 | **A** | persona laws |

---

## Chosen rules → 100-day conclusion protocol

> Fill letters (or accept Recommended pack). Then the protocol below is binding.

### Chosen letters

```text
ISS-01: __   ISS-02: __   ISS-03: __   ISS-04: __
ISS-05: __   ISS-06: __   ISS-07: __   ISS-08: __
ISS-09: __   ISS-10: __   ISS-11: __   ISS-12: __
ISS-13: __   ISS-14: __   ISS-15: __   ISS-16: __
ISS-17: __   ISS-18: __   ISS-19: __   ISS-20: __
ISS-21: __   ISS-22: __   ISS-23: __   ISS-24: __
ISS-25: __   ISS-26: __
```

**Recommended pack applied:** all **A** (ISS-05 optional **C** if ten-pair regression also required).

### Protocol text (with Recommended pack filled in)

| Element | Definition |
|---------|------------|
| **Brain** | `multi_pair_consistent_v1.pt` (or successor) + dials file + **decode=heuristic**, hashes pinned (ISS-07 A, ISS-12 A) |
| **Meaning** | Frozen meaning_version hash (ISS-13 A) |
| **Shell** | Locked: heat, floor-scale, every-bar marks, one signal, bank, breach death (ISS-10 A); ban R1 package (ISS-11 A) |
| **Data** | Real M1 XAUUSD, min_bars=900, **≥100** chronological eligible days (ISS-03 A); phase-1 symbol (ISS-18 A) |
| **100 days** | One contiguous window of 100 eligible days (ISS-02 A) — **not** “100 clears required” unless ISS-02 B chosen |
| **Random inputs** | Each day *d*: sample `(target_d, risk_d)` from support box [1.0–3.5]×[1.5–4.0] discrete 0.5 with risk not absurdly below target; seed S (ISS-05 A, ISS-06 A) |
| **Fit hygiene** | Dials searched only on practice **before** freeze; score window not in search set (ISS-14 A, ISS-15 A) |
| **Day score** | `cleared` ⇔ equity% ≥ target% and never ≤ −risk%; `breached` ⇔ touched floor (ISS-01 A, ISS-25 A) |
| **Window pass** | **breach% = 0** on all 100 days **and** **clear% ≥ 50%** (ISS-04 A); report streak (ISS-09 A); optional hard-target strata (ISS-16 A) |
| **No retrain** | Zero train steps between day 1 and day 100 of the score (ISS-07 A) |
| **Repro** | Run twice with seed S; metrics match (ISS-24 A) |
| **Artifact** | Single conclusion JSON + human table (ISS-19 A) |
| **Tutor** | Speaks this stack only; full-run verdicts; no R1 (ISS-20 A, ISS-26 A) |
| **Champion** | Promotion still via `prove_it` (ISS-21 A); Channel1 isolated (ISS-22 A); meta not required for v1 (ISS-23 A) |

### Pass / fail sentence (Recommended pack)

> **PASS** if and only if: the same pinned multi-pair-tutor brain (heuristic + locked shell + dials) is scored on **100 real eligible XAUUSD days**, each day with a **seeded random (target%, risk%)** from the support box, with **breach% = 0**, **clear% ≥ 50%** (or stricter if chosen), streak reported, no retrain mid-window, meaning/dials hashes match freeze, double-run matches, and R1 family absent.  
> **FAIL** if any breach, clear below floor, hash drift, mid-window retrain, or non-reproducible metrics.  
> **NOT YET MEASURABLE** if fewer than 100 eligible real days (current curriculum = **90**) until data extended or interim 90-day flag accepted (ISS-03 B).

### Current evidence vs PASS

| Check | Status today |
|-------|----------------|
| Breach 0 on claim 90d × 10 pairs | **YES** |
| Clear ≥50% on all 10 frozen pairs claim | **YES** (lowest 44.4% on pair10 — **fails 50% on hardest frozen pair**) |
| Clear ≥50% pair10 claim | **NO** (44.4%) |
| Forward hard pairs ≥50% clear | pair10 **30%** — **NO** |
| 100 eligible days | **NO** (90) |
| Random per-day inputs scored | **NOT BUILT** |
| Practice-only dial search | **NO** (all-day search in code) |
| Meaning hash gate | **NO** |
| Streak in multi-pair JSON | **NO** |

**Honest bottom line:** multi-pair tutor **already proves** runtime multi-pair + **breach 0** on real data for frozen pairs. It does **not** yet meet a **100-day random-input** conclusion under the Recommended pack — mainly due to **data length (90&lt;100)**, **missing random-day protocol**, **hard-target clear shortfall**, and **unseen/leak hygiene**. Those are the build targets; not a jar lid.

---

## What /multi-pair-tutor needs next (build order if rules accepted)

Do not implement in this document’s goal scope; this is the fulfilment order implied by open issues:

| Step | Closes | Build |
|------|--------|-------|
| 1 | ISS-03 | ≥100 real eligible days loader path |
| 2 | ISS-14,15 | Practice-only dial search + leak test |
| 3 | ISS-13 | meaning_manifest + score hash |
| 4 | ISS-05,06,24 | Random-day scorer + seed + double-run |
| 5 | ISS-01,04,09,19 | Conclusion JSON: clear/breach/streak + KEEP/REJECT |
| 6 | ISS-10,11 | SHELL_LOCKED + ban-list test |
| 7 | ISS-16,17 | Stratified + near_breach diagnostics |
| 8 | ISS-20,26 | Tutor walk full verdict default |
| 9 | ISS-23+ | Meta only after regime/sensor reports |

---

## Non-goals (this catalog)

- Claiming 100/100 clears already  
- Overwriting `models/PROVEN_*.pt`  
- Redefining GOAL.md mission  
- Implementing the scorer in this file  

---

## Source index (every issue cites from here)

| Path | Used for |
|------|----------|
| `GOAL.md` | Mission, clear/breach/streak/no-retrain |
| `lineages/.../equity_day.py` | Shell physics, split, load days |
| `lineages/.../score_ten_pairs.py` | Score modes, pass_30_clear |
| `lineages/.../train_multi_pair.py` | All-day dial search |
| `lineages/.../ten_pairs.json` | Pairs, 50/40, seed, data_source |
| `lineages/.../checkpoints/ten_pair_score_all.json` | Claim numbers |
| `lineages/.../checkpoints/ten_pair_score_forward.json` | Forward 5/10 |
| `lineages/.../checkpoints/multi_pair_dials.json` | Dials + decode |
| `lineages/.../PRINCIPLES_OF_SUCCESS.md` | P1–P10, R1–R6 |
| `lineages/.../UNSEEN_CONSISTENCY_RECIPE.md` | G01–G25 |
| `lineages/.../STATUS.md` | Claim table, Channel1 freeze |
| `lineages/.../agents/MULTI_PAIR_TUTOR_PERSONA.md` | Tutor identity |
| `lineages/.../tutor_day_walk.py` | Partial walk G22 |
| `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md` | IRAC KEEP/REJECT |
| `references/plans/GOAL_FROM_TEN_PAIR_IRAC.md` | Two tracks |
| `outputs/reports/GOAL_PROVE_MULTI_PAIR_2026-07-31.md` | PROVEN multi-pair prove_it |
| `models/00_CHAMPION.md` | Champion yardstick |
| `.grok/skills/multi-pair-tutor/SKILL.md` | Skill load rules |
| `data/raw/XAUUSD_curriculum_2026.csv` | Real price data |

---

*End of catalog. Choosing the MC letters (or the Recommended pack) yields a complete, evidence-backed conclusion protocol for 100 consistent days of random inputs under GOAL.md on real price data.*
