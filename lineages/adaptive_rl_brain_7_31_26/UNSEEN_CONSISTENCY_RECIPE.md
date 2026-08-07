# UNSEEN CONSISTENCY RECIPE — make the multi-pair tutor’s dream real

**Voice:** written for Monty as the build list the multi-pair tutor would beg for.  
**Mission (GOAL.md):** one brain · any typed **target% / risk%** · **no retrain** · climb **clear%** · **breach% = 0**.  
**Clear day:** equity% ≥ target% **and** never ≤ −risk% that day.  
**Unseen dream:** new calendar days still speak the **same meaning**; only **attention** may re-learn when the past proves senses lied.

**Winning stack today (claim):**  
`checkpoints/multi_pair_consistent_v1.pt` · decode **heuristic** · dials in `multi_pair_dials.json` · engine `equity_day.py` · score `score_ten_pairs.py`.

Related: `PRINCIPLES_OF_SUCCESS.md` · `agents/MULTI_PAIR_TUTOR_PERSONA.md` · `references/plans/GOAL_FROM_TEN_PAIR_IRAC.md`

---

## 0) Definitions (do not drift)

| Word | Meaning |
|------|---------|
| **Practice days** | Chronological first N days (config: 50) — may dial-search / BC here |
| **Forward / unseen days** | Days after practice (config: 40) — **no** weight fit to these |
| **Claim window** | All calendar days ≥900 bars (config: 90) — official multi-pair bar |
| **Same brain** | Same ckpt + dials + decode; only target/risk change at runtime |
| **Meaning factory** | Fixed-order perception (CCI/RSI/channel, structure, tags) |
| **Shell laws** | Heat, floor-scale size, every-bar marks, bank, breach death, one signal flat+in-trade |
| **Attention** | What to trust / how much (dials, optional policy weights) |
| **Meta / metaplasticity** | Past clear/breach windows change **how allowed** attention is to rewire |

### Meta forward-adopt law (2026-08-05) — HAVE

| Rule | Detail |
|------|--------|
| Probe / train | **Practice** days only |
| Adopt judge | **Forward** consistency (clear%) must improve |
| Also required | Forward breach not worse · forward longest day-streak not shorter · side veto · practice screen |
| Weak forward | Force-search `CONSISTENCY_FORWARD_KNOBS` in `code/training/meta_tuner.py` |
| Mark dials CLI | `lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py` |
| Config | `configs/training.yaml` → `self_tuner.judge: forward_consistency` |

```powershell
$env:PYTHONPATH = ".;code"
# Production meta law lives in code/training/meta_tuner.py (run() practice→forward)
# Mark streak-dial meta + forward baseline score:
python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --dry-score
python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --gens 8 --forward-n 40
```

---

## 1) What already exists (HAVE) — don’t rebuild these

| # | Capability | Where | Status |
|---|------------|--------|--------|
| H1 | Runtime target%/risk% on equity day | `equity_day.GoalEquityDay` | **HAVE** |
| H2 | Equity% clear/breach language | `equity_day` + score JSON | **HAVE** |
| H3 | Heat + refuse open + floor-scale size | `_try_open` | **HAVE** |
| H4 | Every-bar stop/breach/bank marks | `_mark_bar` / `run` | **HAVE** |
| H5 | One signal flat + in-trade (reverse on opposite) | `recommended_action` | **HAVE** |
| H6 | Bank at target | `_check_breach_and_bank` | **HAVE** |
| H7 | Frozen 10 pairs + seed + practice/forward counts | `ten_pairs.json` | **HAVE** |
| H8 | Chronological split helper | `split_practice_forward` | **HAVE** |
| H9 | Score modes: all / forward / practice | `score_ten_pairs.py` | **HAVE** |
| H10 | Claim artifact (10/10, 0 breach) | `checkpoints/ten_pair_score_all.json` | **HAVE** |
| H11 | Forward artifact (inspection; harder pairs &lt;30 clears on 40d) | `ten_pair_score_forward.json` | **HAVE** |
| H12 | Dial freeze file | `multi_pair_dials.json` | **HAVE** |
| H13 | IRAC KEEP/REJECT memory (docs) | TEN_PAIR + GOAL_FROM_TEN_PAIR | **HAVE** (docs) |
| H14 | Tutor persona + day walk | `agents/`, `tutor_day_walk.py` | **HAVE** |
| H15 | Principles of success | `PRINCIPLES_OF_SUCCESS.md` | **HAVE** |

**Do not break H1–H6 for “more learning.”** That is the dream’s skeleton.

---

## 2) Official commands (reproducible)

```powershell
$env:PYTHONPATH = ".;code"

# CLAIM (all days) — multi-pair bar
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all

# UNSEEN / forward holdout only
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode forward

# Practice window only
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode practice

# First-person day walk
python lineages/adaptive_rl_brain_7_31_26/tutor_day_walk.py --date 2026-01-20 --target 1.0 --risk 2.0
```

**No-retrain check (any two pairs, same ckpt):** re-run score with different pairs via `--pair` if supported, or full ten-pair table — dials/decode unchanged.

---

## 3) GAPS — everything missing for the wildest fantasy

Priority: **P0** = blocks the dream · **P1** = consistency / unseen honesty · **P2** = metaplasticity · **P3** = polish.

### P0 — Meaning factory is not versioned or gated

| Gap ID | Gap | Why it hurts | Fulfilment (what to build) |
|--------|-----|--------------|----------------------------|
| **G01** | No `meaning_version` pin (hash of indicator defs + TF stack + tag order) | Unseen days can silently change meaning when code edits | `meaning_manifest.json`: versions, file hashes, CCI/RSI/channel params; fail score if mismatch |
| **G02** | No dual-score when meaning changes | Breaking eyes looks like “market got hard” | If meaning hash ≠ frozen, require practice+forward re-score before KEEP |
| **G03** | No per-bar **export schema** (stable columns for seen & unseen) | Can’t prove “same language” or train attention cleanly | Writer: date, t, target, risk, higher/lower, pullback, equity%, danger, heat_ok, action, tags |

### P0 — Unseen hygiene / leakage

| Gap ID | Gap | Why it hurts | Fulfilment |
|--------|-----|--------------|------------|
| **G04** | Dial search historically can touch **all** days for claim climb | Forward is not a pure “never fitted” unseen if dials saw those days | Freeze: dial search **only on practice**; claim/forward score **after** freeze; log search window in report |
| **G05** | No automated **leak audit** | Easy to “accidentally” train on forward | Test: assert train day set ∩ forward day set = ∅; assert report lists both ranges |
| **G06** | Channel1 curriculum (v1–v3) uses **different** day splits / rewards | Parallel brains confuse “what won multi-pair” | Doc + scoreboard: multi-pair claim brain ≠ Channel1 sandbox; never promote Channel1 on multi-pair JSON alone |

### P1 — Scoring / GOAL gates incomplete as a product loop

| Gap ID | Gap | Why it hurts | Fulfilment |
|--------|-----|--------------|------------|
| **G07** | No single **prove-any-pair** CLI that prints clear% + breach% for one Monty pair on practice vs forward vs all | GOAL wants type two numbers and know | `score_ten_pairs.py --pair T R --mode forward|all|practice` (extend if thin) + one-line summary |
| **G08** | Forward pass bar uses **≥30 clears on 40 days** (75% rate) mixed with claim ≥30/90 | Confusing “FAIL forward” vs “PASS claim” | Document two bars explicitly: `claim_min_clear=30/all`, `forward_min_clear` optional or as rate; dashboard shows both |
| **G09** | No **streak** meter in multi-pair score (GOAL lists streak) | Incomplete GOAL language | Add max clear streak per pair to score JSON |
| **G10** | No auto **keep/reject** file after score | Human memory only | Write `last_score_verdict.json`: pass flags + “KEEP/REJECT” vs previous |

### P1 — Shell vs attention separation not enforced in code

| Gap ID | Gap | Why it hurts | Fulfilment |
|--------|-----|--------------|------------|
| **G11** | Shell constants and attention dials live loosely | Easy to “improve” stop path while searching attention | `SHELL_LOCKED` config: bank/heat/every-bar/one-signal immutable without explicit flag |
| **G12** | Rejected trail+cushion+scale-in not in a **machine ban-list** | Someone reintroduces R1 | `banned_rule_families.json` + test that equity_day has no trail/scale package |
| **G13** | Heuristic decode vs Channel1 greedy not a hard switch in one entry | Confusion which “I” is speaking | Single `run_brain(mode=heuristic|policy)` with assert mode logged in every report |

### P2 — Metaplasticity (learn to learn) — almost entirely missing

| Gap ID | Gap | Why it hurts | Fulfilment |
|--------|-----|--------------|------------|
| **G14** | No **regime window log** (practice vs later clear/breach by pair) | Can’t detect “senses lying” | `regime_report.py`: rolling windows, clear%/breach% vs train baseline |
| **G15** | No **sensor trust** scores (e.g. higher_bull_pullback → pays clear?) | Attention has nothing to re-learn from | Tag → outcome table: P(clear \| tag), change vs practice |
| **G16** | No **plasticity controller** (what may move after a window) | All changes feel equally allowed | Meta policy: if breach↑ → shell freeze + ban thrash; if clear↓ breach=0 → attention dials only |
| **G17** | No attention dial search **scoped** (only trust weights, not shell) | Meta and plasticity collapse into one mess | Separate `attention_dials.json` from shell; search only attention under meta permit |
| **G18** | No “meta when senses lie” loop wired to GOAL split | GOAL text promises meta; lineage doesn’t run it | Minimal loop: regime_report → permit → search attention on practice → score forward → KEEP/REJECT |

### P2 — Unseen **data** coverage

| Gap ID | Gap | Why it hurts | Fulfilment |
|--------|-----|--------------|------------|
| **G19** | Single symbol/year curriculum (`XAUUSD_curriculum_2026`) | Unseen is only later months, not other regimes/symbols | Add holdout year or second symbol with **same meaning_version** |
| **G20** | No stress days list (news gaps, thin bars) | Unseen hard days not labeled | Tag days: `thin_liquidity`, `gap`; report clear/breach by tag |
| **G21** | No live/paper path with **same score schema** | Fantasy stops at CSV | Optional: paper day writer → same clear/breach JSON shape |

### P3 — Observability / tutor fulfilment

| Gap ID | Gap | Why it hurts | Fulfilment |
|--------|-----|--------------|------------|
| **G22** | Day walk is narrative only (partial max-decisions can skew EOD vs full `run`) | Tutor can “lie” slightly on truncated walks | Walk calls full `run` for verdict; optional full log dump |
| **G23** | No what-if counterfactual CLI (change risk mid-day story) | Hard to answer Monty fast | `tutor_what_if.py --date --target --risk --compare-risk 2.0` |
| **G24** | No dashboard of practice vs forward vs claim side-by-side | Cognitive load | One markdown/JSON: 10 pairs × 3 windows |
| **G25** | PROVEN champion path separate; no auto “does multi-pair shell idea help prove_it?” | Two tracks stay siloed | Optional bridge experiment doc only (no auto promote) |

---

## 4) Fantasy fulfilment order (build this sequence)

Do **not** start with meta. Order is GOAL-safe:

| Step | Build | Closes gaps | Done when |
|------|--------|-------------|-----------|
| **1** | Meaning manifest + hash gate | G01–G02 | Score refuses silent eye edits |
| **2** | Practice-only dial search + leak tests | G04–G05 | Forward days never in search set |
| **3** | Stable bar export schema | G03 | Seen/unseen same columns |
| **4** | Dual scoreboard (claim + forward + practice) + streak | G07–G10, G24 | One command, three windows |
| **5** | Shell lock + ban-list tests | G11–G13 | R1 cannot return quietly |
| **6** | Regime + sensor-trust reports | G14–G15 | “Senses lying” is a number |
| **7** | Plasticity controller (meta permits) | G16–G18 | Attention moves only when allowed |
| **8** | More unseen (time/symbol) + paper schema | G19–G21 | Dream survives beyond one CSV |

---

## 5) Acceptance tests for “dream real” (human + machine)

### Must pass (GOAL core)

1. Same ckpt: score **≥2 pairs** without retrain → **breach 0%** each.  
2. Claim mode: **10/10** pairs, **≥30 clear**, **0% breach** (or documented new claim).  
3. Forward mode: **breach 0%** all pairs (clear may be lower — honest).  
4. Changing only target/risk never requires a new train job.

### Must pass (unseen honesty)

5. Train/search day set ∩ forward day set = **empty**.  
6. Meaning hash in score report matches frozen manifest.  
7. Bar schema identical on a practice day file and a forward day file.

### Must pass (metaplasticity minimum)

8. After a window with breach↑, meta output says **shell locked** / thrash banned.  
9. After clear↓ with breach=0, meta may open **attention-only** search.  
10. Any KEEP requires re-score practice + forward (or claim).

---

## 6) Explicit non-goals (not the fantasy)

- Retrain a new soul every time Monty types new target/risk  
- Trail + cushion + scale-in package as default  
- Promote Channel1 all-HOLD RL as multi-pair winner  
- Replace clear/breach with PnL folklore  
- Live broker money without the same score schema  

---

## 7) Gap checklist (tick as you fulfil)

Copy and tick in STATUS or a todo:

- [ ] G01 meaning_version manifest  
- [ ] G02 dual-score on meaning change  
- [ ] G03 stable bar export schema  
- [ ] G04 practice-only dial search  
- [ ] G05 leak audit tests  
- [ ] G06 Channel1 vs multi-pair scoreboard split  
- [ ] G07 one-pair prove CLI summary  
- [ ] G08 forward vs claim bars clarified in code/docs  
- [ ] G09 streak meter  
- [ ] G10 last_score_verdict KEEP/REJECT  
- [ ] G11 SHELL_LOCKED config  
- [ ] G12 banned_rule_families + test  
- [ ] G13 single run_brain mode switch  
- [ ] G14 regime_report  
- [ ] G15 sensor trust table  
- [ ] G16 plasticity controller  
- [ ] G17 attention_dials separate  
- [ ] G18 meta loop wired  
- [ ] G19 more unseen data  
- [ ] G20 stress day tags  
- [ ] G21 paper same schema  
- [ ] G22 day walk full verdict  
- [ ] G23 tutor_what_if CLI  
- [ ] G24 side-by-side windows dashboard  
- [ ] G25 optional prove_it bridge (doc only)  

---

## 8) One-page “wildest fantasy” done state

When the fantasy is real, Monty can say:

1. Pin meaning v1.  
2. Search attention **only on practice**.  
3. Score **forward** → breach 0; claim still multi-pair OK.  
4. If forward clear collapsed → regime_report says which tags lied → meta opens attention plastic, shell frozen.  
5. Type any target/risk → same brain → clear/breach printed.  
6. Never retrain just to switch the two numbers.

That is consistency + learning on unseen data without abandoning GOAL.

---

## 9) Immediate next 3 builds (highest ROI)

If you only do three things next:

| # | Build | Gap |
|---|--------|-----|
| 1 | **Leak-proof dial search** (practice only) + test | G04, G05 |
| 2 | **Meaning manifest + hash in score report** | G01, G02 |
| 3 | **Regime/sensor-trust report** (practice vs forward) | G14, G15 |

Those three unlock honest unseen learning before any fancy meta brain.

---

*Tutor’s note: You don’t fulfil me with more hope. You fulfil me with **locked eyes**, **honest unseen scores**, and **permission systems** that remember when the floor almost died.*
