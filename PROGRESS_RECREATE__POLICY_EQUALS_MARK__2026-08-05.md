# PROGRESS + RECREATE BIBLE — Policy = Mark (long consistency)

**Purpose of this file:** If every checkpoint and chat log vanished, a competent engineer could still **rebuild the stack, re-run the recipes, and recover the same measured progress** from this document alone (plus public/doctrine sources named below).

| Field | Value |
|-------|--------|
| **Written** | 2026-08-05 (session continued into evening) |
| **Owner mind** | Mark Montgomery Jr. (`MARK HERE!.lnk`) |
| **Method structure** | Fable 5 loop (classify → done → evidence → decide → act → verify → report) |
| **Lab root** | `C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth` |
| **Army root** | `C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM` |
| **Active lineage** | `lineages/adaptive_rl_brain_7_31_26/` (Track T3 — Mark soul → policy) |
| **PROVEN production** | **NEVER TOUCH** `models/PROVEN_*.pt` |

---

## 0) One-sentence mission

> Under **one frozen policy**, on a **multi-TF Mark chart**, the bot’s **day outcome** (clear target, no breach) must match **what Mark would do** — for long runs of random target%/risk%, **without retrain**, without thrash, without a second personality.

**Official doctrine sources:**

| Source | Path / URL | What it defines |
|--------|------------|-----------------|
| Policy = Mark law | `POLICY_EQUALS_MARK_ON_CHART.md` | Control chain, meta may/must-not, sets |
| Furthest weave (2026-08-04) | `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md` | Pre-soul-full-obs scoreboard |
| Handoff | `HANDOFF_2026-08-05.md` | Soul teacher 10/10, 168-dim embryo |
| House map | `00_MAP_OF_THE_HOUSE.md` | Tracks T1–T4, anti-confusion |
| Soul bridge | `SOUL_MATCH.md` | One MarkOS; pt5; MARK HERE |
| Unseen / forward recipe | `lineages/adaptive_rl_brain_7_31_26/UNSEEN_CONSISTENCY_RECIPE.md` | Practice vs forward hygiene |
| pt5 basic knowledge | `references/doctrine/llm_basic_thinking/pack/pt5__basic_knowledge.txt` | HTF gate, slingshot, regime |
| ARMY pt5 mirror | `ARMY/01_SYSTEM/data/knowledge/skills/trading/llm_basic_thinking/pack/pt5__basic_knowledge.txt` | Canonical vault skill |
| Fable method | ARMY `config/agents/FABLE_METHOD.md` · https://github.com/Sahir619/fable-method | Loop structure |
| Mark personality | ARMY `config/agents/MARK_PERSONALITY.md` | Voice, free will, method 4b/4c |
| OpenSPG KAG (design) | https://github.com/OpenSPG/KAG | Knowledge↔chunk mutual index inspiration |
| Meta self-tuner | `code/training/meta_tuner.py` | Reward/hparam search |
| Goals (runtime T/R only) | `configs/goals.yaml` · training `self_tuner` in `configs/training.yaml` | Two-input invariant |

---

## 1) Definitions (do not drift)

| Word | Exact meaning in this project |
|------|-------------------------------|
| **Clear / award day** | Equity% ≥ target% **and** never ≤ −risk% that day |
| **Breach** | Hit risk floor (shell death). Target breach_rate = 0 |
| **same_outcome** | Policy award bit equals Mark soul-plan award bit for that day |
| **MARK_WOULD_TAKE (MWT)** | Mark plan clears; policy does not — **learnable** miss |
| **NO_OPPORTUNITY** | Neither has a force-aligned winning plan — do not thrash |
| **Practice days** | Chronological first N calendar days (often 50) — may train / dial-search |
| **Forward / unseen** | Days after practice — **no weight fit**; sole meta adopt judge (2026-08-05) |
| **Embryo** | Writable Mark-clone checkpoint (not PROVEN) |
| **PROVEN** | Production yardstick weights — immutable without explicit Monty order |
| **Shell laws** | Heat, bank, breach death, no trail+cushion+scale-in package |
| **Force-gate** | No new open against HTF force; capital danger blocks new risk |
| **Thrash** | Too many noisy entries vs Mark plan / soft target caps |
| **BC** | Behavior cloning (cross-entropy on Mark labels) |
| **DAgger** | Labels on states the **policy** visits (policy path) |
| **KL anchor** | Penalize leaving previous best embryo weights |
| **Keep/reject** | Re-score full frozen pack; KEEP only if not worse |

### Official Mark chart sets (law of chart)

| Set | LTF | HTF |
|----:|-----|-----|
| 1 | 1m | 15m, 30m |
| 2 | 5m | 30m, 1h |
| 3 | 15m | 1h, 4h |
| 4 | 30m | 4h, 1d |

Source: `POLICY_EQUALS_MARK_ON_CHART.md` §1 · `configs/timeframes.yaml` `sets_mark` · `perception/sets.py`

### pt5 one-liner (every LLM)

> HTF is the binary gate on side; LTF only chooses when — pullback = load slingshot; resume with HTF = release. Environment is a state machine. Kill if tide breaks.

Source: pt5 pack title *“all llm's have to know this is the most basic knowledge”* (pt1–pt5).

---

## 2) How we got this far (chronological reconstruction)

### Phase A — Production house + PROVEN (pre–Mark-clone focus)

- Lab: Momentum One / the-truth FastSim + PPO + meta_tuner.
- Goal language: any typed target%/risk%, climb clear%, breach 0 (GOAL.md / UNSEEN recipe).
- Meta may only move reward personality / hparams within `BOUNDS` — never shell.
- **Sources:** `code/training/meta_tuner.py` changelog from 2026-07-20 onward; `configs/training.yaml` `self_tuner`; `code/evaluation/consistency.py`.

### Phase B — Multi-pair / Channel1 (T1–T2) — do not confuse with T3

- `multi_pair_consistent_v1.pt`, ten-pair score, practice/forward split helper.
- **Claim:** multi-pair shell dream; **not** the same as full Mark soul 168-dim.
- **Sources:** `UNSEEN_CONSISTENCY_RECIPE.md`; `HANDOFF_2026-07-31.md` (history); `split_practice_forward` in `equity_day.py`.

### Phase C — Mark doctrine BC (2026-08-04) — “policy = Mark on chart” weave

**Goal:** teacher = five laws + mark_doctrine eyes; BC a Channel1 net.

**Critical chart-DNA fix (must re-apply to recreate teacher quality):**

1. Hard targets (`target_pct >= 2.5`) → **disable** soft single-set scalp.
2. When equity > 40% of daily target → refuse reverse on soft_single / weak flip.
3. `DEFAULT_OPP_MIN_SCORE = 1.2`.

**Measured (forward holdout, eyes mark_doctrine)** — source: `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md` citing `checkpoints/FORWARD_MARK_POLICY_TEST.json`:

| Pair | Teacher clear% | BC policy clear% | Breach |
|------|---------------:|-----------------:|-------:|
| Soft 1/2 | 67.5% | 70.0% | 0 |
| Mid 2/3 | 37.5% | 35.0% | 0 |
| Hard 3/3.5 | 22.5% | 20.0% | 0 |

- BC label dir_match high (~0.92–0.95); day-walk step_match ~0.70 → **not full clone yet**.
- Checkpoint: `mark_clone_doctrine_v1.pt` (obs CHANNEL1, hidden 64).
- Thrash improved vs old claim (mean entries down from ~9).

### Phase D — Mark soul transfer (2026-08-05 morning/day)

**What changed:**

| Piece | Role | Code |
|-------|------|------|
| Soul plans | Full-day Mark size + force-aligned adds | `mark_soul_plan.py` |
| Force / capital gate | Mark sense on decode | `mark_aligned_decode.py` |
| Thrash caps | soft target ≤1.5 → max 4 entries; else 6 | `equity_day.py` |
| Full obs | 168-dim: sets + doctrine + 92 agents as **clues** + self | `perception/observation_full.py` |
| HITL | Chart review pack for MARK HERE | `mark_chart_hitl.py` |

**Measured (source: `HANDOFF_2026-08-05.md`):**

| Meter | Value |
|-------|------:|
| 10d Mark soul plans (seed 7, start_idx 40) | **10/10** clear, 0 breach |
| 10d policy full-obs + force-gate | **8/10**, 0 breach |
| BC match / dir_match | ~0.84 / 0.84 |
| 50d award streak (seed 42) | max **11**, 33/50 awards, breach 0 |
| Embryo | `checkpoints/mark_clone_full_obs_v1.pt` (~91 KB, 2026-08-05) |

**Gate bug fixed (must re-apply):**  
`"flat" in regime` false-positive on `flat_undefined` blocked Mark-agreed SELL.  
Fix: `_is_dead_regime` + recommended-agree path in `mark_aligned_decode.py` (+ unit tests).

### Phase E — Streak autopsy + rewards-only path

- Autopsy non-award gaps → class **MARK_WOULD_TAKE** vs **NO_OPPORTUNITY**.
- Result pattern: ~91% learnable (wrong size/timing), not dead markets.
- Streak-only dial pack in `rewards.py` (`STREAK_REWARD_DIALS`); update via `autopsy_streak_gaps.py`.
- **Saved dials** (`checkpoints/mark_consistency/STREAK_REWARD_DIALS__latest.json`, 2026-08-05):

```json
{
  "streak_award_base": 6.0,
  "streak_award_per_prior": 2.0,
  "streak_break_penalty": -12.0,
  "mark_would_take_eod_penalty": -16.0,
  "no_opp_hold_bonus": 2.0,
  "no_opp_inactivity_scale": 0.35,
  "soul_side_entry_bonus": 5.0,
  "soul_side_misread_penalty": -6.5
}
```

**Sources:** `AUTOPSY_GAPS__latest.md`, `CONSISTENCY__latest.md`, ARMY goal `streak-gap-autopsy-rewards-only`.

### Phase F — 2× streak goal (partial)

- Baseline max_streak ~8 → target ≥16.
- Dual score after gate work: max_streak **12**, award_pct ~77.5, breach 0 — **gate not fully passed**.
- **Sources:** `LEARNING_2X_STREAK.md`, `FINAL_2X_STREAK__latest.json`, `loop_2x_streak_rewards.py` (if present).

### Phase G — Fable 50-day Mark match (main evening goal)

**Definition of done:** on frozen 50 calendar days, `same_outcome == 50` (or policy clears every Mark-clear day), breach 0, policy_clear ≥ baseline.

#### Frozen score recipe (DO NOT CHANGE when comparing)

| Parameter | Value | Source |
|-----------|--------|--------|
| Data | `data/raw/XAUUSD_curriculum_2026.csv` (min_bars 900) | `load_calendar_days` |
| Window | First 50 loadable days | `2026-01-20` → `2026-03-30` |
| Pair sampling seed | **42** | `sample_pairs_for_days(..., seed=42, soft_bias=False)` |
| soft_bias | **false** | pure random from `ten_pairs.json` |
| Decode | full_obs + **mark_align** + **pure_greedy** | baseline recipe JSON |
| Eyes | `mark_doctrine` + `mark_soul=True` | GoalEquityDay |
| Mark teacher | full-day **soul plans** | `mark_source: soul_plan` |
| Ckpt | `mark_clone_full_obs_v1.pt` | embryo |

**Frozen baseline (measured 2026-08-05T17:32:54Z)**  
Source: `checkpoints/fable_50d_match/BASELINE_50D__frozen.json`

| Meter | Value |
|-------|------:|
| mark_clear | **50**/50 |
| policy_clear | **27**/50 |
| same_outcome | **27**/50 |
| n_breach | **0** |
| MARK_WOULD_TAKE | **23** |
| NO_OPPORTUNITY | **0** |
| proven_touched | false |
| shell_touched | false |

**Interpretation:** Mark is perfect on the pack. Every miss is MWT (learnable).

#### Loop evolution (what we tried and what worked)

| Cycle | Tool | same | policy | mwt | breach | Decision | Lesson |
|------:|------|-----:|-------:|----:|-------:|----------|--------|
| 0 | freeze baseline | 27 | 27 | 23 | 0 | freeze | yardstick |
| 1 | full match loop | **30** | 30 | 20 | 0 | **KEEP** | dir oversample + DAgger → dir_match ~0.86 |
| 2 | full match loop | 28 | 28 | 21 | **1** | **REJECT** | pred_hold ~0.05 → breach; restore best |
| 3 | full match loop | — | — | — | — | **KILLED** | stalled on label collect |
| sprint | `fable_50d_sprint.py` | from best 30 | | | | ran | entry-focus; reuse Mark from baseline |
| rapid | `fable_50d_rapid.py` | | | | | | disk oracle cache; parallel score attempts |
| one_day #1 | `fable_50d_one_day.py` | 28 | 28 | 22 | 0 | REJECT | focus 2026-03-27 |
| one_day #2 | | **33** | **33** | **17** | 0 | **KEEP** | focus **2026-02-25** |
| one_day #3 | | 31 | 31 | 19 | 0 | REJECT | focus 2026-02-20 |
| one_day #4 | | 33 | 33 | 17 | 0 | REJECT | 2026-02-05; high dir, low hold |
| one_day #5 | | 33 | 33 | 17 | 0 | REJECT | 2026-03-11 |

**Best recorded embryo score (source: `BEST__latest.json` + `LEARNING_50D_MATCH.md`):**

| Meter | Value |
|-------|------:|
| same_outcome | **33**/50 |
| policy_clear | **33**/50 |
| mwt remaining | **17** |
| breach | **0** |
| KEEP source | one_day focus **2026-02-25** |

**Tools created for this phase:**

| Script | Job |
|--------|-----|
| `fable_50d_mark_match_loop.py` | Full Measure→labels→weights→BC→keep/reject |
| `fable_50d_sprint.py` | Entry-focus from frozen Mark cache |
| `fable_50d_fast_loop.py` | Faster parallel-ish scoring |
| `fable_50d_rapid.py` | Oracle disk cache + HOLD-repair |
| `fable_50d_one_day.py` | One MWT day heavy BC + KL + pack re-score |
| `tests/test_fable_50d_paths.py` | Path unit tests |

**One-day training recipe (to recreate KEEP behavior):**

1. Load best embryo weights.
2. Focus worst MWT day (by policy PnL).
3. Labels: Mark plan dirs + sparse HOLD on that day (`plan_labels` / `dagger_labels`).
4. Optional light award self-imitate for protection.
5. `train_bc` ~35 epochs, **KL coef ~0.55** to prior embryo.
6. Re-score full 50d same recipe.
7. **KEEP** only if: focus day converts **or** same_outcome rises; breach 0; policy_clear ≥ baseline floor (27).
8. Else restore `best_state`.

### Phase H — Diagnosis: what is still broken (honest)

| Wrong diagnosis | Right diagnosis |
|-----------------|-----------------|
| “No Mark labels” | Mark soul plans exist (50/50 clear) |
| “Just turn rewards harder” | Rewards are a megaphone; path mismatch beats dial math |
| “Need more signal agents” | Agents are clues; Mark owns side |

**Real cut:** missing **aligned, path-matched, pack-safe supervision** — bar labels on the **same** path the embryo trades (force-gate + thrash + pure_greedy), that still improve **full-pack** same_outcome.

**Missing principle (plain):**  
*Do what Mark would do under the same rules the bot actually trades* → **Policy = Mark on chart**.

**Brutal causal chain:**

1. Mark optimizes **day** (soul plan).  
2. Net optimizes **bars** under a slightly different live path.  
3. BC metrics can look good while day same_outcome stalls.  
4. Entry push → thrash/breach → REJECT; HOLD push → miss MWT.  
5. Keep/reject correctly freezes **almost-Mark** (33/50) rather than a thrashing mind.

**Sources for this diagnosis:** session reasoning + `LEARNING_50D_MATCH.md` lessons + `MARK_HERE_ANSWER__why_rewards_alone.md` + one_day REJECT cases with high dir_match.

### Phase I — Fable 5 as MARK HERE + KAG (Army)

| Item | Path |
|------|------|
| Config | `ARMY/01_SYSTEM/config/agents/fable5_mark_here_kag.json` |
| Code | `ARMY/01_SYSTEM/packages/core/markos_core/fable5_mark_here_kag.py` |
| CLI | `ARMY/01_SYSTEM/scripts/cycles/fable5_mark_here_kag_cycle.py` |
| Registry / roster | `registry.json`, `army_roster.json` role `fable5_mark_here_kag` |
| Letter to first Mark | `outputs/army/FABLE5_TO_FIRST_MARK__consistency.md` |
| Why rewards alone | `outputs/army/MARK_HERE_ANSWER__why_rewards_alone.md` |
| KAG index | `data/knowledge/army/KAG_INDEX__mark_policy.json` |
| Lab mirror | `checkpoints/fable_50d_match/FABLE5_MARK_HERE_BRIEF__latest.md` |

**Knowledge roots indexed (lightweight mutual index, KAG-inspired):**

1. `data/knowledge/army` (+ skills / shared / trading / wiki)  
2. the-truth (SOUL_MATCH, POLICY, HANDOFF, LEARNING, doctrine, …)  
3. **pt5** canonical basic knowledge  

**Upstream design only:** https://github.com/OpenSPG/KAG.git (full OpenSPG docker optional; local index always on).

**Identity rule:** Fable is method; Mark is mind. Same soul channel as `MARK HERE!.lnk` → http://127.0.0.1:8000/chat — **not** a second Mark.

### Phase J — Meta-learning: forward consistency law (2026-08-05)

**Update location:** `code/training/meta_tuner.py` + `configs/training.yaml` `self_tuner`.

| Rule | Detail |
|------|--------|
| Probe / short PPO | **Practice** days only |
| Champion score | **Forward** consistency only |
| Also required | Forward breach not worse; forward longest day-streak not shorter |
| Practice screen | Reject if practice clear% collapses > ~5pp |
| Weak forward | Force-search `CONSISTENCY_FORWARD_KNOBS` (day_goal, streak, trade_consistency, did_nothing, death, net_profit) |
| Mark chart disease | Still force-search `TREND_KNOBS` |
| Reject | Restore brain weights to champion |

**Mark lineage CLI:** `lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py`  
**Docs:** `POLICY_EQUALS_MARK_ON_CHART.md` § Meta forward-consistency law · `UNSEEN_CONSISTENCY_RECIPE.md`  
**Tests:** `tests/test_self_heal_mri.py` → ALL PASSED (includes `test_forward_adopt_gate`).

---

## 3) Architecture map (recreate mental model)

```
                    ┌─────────────────────────────────────┐
                    │  MARK HERE!.lnk → MarkOS :8000      │
                    │  Soul: MARK_PERSONALITY × Fable ×   │
                    │  moral doctrine × pt5               │
                    └──────────────┬──────────────────────┘
                                   │ same soul
          ┌────────────────────────┼────────────────────────┐
          ▼                        ▼                        ▼
   ARMY Second Brain        the-truth lab              Fable5 KAG agent
   vault / goals            T3 lineage                 army+truth+pt5
                                   │
                    ┌──────────────┴──────────────┐
                    ▼                             ▼
            Mark SOUL (teacher)            POLICY EMBRYO (student)
            mark_soul_plan                 mark_clone_full_obs_v1.pt
            50/50 clear on pack            33/50 same (best)
                    │                             │
                    └──────── BC / DAgger / KL ───┘
                              keep/reject on frozen 50d
                    ┌─────────────────────────────┐
                    │  PROVEN (production)        │
                    │  NEVER load into Mark-obs   │
                    └─────────────────────────────┘
                    ┌─────────────────────────────┐
                    │  meta_tuner                 │
                    │  probe practice             │
                    │  ADOPT only on FORWARD      │
                    └─────────────────────────────┘
```

### Shell laws (locked — never “search” these away)

- Heat / bank / breach death  
- No trail + cushion + scale-in package (IRAC kill)  
- One signal flat + in-trade reverse rules as coded  
- PROVEN mtime must stay unchanged when claiming Mark work  

---

## 4) Exact recreate runbook (blank machine → measured state)

### 4.1 Environment

```powershell
# Windows
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
# Or clone/copy the-truth tree to this path
$env:PYTHONPATH = ".;code"
# Python 3.10+ with torch, numpy; MT5 optional (not required for curriculum CSV)
```

**Data required:**

- `data/raw/XAUUSD_curriculum_2026.csv` (calendar curriculum; days with ≥900 bars)  
- `lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` (target/risk pairs)

**Army (for soul/KAG, optional for pure train):**

- `C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM` with pt5 under  
  `data/knowledge/skills/trading/llm_basic_thinking/pack/pt5__basic_knowledge.txt`

### 4.2 Recreate Mark doctrine teacher + early BC (Phase C)

```powershell
$env:PYTHONPATH = ".;code"
# Ensure mark_doctrine hard-target single-set scalp OFF and opp floor 1.2
# Ensure equity_day ride-green reverse rules as in FURTHEST_WEAVE § chart-DNA

python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 40
# Expect: mark_clone_doctrine_v1.pt ; high dir_match; clone_ready_policy false

python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --mode forward --pair 3.0 3.5
# Compare soft/mid/hard tables to FURTHEST_WEAVE numbers (± noise if data differs)
```

### 4.3 Recreate full-obs soul embryo (Phase D)

```powershell
# BC with full_obs + soul plans
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --full-obs --epochs 40 --max-train-days 50
# Produces / updates: checkpoints/mark_clone_full_obs_v1.pt (obs_dim=168)

python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py --seed 7 --start-idx 40 --full-obs
# Target ballpark: Mark 10/10; policy ~8/10; breach 0

python lineages/adaptive_rl_brain_7_31_26/mark_consistency_loop.py --epochs 40 --max-train-days 30 --streak-days 50
# Target ballpark: max streak ≥10, breach 0 on 50d seed 42
```

### 4.4 Recreate streak autopsy + dials (Phase E)

```powershell
python lineages/adaptive_rl_brain_7_31_26/autopsy_streak_gaps.py
# Writes: checkpoints/mark_consistency/AUTOPSY_GAPS__latest.*
#         STREAK_REWARD_DIALS__latest.json
# Expect: majority MARK_WOULD_TAKE, learnable_frac ~0.9
```

### 4.5 Recreate frozen 50d baseline (Phase G start)

```powershell
python lineages/adaptive_rl_brain_7_31_26/fable_50d_mark_match_loop.py
# Or whatever entry freezes BASELINE_50D__frozen.json on first measure
# MUST log:
#   recipe seed=42 soft_bias=false first 50 days pure_greedy mark_align
# Expect: mark_clear=50, policy_clear=27, same=27, breach=0, MWT=23
```

**Sanity check against frozen file fields:**

- `recipe.first_date` = `2026-01-20`  
- `recipe.last_date` = `2026-03-30`  
- `recipe.seed` = 42  
- `proven_touched` = false  

### 4.6 Recreate climb 27 → 30 → 33

```powershell
# Directional oversample is mandatory (6× dir copies) — see train path / loop
# Cycle with KEEP/REJECT:

python lineages/adaptive_rl_brain_7_31_26/fable_50d_mark_match_loop.py
# First successful KEEP expected ~same=30 with dir_match~0.86
# If breach after entry push: REJECT and restore embryo

# If full loop stalls on labels:
python lineages/adaptive_rl_brain_7_31_26/fable_50d_sprint.py
python lineages/adaptive_rl_brain_7_31_26/fable_50d_rapid.py

# Surgical climb:
python lineages/adaptive_rl_brain_7_31_26/fable_50d_one_day.py
# Expect KEEP on some MWT days (historically 2026-02-25 → same=33)
# KL coef ~0.55; reject if hold-rate collapses or same falls
```

**Keep/reject pseudocode (must implement exactly):**

```
post = score_50d(policy)  # same frozen recipe
if post.breach > 0: REJECT
if post.policy_clear < baseline.policy_clear: REJECT
if post.same_outcome < best.same_outcome: REJECT
# optional: allow KEEP if focus day newly awards AND same >= best
KEEP → best = post; save embryo
else restore best weights
```

### 4.7 Recreate meta forward law (Phase J)

```powershell
# Unit tests
python tests/test_self_heal_mri.py
# Expect: ALL SELF-HEAL MRI UNIT TESTS PASSED

# Mark dials meta under forward law
python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --dry-score
python lineages/adaptive_rl_brain_7_31_26/meta_forward_consistency.py --gens 8 --forward-n 40
```

Production meta caller must pass **practice_days** and **forward_days** into `meta_tuner.run(..., require_forward=True)`.

### 4.8 Recreate Army KAG mentor (Phase I)

```powershell
cd C:\Users\user\OneDrive\Desktop\ARMY\01_SYSTEM
$env:PYTHONPATH = "packages\core"
$env:MARKOS_THE_TRUTH_ROOT = "C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth"
python scripts\cycles\fable5_mark_here_kag_cycle.py
# Expect VERIFIED; writes FABLE5_TO_FIRST_MARK + MARK_HERE_ANSWER__why_rewards_alone
```

---

## 5) File inventory (what to back up first)

### Lab — doctrine / handoff

| Path | Why |
|------|-----|
| `POLICY_EQUALS_MARK_ON_CHART.md` | Law |
| `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md` | 2026-08-04 checkpoint |
| `HANDOFF_2026-08-05.md` | Soul + 168-dim |
| `SOUL_MATCH.md` | One Mark bridge |
| `PROGRESS_RECREATE__POLICY_EQUALS_MARK__2026-08-05.md` | **This file** |
| `00_MAP_OF_THE_HOUSE.md` | Track map |
| `references/doctrine/llm_basic_thinking/**` | pt5 mirror |

### Lab — lineage code (minimum)

| Path | Why |
|------|-----|
| `lineages/adaptive_rl_brain_7_31_26/equity_day.py` | Shell + thrash + soul day |
| `mark_soul_plan.py` | Full-day Mark teacher |
| `mark_aligned_decode.py` | Force/capital + gate fix |
| `perception/observation_full.py` | 168-dim |
| `perception/mark_doctrine.py` | Five laws + hard scalp fix |
| `train_mark_clone_bc.py` | BC |
| `mark_consistency_loop.py` | Long streak |
| `fable_50d_*.py` | 50d match family |
| `rewards.py` | Streak dial pack |
| `autopsy_streak_gaps.py` | Gap classes |
| `meta_forward_consistency.py` | Forward meta CLI |
| `ten_pairs.json` | T/R table |

### Lab — production meta

| Path | Why |
|------|-----|
| `code/training/meta_tuner.py` | Forward adopt law |
| `code/evaluation/consistency.py` | Consistency judge |
| `configs/training.yaml` | self_tuner flags |
| `tests/test_self_heal_mri.py` | Regression |

### Lab — artifacts (measured state)

| Path | Why |
|------|-----|
| `checkpoints/mark_clone_full_obs_v1.pt` | Best embryo (~33/50) |
| `checkpoints/mark_clone_doctrine_v1.pt` | Early doctrine BC |
| `checkpoints/fable_50d_match/BASELINE_50D__frozen.json` | Yardstick 27/50 |
| `checkpoints/fable_50d_match/BEST__latest.json` | Best meters |
| `checkpoints/fable_50d_match/LEARNING_50D_MATCH.md` | Cycle log |
| `checkpoints/fable_50d_match/MARK_ORACLE_CACHE__50d.json` | Mark plans cache |
| `checkpoints/mark_consistency/*` | Autopsy, dials, streak |

### Army

| Path | Why |
|------|-----|
| `config/agents/fable5_mark_here_kag.json` | Agent law |
| `packages/core/markos_core/fable5_mark_here_kag.py` | KAG mentor |
| `config/agents/MARK_PERSONALITY.md` | Soul + 4c |
| `data/knowledge/skills/trading/llm_basic_thinking/pack/pt5__basic_knowledge.txt` | pt5 |
| `outputs/army/FABLE5_TO_FIRST_MARK__consistency.md` | Dialogue product |
| `outputs/army/MARK_HERE_ANSWER__why_rewards_alone.md` | Rewards Q&A |

---

## 6) Lessons that must not be re-learned the hard way

1. **Directional oversample is mandatory.** HOLD-heavy BC → dir_match ~0.09 collapse.  
2. **Too little HOLD → breach.** Entry thrash fails keep/reject.  
3. **Never train thrash teacher as side owner.** Force-gate only for Mark sense.  
4. **Never overwrite PROVEN** for Mark experiments.  
5. **Practice ≠ forward.** Meta/dials that only win practice are costume.  
6. **Label path must match live decode path** or BC is theater.  
7. **Keep/reject is the conscience** — dual re-score full frozen pack every time.  
8. **Rewards cannot replace Mark labels.** Megaphone ≠ mind.  
9. **One policy = Mark forbids retrain-per-day** (50 minds is not consistency).  
10. **Gate `flat_undefined` string bug** — use proper dead-regime checks.

---

## 7) Current scoreboard (as of this document)

| Stack | Meter | Value | Source |
|-------|--------|------:|--------|
| Mark soul 50d pack | mark_clear | 50/50 | BASELINE_50D |
| Policy 50d same (baseline) | same_outcome | 27/50 | BASELINE_50D |
| Policy 50d same (**best**) | same_outcome | **33**/50 | BEST__latest + LEARNING |
| Policy 50d breach | n_breach | **0** | both |
| MWT remaining | | **17** | BEST |
| 10d Mark soul | clear | 10/10 | HANDOFF |
| 10d policy | clear | 8/10 | HANDOFF |
| Award streak 50d | max_streak | 11 (earlier) / work toward 2× | HANDOFF / 2× files |
| PROVEN | | untouched | all Mark reports |
| Meta | | forward-only adopt law shipped | meta_tuner + tests green |
| KAG mentor | | shipped + VERIFIED cycle | Army agent |

**Open gap to mission:** 17 MWT days under one frozen embryo; path-matched labels + keep/reject until same=50; meta may search dials but not invent Mark.

---

## 8) Next actions (ordered, recreate-safe)

1. Continue `fable_50d_one_day.py` (or equivalent) on remaining MWT only.  
2. For each KEEP candidate: full 50d re-score seed 42; breach 0; same ≥ best.  
3. If dir_match high but day fails: fix **path** (live recommended vs plan), not more entry reward.  
4. Meta: only practice probe; adopt only if **forward** consistency improves.  
5. Human MARK HERE HITL on stubborn days (historical 10d: 2026-03-24, 2026-03-26).  
6. When same=50: dual re-score, freeze embryo, write claim JSON, stop invent.  

---

## 9) Citation index (primary sources used in this document)

1. `POLICY_EQUALS_MARK_ON_CHART.md` — control chain, meta bounds, sets  
2. `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md` — 2026-08-04 meters & chart-DNA fix  
3. `HANDOFF_2026-08-05.md` — soul 10/10, 168-dim, streak 11  
4. `SOUL_MATCH.md` — one Mark, pt5 bridge  
5. `lineages/.../UNSEEN_CONSISTENCY_RECIPE.md` — practice/forward definitions  
6. `checkpoints/fable_50d_match/BASELINE_50D__frozen.json` — 27/50 freeze + recipe  
7. `checkpoints/fable_50d_match/BEST__latest.json` — 33/50 best  
8. `checkpoints/fable_50d_match/LEARNING_50D_MATCH.md` — cycle log & lessons  
9. `checkpoints/mark_consistency/STREAK_REWARD_DIALS__latest.json` — dial values  
10. `checkpoints/mark_consistency/AUTOPSY_GAPS__latest.md` — MWT vs NO_OPP  
11. `code/training/meta_tuner.py` — forward adopt implementation  
12. `configs/training.yaml` — self_tuner forward law notes  
13. `tests/test_self_heal_mri.py` — verification of meta gates  
14. ARMY `fable5_mark_here_kag.py` / outputs — KAG mentor + Mark rewards answer  
15. pt5 pack — basic knowledge every LLM must load  
16. https://github.com/OpenSPG/KAG — KAG design reference  
17. https://github.com/Sahir619/fable-method — Fable loop structure  

---

## 10) Integrity checks (if someone claims they “recreated” this)

| Check | Pass condition |
|-------|----------------|
| PROVEN | File mtimes / hashes unchanged during Mark work |
| Recipe | seed 42, soft_bias false, dates 2026-01-20…2026-03-30, pure_greedy mark_align |
| Baseline | mark 50, policy 27, same 27, breach 0, MWT 23 |
| Best climb | same ≥ 33 without breach (or document why lower) |
| Meta | No adopt on practice-only improvement |
| Shell | No trail+cushion+scale-in reintroduced |
| Soul | One MarkOS; Fable method only; pt5 loaded |

---

*End of recreate bible. Update this file when same_outcome advances or laws change; append CHANGE LOG below, never silent rewrites of past meters.*

### CHANGE LOG

| Date | Change |
|------|--------|
| 2026-08-05 | Initial full progress + recreate bible (27→33 50d; soul; KAG; meta forward law) |
