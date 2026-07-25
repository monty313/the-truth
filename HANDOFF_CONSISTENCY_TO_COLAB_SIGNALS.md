# HANDOFF — How We Got Higher Consistency → Colab GPU + Signal Agents ON

**Date:** 2026-07-25  
**Audience:** Next LLM / engineer writing **Google Colab GPU training instructions** with **signal-agent slots enabled**  
**Repo:** https://github.com/monty313/the-truth  
**Local (Windows):** `C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth`  
**Owner doctrine:** Monty / Momentum One  

**Purpose of this file:** Single, source-cited briefing so another model can author correct Colab training instructions **without re-deriving history or repeating past mistakes**.

---

## 0. Executive summary (read this first)

### What “higher consistency” meant in history (measured RL)

| Metric | Peak measured (PROVEN lineage) | Source |
|--------|--------------------------------|--------|
| Clear rate (hit target, no breach) | **~27%** (24/90 days) | `doctrine/flea-jar/evidence/record_brain_scoreboard.txt`; `doctrine/SUCCESS_LEDGER.md` |
| Longest clear streak (row) | **4 days** | Same |
| Breach rate | **0%** | Same |
| Checkpoint | `artifacts/checkpoints/PROVEN_SPRINT_row04_clear24_2026-07-20.pt` | SUCCESS_LEDGER |
| Journey | **1/90 → 24/90** after scoreboard cures + lift + consistency sprint | `doctrine/flea-jar/THE_BOTS_ISSUE_AND_CURE_IRAC.md`; `PERFORMANCE_IS_POSSIBLE_PART3.md` |

### What is **not** a past RL clear-rate of 70–80%

| Claim | Actual meaning | Source |
|-------|----------------|--------|
| **+70%/day possible** | Hindsight/oracle **physics bound** on **90/90** curriculum days under ≤4.5% DD, lev≤100 — **not** RL clear % | `scripts/prove_70.py`; `doctrine/flea-jar/evidence/the_70_sweep_2026-07-21.txt`; `doctrine/flea-jar/EXHIBIT_D_LEDGER.md` |
| **~70–81%** | Forward hit rate of **agreement signal agents** (slots 80–83) on signal events | `PERFORMANCE_IS_POSSIBLE_PART4.md`; `signals/agree.py` |
| **Climb toward 80% clear** | **Goal** for future training (HOST_RUN, CONSISTENCY_PLAN) — **not yet a logged prove_it result** | `doctrine/HOST_RUN.md`; `CONSISTENCY_PLAN.md` |

### Current baseline (2026-07-25, after TF realign + engine restore)

| Metric | Value | Source |
|--------|-------|--------|
| Brain | `PROVEN_SPRINT_row04_clear24_2026-07-20` | prove_it run 2026-07-25 |
| Clear / breach / row | **21% / 0% / 2** @ 3.0% / 3.5% | `HANDOFF.md`; SUCCESS_LEDGER dated win |
| frame_dim | **1820** = 10 × (170 market cols + 12 self) | prove_it printout; `training/env.py` FRAME=10, SELF_DIM=12 |
| Signal slots in obs | **OFF** (`include_signal_agent_slots: false`) | `configs/features.yaml` |
| Disease | **Policy** — setups visible, policy holds | IRAC JSON + Mind Probe |
| IRAC (6 days) | policy_hold_on_setup=**404**, high_miss_pull=**54**, mask_veto=**4** | `artifacts/llm_curriculum/irac_PROVEN_SPRINT_row04_clear24_2026-07-20.json` |
| Reward already applied | `w_pullback_with_htf: 0.25` (was 0.02) | `configs/rewards.yaml`; `doctrine/IRAC_PULLBACK_2026-07-24.md` |

### Critical rule for Colab + signals

> **Turning signal slots ON expands observation (~1820 → ~6820 frame_dim).**  
> **Old PROVEN_* checkpoints cannot load** (mat1/mat2 shape mismatch).  
> Signal-ON training = **NEW brain lineage**. Do not force PROVEN weights into expanded input.  
> Sources: `configs/features.yaml` comments; historical error `90x6820 and 1820x128`; `HANDOFF.md`.

---

## 1. Project north star (non-negotiable)

1. Hit a **daily profit target %** and never breach a **daily risk floor %**.  
2. Target and floor are **runtime inputs** (brain sees them in obs self-state). Changing them does **not** require retrain.  
   - Sources: `DO_THIS.md`; `configs/goals.yaml`; `HANDOFF.md`; SUCCESS_LEDGER “goal/floor = runtime inputs”.
3. Champion metric = **consistency** = % days with goal hit **and** no floor breach.  
   - Source: `docs/adr/ADR-0006-training-protocol.md`.
4. Evolve via **reward shaping + policy practice + meta-tuner** — not wiping weights for every goal change.  
   - Sources: `doctrine/STANDING_LAWS.md` (optimization constraint); `training/meta_tuner.py`.
5. **Flea-jar / Antibody Law:** never call a day impossible without a **measured bound**.  
   - Sources: `doctrine/flea-jar/THE_FLEA_CURE.md`; `doctrine/flea-jar/README.md`.
6. **Never delete:** PERFORMANCE_IS_POSSIBLE*, SUCCESS_LEDGER, flea-jar, PROVEN checkpoints.  
   - Sources: SUCCESS_LEDGER “Never delete”; HANDOFF hard rules.

---

## 2. How consistency was raised historically (detailed timeline)

Primary narrative sources (read in this order for the next LLM):

1. `doctrine/flea-jar/README.md` — index of the case file  
2. `doctrine/flea-jar/THE_FLEA_CURE.md` — trainer sickness (assumed lids)  
3. `doctrine/flea-jar/THE_BOTS_ISSUE_AND_CURE_IRAC.md` — bot issue = intent drift in feedback  
4. `PERFORMANCE_IS_POSSIBLE_PART3.md` — medical record / treatment log  
5. `PERFORMANCE_IS_POSSIBLE_PART2.html` — lift evidence (4/5 banked)  
6. `doctrine/flea-jar/evidence/record_brain_scoreboard.txt` — frozen 27% scoreboard  
7. `PERFORMANCE_IS_POSSIBLE_PART4.md` — agreement signals (70–81%) for **later** obs expansion  

### 2.1 Starting patient (pre-cure)

- **1/90 cleared**, ~+0.04%/day, motionless on rich days.  
- **0 breaches** always — body/safety OK; feedback broken.  
- Source: `THE_BOTS_ISSUE_AND_CURE_IRAC.md` §I; `PERFORMANCE_IS_POSSIBLE_PART3.md` admission notes.

### 2.2 Four infections (all scoreboard / trainer — Mirror Law)

**Mirror Law** (doctrine): *A learning bot becomes exactly what its feedback measures — never what owners intend.*  
Source: `THE_BOTS_ISSUE_AND_CURE_IRAC.md` §R.

| # | Name | Mechanism | Cure | Measured effect | Sources |
|---|------|-----------|------|-----------------|--------|
| 1 | **Pay-cliff** | Win-lock parked day at goal−fee (~2.9% on 3% target); day-end judge paid **0** for near-goal | LAW FIX in `training/fastsim.py` + `backtesting/simulator.py` (search “LAW FIX”) | Fix alone: 1/90 → 3/90 | IRAC infections; PART3 Tick 1 |
| 2 | **Counterfeit streak applause** | Batch bug gave streak credit to all envs when any finished | Fix + pin `tests/test_ratchet_and_streak.py` | Records counterfeit-proof | IRAC Infection 2; PART3 Tick 2 |
| 3 | **Forbidden classroom** | Trainer zero-weighted 59/90 days as “unwinnable” by range | Every day practices; winnability = **swing-capture bound** | **90/90 winnable**; ceiling 90 in a row | `THE_FLEA_CURE.md`; IRAC Infection 3; `consistency_sprint.py` comments |
| 4 | **Inherited flinch** | Warm-start from flat ancestor first | Warm-start profitable seed (`lift_best` → PROVEN_LIFT / PROVEN_SPRINT) | Lineage banks target | IRAC Infection 4; PART3 Tick 4 |

### 2.3 Treatment sequence that produced 24/90 + row 4

From `PERFORMANCE_IS_POSSIBLE_PART3.md` treatment log:

| Phase | Action | Script / system | Response |
|-------|--------|-----------------|----------|
| Midday | Tick 1 law fix | fastsim + simulator | 1/90 → 3/90 |
| Afternoon | **Ladder training** | `scripts/lift_demo.py` | Master 1 rich day, then 5-day pool; first banked clear ~35 updates; 4/5 banked |
| Evening | Ticks 2–4 + tests | ratchet tests, warm-start order | Counterfeit-proof records |
| Night | Remove range lid | `consistency_sprint.py` all-day weights | 90/90 winnable map |
| Final | **Consistency sprint** | `scripts/consistency_sprint.py` | **24/90 clear, row 4, 0 breach** |

**Lift ladder details** (`scripts/lift_demo.py` docstring + code):

- Phase 1: master ONE rich day (index for 2026-01-30, large range) to ≥ +3.0% no breach.  
- Phase 2: extend to 5-day rich pool `POOL = [8, 9, 7, 44, 42]`.  
- Serial-stamped saves (sha256[:12] in filename).  
- PART2 HTML: “4 of 5 banked past +3%”, reproducible.

**Consistency sprint mechanics** (`scripts/consistency_sprint.py`):

- Warm-start PROVEN lineage (lift_best → PROVEN_SPRINT → PROVEN_LIFT → PROVEN_2x).  
- Measure clear count + streak on all 90 days (greedy).  
- Day weights: cleared retention; near-miss heavy; **chain-break day ×5**.  
- **No day zero-weighted by assumption** (flea cure).  
- PPO polish: gentle lr ~1.2e-4, low entropy, periodic measure.  
- **Ratchet:** on record → save `sprint_rowXX_clearYYof90_SN-*.pt` under `artifacts/checkpoints/history/` and copy to `lift_best.pt`.  
- Snap back on real collapse.  
- Host recipe for climb: **`--minutes 600 --envs 256`** (not 64).  
  - Sources: `doctrine/HOST_RUN.md`; `TRAINING.md`; `DO_THIS.md`; `PERFORMANCE_IS_POSSIBLE.md`.

### 2.4 Frozen proof (cite this in Colab docs as the “old world” ceiling we beat once)

```
# doctrine/flea-jar/evidence/record_brain_scoreboard.txt
BRAIN: PROVEN_SPRINT_row04_clear24_2026-07-20
cleared: 27% of days
breached: 0% of days
longest cleared streak: 4 days
```

Serial mentioned in IRAC/PART3: `c49091b393ca` (row of +3.04 / +3.19 / +3.10 / +3.13 Mar 18→23).

### 2.5 After the peak: TF lock + policy_hold

- Sets locked (Monty): A=1m/15m/30m, B=5m/1h/4h, C=15m/4h/1d.  
  - Sources: `features/engine.py` SETS; `doctrine/STANDING_LAWS.md` / HANDOFF Gravity table.  
- Post realign: clear **~21%**, breach 0% (semantic obs shift).  
  - Source: SUCCESS_LEDGER 2026-07-24.  
- IRAC 2026-07-24: stuck ~27% while **policy_hold** on HTF trend + LTF pull/cont; not pure Perception.  
  - Source: `doctrine/IRAC_PULLBACK_2026-07-24.md`.  
- Cure applied: `w_pullback_with_htf` **0.02 → 0.25**.  
  - Source: `configs/rewards.yaml` changelog.  
- Disease still open 2026-07-25: Mind Probe shows pull flags **acted 0 / held all**; policy_hold dominates mask_veto.  
  - Source: IRAC JSON under `artifacts/llm_curriculum/`.

### 2.6 Parallel evidence: agreement signals (for signal-ON phase)

**Not used in PROVEN 1820-dim brains.** Ready for expanded obs.

| Slot | Kind | Approx forward hit | Sources |
|------|------|--------------------|---------|
| 80 | `agree_seA_r2A` | ~75% @ 10 bars | PART4 table; `configs/signal_slots.yaml` |
| 81 | `agree_seB_r2B_epB` | ~70–72% | PART4 |
| 82 | `agree_2of_top4` | ~76% @ 10 / ~71% @ 20 | PART4 |
| 83 | `agree_seA_r2A_atr` | ~78–81% (rarer) | PART4 |

Code path:

- Handlers: `signals/agree.py`  
- Registry: `configs/signal_slots.yaml` filled 80–83  
- Encode bus: `signals/encode.py` (`append_signal_obs`, N_SLOTS=500)  
- Gate: `features/engine.py` only if `configs/features.yaml` → `include_signal_agent_slots: true`  
- Slot map overview: `signals/README.md` (0–9 native, 10–27 Camillion, 28–499 free; 80–83 agree)

Skill rule (doctrine): non-zero agreement under firm HTF Gravity = high-value engage; holding while they fire = same Policy disease as holding bread-and-butter pull.  
Source: `doctrine/policy_skill.md` §Agreement suggestions.

---

## 3. Hard gates (copy into any Colab instructions)

| # | Gate | Why | Source |
|---|------|-----|--------|
| G1 | **Breach must stay 0%** on `prove_it` | Sacred floor | SUCCESS_LEDGER; ADR goal/floor; HANDOFF |
| G2 | Measure only with `python scripts/prove_it.py <brain> <target> <risk>` | Single scoreboard | `PERFORMANCE_IS_POSSIBLE.md`; `scripts/prove_it.py` |
| G3 | Target/risk are **CLI/runtime**, not retrain triggers | Self-state + goals.yaml | DO_THIS.md |
| G4 | Warm-start from **PROVEN / profitable** lineage when dims match | Avoid inherited flinch | IRAC Infection 4 |
| G5 | **Signal OFF** keeps 1820-dim PROVEN usable | Dim compatibility | features.yaml |
| G6 | **Signal ON** ⇒ **new train**, new obs_dim, **do not load 1820 into 6820** | Shape error | features.yaml; engine gate |
| G7 | Never delete PERFORMANCE* / SUCCESS_LEDGER / flea-jar / PROVEN_* | Precedence | SUCCESS_LEDGER |
| G8 | Never zero-weight days as impossible | Flea cure | THE_FLEA_CURE.md; consistency_sprint |
| G9 | One material change per cycle; prove before/after | Consistency climb discipline | CONSISTENCY_PLAN.md |
| G10 | Prefer practice + rewards over new weak indicators | Cure order | SYSTEM_DOCTRINE_CMO; IRAC |

**Standing Laws tension (document honestly):**  
`doctrine/STANDING_LAWS.md` says do not alter observation space for frozen evolution.  
**Exception for this Colab mission:** signal-ON is an **explicit new-brain phase** already planned in HANDOFF / CONSISTENCY_PLAN / features.yaml. It does **not** mutate frozen PROVEN files; it creates a **parallel lineage** with larger `obs_dim`.

---

## 4. Observation dimensions (must be correct in Colab)

### 4.1 Formula

From `training/env.py` / `training/gpu_rollout.py` / `scripts/prove_it.py`:

```text
FRAME = 10                    # features.yaml frame_stack
SELF_DIM = 12                 # goal, floor, dists, ratchet, etc.
market_cols = len(obs_columns(F))   # set* flags + obs::* + masks

frame_dim = FRAME * (market_cols + SELF_DIM)
```

### 4.2 Known operating points

| Mode | `include_signal_agent_slots` | market_cols (approx) | frame_dim (approx) | Compatible brains |
|------|------------------------------|----------------------|--------------------|-------------------|
| PROVEN / current | `false` | **170** | **1820** | All PROVEN_*, lift_best from 1820 sprints |
| Signal agents ON | `true` | **~670** (170+500) | **~6820** | **Only brains trained with slots ON** |

Cache rule: if you change the flag or feature engine, **delete**  
`artifacts/gpu_cache_XAUUSD_curriculum_2026.npz` before rebuild.  
Source: `scripts/prove_it.py` docstring; HANDOFF.

### 4.3 How to enable signals (code path)

1. Set in `configs/features.yaml`:
   ```yaml
   include_signal_agent_slots: true
   ```
2. Ensure `features/engine.py` reads that flag and calls `signals.encode.append_signal_obs` when true (restored full engine with gate — 2026-07-25).  
3. Rebuild feature cache.  
4. Construct `Brain(new_obs_dim)` — **random init or special transfer** (see §6); **do not** `load_state_dict` from 1820 PROVEN.  
5. Train with GPU edition / consistency_sprint / gpu_train as appropriate.  
6. Gate with `prove_it` on the **new** brain name; expect frame_dim print ~6820.

---

## 5. Gravity / strategy context (what the brain should act on)

### 5.1 Locked TF sets

| Set | LTF | HTFs | Source |
|-----|-----|------|--------|
| A | 1m | 15m, 30m | engine SETS; HANDOFF |
| B | 5m | 1h, 4h | same |
| C | 15m | 4h, 1d | same |

Bread-and-butter: **LTF pullback while both HTFs strongly trending**.  
Sources: STANDING_LAWS; policy_skill.md; IRAC_PULLBACK.

### 5.2 Policy disease (why clear rate stalls)

- Mind Probe / Ghosts: skip_reason=`policy_hold` when pull/cont + HTF bias present.  
- Mask veto rare relative to hold.  
- Cure: frontier practice under `w_pullback_with_htf` (+ optional gated nudges), **not** more ~55% single-family stacks.  
- Sources: `doctrine/IRAC_PULLBACK_2026-07-24.md`; `telemetry/mind_probe.py`; `telemetry/ghost_trades.py`; `scripts/give_llm_what_it_needs.py`.

### 5.3 Active rewards (starting point for training)

From `configs/rewards.yaml` (verify live file; do not invent):

- `w_net_profit: 6.0`  
- `w_day_goal_hit: 2.0`  
- `w_pullback_with_htf: 0.25`  ← Policy cure  
- `w_did_nothing: -6.0`  
- `w_death_penalty: -10.0`  ← do not weaken while climbing clear  
- `w_streak_per_day: 0.15`  

Meta-tuner bounds: `training/meta_tuner.py` BOUNDS (includes `w_pullback_with_htf` 0–1).

---

## 6. Training recipes (cite + use in Colab instruction authoring)

### 6.1 Host / CPU–GPU consistency climb (1820-dim, signals OFF) — what we used to hit 27%

```bash
# Sources: doctrine/HOST_RUN.md, TRAINING.md, DO_THIS.md, PERFORMANCE_IS_POSSIBLE.md
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <sprint_or_lift_best_name> 3.0 3.5
# optional multi-pair after gains:
python scripts/prove_it.py <brain> 2.5 2.5
python scripts/meta_train.py --minutes 600
```

**Note on envs:** doctrine says **256**. Local 2026-07-25 experiment used **64** for capacity; wider batch is the written recipe for climb speed.

### 6.2 Lift ladder (seed banking ability)

```bash
# Source: scripts/lift_demo.py; PART2/PART3
python scripts/lift_demo.py          # phase 1 one rich day
python scripts/lift_demo.py --p2     # phase 2 five-day pool (uses lift_best)
```

### 6.3 Diagnostic loop (Policy vs Perception)

```bash
# Sources: scripts/self_heal_epoch.py, give_llm_what_it_needs.py, START_FROM_TODAY.md
python scripts/self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
# or
python scripts/give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
```

Outputs: `artifacts/llm_curriculum/irac_*.json`, Mind Probe dumps.

### 6.4 Colab GPU Edition (mass parallel) — existing entrypoint

**Docs:** `GPU_EDITION/START_HERE_README.md`  
**Notebooks:**  

- `GPU_EDITION/Momentum_One_GPU.ipynb` — primary multi-instance train  
- `GPU_EDITION/Momentum_One_SelfTuner.ipynb` — self-tuner path  
- Guide: `GPU_EDITION/GUIDE.html`  

**Plain-language design (START_HERE):**

- Thousands of practice markets in parallel (default narrative: **8000 instances**).  
- Random target in **2.5%–70.3%**, risk **1%–4.4%** (ranges in README — verify notebook cells).  
- Record = longest **cleared days in a row**; save with pass count in filename.  
- Finish line narrative: **365 cleared days in a row**.  
- Hardware: Colab **L4 GPU**.  
- OOM: reduce `--instances` 8000 → 4000 → 2000.  
- Saves: `artifacts/checkpoints/history/`, `gpu_best.pt`, Drive `MyDrive/momentum_gpu/`.  
- Training uses **FastSim** (batched twin); **real DaySim** judges proofs (`scripts/gpu_validate.py` fidelity notes in `training/fastsim.py`).  
- “Never overwrites proven brain; starts from a copy.”

**Related scripts:**

- `scripts/gpu_train.py` — CLI GPU train  
- `scripts/gpu_validate.py` — FastSim vs DaySim  
- `scripts/consistency_sprint.py` — frontier polish on curriculum (often after GPU)  

### 6.5 Recommended two-track plan for the next LLM’s Colab instructions

**Track A — Recover / climb on 1820 (optional baseline hygiene)**  
Keep signals OFF; warm-start PROVEN_SPRINT; long sprint or GPU notebook **without** expanding obs; prove_it until clear ≥27% row≥4 breach 0 again.  
Purpose: sanity + Safety of gates before expanding dim.

**Track B — Signal-ON new lineage (the requested Colab mission)**  

1. Fork config: `include_signal_agent_slots: true`.  
2. Delete GPU/feature caches so columns rebuild with `obs::sig_000`…`sig_499`.  
3. Confirm `obs_columns` length and frame_dim (~6820).  
4. **Initialize new Brain(6820)** — do not load 1820 PROVEN state_dict.  
   - Optional research: partial weight copy of non-input layers is **not** implemented as standard path; prefer clean train or carefully engineered transfer later.  
5. Warm-start **behaviorally** via rewards + skill doctrine (pullback weight, did_nothing, death penalty), not via wrong-shaped weights.  
6. Train on Colab GPU (notebook or `gpu_train`) with high instance count; save serial-stamped checkpoints with **obs_dim in filename/meta**.  
7. Every record: run `prove_it <new_brain> 3.0 3.5` — ACCEPT only if breach==0 and clear improves.  
8. Reward/skill emphasis: when `obs::sig_080`…`sig_083` ≠ 0 under Gravity, **engage** (policy_skill + PART4).  
9. After clear climb, meta_train across goal/floor ranges (`configs/goals.yaml`).  
10. Never promote a signal-ON brain into 1820 live path without matching code flag.

### 6.6 Phased clear-rate targets (planning only; always prove_it)

From `CONSISTENCY_PLAN.md` (aligned with historical peak + ambition):

| Phase | Clear % | Row | Breach | Notes |
|-------|---------|-----|--------|-------|
| P0 | 21% | 2 | 0% | Post-realign migration floor (2026-07-25) |
| P1 | ≥27% | ≥4 | 0% | Match historical PROVEN peak |
| P2 | ≥35% | ≥6 | 0% | Sprint + practice |
| P3 | ≥50% | ≥10 | 0% | Meta-tuner territory |
| P4 | ≥70% → ~80% | stretch | 0% | Ambition; physics bound supports possibility, not entitlement |

---

## 7. Physics / possibility evidence (for Colab “why 80% is allowed”)

| Evidence | Result | How to re-run | Source |
|----------|--------|---------------|--------|
| Swing-capture bound ≥3% | **90/90** days | consistency_sprint header / flea cure | THE_FLEA_CURE.md |
| +70% day possible ≤4.5% DD | **90/90** days | `python scripts/prove_70.py` | the_70_sweep; prove_70.py |
| Jan 29 exhibit | $10k → $18.4k in 15 min, 3 trades, floor law | `python scripts/exhibit_d_ledger.py` | EXHIBIT_D_LEDGER.md |
| Agreement signals | 70–81% forward | score_signal_slots / PART4 tables | PART4 |

These prove **the jar lid is not physics**. They do **not** replace RL practice.

---

## 8. Files the next LLM must open (checklist)

### Doctrine & evidence (history)

- [ ] `doctrine/flea-jar/README.md`  
- [ ] `doctrine/flea-jar/THE_FLEA_CURE.md`  
- [ ] `doctrine/flea-jar/THE_BOTS_ISSUE_AND_CURE_IRAC.md`  
- [ ] `doctrine/flea-jar/THE_CLOSING_ARGUMENT.md`  
- [ ] `doctrine/flea-jar/EXHIBIT_D_LEDGER.md`  
- [ ] `doctrine/flea-jar/evidence/record_brain_scoreboard.txt`  
- [ ] `doctrine/flea-jar/evidence/the_70_sweep_2026-07-21.txt`  
- [ ] `PERFORMANCE_IS_POSSIBLE.md` (+ PART2 html, PART3, **PART4**)  
- [ ] `doctrine/SUCCESS_LEDGER.md`  
- [ ] `doctrine/IRAC_PULLBACK_2026-07-24.md`  
- [ ] `doctrine/HOST_RUN.md`  
- [ ] `doctrine/policy_skill.md`  
- [ ] `doctrine/STANDING_LAWS.md`  
- [ ] `doctrine/SYSTEM_DOCTRINE_CMO.md`  

### Ops & plans

- [ ] `DO_THIS.md`, `START_FROM_TODAY.md`, `TRAINING.md`  
- [ ] `CONSISTENCY_PLAN.md`  
- [ ] `HANDOFF.md` (2026-07-25 live state)  
- [ ] **This file** `HANDOFF_CONSISTENCY_TO_COLAB_SIGNALS.md`  

### Code / config for Colab authoring

- [ ] `GPU_EDITION/START_HERE_README.md` + `Momentum_One_GPU.ipynb` + `Momentum_One_SelfTuner.ipynb`  
- [ ] `scripts/consistency_sprint.py`, `scripts/prove_it.py`, `scripts/gpu_train.py`, `scripts/lift_demo.py`  
- [ ] `scripts/give_llm_what_it_needs.py`, `scripts/self_heal_epoch.py`  
- [ ] `configs/features.yaml` (**signal flag**), `configs/rewards.yaml`, `configs/goals.yaml`, `configs/signal_slots.yaml`  
- [ ] `features/engine.py` (build_features + signal gate + masks)  
- [ ] `signals/encode.py`, `signals/agree.py`, `signals/README.md`  
- [ ] `training/fastsim.py`, `training/policy.py`, `training/gpu_data.py`, `training/meta_tuner.py`  
- [ ] `inference/loader.py` (load_brain requires matching obs_dim)  
- [ ] `tests/test_ratchet_and_streak.py` (pins)  

### Data

- [ ] `data/XAUUSD_curriculum_2026.csv` (primary 90-day curriculum)  
- [ ] Cache: `artifacts/gpu_cache_XAUUSD_curriculum_2026.npz` (delete on feature change)  

---

## 9. Prompt contract for the next LLM (suggested)

Use this as the system/task preamble when asking another model to write Colab instructions:

```text
You are writing Google Colab training instructions for Momentum One (repo the-truth).

Read HANDOFF_CONSISTENCY_TO_COLAB_SIGNALS.md end-to-end and cite its source files.
Mission: train a NEW brain with include_signal_agent_slots: true on Colab L4 GPU,
using GPU_EDITION notebooks and/or scripts/gpu_train.py + prove_it gate.

Hard constraints:
- breach 0% on prove_it or REJECT
- do NOT load PROVEN_* 1820-dim weights into signal-ON ~6820-dim Brain
- delete feature cache after enabling signals
- preserve PROVEN checkpoints and PERFORMANCE/flea-jar docs
- warm-start philosophy: profitable lineage for 1820 only; signal-ON is new lineage
- emphasize agreement slots 80–83 as engage cues (PART4), rewards w_pullback_with_htf=0.25,
  never zero-weight days, serial-stamped saves with obs_dim in meta
- measurement: prove_it <brain> 3.0 3.5 (and spot-check 2.5 2.5)

Deliverable: step-by-step Colab cells / shell commands, config diffs, acceptance criteria,
and a failure checklist (shape mismatch, cache stale, breach >0, OOM instances).
```

---

## 10. Current engineering state (2026-07-25) relevant to handoff

| Item | State |
|------|--------|
| Engine | Full masks + S1_perm/trig restored after bad rewrite `d6313e9` |
| Signal gate | YAML default false; engine respects flag |
| Local commits | Engine restore + CONSISTENCY_PLAN pushed earlier; sprint may still be running |
| Live sprint (if still up) | `consistency_sprint --minutes 600 --envs 64` on 1820-dim; best ~20–22/90, row 4 in sprint measure — still needs prove_it ACCEPT |
| IRAC | Policy class confirmed with high policy_hold counts |

---

## 11. One-page “how we got higher consistency” (for slides / short prompt)

1. **Safety was never the problem** (0 breaches) — feedback was.  
2. Fixed **pay-cliff**, **fake streaks**, **forbidden days**, **bad warm-starts** (Mirror Law).  
3. **Lift** one rich day → five-day pool until banking +3% was real.  
4. **Consistency sprint** on all 90 days with chain-repair weights + ratchet.  
5. Measured **24/90 clear, row 4, 0 breach** — frozen PROVEN_SPRINT.  
6. After TF realign, **21%** is the migration floor; disease is **policy_hold**.  
7. Raised **w_pullback_with_htf to 0.25**; still need **GPU hours**.  
8. **Agreement agents 80–83** give 70–81% signal evidence for a **new** obs space.  
9. Colab path: **signals ON → new Brain → GPU train → prove_it only** — never glue old weights to new dim.  
10. Physics allows high clears (90/90 bound, 70% oracle days); **consistency is practice under a clean scoreboard**.

---

## 12. Source index (alphabetical by path)

| Path | Role in this handoff |
|------|----------------------|
| `CONSISTENCY_PLAN.md` | Phased climb 21→27→35→50→70/80 |
| `DO_THIS.md` / `START_FROM_TODAY.md` / `TRAINING.md` | Daily ops commands |
| `GPU_EDITION/*` | Colab GPU mass training |
| `HANDOFF.md` | Live 2026-07-25 status |
| `PERFORMANCE_IS_POSSIBLE.md` (+2/3/4) | Possibility + lift + diagnosis + agreement |
| `configs/features.yaml` | Signal flag; frame_stack |
| `configs/rewards.yaml` | Pullback weight 0.25 |
| `configs/signal_slots.yaml` | Slots 80–83 registry |
| `doctrine/HOST_RUN.md` | 600 min × 256 envs host climb |
| `doctrine/IRAC_PULLBACK_2026-07-24.md` | Policy disease IRAC |
| `doctrine/SUCCESS_LEDGER.md` | Standing proofs |
| `doctrine/flea-jar/*` | Full consistency rise case file |
| `features/engine.py` | Features + mask + signal gate |
| `scripts/consistency_sprint.py` | Frontier polish |
| `scripts/lift_demo.py` | Ladder to bank target |
| `scripts/prove_it.py` | Official measurement |
| `scripts/prove_70.py` | 70% day physics sweep |
| `signals/agree.py` / `encode.py` | Signal agents |
| `training/fastsim.py` | GPU twin + law fidelity notes |
| `tests/test_ratchet_and_streak.py` | Streak law pin |

---

*End of handoff. Next LLM: author Colab GPU + signal-ON training instructions from §6.5 Track B and §9 prompt contract; do not invent unmeasured clear-rate history.*
