# HANDOFF — Momentum One / the-truth

**Date:** 2026-07-30  
**Previous handoff:** 2026-07-25  
**Owner:** Monty  
**Repo:** https://github.com/monty313/the-truth  
**Local path (Windows):** `C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth`

> **Next Grok / LLM session: start here.** Then open the PERFORMANCE series (map below) so you do not re-learn the flea-jar story from scratch. Daily commands also live in `DO_THIS.md`. Folder map: `MAP.md`.

---

## Stay on purpose (session discipline)

**Purpose only:** raise **clear %** at Monty’s target/risk with **breach 0%**, via Gravity + Shell + self-heal (measure with `prove_it`). Everything else is noise until that climbs.

**Coding habit (ponytail-style — no extra repo):**

```text
Minimal change. Reuse existing. No new deps unless required.
Do not invent frameworks, parallel pipelines, or “nice-to-have” tools.
If it does not move prove_it clear% or protect breach 0%, skip it.
```

**Out of scope unless Monty asks:** new agent plugins (e.g. full ponytail install), UI polish, unrelated indicators, obs expansion for curiosity, long design essays without a prove_it gate.

**In scope:** HANDOFF → diagnose (IRAC) → dials/practice/masks → prove_it → accept or reject.

---

## What this project is

Self-healing RL scalper (Momentum One):

1. Hit a **daily profit target %** without breaching a **daily risk floor %** (both are **runtime inputs** — no full retrain when they change).
2. Multi-timeframe **Gravity Framework** + Shell forever-masks.
3. Improve by **reward shaping + meta_tuner + self-heal**, not wiping neural weights.
4. Diagnostics (Mind Probe, Ghosts, IRAC) name the disease; meta **searches dials** — humans install tools, not forever freeze the cure.

**North star:** consistency (clear target, breach 0%, day after day).

**Product rule (2026-07-30):** the bot must **learn to learn**. Install **rulers + dials at zero + proposal loop**. Do **not** hardcode final reward answers as the permanent fix.

---

## PERFORMANCE_IS_POSSIBLE series — read order for next session

These files are **kept on purpose**. They are evidence that the ceiling is learned, not physical. **Do not delete.**

| File | Role | What next session must take from it |
|------|------|-------------------------------------|
| **[PERFORMANCE_IS_POSSIBLE.md](PERFORMANCE_IS_POSSIBLE.md)** | **Part 1 — founding brief** | Daily target is reachable on most XAUUSD curriculum days. Measure only with `prove_it`. Floor sacred; clear rate is the climb. Flea-jar: do not label days impossible. |
| **[PERFORMANCE_IS_POSSIBLE_PART2.html](PERFORMANCE_IS_POSSIBLE_PART2.html)** | **Part 2 — colorful record** (canonical `.md` twin may be missing on disk) | Win-lock forgot exit fee → banked ~2.9% on 3.0% target. Law fix + ladder → **1/90 → 21/90** clear, 0 breach. Artifact: `PROVEN_LIFT_2026-07-20.pt`. Possibility is learned skill, not luck. |
| **[PERFORMANCE_IS_POSSIBLE_PART3.md](PERFORMANCE_IS_POSSIBLE_PART3.md)** | **Part 3 — discharge papers** | Four “ticks”: pay-cliff, counterfeit applause, forbidden classroom, inherited flinch. Scoreboard was the disease; bot obeyed a lying scoreboard. Discharge: bank target, 0 breaches ever structural, food supply 90/90 winnable. Serial evidence under `doctrine/flea-jar/`. |
| **[PERFORMANCE_IS_POSSIBLE_PART4.md](PERFORMANCE_IS_POSSIBLE_PART4.md)** | **Part 4 — agreement evidence** | Independent families agreeing → **70–81%** hit rates (slots 80–83). Compose, don’t polish weak singles. Signal hit rate ≠ daily clear — still judge with `prove_it`. Holding while 80–83 fire + Gravity = Policy disease. |

**How next Grok should use them:**

1. Skim **Part 1** for doctrine (measure, floor, flea cure).  
2. Skim **Part 3** if diagnosing “why was it flat?” (scoreboard parasites).  
3. Open **Part 4** only when working on signal agents / SIGON / agreement slots.  
4. Open **Part 2** for the lift history and law-fix narrative.  
5. **Never** re-argue “is 3% possible?” — Parts 1–3 already closed that. Climb clear % under breach 0.

Companions: `doctrine/flea-jar/`, `doctrine/SUCCESS_LEDGER.md`, `CONSISTENCY_PLAN.md`.

---

## Scoreboard — current (2026-07-30)

**Champion brain:** `PROVEN_SPRINT_row04_clear24_2026-07-20`  
**Measure:** `python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5`  
**Obs:** 170 market cols, **frame_dim=1820** (`include_signal_agent_slots: false`)

| Metric | 2026-07-25 baseline | **2026-07-30 after CCI+SMA masks** |
|--------|---------------------|-------------------------------------|
| Clear % (target hit, no breach) | 21% | **24%** |
| Breach % | 0% | **0%** |
| Longest clear streak | 2 days | 2 days |
| Avg day | +0.17% | **+0.29%** |
| Median day | −0.40% | **−0.17%** |
| Green days | 46% | 46% |
| Best / worst | +6.46% / −3.47% | +6.19% / −3.42% |

**Interpretation:** Shell masks improved clear and avg day without trading the floor. Policy **logits** still short-biased under bull cont — masks block bad opens; they do **not** retrain the brain. Climb to ≥27% still needs self-heal dials + GPU practice.

Historical peak on lineage: ~27% clear (pre TF-set realign). Recover that, then climb.

---

## Progress made this chapter (2026-07-25 → 2026-07-30)

### A. Diagnosis (trend / side)

- Tested how well PROVEN **identifies HTF trend** (cont flags vs action mass).  
- **Feature layer OK** — `cont_buy` / `cont_sell` fire.  
- **Policy layer broken / asymmetric:** under bull cont, prefers **shorts** (`side_bias_bull ≈ −0.08`); under bear cont, shorts OK.  
- **Not** mainly “holds forever” — P(hold) ~0.10; wrong side > hold on match pulls.  
- Causes (simple): lagging set1 cont teacher; set3/set4 cont ≈ **0%**; rewards side-symmetric profit-only; op_head short lean.  
- **Mind Probe bug fixed:** was treating `Categorical` as logits → silent pure-hold → fake IRAC Policy. Now uses `.probs` + dim assert.

### B. Self-heal toolkit (learn to learn)

| Tool | What | Where |
|------|------|--------|
| Honest MRI | Categorical.probs, side_bias, wrong_side | `telemetry/mind_probe.py` |
| Tags at open | `with_trend`, `against_trend`, `pullback`, firm cont | `training/env.py`, `training/fastsim.py` |
| Dials default **0** | `w_with_trend_close`, `w_against_trend_close`, `w_quick_pull_close`, `w_setup_skip` | `configs/rewards.yaml`, RewardEngine, meta_tuner BOUNDS |
| IRAC classes | **WrongSide** / Policy / Perception / Shell | `scripts/self_heal_epoch.py`, `give_llm_what_it_needs.py`, `diagnose_day.py` |
| Skill memory | Dial procedures | `doctrine/policy_skill.md` |

Humans do **not** freeze final with-trend weights. Meta/self-heal **searches** dials; adopt only if prove_it clear ≥ baseline and breach 0.

### C. Shell masks (live law — 2026-07-30)

OR’d into `mask_buy_blocked` / `mask_sell_blocked` (obs dim unchanged):

1. **Legacy envelope** — 15m/30m/1h high/low (keep).  
2. **CCI dual** — both CCI(30) & CCI(100) > 0 and each > SMA(2)+2 on **5m OR 30m** → **no sells**; mirror below 0 → **no buys**.  
3. **Price SMA** — close vs SMA(4)+4 on **1m AND 15m** → block buy if both under, sell if both over.

Code: `features/engine.py`, docs: `configs/masks_shell.yaml`, `doctrine/STANDING_LAWS.md`.  
Tests: `tests/test_masks.py` (pass).

**prove_it after masks:** clear **21% → 24%**, breach **0%**, avg **+0.17% → +0.29%**.

### D. Config gotcha

```yaml
# configs/features.yaml
include_signal_agent_slots: false   # REQUIRED for PROVEN_* (1820)
# true → ~6820 frame_dim → shape crash loading PROVEN
```

After mask/engine changes: **delete** `artifacts/gpu_cache_*.npz` and rebuild on next prove_it/train.

---

## Disease map (current)

| Class | Evidence | Tool response |
|-------|----------|----------------|
| **WrongSide** | side_bias_bull &lt; 0; shorts under cont_buy | Shell masks (live) + search `w_with_trend_close` / `w_against_trend_close` |
| **Policy hold** | setup visible, hold | search pullback / setup_skip + GPU sprint |
| Not “impossible days” | swing bound / Part 1–3 | never zero-weight days by assumption |

---

## Gravity / TF sets (locked)

| Set | LTF | HTFs |
|-----|-----|------|
| A / set1 | 1m | 15m, 30m |
| B / set2 | 5m | 1h, 4h |
| C / set3 | 15m | 4h, 1d |
| set4 | 30m | 4h, 1d |

**Bread-and-butter:** LTF pull under dual HTF trend.  
Note: set3/set4 **cont** rates ~0% on current curriculum (higher stacks weak) — dual CCI masks partly substitute as firm regime.

---

## Key checkpoints

| Name | Role |
|------|------|
| `PROVEN_SPRINT_row04_clear24_2026-07-20` | Current climb baseline (**24% clear** post-masks) |
| `PROVEN_LIFT_2026-07-20` | First banking brain (Part 2) |
| `PROVEN_2x_2026-07-19` | Earlier ancestor |

Path: `artifacts/checkpoints/`.

---

## Commands — next session playbook

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth

# 0) Orient
# Read this HANDOFF + PERFORMANCE_IS_POSSIBLE.md (Part 1)
# If climbing clear %: CONSISTENCY_PLAN.md
# If signals/SIGON: PERFORMANCE_IS_POSSIBLE_PART4.md + HANDOFF_CONSISTENCY_TO_COLAB_SIGNALS.md

# 1) Preflight
python scripts\preflight_train.py

# 2) Baseline judge (only score that counts)
python scripts\prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
# Expect ~24% clear, 0% breach after masks (2026-07-30). If cache stale, delete gpu_cache and re-run.

# 3) Diagnose (honest MRI + IRAC class)
python scripts\give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
# Read artifacts/llm_curriculum/irac_*.json → class WrongSide | Policy | ...

# 4) Self-heal epoch (proposes dials; optional apply after gate)
python scripts\self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12
# With train + auto skill + reward search (GPU preferred for sprint):
# python scripts\self_heal_epoch.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 12 --sprint-minutes 120 --auto-accept-skill --apply-reward-nudge

# 5) Frontier practice
python scripts\consistency_sprint.py --minutes 600 --envs 256
python scripts\prove_it.py <new_brain> 3.0 3.5
# Accept only if clear ≥ previous AND breach == 0

# 6) Meta search (any-X consistency)
python scripts\meta_train.py --minutes 600

# 7) Unit tests after reward/mask/MRI edits
python tests\test_self_heal_mri.py
python tests\test_masks.py
python tests\test_rewards.py
```

**SIGON / signals ON (new lineage only):**

```text
configs/features.yaml → include_signal_agent_slots: true
Delete gpu_cache_*.npz
Train NEW brain — do NOT load PROVEN 1820 into 6820
See: HANDOFF_CONSISTENCY_TO_COLAB_SIGNALS.md, PERFORMANCE_IS_POSSIBLE_PART4.md
```

---

## Next work (priority for next Grok)

1. **Run self-heal / IRAC** with fixed MRI → confirm class **WrongSide** (or Policy).  
2. **Let dials move** via `--apply-reward-nudge` only after prove_it gate — do not hand-set permanent “final” weights as the product answer.  
3. **GPU consistency_sprint** hours under current Shell + rewards; prove_it every winner.  
4. Climb clear **24% → ≥27% → 35%** (see `CONSISTENCY_PLAN.md`); breach always **0%**.  
5. Optional: SIGON lineage with agreement slots 80–83 (Part 4) — separate obs_dim.  
6. Commit/push: engine masks, rewards dials, mind_probe, self_heal, HANDOFF, tests (if not already).

**Do not:**

- Relitigate “is performance possible?” → Parts 1–3.  
- Expand obs casually on PROVEN lineage.  
- Soften `w_death_penalty` for clear rate.  
- Delete PERFORMANCE_* / flea-jar / PROVEN checkpoints.

---

## Non-negotiable design rules

1. Target % and risk % are **runtime inputs** (`prove_it <brain> <tgt> <risk>`).  
2. Evolve via rewards + practice + meta — **no catastrophic weight wipe**.  
3. **Breach 0% sacred.**  
4. **Never delete** PERFORMANCE_IS_POSSIBLE*, SUCCESS_LEDGER, flea-jar, proven checkpoints.  
5. Flea-jar: nothing impossible; make it consistent.  
6. **Self-correcting:** instruments + dials + prove_it gate; humans build the school, not the forever exam answer.

---

## One-line summary (2026-07-30)

> prove_it @ 3.0/3.5 on PROVEN_SPRINT: **clear 24% (was 21%), breach 0%, avg +0.29%** after CCI+SMA Shell masks; MRI fixed; self-heal dials at 0 for WrongSide search; signals OFF (1820); next = IRAC self-heal + GPU sprint under PERFORMANCE doctrine Parts 1–4.

*Handoff written 2026-07-30 for continuous Grok sessions.*
