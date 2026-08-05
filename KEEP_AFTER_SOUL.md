# KEEP — eyes on the board (agents + obs)

**Update 2026-08-05 (Fable full clone):**  
Agents and context obs are **on the policy board now** (`full_obs` → 168-dim).  
They remain **sensors / pattern clues** — **Mark soul still owns side.**

**Never delete** this inventory.  
Bridge map: `00_MAP_OF_THE_HOUSE.md`  
Full clone protocol: `lineages/.../FABLE_FULL_CLONE_INSTRUCTIONS.md`  
Soul status: `lineages/.../MARK_SOUL_TRANSFER.md`

---

## How they enter the clone

| Mode | Dim | Contents |
|------|----:|----------|
| Legacy Channel1 | 32 | sets + structure + progress/danger |
| **`full_obs=True`** | **168** | + doctrine + majority + **92 agent votes** + self/goal |

Train: `train_mark_clone_bc.py --full-obs`  
Ckpt: `mark_clone_full_obs_v1.pt`

---

## Why soul still leads

Agents without Mark soul = thrash with better sensors.  
Full board + Mark teacher (soul plans) + HITL chart checks via **MARK HERE!.lnk**.

---

## A) Signal agents — KEEP ALL

| What | Path |
|------|------|
| Registry SSOT | `configs/signal_slots.yaml` (92 filled / 500 capacity) |
| Index | `code/signals/00_ALL_92_AGENTS.md` |
| Code | `code/signals/` (`encode.py` + family modules) |
| Manager | `scripts/manage_signal_slots.py` |
| Accuracy schema | `configs/signal_accuracy_schema.yaml` |
| Lineage majority helper | `lineages/.../signal_majority.py` |

**After soul:** majority / slot votes become **confirm**, not the policy soul.  
Mark (force→regime→velocity) still owns side.

---

## B) Observation / features — KEEP ALL

| What | Path |
|------|------|
| Production feature engine | `code/features/` (`engine.py`, indicators, sets) |
| Sets lock config | `configs/features.yaml` · `configs/timeframes.yaml` |
| Lineage Channel1 obs (32) | `lineages/.../perception/observation.py` |
| Full perception package | `lineages/.../perception/` |
| Goals / progress slots | `configs/goals.yaml` (runtime target/risk) |

**Dual lock reminder**

| `sets_lock` | Use |
|-------------|-----|
| `proven_legacy` | PROVEN / T1 only |
| `mark` | New Mark trains only — do not warm-start PROVEN |

---

## C) Telemetry / mind probe — KEEP ALL

| What | Path |
|------|------|
| Mind probe | `code/telemetry/mind_probe.py` |
| Ghosts / tracer / regime language | `code/telemetry/` |
| Logging | `code/telemetry/logging_setup.py` |

Used to **measure** whether policy acts like Mark — not to replace Mark.

---

## D) Production RL stack — KEEP (T1, separate)

| What | Path |
|------|------|
| Brain / PPO / FastSim | `code/training/` |
| Meta tuner | `code/training/meta_tuner.py` |
| Rewards dials | `configs/rewards.yaml` |
| prove_it | `scripts/prove_it.py` · `USE/1_prove.bat` |
| PROVEN weights | `models/PROVEN_*.pt` |

Do **not** load PROVEN into Mark-obs.  
Do **not** promote lineage over PROVEN without Mark’s explicit order.

---

## E) Personas / agent docs — KEEP

| What | Path |
|------|------|
| Multi-pair tutor persona | `lineages/.../agents/MULTI_PAIR_TUTOR_PERSONA.md` |
| Reference agent prompts | `references/agents/` · `references/prompts/` |
| ARMY Mark soul files | via `SOUL_MATCH.md` (ARMY owns personality) |

---

## When to unlock (definition of done for step 1)

Unlock agents/obs into T3 policy only when **all** hold:

| # | Gate |
|---|------|
| 1 | Mark soul teacher ≥ 9/10 on random-pair 10d, breach 0 |
| 2 | Pure policy ≥ 9/10 same pack, breach 0 |
| 3 | Mean entries near Mark (not 14–18 thrash on soft days) |
| 4 | BC dir_match ≥ 0.85 on soul-plan labels |
| 5 | Hard pair forward clear not collapsed vs baseline |

Then: wire slots as **confirm / refuse**, re-score, keep soul dials.

---

## Explicitly NOT delete when “organizing”

- Anything under `code/signals/`
- Anything under `code/telemetry/`
- Anything under `code/features/`
- `configs/signal_slots.yaml`
- `lineages/.../perception/`
- PROVEN models
- flea-jar doctrine

Archive **duplicates** only (`_archive/`, `references/doctrine/_from_root/` mirrors) — never the live agent/obs trees.
