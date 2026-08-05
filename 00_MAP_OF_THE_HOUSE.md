# MAP OF THE HOUSE — the-truth

**Owner:** Mark Montgomery Jr.  
**Method:** Fable 5 (structure) × Mark mind (soul)  
**Date:** 2026-08-05  
**Read this if you feel lost.** One page. No second opinion.

---

## Who is talking

| Thing | What it is | What it is NOT |
|-------|------------|----------------|
| **Mark / MarkOS** | You. Second brain via **`MARK HERE!.lnk`** | A second personality living only in this folder |
| **Fable 5** | How work is done (ask → evidence → act → verify) | The trader / the soul |
| **pt5 basic knowledge** | Trading **principles** (HTF side, LTF time, regime, capital, slingshot) | Indicator names, PROVEN weights, 92 agents |

Soul bridge: **`SOUL_MATCH.md`**  
Principles full text: **`references/doctrine/llm_basic_thinking/pack/pt5__basic_knowledge.txt`**

---

## One mission (never invent another)

**`GOAL.md`**

One bot that hits **whatever target% / risk% you type** — **no retrain to switch pairs**.  
Clear % **up**. Breach % **0**.

---

## Two brains only

| Brain | Folder | Touch? |
|-------|--------|--------|
| **PROVEN** (champion yardstick) | `models/` | **Never overwrite** |
| **Mark-soul lineage** (sandbox) | `lineages/adaptive_rl_brain_7_31_26/` | Safe experiments |

---

## Four tracks — always name which one

| ID | Name | Where | Scoreboard | Status (2026-08-05) |
|----|------|-------|------------|---------------------|
| **T1** | PROVEN production | `models/` + `code/training/` + `scripts/prove_it.py` | `prove_it` clear/breach | Champion ~24% @ 3.0/3.5 — **frozen yardstick** |
| **T2** | Multi-pair tutor | lineage shell + heuristic | `score_ten_pairs` | Claim multi-pair win (heuristic path) |
| **T3** | **Mark soul → policy** | mark_doctrine + soul plans + BC | 10d Mark vs policy | **Teacher 10/10 · policy ~8/10 · active** |
| **T4** | Channel1 pure RL sandbox | old curriculum / hold probes | greedy hold | **Parked** — not champion |

**If you say “the bot” without T1–T4, you will get confused.**

---

## Work order (long-term consistency)

```
1) SOUL IN POLICY (T3)          ← DONE (teacher 10/10; policy climbing)
   - Mark sees chart (pt5 + sets law)
   - Goal-relative size + force-aligned adds
   - Soul-plan BC → policy matches Mark

2) FULL EYES ON CLONE           ← ACTIVE (Fable full clone)
   - 168-dim obs: sets + doctrine + 92 agents + self
   - Meta RL policy = Mark clone (attention over full board)
   - Chart HITL via MARK HERE!.lnk (mark_chart_hitl.py)
   Protocol: lineages/.../FABLE_FULL_CLONE_INSTRUCTIONS.md

3) LONG CONSISTENCY
   - More BC + thrash penalty + Mark chart audits
   - Award streak under random pairs, no retrain
   - Promote only if beats gates (never silent PROVEN overwrite)
```

**Mark owns side. Agents are pattern clues on the board.**

Eyes inventory: **`KEEP_AFTER_SOUL.md`** · HITL: **`mark_chart_hitl.py`**

---

## Doors (only these when lost)

| # | Open | Job |
|---|------|-----|
| 1 | **This file** | Where am I |
| 2 | **`GOAL.md`** | Mission |
| 3 | **`MARK HERE!.lnk`** | Talk to Mark |
| 4 | **`USE/`** | Run buttons |
| 5 | **`models/00_CHAMPION.md`** | PROVEN identity |
| 6 | **T3 door** | `lineages/adaptive_rl_brain_7_31_26/00_START_HERE.md` |
| 7 | **Soul status** | `lineages/.../MARK_SOUL_TRANSFER.md` |
| 8 | **pt5 laws** | `references/doctrine/llm_basic_thinking/00_INDEX.md` |
| 9 | **Signals (later)** | `code/signals/00_ALL_92_AGENTS.md` + `configs/signal_slots.yaml` |

Older handoffs (history only): `HANDOFF_2026-07-31.md`, lineage `00_7_31_26_HANDOFF.md`  
**Current sit-down:** `HANDOFF_2026-08-05.md`

---

## pt5 → code (principles only)

| Law | One line | Code (T3) |
|-----|----------|-----------|
| 1 Dominant trends | HTF side; LTF time only | `mark_doctrine` · `MARK_SETS_LAW` · `sets.py` |
| 2 Breath vs launch | Pullback = load; resume = fire | `mark_doctrine` play states |
| 3 Regime | Wrong playbook = damage | doctrine regime · shell flat |
| 4 Capital | Floor sacred; size from remaining day | `equity_day` shell + soul size |
| 5 Speed vs weight | Force > velocity | force gate before entry |

Full map: `references/doctrine/llm_basic_thinking/HOW_PT5_MAPS_HERE.md`

---

## House layout (plain)

| Folder | English |
|--------|---------|
| **USE/** | Buttons |
| **models/** | PROVEN only |
| **lineages/** | Mark soul / experiments |
| **code/** | Production Python (training, **signals**, features, telemetry) |
| **configs/** | YAML (goals, rewards, **signal_slots**, sets) |
| **data/** | Prices |
| **scripts/** | CLI (`prove_it`, …) |
| **references/** | Doctrine forever (flea-jar, pt5, plans) |
| **outputs/** | Run junk |
| **_archive/** | Old copies — do not develop |
| **mark_here/** | MarkOS launcher |

Real package root is **`code/`** (not a root `training/` folder — ignore stale docs that say that).

---

## Forbidden (confusion killers)

| Do not | Why |
|--------|-----|
| Overwrite PROVEN | Yardstick dies |
| Mix T1 weights into Mark-obs without retrain | Dim / sets mismatch |
| Call multi-pair tutor win “pure RL win” | Different track |
| Delete 92 agents or telemetry “to clean up” | Needed **after** soul |
| Trail + cushion + scale-in package | IRAC killed multi-pair |
| Invent a second Mark personality in this folder | One soul: ARMY MarkOS |

---

## Right now (2026-08-05)

| Meter | Number |
|-------|-------:|
| Mark soul plans (10d seed7) | **10/10** clear |
| Policy after soul BC | **~8/10** clear |
| Full obs board | **168-dim** (sets+doctrine+92 agents+self) |
| Gap | Soft thrash + need full-obs BC + Mark chart HITL |

**Next (full Mark clone = meta policy):**

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --full-obs --epochs 40 --max-train-days 30 --hidden 128
python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py --seed 7 --start-idx 40 --full-obs
python lineages/adaptive_rl_brain_7_31_26/mark_chart_hitl.py --seed 7 --start-idx 40 --full-obs
# Double-click MARK HERE!.lnk → open checkpoints/mark_chart_hitl/HITL__latest.md + chart
```
