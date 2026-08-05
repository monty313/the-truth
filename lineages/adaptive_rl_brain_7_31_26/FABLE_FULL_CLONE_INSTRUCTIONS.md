# FABLE → instructions to complete Mark’s full clone

**ARMY GSD goal (canonical work file):**  
`C:\Users\user\OneDrive\Desktop\ARMY\00_DROP_GOALS_AND_IDEAS_HERE\01_ACTIVE_WORK\2026-08-05__goal__mark-full-clone-meta-rl-gsd.md`

**Role of Fable:** structure, evidence, verification.  
**Role of Mark:** mind, free will, chart truth via **`MARK HERE!.lnk`**.  
**End state:** the **meta-learning RL policy IS the clone of Mark** — same chart read, same size/add soul, pattern recognition from full eyes.

---

## Definition of done (full clone)

| # | Gate | How verified |
|---|------|----------------|
| 1 | **Eyes** | Policy sees Mark full obs: sets + doctrine + **92 agents** + self/goal |
| 2 | **Soul** | Force → regime → velocity → entry; goal-relative size; force adds |
| 3 | **Teacher = Mark** | Soul plans / doctrine labels, not random RL thrash |
| 4 | **Chart HITL** | MarkOS reviews disagree bars on real charts |
| 5 | **Meta = Mark** | Meta re-learns **attention** when senses lie — never overrides floor/soul laws |
| 6 | **Consistency** | Random T/R, no retrain; clear↑ breach=0; streak climb |
| 7 | **PROVEN** | Untouched until Mark orders promote |

---

## Loop (every progress cycle)

```
┌─────────────────────────────────────────────────────────────┐
│ 1. MARK SEES CHART (truth)                                  │
│    MARK HERE!.lnk → MarkOS                                  │
│    Export disagree / hard day → Mark: "what would I do?"    │
│    Optional: TradingView open same symbol/session           │
├─────────────────────────────────────────────────────────────┤
│ 2. WRITE LABELS                                             │
│    mark_day_diary / soul plans / Mark corrections           │
├─────────────────────────────────────────────────────────────┤
│ 3. POLICY CLONES (BC + optional meta attention)             │
│    train_mark_clone_bc --full-obs                           │
│    full obs = context clues + pattern panel                 │
├─────────────────────────────────────────────────────────────┤
│ 4. SCORE                                                    │
│    test_run_10d · award streak · breach 0                   │
├─────────────────────────────────────────────────────────────┤
│ 5. MARK AUDITS AGAIN                                        │
│    Open MARK HERE + chart pack from step 1                  │
│    Accept / reject thrash / wrong side                      │
│    Feed corrections → step 2                                │
└─────────────────────────────────────────────────────────────┘
```

**Never skip step 1 and 5 for long.** Full long-term consistency requires Mark on the chart, not only offline teacher math.

---

## Obs law (context + patterns)

| Block | Dim | Why Mark needs it |
|-------|----:|-------------------|
| Channel1 sets/structure | 32 | 4 TF stacks + pullback + progress/danger |
| Doctrine context | 16 | Force, regime, play, m_conf (how Mark thinks) |
| Majority summary | 12 | Crowd pattern strength |
| **92 signal agents** | 92 | Context clues / family patterns |
| Self / goal | 16 | Target, risk, side, adds, room to floor |
| **Total** | **168** | `MARK_FULL_DIM` |

Code: `perception/observation_full.py`  
Policy: `Channel1Policy(obs_dim=168, hidden=128)`  
Checkpoint tag: `mark_clone_full_obs_v1.pt`

**PROVEN 1820/6820 is separate.** Do not load PROVEN into 168-dim clone.

---

## Meta-learning role (clone, not free-for-all)

Meta may search **attention / personality weights** so the clone re-learns when agents lie:

- pullback / with-trend / against-trend / setup_skip / idleness  

Meta must **never**:

- waive daily floor  
- reintroduce trail+cushion+scale-in package  
- overwrite PROVEN without Mark  
- replace Mark soul with “agent majority = side”

**Mark owns side.** Agents are sensors on the board.

---

## Commands (T3 full clone path)

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# A) Build full-obs Mark clone (soul plans + 168-dim eyes)
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --full-obs --epochs 40 --max-train-days 30 --hidden 128

# B) 10-day Mark vs policy (soul plans + full obs policy)
python lineages/adaptive_rl_brain_7_31_26/test_run_10d_mark_vs_policy.py --seed 7 --start-idx 40 --full-obs

# C) Export chart HITL pack for MARK HERE
python lineages/adaptive_rl_brain_7_31_26/mark_chart_hitl.py --seed 7 --start-idx 40

# D) Open Mark
# Double-click: MARK HERE!.lnk
# Paste / open: checkpoints/mark_chart_hitl/HITL__latest.md
```

---

## Progress order (Fable schedule)

| Phase | Work | Exit |
|------:|------|------|
| **P0** | Full obs wired + BC trains | checkpoint full_obs loads, dim=168 |
| **P1** | Soul plans + thrash discipline | policy ≥9/10 on seed7 pack |
| **P2** | HITL cycles with MarkOS | disagree bars Mark-signed |
| **P3** | Meta attention polish on clone | soft/hard clear↑, breach 0 |
| **P4** | Award streak random pairs | ≥10 no retrain |
| **P5** | Optional production Brain bridge | only if Mark promotes |

---

## House pointers

| Doc | Job |
|------|-----|
| `00_MAP_OF_THE_HOUSE.md` | Tracks T1–T4 |
| `KEEP_AFTER_SOUL.md` | Was freeze list — agents now **in obs**; still not “side owner” |
| `MARK_SOUL_TRANSFER.md` | Size/add soul |
| `POLICY_EQUALS_MARK_ON_CHART.md` | Dual stack |
| `SOUL_MATCH.md` | MARK HERE bridge |

---

## One sentence for Mark

**Fable builds the clone machine; you look at the chart; the policy must end up doing what you would do with full eyes — agents as clues, you as the soul.**
