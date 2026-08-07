# Grow up like Mark — learn to learn for forward

**When:** 2026-08-06T09:16:45.367809+00:00  
**Child body (50d pack):** same=None mwt=None breach=None  
**Adult principle cycle decision:** **KEEP**  
**learn≠copy:** {'gate': 'learn_not_copy', 'passed': True, 'metrics': {'act_match': 1.0, 'topology_match': 1.0, 'role_map_match': 1.0, 'wait_match': 1.0, 'copying': False, 'freeze_copy': False}, 'reason': 'OK principles track with act'}


## How Mark learns (the process the student must internalize)

| # | Mark | Student learns | Code product |
|---|------|----------------|--------------|
| 1 | Reads HTF then LTF | Side only with force permission | force-gate / mark_align |
| 2 | Waits loaded, not frozen | phase before_first_fire + wait_loaded | shadow phase/event heads |
| 3 | Fires with size to goal | fire/add + size_bucket | shadow size head |
| 4 | Names the mistake class | early/late/thrash/miss — not the date | L2L path classes + memory |
| 5 | Ignores lying sensors | attention / who to trust | clue_gate |
| 6 | Undoes bad days | if pack dies, restore | KEEP/REJECT |
| 7 | Same mind on new days | no retrain at score | forward principle + score_forward_100d |

**You lead** with principles, HITL spines, and conscience.  
**He grows up** when forward works without fitting those days.


## Adult cycle result (principles — not day IDs)

| Split | Metrics |
|-------|---------|
| Practice | `{"n": 32, "act_match": 1.0, "topology_match": 1.0, "wait_match": 1.0, "tide_match": 1.0, "role_map_match": 1.0, "principle_acc": 1.0}` |
| **Forward (new tasks, no fit)** | `{"n": 32, "act_match": 1.0, "topology_match": 1.0, "wait_match": 1.0, "tide_match": 1.0, "role_map_match": 1.0, "principle_acc": 1.0}` |
| Cold forward (untrained baseline) | `{"n": 32, "act_match": 0.09375, "topology_match": 0.0, "wait_match": 0.03125, "tide_match": 0.3125, "role_map_match": 0.0, "principle_acc": 0.109375}` |
| Decision | KEEP |

**Meaning:**  
- If decision is KEEP/adopt → he internalized principles enough to **transfer** (adult step).  
- If REJECT → still a child/teen on principles; do **not** promote day-oracle spam.

## Path-class memory (mistake vocabulary)

- Rows: 0  
- Recent dominant classes: None  
- These are *how Mark names errors* so the same early-fire bug is one skill, not 15 day memos.

## Spine Shadow heads (day structure)

Already shipped in `mark_shadow_policy.py`:

| Head | Adult skill |
|------|-------------|
| phase | where am I in the day |
| event | wait / fire / add / hold |
| size | how hard |
| clue_gate | who to trust (meta) |

Train: `train_spine_shadow_full.py` · Path L2L: `learn_to_learn_path.py`

## Your role (Mark) vs his role (student)

| You lead | He must do alone later |
|----------|-------------------------|
| Principles, HITL spines, KEEP conscience | Decide under force-gate |
| Name error classes on chart | Apply class on **unseen** days |
| Refuse pack-killing updates | Self-limit thrash (HOLD skill) |
| Forward exam | No retrain at score time |

## Stages

| Stage | What he does | Forward |
|-------|--------------|---------|
| Child | Act-only copy practice days | Weak |
| Teen | Path classes + multi-head spine | Improving if pack safe |
| **Adult** | Principles + attention + KEEP internalized + **forward gate PASS** | **Goal** |

## Commands

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# 1) Adult principles (this script)
python lineages/adaptive_rl_brain_7_31_26/grow_up_mark_style.py

# 2) Path mistake vocabulary (meta boost)
python lineages/adaptive_rl_brain_7_31_26/learn_to_learn_path.py --max-rounds 6

# 3) Full day structure heads
python lineages/adaptive_rl_brain_7_31_26/train_spine_shadow_full.py --max-rounds 4

# 4) Honest adult exam — calendar holdout
python lineages/adaptive_rl_brain_7_31_26/score_forward_100d.py --n-days 100 --partial 20
```

## Why this is “like I learn”

You do not re-memorize every tick of every past day when the market is new.  
You re-use **principles**, **mistake types**, and **attention**.  
That is learn-to-learn. That is what survives **forward testing**.

Day-oracle BC alone raises practice then dies forward and kills the pack — we already measured that (35→32).  
Adult path: principles + classes + clue_gate + conscience + forward gate.
