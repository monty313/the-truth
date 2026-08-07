# kag_teachers — Teacher 1 home (+ Teacher 2 bridge)

**You are Teacher 1 (trade / policy=Mark).**  
ARMY Reason Teacher is **Teacher 2** — partner, not competitor.

## This folder (lineage)

| File | Role |
|------|------|
| `teachers.py` / `decision_chain.py` / `lesson.py` | Teacher 1: principle_application acts for BC |
| `student_interface.py` / `novel_protocol.py` | Student gates (learn≠copy) |
| `reason_teacher_bridge.py` | Optional call into ARMY Teacher 2 |
| `full_obs_reason_hook.py` | Collect 168-obs + multi-head labels for BC |

## Checkpoints (do not clutter root)

| Path | Role |
|------|------|
| `../checkpoints/partner_bus/` | messages to/from ARMY partner lane |
| `../checkpoints/full_obs_reason/` | full-obs reason labels for multi-head BC |
| `../checkpoints/mark_clone_full_obs_v1.pt` | **your** embryo |
| `../checkpoints/forward_principle_learn/` | principle student (lane B) |

## ARMY house (Teacher 2 products)

```
ARMY/01_SYSTEM/outputs/army/teachers/
  partner/           shared alignment + letter to you
  teacher2_reason/   their reason products
  learn_to_learn/    labels you may consume
  teacher1_trade/    gifts staged for you
```

## Hard locks (shared)

Sets · chain · wait=loaded · learn≠copy · agents=sensors · **never overwrite PROVEN**

```powershell
# Your BC + optional reason labels
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --full-obs --reason-labels
```
