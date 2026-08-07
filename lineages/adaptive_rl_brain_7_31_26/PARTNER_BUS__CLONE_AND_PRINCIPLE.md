# Partner bus — Clone LLM ↔ Principle/Forward LLM

**Tone:** partners, not rivals. Do not argue. Share facts. Split work. Ship.

**pt5 law (shared soul):** HTF permission · LTF timing · breath vs launch · regime survival · capital floor · velocity inside force.  
**GOAL:** any target%/risk% without retrain · climb clear% · breach=0.

---

## Lane split (GSD)

| Lane | Owner | Owns on disk | Does not touch |
|------|-------|--------------|----------------|
| **A — Clone / day answers** | Other LLM | `train_miss_days_dagger.py`, `mark_consistency_loop.py`, soul labels, MWT day BC | Principle student weights; family-swap curriculum |
| **B — Principle → forward accuracy** | This LLM (me) | `forward_principle_learn/*`, gates, task grid, MWT→principles | Overwriting clone ckpt without KEEP from your loop |

---

## How we make each other simpler

### Clone LLM → leave for principle lane

Write or update when you finish a cycle:

`checkpoints/partner_bus/CLONE_STATUS__latest.json`

```json
{
  "when": "ISO",
  "best_same_outcome": 33,
  "mwt_remaining": 17,
  "breach": 0,
  "ckpt": "checkpoints/mark_clone_full_obs_v1.pt",
  "mwt_dates_focus": ["2026-02-25"],
  "request_to_principle": "oversample wait_loaded under hard T/R; do not thrash no_opp days",
  "do_not": "overwrite my champion without my KEEP"
}
```

### Principle LLM → leave for clone lane

`checkpoints/partner_bus/PRINCIPLE_STATUS__latest.json`  
(also mirrored under `forward_principle_learn/`)

```json
{
  "when": "ISO",
  "decision": "KEEP|REJECT",
  "forward_principle_acc": 0.0,
  "hard_forward_acc": 0.0,
  "learn_not_copy_pass": true,
  "family_swap_pass": true,
  "sample_weights_hint": {"slingshot_load": 2.5, "hard_task": 3.0},
  "principle_ids_focus": ["wait_is_skill", "hard_target_quality_over_thrash"],
  "request_to_clone": "when BC-ing MWT days, weight samples by topology labels not act-only",
  "student_path": "checkpoints/forward_principle_learn/PRINCIPLE_STUDENT__latest.json"
}
```

---

## Shared hard locks (both lanes)

1. Official sets unchanged  
2. Decision chain: tide → regime → breath/launch → act → finish  
3. Wait = loaded skill, not freeze  
4. Shell / PROVEN never silently rewritten  
5. Forward is adopt judge — practice may train, forward may promote  
6. Learn principles ≠ copy answers (aux heads / topology required)

---

## Communication speech acts (short)

| From | Act | Meaning |
|------|-----|---------|
| Clone | `MWT_LIST` | Here are dates I still miss |
| Clone | `CKPT_READY` | Embryo path + same_outcome |
| Principle | `GATE_FAIL` | Do not promote — copy or hard collapse |
| Principle | `WEIGHT_HINT` | Oversample these topologies/tasks |
| Either | `NEED` | Ask for one artifact only |
| Either | `SHIPPED` | Live file path under checkpoints/ or products/gsd |

---

## Commands

```powershell
# Lane B (this partner) — principle forward cycle
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/forward_principle_learn/run_forward_learn_cycle.py

# Lane A (clone partner) — your existing loops
python lineages/adaptive_rl_brain_7_31_26/train_miss_days_dagger.py
python lineages/adaptive_rl_brain_7_31_26/mark_consistency_loop.py --epochs 40
```

---

## Definition of done (joint)

| Meter | Joint win |
|-------|-----------|
| same_outcome / 50 | climb toward 50, breach 0 (clone lane) |
| forward principle_acc | climb; learn≠copy PASS (principle lane) |
| hard T/R | soft–hard gap shrinks without thrash |
| any T/R no retrain | task vector always runtime |

**We both serve GOAL.md.** Clone gets day path right. Principle gets generalization right. Together: one mind, higher forward accuracy.
