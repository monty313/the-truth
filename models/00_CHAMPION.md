# CHAMPION BRAIN

**This is the active brain. Start here in `models/`.**

**Mission (from GOAL.md):** one bot that solves for **any** target % / risk % you type in — **without retrain**.  
Numbers below are the **yardstick** pair (3.0 / 3.5). Re-score at other pairs anytime (same file).

---

## Current

| Field | Value |
|-------|--------|
| **Name** | `PROVEN_SPRINT_row04_clear24_2026-07-20` |
| **File** | `models/PROVEN_SPRINT_row04_clear24_2026-07-20.pt` |
| **Clear @ 3.0/3.5** | ~24% |
| **Breach @ 3.0/3.5** | 0% |
| **Date** | 2026-07-30 |

---

## Score it (any target / risk)

```text
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 2.5 2.5
```

Or: **`USE/1_prove.bat`** (change numbers if you want a different pair)

---

## Other proof brains (do not delete)

| File | Note |
|------|------|
| `PROVEN_LIFT_2026-07-20.pt` | Lift history |
| `PROVEN_2x_2026-07-19.pt` | Early proof |
| `history/` | Old sprint copies |

---

## Rebuild from blank (IRAC)

**`HOW_TO_ACHIEVE_GOAL_FROM_BLANK_IRAC.md`** — what works / fails; steps to get multi-pair prove_it (breach 0) again from a new policy.

**GOAL playbook from multi-pair IRAC** (equity shell KEEP/REJECT, checklist, score commands):  
`references/plans/GOAL_FROM_TEN_PAIR_IRAC.md`

## Rule for AI

When a new brain **beats** champion on `prove_it` (higher clear, breach still 0):

1. Save under `models/` with a clear name  
2. Update **this file**  
3. Update **GOAL.md** scoreboard  
4. Note win in `references/doctrine/SUCCESS_LEDGER.md`
