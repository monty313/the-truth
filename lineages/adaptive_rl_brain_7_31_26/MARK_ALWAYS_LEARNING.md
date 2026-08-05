# MARK ALWAYS LEARNING — Mark first, then policy

**Soul:** MARK HERE looks at the **chart (price path) before the policy acts as authority**.  
Then he **updates the policy** so it moves like him.  
Then he **harvests** other trained brains for benefits he would keep.  
He never stops learning. PROVEN is yardstick only — never silent overwrite.

---

## Loop (non-negotiable order)

```
1. MARK SEES THE CHART
   - Load day price (curriculum M1)
   - MARK SETS LAW: all 4 stacks
   - LTF = pullback / continuation / add
   - HTF×2 = trend confirm
   - Doctrine teacher decides (what Mark would do)
   - Write mark_labels for the day

2. POLICY ACTS (student)
   - Same obs, pure greedy weights
   - Record policy actions

3. COMPARE & CORRECT
   - Disagree bars = learning targets
   - BC / regret only on Mark-labeled truth
   - Never “policy wins” without Mark sign-off

4. HARVEST OTHER POLICIES
   - multi_pair_consistent, channel1_*, other lineage ckpts
   - Keep only behaviors that:
     a) match Mark labels on the day, OR
     b) improve award/clear without breach AND Mark would allow (teacher or shell law)
   - Distill agree-with-Mark bars into embryo weights
   - Reject thrash / anti-HTF / trail packages

5. META (always)
   - target% / risk% runtime inputs — no retrain to switch pair
   - Log streak under random pairs
   - Next day: repeat
```

---

## Roles

| Role | Who | Job |
|------|-----|-----|
| Chart eyes | Mark doctrine + sets law | Sees first |
| Student | `mark_clone_doctrine_v1.pt` | Imitates Mark |
| Shell | GoalEquityDay | Bank / floor / heat |
| Museum | Other `.pt` under lineage | Harvest candidates |
| Yardstick | PROVEN (models/) | Compare only — do not overwrite |

---

## Commands

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# One full Mark-first cycle (see chart → update policy → harvest)
python lineages/adaptive_rl_brain_7_31_26/mark_first_learn_cycle.py

# Specific days + optional more epochs
python lineages/adaptive_rl_brain_7_31_26/mark_first_learn_cycle.py --dates 2026-04-02,2026-04-01 --epochs 20
```

Artifacts:
- `checkpoints/mark_first_labels/` — Mark saw first
- `checkpoints/mark_first_cycle_report.json` — diffs + harvest + streak
- `checkpoints/mark_clone_doctrine_v1.pt` — updated embryo

---

## Always-learning test

Mark is learning only if **each cycle** either:
- reduces disagree rate vs prior cycle, OR
- improves award streak under random T/R, OR
- harvests a measured benefit that Mark (teacher) still allows  

If none of these, the cycle is thrash — change one thing, re-run.
