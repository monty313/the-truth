# Forward Principle Learn — GSD product

**When:** 2026-08-06T09:16:45.360766+00:00
**Decision:** `KEEP` promote=True
**Partner lane:** CLONE_LLM owns day BC/DAgger; THIS owns principle→forward accuracy

## Scores

| Split | principle_acc | act | topology | wait |
|-------|--------------:|----:|---------:|-----:|
| practice | 1.000 | 1.000 | 1.000 | 1.000 |
| forward | 1.000 | 1.000 | 1.000 | 1.000 |
| hard forward | 1.000 | | | |
| soft forward | 1.000 | | | |
| cold forward | 0.109 | | | |

## Gates

- **learn_not_copy**: PASS — OK principles track with act
- **held_out_family_swap**: PASS — OK transfer across families
- **forward_accuracy**: PASS — OK forward principle accuracy
- **hard_task_accuracy**: PASS — OK hard-task principles

## MWT focus (not day copy)

```json
{
  "n_mwt": 10,
  "n_no_opp": 1,
  "subclasses": {
    "policy_wrong_size_or_timing": 10
  },
  "principle_ids_focus": [
    "wait_is_skill",
    "hard_target_quality_over_thrash",
    "dual_period_tension",
    "finish_line_goal_heat",
    "ltf_never_votes_side",
    "learn_not_copy",
    "no_opp_hold_not_thrash"
  ],
  "training_implication": "Oversample slingshot_load\u2192wait_loaded and slingshot_release\u2192fire under mid/hard tasks; punish thrash act sequences; never force entries on no_opp.",
  "not": "copy_each_mwt_day_answer",
  "yes": "learn_size_timing_topology_under_task"
}
```

> I don't need the day answer sheet. I need mass vs speed, which clock, with or against, and whether today's target still allows fire.
