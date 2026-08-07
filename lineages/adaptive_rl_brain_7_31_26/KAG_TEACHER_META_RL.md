# KAG Teachers → Meta-RL (the-truth)

Additive. Does not rewrite `perception/sets.py`, PROVEN weights, or five laws.

| Piece | Path |
|-------|------|
| Doctrine pack | `references/doctrine/kag_mark_doctrine/` |
| Teachers code | `lineages/adaptive_rl_brain_7_31_26/kag_teachers/` |
| Sets law pin | `perception/sets.py` · `MARK_SETS_LAW.md` |
| Tests | `tests/lineages/adaptive_rl_brain_7_31_26/test_kag_teachers.py` |
| ARMY twin | `markos_core.kag_mark` (monty313/ARMY) |

## One bar

```python
from lineages.adaptive_rl_brain_7_31_26.kag_teachers import (
    bread_and_butter_obs,
    teach_one_bar,
    StudentAuxHeads,
    train_step_one_bar,
)

taught = teach_one_bar(bread_and_butter_obs())
lesson = taught["primary_lesson"]
student = StudentAuxHeads(
    act=lesson["act"],
    topology=lesson["topology"],
    role_map={s["name"]: s["role"] for s in lesson["sensors"]},
)
print(train_step_one_bar(lesson, student))
```

## Feel

> I don’t need to have trained on this line. I need mass vs speed, which clock, with or against, and whether the day still allows fire.
