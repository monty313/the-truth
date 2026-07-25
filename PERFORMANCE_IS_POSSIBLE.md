# PERFORMANCE IS POSSIBLE

> Kept on purpose. Not required for daily train — see DO_THIS.md for commands.

This file and its PART2 / PART3 companions document that the daily target is reachable under measured swing bounds. Do **not** delete these files when organizing the repo.

## How this fits the organized repo

| File | Role |
|------|------|
| DO_THIS.md | What you run |
| MAP.md | Where folders are |
| UPDATE_RULES.md | How to edit without mess |
| **PERFORMANCE_IS_POSSIBLE*.md** | **Evidence / narrative that the goal is reachable** |

## Training

```bash
python scripts/preflight_train.py
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <brain> 3.0 3.5
```

Only `prove_it` clear % and breach % count as the score.

See also: PERFORMANCE_IS_POSSIBLE_PART2.md, PART3.md, and the HTML companion.
