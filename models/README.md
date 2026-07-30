# models — trained brains

Cookiecutter **models/** folder.

| File | Role |
|------|------|
| `PROVEN_SPRINT_row04_clear24_2026-07-20.pt` | Current champion |
| `PROVEN_LIFT_2026-07-20.pt` | Historical lift proof |
| `PROVEN_2x_2026-07-19.pt` | Early proof |
| `history/` | Named sprint / SIGON history |

Load with:

```python
from inference.loader import load_brain
brain, meta = load_brain("PROVEN_SPRINT_row04_clear24_2026-07-20")
```

Legacy path `artifacts/checkpoints/` still works as a fallback.
