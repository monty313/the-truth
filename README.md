# Momentum One

Self-healing RL scalper. You set daily **target %** and **floor %**.  
The bot trains to **hit the target often** without breaking the floor.

---

## Confused? Open only this

### → [DO_THIS.md](DO_THIS.md)

Three steps: setup → train / self-heal → measure with `prove_it`.

---

## Three files (daily)

| File | Use |
|------|-----|
| [DO_THIS.md](DO_THIS.md) | Commands to run |
| [MAP.md](MAP.md) | Where every folder is |
| [UPDATE_RULES.md](UPDATE_RULES.md) | How to edit without mess |

## Three files (mindset / proof)

| File | Use |
|------|-----|
| [doctrine/SUCCESS_LEDGER.md](doctrine/SUCCESS_LEDGER.md) | Wins already proven |
| [doctrine/flea-jar/THE_FLEA_CURE.md](doctrine/flea-jar/THE_FLEA_CURE.md) | Nothing is impossible |
| [PERFORMANCE_IS_POSSIBLE.md](PERFORMANCE_IS_POSSIBLE.md) | Performance proof (never delete) |

Everything else is support. Skip until you need it.

---

## Folders (one line each)

| Folder | What |
|--------|------|
| `scripts/` | Commands you type |
| `configs/` | Numbers (goals, rewards, TFs) |
| `training/` | RL brain + meta_tuner |
| `features/` | Indicators + Gravity sets |
| `telemetry/` | Mind probe + ghosts |
| `doctrine/` | Laws + CMO + flea cure |
| `data/` | Price CSVs |
| `artifacts/` | Saved brains + epoch logs |
| `docs/` | Old notes + ADRs (skip daily) |
| `GPU_EDITION/` | Notebooks for cloud GPU |

```bash
python scripts/preflight_train.py
```
