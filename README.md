# Momentum One

Trading bot. You set a daily profit target and a max loss. It tries to hit the target without breaking the floor.

---

## If you are confused, read only this

1. Open **[DO_THIS.md](DO_THIS.md)**
2. Run the commands in order
3. The only score that matters is what `prove_it` prints

---

## Three files that matter

| File | When to open it |
|------|-----------------|
| [DO_THIS.md](DO_THIS.md) | Every time you train |
| [MAP.md](MAP.md) | When you forget where a folder is |
| [UPDATE_RULES.md](UPDATE_RULES.md) | When you change code or configs |

Everything else is support. You can ignore it until you need it.

---

## Folders in one line each

| Folder | In one line |
|--------|-------------|
| `scripts/` | Commands you type |
| `configs/` | Numbers (target, risk, rewards) |
| `training/` | The learning code + meta_tuner |
| `features/` | Chart indicators |
| `telemetry/` | “What was the bot thinking?” |
| `doctrine/` | Rules written down |
| `data/` | Price history |
| `artifacts/` | Saved brains |
| `docs/history/` | Old notes — skip these |

```bash
python scripts/preflight_train.py
```
