# Momentum One

Self-healing RL trading bot. You set daily target % and risk %.  
It aims to hit the target without breaching the floor.

---

## Confused? Read only this

1. **[START_FROM_TODAY.md](START_FROM_TODAY.md)** — this chapter  
2. **[DO_THIS.md](DO_THIS.md)** — commands  
3. **[doctrine/SUCCESS_LEDGER.md](doctrine/SUCCESS_LEDGER.md)** — what already worked  
4. **[MAP.md](MAP.md)** — where everything is  

The only score that counts is what **`prove_it`** prints (clear % and breach %).

---

## Folders in one line

| Folder | One line |
|--------|----------|
| `scripts/` | Commands you type |
| `configs/` | Numbers (target, risk, rewards) |
| `training/` | Learning code + meta_tuner |
| `features/` | Indicators + Gravity |
| `telemetry/` | What the bot was thinking |
| `doctrine/` | Laws, CMO, wins, flea-jar cure |
| `data/` | Prices |
| `artifacts/` | Saved brains and logs |
| `prompts/` | CMO system prompt |
| `docs/history/` | Old notes |

---

## First commands

```bash
python scripts/restore_meta_tuner.py
python scripts/preflight_train.py
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

**LLM persona:** `doctrine/SYSTEM_DOCTRINE_CMO.md`  
**Flea-jar cure:** lid is gone — nothing impossible; only consistency remains.
