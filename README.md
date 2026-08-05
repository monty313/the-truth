# Momentum One (`the-truth`)

Self-healing RL scalper: hit **your** daily target without floor breach.

## Start

1. **[00_MAP_OF_THE_HOUSE.md](00_MAP_OF_THE_HOUSE.md)** — tracks, order, freeze (anti-confusion)  
2. **[00_START_HERE.md](00_START_HERE.md)** — short doors  
3. **[HANDOFF_2026-08-05.md](HANDOFF_2026-08-05.md)** — where we left off  

**Mark:** [MARK HERE!.lnk](MARK%20HERE!.lnk) · [SOUL_MATCH.md](SOUL_MATCH.md)  
**Agents/obs (keep, wire later):** [KEEP_AFTER_SOUL.md](KEEP_AFTER_SOUL.md)

## Layout (cleaner than FinRL’s flat sprawl)

| Path | Role |
|------|------|
| `USE/` | One-click buttons |
| `models/` | PROVEN champion checkpoints |
| `lineages/` | Parallel experiment brains |
| `code/` | All Python packages (`training`, `signals`, `features`, …) |
| `configs/` | YAML settings |
| `data/` | Price data |
| `scripts/` | CLI entrypoints |
| `tests/` | Unit tests |
| `docs/` | Long docs |
| `references/doctrine/flea-jar/` | Active doctrine (**keep**) |
| `outputs/` | Caches, reports, logs |
| `tools/` | Colab, notebooks, HUD |

## Hard rules

- **Never overwrite PROVEN** without an explicit human order  
- **Flea-jar doctrine stays**  
- New experiments go under **`lineages/`** only  

## Run

```text
USE/1_prove.bat
USE/6_new_brain_tests.bat
```
