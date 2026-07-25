# Momentum One

Self-healing RL trading bot. You type target % and risk %. It aims to hit the target without breaching the floor.

## Start here
Open **[DO_THIS.md](DO_THIS.md)** — three command blocks only.

| File | What |
|------|------|
| [DO_THIS.md](DO_THIS.md) | Commands to run |
| [UPDATE_RULES.md](UPDATE_RULES.md) | How to change the repo without making a mess |
| [TRAINING.md](TRAINING.md) | Longer train notes |
| [doctrine/LLM_REGIME_DEFINITIONS.yaml](doctrine/LLM_REGIME_DEFINITIONS.yaml) | How the LLM defines regimes |

## Folders (short)
| Folder | What lives here |
|--------|-----------------|
| `configs/` | All numbers (goals, rewards, TFs) |
| `scripts/` | Commands you run |
| `training/` | Brain + PPO + meta_tuner |
| `features/` | Indicators + Gravity sets |
| `telemetry/` | Mind probe, ghosts |
| `doctrine/` | Laws and regime definitions |
| `data/` | Price CSVs |
| `artifacts/` | Brains and caches |
| `docs/history/` | Old notes (not day-to-day) |

```bash
pip install -r requirements.txt
python scripts/preflight_train.py
```
