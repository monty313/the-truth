# src — all project code

Cookiecutter-mlops **source** root. Import packages from here.

| Package | Job |
|---------|-----|
| `core/` | Config door (`configs/*.yaml`) |
| `data_io/` | Load and validate prices |
| `features/` | Indicators + Gravity |
| `training/` | Env, PPO, rewards, meta_tuner |
| `signals/` | Signal agents |
| `evaluation/` | Consistency / day scores |
| `inference/` | Load `.pt` brains |
| `execution_bridge/` | MT5 |
| `telemetry/` | Mind probe, ghosts |
| `backtesting/` | Simulator + gauntlet |
| `dashboards/` | HUD |
| `alerts/` | Notify |
| `experiments/` | Run cards / tracker |

Scripts add `src` to `PYTHONPATH`. VS Code does too (see `.vscode/settings.json`).
