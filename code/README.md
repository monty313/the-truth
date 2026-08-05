# code/ — all bot library packages

Like FinRL’s single `finrl/` tree: **one place for Python code**.

| Package | What it does |
|---------|----------------|
| `training/` | RL train, FastSim, meta_tuner |
| `features/` | Indicators, feature engine |
| `signals/` | Signal agents |
| `evaluation/` | prove scores, consistency |
| `inference/` | Load brains |
| `telemetry/` | Mind probe, logging |
| `backtesting/` | DaySim, gauntlet |
| `data_io/` | Load / resample price data |
| `core/` | Config door |
| `execution_bridge/` | Live broker bridge |
| `alerts/` | Notifications |
| `experiments/` | Experiment tracker |

**Imports stay the same:** `from training.x import y`  
(as long as `code/` is on PYTHONPATH — USE buttons set this).

**Not here:** configs, data, models, scripts, tests, lineages, USE.
