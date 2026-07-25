# configs/

**All tunable numbers live here.** Not in random Python files.

## How to update
- Change a weight or goal → edit the matching YAML.
- Add a new knob → add it here first, then read it via `core.configs.load`.
- After reward changes, run `python scripts/preflight_train.py`.
- CHANGE LOG: put a dated comment at the top of the YAML you edited.

| File | Holds |
|------|--------|
| goals.yaml | target %, floor % |
| rewards.yaml | reward / penalty weights |
| timeframes.yaml | TF sets (must match `features/engine.py` SETS) |
| training.yaml | PPO / self-tuner settings |
| features.yaml | indicator list for obs |
