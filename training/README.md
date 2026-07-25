# training/

RL brain and the meta-tuner wrapper.

## How to update
- Policy / PPO / sim changes → edit the module; append CHANGE LOG (date + WHY) at top.
- **meta_tuner.py** is required. If missing: `python scripts/restore_meta_tuner.py`.
- Do not add `meta_tuner_v2.py`. Edit the real file.
- `meta_optimizer.py` is legacy propose-only; new work goes through **meta_tuner**.
- Never change observation size here without an explicit Monty decision.

| File | Role |
|------|------|
| policy.py | Network |
| ppo.py / gpu_rollout.py | Learning step |
| fastsim.py / gpu_data.py | Fast sim + data |
| meta_tuner.py | Reward/penalty + hparam wrapper |
| rewards.py / env.py | Reward engine / env |
