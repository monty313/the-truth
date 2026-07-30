# training/ — RL brain

| File | Role |
|------|------|
| **meta_tuner.py** | Required. Reward/penalty + hparam wrapper. If missing: `python scripts/restore_meta_tuner.py` |
| policy.py | Network |
| ppo.py / gpu_rollout.py | Learning |
| fastsim.py / gpu_data.py | Sim + data |
| rewards.py / env.py | Rewards / env |
| meta_optimizer.py | Legacy propose-only |

Warm-start from `models/PROVEN_*`. No from-scratch wipe.
