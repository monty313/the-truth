# training/ — RL core + meta wrapper

| Module | Role |
|--------|------|
| `policy.py` | Brain (network) — warm-start only |
| `ppo.py` / `gpu_rollout.py` | PPO updates + rollout |
| `fastsim.py` | Fast vectorized sim |
| `gpu_data.py` | Day tensors + feature cache |
| `rewards.py` / `env.py` | Reward engine + gym env |
| **`meta_tuner.py`** | **Self-tuner wrapper**: reward/penalty + lr/entropy; adopt gate |
| `meta_optimizer.py` | Legacy propose-only search (no auto-adopt) |
| `canary.py` / `trophy_case.py` | Canary / trophy lineage |

**Train via scripts:**
- `scripts/preflight_train.py` — must pass first
- `scripts/consistency_sprint.py` — consistency climb
- `scripts/meta_train.py` — meta-tuner loop

If `meta_tuner.py` is missing: `python scripts/restore_meta_tuner.py`
