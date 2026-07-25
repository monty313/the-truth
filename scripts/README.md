# scripts/ — entry points

## Before every train
| Script | Purpose |
|--------|---------|
| `preflight_train.py` | Verify meta_tuner, SETS, rewards, telemetry, data, checkpoints |
| `restore_meta_tuner.py` | Restore `training/meta_tuner.py` with Phase-4 unlocks if missing |
| `align_tf_sets.py` | Align engine SETS to Monty TF lock |

## Training (GPU preferred)
| Script | Purpose |
|--------|---------|
| `consistency_sprint.py` | Frontier-weighted PPO climb; ratchet keeps best clear/row |
| `meta_train.py` | Meta-tuner wrapper: reward/penalty + lr/entropy under adopt gate |
| `curriculum_train.py` | Curriculum path |
| `train_bootcamp.py` | Bootcamp drill |
| `gpu_train.py` / `gpu_validate.py` | GPU train / validate |

## Measure / diagnose (LLM layer)
| Script | Purpose |
|--------|---------|
| `prove_it.py` | Clear rate / row / breach at your target/risk |
| `give_llm_what_it_needs.py` | Cache + multi-day probe + ghosts + IRAC pack |
| `mind_probe_day.py` | Single-day MRI |
| `diagnose_day.py` | IRAC on one day |
| `perception_scoreboard.py` | Perception metrics |

## Live / ops
| Script | Purpose |
|--------|---------|
| `run_live.py` | Live bridge |
| `run_hud.py` | HUD |
| `run_gauntlet.py` | Feasibility gauntlet |

## Doctrine
Regime SSOT: `doctrine/LLM_REGIME_DEFINITIONS.yaml`  
Train playbook: `TRAINING.md` + `doctrine/HOST_RUN.md`
