# training/ — RL brain + meta (production stack)

**North star:** [POLICY_EQUALS_MARK_ON_CHART.md](../../POLICY_EQUALS_MARK_ON_CHART.md)

| File | Role vs Mark-on-chart |
|------|------------------------|
| **meta_tuner.py** | Meta-learn **where** to search reward/hparams. When wrong-side / hold-on-setup hot → force Mark knobs (`w_pullback_with_htf`, with/against trend, quick_pull, setup_skip). Never freezes dial answers. |
| policy.py | Production `Brain` (GRU). Learns personality from rewards. |
| ppo.py / gpu_rollout.py | Learning |
| fastsim.py / gpu_data.py | Sim + data — respect `features.sets_lock` |
| rewards.py / env.py | Closed-trade pay; tags pullback/with_trend/against_trend = Law 1–2 sensors |
| meta_optimizer.py | Legacy propose-only (ADR-0007) |

### Two locks

| `features.sets_lock` | Use |
|----------------------|-----|
| `proven_legacy` (default) | Warm-start `models/PROVEN_*`; prove_it |
| `mark` | New Mark-on-chart trains only — wipe `gpu_cache_*.npz`; **no** PROVEN warm-start |

### Parallel Mark clone (lineage)

Explicit teacher + BC (same mission, different obs dim):

`lineages/adaptive_rl_brain_7_31_26/` — `mark_doctrine` · `train_mark_clone_bc.py`

Warm-start from `models/PROVEN_*` only on `proven_legacy`. No from-scratch wipe of PROVEN.
