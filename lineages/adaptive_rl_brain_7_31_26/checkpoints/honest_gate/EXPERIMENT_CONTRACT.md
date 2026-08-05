# Multi-pair tutor — experiment contract (pre-training gate)

**Experiment ID:** `multi_pair_tutor_honest_v1_pretrain_gate`  
**Generated (UTC):** 2026-07-31T19:26:50.246970+00:00  
**Gate verdict:** **GATE_PASS**  
**Track:** multi_pair_tutor (NOT PROVEN, NOT Channel1 claim)

## Mission

- Runtime **target%** / **risk%** without retrain.
- **Clear** = banked equity% ≥ target% after costs path AND floor never hit.
- **Breach** = floor touched → day fails.
- Official meters: clear count, breach count, clear streak.
- PnL / entries / “looks good” = diagnostics only.

## Identity pins

| Item | Path | SHA-256 |
|------|------|---------|
| checkpoint | `C:/Users/user/Fable5_Foundation/MOMENTUM_ONE/the-truth/lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_consistent_v1.pt` | `cd89f6441d50ee06fe59d39291083ed867ed497f3b5dd4bb31568ad5dc7e112c` |
| dials | `C:/Users/user/Fable5_Foundation/MOMENTUM_ONE/the-truth/lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_dials.json` | `327d190a6f13319a44cacfe65463ae2153673a098e79a8fe4a9d7013eaba0daf` |
| data | `C:/Users/user/Fable5_Foundation/MOMENTUM_ONE/the-truth/data/raw/XAUUSD_curriculum_2026.csv` | `2c3826af2623872d762dfc6d7a3289afd3957740192dbf7f93e29ac2550ddd20` |
| meaning_manifest | `C:/Users/user/Fable5_Foundation/MOMENTUM_ONE/the-truth/lineages/adaptive_rl_brain_7_31_26/checkpoints/honest_gate/meaning_manifest.json` | `aed6d20c42fb2bc3cce9104a85cb48f5dcd759228d9ebd76ba8b945f4432c2cd` |
| banned_rule_families | `C:/Users/user/Fable5_Foundation/MOMENTUM_ONE/the-truth/lineages/adaptive_rl_brain_7_31_26/checkpoints/honest_gate/banned_rule_families.json` | `f26211c01da64487eab43d5f01bc10215a1a18c2663764deb89c274aeee4b4e3` |
| data_contract | `C:/Users/user/Fable5_Foundation/MOMENTUM_ONE/the-truth/lineages/adaptive_rl_brain_7_31_26/checkpoints/honest_gate/data_contract.json` | `0738f6ed70fe0170b3c8436fd448757f40721d5a010049a84573e0d0852e2445` |

| decode | (claim path) | **heuristic** |
| seed | | **42** |
| meaning_version | | `mp_tutor_meaning_v1_8f75acadba92` |
| meaning_hash | | `8f75acadba924f016fec370250263073c2c59686b103f906e29ffab345ba7c6c` |

## Data split (chronological)

- Source: `XAUUSD_curriculum_2026.csv` sha256=`2c3826af2623872d762dfc6d7a3289afd3957740192dbf7f93e29ac2550ddd20`
- Eligible days (min_bars≥900): **90**
- Practice: **50** days (2026-01-20 → 2026-03-30)
- Forward: **40** days (2026-03-31 → 2026-05-26)
- Overlap: **0** (must be 0)
- 100-day conclusion: **NOT_YET_MEASURABLE**

## Shell

- SHELL_LOCKED: **True** ok=True
- Laws: heat_refuse_open, floor_scaled_sizing, every_bar_marks, bank_at_target, breach_termination, one_signal_flat_and_in_trade

## Prior claim honesty

- `ten_pair_score_all.json` = **IN_SAMPLE_CLAIM** (dials may have seen all days historically).
- Do **not** call prior forward JSON pure unseen if dials were fit with all-day search.
- New training: dial search **practice only**; score forward once after freeze.

## Allowed vs forbidden (first training cycle)

**Allowed on practice only:** risk_use_frac, stop_atr_mult, per_trade_cap_pct, channel1_policy_weights, bc_epochs, bc_lr, hidden_size, seed, practice_day_subset_for_bc

**Forbidden:** target_pct_baked_into_weights, risk_pct_baked_into_weights, trail_stop_enabled, floor_cushion_pct, scale_in_on_claim_path, decision_bar_only_stops, weaker_in_trade_eyes, decode=pure_greedy_as_claim_default, forward_day_in_dial_search, forward_day_in_reward_selection, forward_day_in_feature_selection, shell_law_mutation_without_unlock, retrain_only_because_target_risk_changed, cross_track_promote_to_PROVEN, cross_track_promote_channel1_to_multi_pair_claim

## Commands to reproduce this gate

```powershell
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/honest_gate/run_gate.py
```

## What we do NOT claim yet

- Not “consistent,” “unseen-proven,” “100-day,” “robust,” or “ready to train for production.”
- Gate PASS means honesty infrastructure is ready — not that clear% is high.

