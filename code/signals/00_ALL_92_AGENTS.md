# All 92 signal agents

**Open this folder:** `code/signals/`  
**Registry (source of truth):** `configs/signal_slots.yaml`  
**Capacity:** 500 slots | **Filled:** 92 | **Enabled:** 91

> **KEEP ALL — wire after Mark soul is full.**  
> See root `KEEP_AFTER_SOUL.md` + `00_MAP_OF_THE_HOUSE.md`.  
> Until then: agents are **sensors**, not the policy soul (T3 Mark doctrine owns side).

These are **slots** (one row each in YAML), not 92 separate folders.  
Code lives in a few Python files; `encode.py` dispatches every kind.

## Where to click

| What | Path |
|------|------|
| Full list of 92 | `configs/signal_slots.yaml` |
| Code folder | `code/signals/` |
| Dispatcher | `code/signals/encode.py` |
| This index | `code/signals/00_ALL_92_AGENTS.md` |
| Manager | `python scripts/manage_signal_slots.py list` |

## Family → code file

| Family | Count | Code file under `code/signals/` |
|--------|------:|--------------------------------|
| momentum_one | 9 | `encode.py` |
| camillion | 18 | `encode.py` |
| decision_tree | 4 | `encode.py` |
| rl_trading_live | 5 | `encode.py` |
| sma_mtf | 8 | `encode.py` |
| rsi_mtf | 11 | `encode.py` |
| stoch_mtf | 8 | `encode.py` |
| stoch_ema | 3 | `encode.py` |
| rsi2_ema | 9 | `rsi2_ema.py` |
| smma_rsi | 4 | `smma_rsi.py` |
| agree | 4 | `agree.py` |
| dvmr | 3 | `dvmr_agent.py` |
| momentum_vector | 3 | `momentum_vector_agent.py` |
| bb_rsi_sma | 3 | `bb_rsi_sma_agent.py` |

## All 92 agents (slot order)

| Slot | On | Name | Kind | Family | Code |
|-----:|:--:|------|------|--------|------|
| 0 | yes | `mo_bread_and_butter_pull_set1` | `pull_set1` | momentum_one | `encode.py` |
| 1 | yes | `mo_continuation_set1` | `cont_set1` | momentum_one | `encode.py` |
| 2 | yes | `mo_bread_and_butter_pull_set2` | `pull_set2` | momentum_one | `encode.py` |
| 3 | yes | `mo_continuation_set2` | `cont_set2` | momentum_one | `encode.py` |
| 4 | yes | `mo_pull_set3` | `pull_set3` | momentum_one | `encode.py` |
| 5 | yes | `mo_continuation_set3` | `cont_set3` | momentum_one | `encode.py` |
| 6 | yes | `mo_rev_set1` | `rev_set1` | momentum_one | `encode.py` |
| 7 | yes | `mo_rev_set2` | `rev_set2` | momentum_one | `encode.py` |
| 8 | yes | `mo_rev_set3` | `rev_set3` | momentum_one | `encode.py` |
| 10 | yes | `cam_gravity_30m_4h` | `cam_gravity_30m_4h` | camillion | `encode.py` |
| 11 | yes | `cam_regime_pulse_trend_5m_30m` | `cam_regime_pulse_trend_5m_30m` | camillion | `encode.py` |
| 12 | yes | `cam_regime_pulse_pullback_5m_30m` | `cam_regime_pulse_pullback_5m_30m` | camillion | `encode.py` |
| 13 | yes | `cam_regime_pulse_trend_30m_4h` | `cam_regime_pulse_trend_30m_4h` | camillion | `encode.py` |
| 14 | yes | `cam_regime_pulse_pullback_30m_4h` | `cam_regime_pulse_pullback_30m_4h` | camillion | `encode.py` |
| 15 | yes | `cam_cci_surge_trend_5m_30m` | `cam_cci_surge_trend_5m_30m` | camillion | `encode.py` |
| 16 | yes | `cam_cci_surge_pullback_5m_30m` | `cam_cci_surge_pullback_5m_30m` | camillion | `encode.py` |
| 17 | yes | `cam_cci_surge_trend_30m_4h` | `cam_cci_surge_trend_30m_4h` | camillion | `encode.py` |
| 18 | yes | `cam_cci_surge_pullback_30m_4h` | `cam_cci_surge_pullback_30m_4h` | camillion | `encode.py` |
| 19 | yes | `cam_sma_stack_trend_5m_30m` | `cam_sma_stack_trend_5m_30m` | camillion | `encode.py` |
| 20 | yes | `cam_sma_stack_pullback_5m_30m` | `cam_sma_stack_pullback_5m_30m` | camillion | `encode.py` |
| 21 | yes | `cam_sma_stack_trend_30m_4h` | `cam_sma_stack_trend_30m_4h` | camillion | `encode.py` |
| 22 | yes | `cam_sma_stack_pullback_30m_4h` | `cam_sma_stack_pullback_30m_4h` | camillion | `encode.py` |
| 23 | yes | `cam_sma_reversion_rally_5m_30m` | `cam_sma_reversion_rally_5m_30m` | camillion | `encode.py` |
| 24 | yes | `cam_sma_reversion_rally_30m_4h` | `cam_sma_reversion_rally_30m_4h` | camillion | `encode.py` |
| 25 | NO | `cam_orb_ny_breakout_indices` | `cam_orb_ny_breakout` | camillion | `encode.py` |
| 26 | yes | `cam_adx_di_align_5m_30m` | `cam_adx_di_align_5m_30m` | camillion | `encode.py` |
| 27 | yes | `cam_adx_di_align_30m_4h` | `cam_adx_di_align_30m_4h` | camillion | `encode.py` |
| 28 | yes | `dt_ftmo_alpha` | `dt_ftmo_alpha` | decision_tree | `encode.py` |
| 29 | yes | `s11_cci` | `s11_cci` | decision_tree | `encode.py` |
| 30 | yes | `s11_pull` | `s11_pull` | decision_tree | `encode.py` |
| 31 | yes | `s11_m15` | `s11_m15` | decision_tree | `encode.py` |
| 32 | yes | `phase_cci_align` | `phase_cci_align` | rl_trading_live | `encode.py` |
| 33 | yes | `phase_hilo_trend` | `phase_hilo_trend` | rl_trading_live | `encode.py` |
| 34 | yes | `phase_bb_mid` | `phase_bb_mid` | rl_trading_live | `encode.py` |
| 35 | yes | `phase_sma_stack` | `phase_sma_stack` | rl_trading_live | `encode.py` |
| 36 | yes | `phase_atr_expand` | `phase_atr_expand` | rl_trading_live | `encode.py` |
| 37 | yes | `sma_mtf_A_mid` | `sma_mtf_A_mid` | sma_mtf | `encode.py` |
| 38 | yes | `sma_mtf_B_mid` | `sma_mtf_B_mid` | sma_mtf | `encode.py` |
| 39 | yes | `sma_mtf_C_mid` | `sma_mtf_C_mid` | sma_mtf | `encode.py` |
| 40 | yes | `sma_mtf_A_outer` | `sma_mtf_A_outer` | sma_mtf | `encode.py` |
| 41 | yes | `sma_mtf_B_outer` | `sma_mtf_B_outer` | sma_mtf | `encode.py` |
| 42 | yes | `sma_mtf_C_outer` | `sma_mtf_C_outer` | sma_mtf | `encode.py` |
| 43 | yes | `sma_mtf_agree_mid` | `sma_mtf_agree_mid` | sma_mtf | `encode.py` |
| 44 | yes | `sma_mtf_agree_outer` | `sma_mtf_agree_outer` | sma_mtf | `encode.py` |
| 45 | yes | `rsi_mtf_A_momentum` | `rsi_mtf_A_momentum` | rsi_mtf | `encode.py` |
| 46 | yes | `rsi_mtf_A_pullback` | `rsi_mtf_A_pullback` | rsi_mtf | `encode.py` |
| 47 | yes | `rsi_mtf_A_combined` | `rsi_mtf_A_combined` | rsi_mtf | `encode.py` |
| 48 | yes | `rsi_mtf_B_momentum` | `rsi_mtf_B_momentum` | rsi_mtf | `encode.py` |
| 49 | yes | `rsi_mtf_B_pullback` | `rsi_mtf_B_pullback` | rsi_mtf | `encode.py` |
| 50 | yes | `rsi_mtf_B_combined` | `rsi_mtf_B_combined` | rsi_mtf | `encode.py` |
| 51 | yes | `rsi_mtf_C_momentum` | `rsi_mtf_C_momentum` | rsi_mtf | `encode.py` |
| 52 | yes | `rsi_mtf_C_pullback` | `rsi_mtf_C_pullback` | rsi_mtf | `encode.py` |
| 53 | yes | `rsi_mtf_C_combined` | `rsi_mtf_C_combined` | rsi_mtf | `encode.py` |
| 54 | yes | `rsi_mtf_agree` | `rsi_mtf_agree` | rsi_mtf | `encode.py` |
| 55 | yes | `rsi_mtf_any` | `rsi_mtf_any` | rsi_mtf | `encode.py` |
| 56 | yes | `stoch_mtf_A_momentum` | `stoch_mtf_A_momentum` | stoch_mtf | `encode.py` |
| 57 | yes | `stoch_mtf_A_pullback` | `stoch_mtf_A_pullback` | stoch_mtf | `encode.py` |
| 58 | yes | `stoch_mtf_A_combined` | `stoch_mtf_A_combined` | stoch_mtf | `encode.py` |
| 59 | yes | `stoch_mtf_B_momentum` | `stoch_mtf_B_momentum` | stoch_mtf | `encode.py` |
| 60 | yes | `stoch_mtf_B_pullback` | `stoch_mtf_B_pullback` | stoch_mtf | `encode.py` |
| 61 | yes | `stoch_mtf_B_combined` | `stoch_mtf_B_combined` | stoch_mtf | `encode.py` |
| 62 | yes | `stoch_mtf_C_momentum` | `stoch_mtf_C_momentum` | stoch_mtf | `encode.py` |
| 63 | yes | `stoch_mtf_agree` | `stoch_mtf_agree` | stoch_mtf | `encode.py` |
| 64 | yes | `stoch_ema_A` | `stoch_ema_A` | stoch_ema | `encode.py` |
| 65 | yes | `stoch_ema_B` | `stoch_ema_B` | stoch_ema | `encode.py` |
| 66 | yes | `stoch_ema_C` | `stoch_ema_C` | stoch_ema | `encode.py` |
| 67 | yes | `rsi2_ema_1m_15m` | `rsi2_ema_1m_15m` | rsi2_ema | `rsi2_ema.py` |
| 68 | yes | `rsi2_ema_1m_30m` | `rsi2_ema_1m_30m` | rsi2_ema | `rsi2_ema.py` |
| 69 | yes | `rsi2_ema_5m_1h` | `rsi2_ema_5m_1h` | rsi2_ema | `rsi2_ema.py` |
| 70 | yes | `rsi2_ema_5m_4h` | `rsi2_ema_5m_4h` | rsi2_ema | `rsi2_ema.py` |
| 71 | yes | `rsi2_ema_15m_4h` | `rsi2_ema_15m_4h` | rsi2_ema | `rsi2_ema.py` |
| 72 | yes | `rsi2_ema_15m_1d` | `rsi2_ema_15m_1d` | rsi2_ema | `rsi2_ema.py` |
| 73 | yes | `rsi2_ema_A` | `rsi2_ema_A` | rsi2_ema | `rsi2_ema.py` |
| 74 | yes | `rsi2_ema_B` | `rsi2_ema_B` | rsi2_ema | `rsi2_ema.py` |
| 75 | yes | `rsi2_ema_C` | `rsi2_ema_C` | rsi2_ema | `rsi2_ema.py` |
| 76 | yes | `smma_rsi_15m_4h_V0` | `smma_rsi_15m_4h_V0` | smma_rsi | `smma_rsi.py` |
| 77 | yes | `smma_rsi_15m_4h_V1` | `smma_rsi_15m_4h_V1` | smma_rsi | `smma_rsi.py` |
| 78 | yes | `smma_rsi_C_V0` | `smma_rsi_C_V0` | smma_rsi | `smma_rsi.py` |
| 79 | yes | `smma_rsi_C_V1` | `smma_rsi_C_V1` | smma_rsi | `smma_rsi.py` |
| 80 | yes | `agree_seA_r2A` | `agree_seA_r2A` | agree | `agree.py` |
| 81 | yes | `agree_seB_r2B_epB` | `agree_seB_r2B_epB` | agree | `agree.py` |
| 82 | yes | `agree_2of_top4` | `agree_2of_top4` | agree | `agree.py` |
| 83 | yes | `agree_seA_r2A_atr` | `agree_seA_r2A_atr` | agree | `agree.py` |
| 84 | yes | `dvmr_champ_1h_1d` | `dvmr_champ_1h_1d` | dvmr | `dvmr_agent.py` |
| 85 | yes | `dvmr_30m_4h_v2` | `dvmr_30m_4h_v2` | dvmr | `dvmr_agent.py` |
| 86 | yes | `dvmr_champ_1h_1d_pulse` | `dvmr_champ_1h_1d_pulse` | dvmr | `dvmr_agent.py` |
| 87 | yes | `mv_best_quality_30m_4h_long` | `mv_best_quality_30m_4h_long` | momentum_vector | `momentum_vector_agent.py` |
| 88 | yes | `mv_strong3_30m_4h_long` | `mv_strong3_30m_4h_long` | momentum_vector | `momentum_vector_agent.py` |
| 89 | yes | `mv_strong4_30m_4h_long` | `mv_strong4_30m_4h_long` | momentum_vector | `momentum_vector_agent.py` |
| 90 | yes | `bb_rsi_sma_A` | `bb_rsi_sma_A` | bb_rsi_sma | `bb_rsi_sma_agent.py` |
| 91 | yes | `bb_rsi_sma_B` | `bb_rsi_sma_B` | bb_rsi_sma | `bb_rsi_sma_agent.py` |
| 92 | yes | `bb_rsi_sma_C` | `bb_rsi_sma_C` | bb_rsi_sma | `bb_rsi_sma_agent.py` |

## Turn them into the observation (main bot)

In `configs/features.yaml`:

```yaml
include_signal_agent_slots: true   # all filled agents → obs (NEW brain size)
# false = PROVEN 1820-compatible; do not load PROVEN with true
```

After flipping, delete GPU feature caches (not .pt brains):
`outputs/artifacts/gpu_cache_*.npz` etc.

## New lineage note

`lineages/adaptive_rl_brain_7_31_26/` does **not** load these 92 yet.
It uses `perception/` (sets/confluence) + dim-32 Channel 1.
Wiring all 92 into that lineage is a separate step.

## Verified

- Registry rows: 92
- Unique kinds: 92
- Enabled: 91 (1 disabled: slot 25 ORB stub)
- Every kind string appears in `code/signals/` handlers
