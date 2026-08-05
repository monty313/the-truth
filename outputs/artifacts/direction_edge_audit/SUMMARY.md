# Direction edge audit — real MT5 XAUUSD

- Range: 2025-07-21 01:05:00 → 2026-07-31 22:58:00 (360,064 M1 bars)
- Metric: right/wrong on sign(close[t+h]-close[t]) only

## Best @ 10 bars (London+NY)

| Signal | n | Accuracy |
|--------|--:|---------:|
| agree_seA_r2A | 343 | 67.6% |
| agree_2of_top4 | 421 | 66.5% |
| single_stoch_ema_A | 1619 | 65.3% |
| agree_seA_r2A_atr | 179 | 64.8% |
| agree_strict3_seA_r2A_epA | 155 | 62.6% |
| single_rsi2_ema_A | 2019 | 62.3% |
| agree_seB_r2B_epB | 1207 | 61.9% |
| single_stoch_ema_B | 1188 | 60.8% |

## Lesson

Always-on CCI/momentum calls every bar → ~50%.
Independent family agreement (PART4) only when they fire → measured lift.

CSV: `direction_accuracy_by_signal.csv`