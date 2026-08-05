# Momentum Vector M — Backtest Analysis

## Strategy confirmed

```
M = |CCI| × tanh((CCI − SMA(CCI,4)) / 40) × Strength_Factor × sign(CCI)
Strength_Factor = clip(1 − |RSI_BB_pos − 0.5| × 1.4, 0.25, 1.4)
```

- CCI(20), RSI(14), BB on RSI(10, 2σ)
- LTF 5m for M timing; HTF 30m or 1h for direction + M filter
- Entry: cross of ±18 (or ±25) + HTF agree
- Exit: M back through ±4 or opposite signal
- Data: EURUSD & US30 M1 curriculum (~2021–2026 / 2020–2026)

## Results (risk 1% / 1.5 ATR, costs applied)

All four primary configs hit the 5% equity floor (~−95% total return).  
Root cause: ~32–42% win rate, avg hold **~2 bars**, negative expectancy after costs, many trades → ruin under 1% risk.

| Symbol | Config | Trades | WR | PF | Total ret | Buy&Hold | Max DD | Sharpe | Avg bars |
|--------|--------|--------|-----|-----|-----------|----------|--------|--------|----------|
| EURUSD | 5m+30m thr18 | 1186 | 32% | 0.44 | −95% | −4.4% | −95% | −2.7 | 1.8 |
| EURUSD | 5m+30m thr25 | 1145 | 32% | 0.44 | −95% | −4.4% | −95% | −2.8 | 1.9 |
| EURUSD | 5m+1h thr18 | 1177 | 32% | 0.43 | −95% | −4.4% | −95% | −2.8 | 1.9 |
| EURUSD | 5m+1h thr25 | 1254 | 34% | 0.46 | −95% | −4.4% | −95% | −3.0 | 1.9 |
| US30 | 5m+30m thr18 | 3995 | 40% | 0.83 | −95% | **+89%** | −95% | −2.0 | 1.9 |
| US30 | 5m+1h thr18 | 3705 | 42% | 0.77 | −95% | **+91%** | −95% | −2.0 | 2.0 |

### Fixed 1-unit price expectancy (diagnostic, no compounding)

Near-zero edge; best is noise-level:

- EURUSD **5m+1h thr18/exit4**: PF **1.01**, sum of trade % ≈ **+2%** over ~5 years  
- All US30 variants: PF **0.95–0.99**, negative sum returns  
- Raising entry to 25 or exit to 0 / −8 does **not** create a real edge  

## Which version is “better”?

**Least bad:** EURUSD **5m + 1h** (slightly higher WR / PF in fixed-unit test).  
**Still not tradeable:** no version beats costs reliably.  
**US30:** buy & hold crushed the strategy; trend market punishes mean-reverting ±4 exits after momentum entries.

## Why it fails

1. **Exit ±4 vs entry ±18** — M mean-reverts in 2–4 bars; winners cut tiny, losers similar → need WR ≫ 50%.  
2. **5m noise** + costs on many round-trips.  
3. **HTF filter** only agrees direction; does not require HTF strength magnitude.  
4. **1% risk × hundreds of −EV trades** → account ruin (honest risk sim).

## Suggestions

1. **Exit redesign:** trail or exit on M cross of 0 / opposite HTF, not +4.  
2. **Higher base TF:** 15m/30m timing instead of 5m.  
3. **Entry:** require |HTF M| > k and rising velocity_factor.  
4. **Cooldown:** fewer trades; atr-filter; session filter.  
5. **Stops:** structure/ATR target 2R+; partial scale-out.  
6. **Do not** promote as RL signal agent until fixed-unit PF > 1.2 OOS on 2+ symbols.

## Code

- `features/momentum_vector.py` — modular M features  
- `scripts/backtest_momentum_vector.py` — MTF backtest + plots  
- Artifacts: `artifacts/momentum_vector/`
