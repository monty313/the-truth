# Regime: FOREVER MASKS (hard Shell gate — permanent)
| Field | Value |
|---|---|
| Summary | Directional no-go zones when the M15/M30/H1 envelope chain fully agrees |
| Detection | SELL-mask: close > SMA(4,+4,High) AND close > SMA(4,+4,Low) on M15 & M30 & H1 simultaneously. BUY-mask: mirrored below. |
| Timeframes | M15, M30, H1 (all three at once) |
| Indicators | SMA(4) shift +4 applied to High; SMA(4) shift +4 applied to Low (MT5 semantics) |
| Masks applied | Total side lockout |
| Allowed | Managing/closing existing positions of the blocked side |
| Blocked | Market orders, pending orders, re-entries, adds, probes, any bypass — on the blocked side |
| Assumptions | MT5 shift = line displaced forward; compare current close vs value 4 native bars back |
| Failure conditions | Indicator warmup (<8 bars) => mask treated ACTIVE (fail-closed) |
| Telemetry | span mask_check per decision; mask flips logged with values of all six lines |
| Notes | Monty verbatim spec 2026-07-18. NEVER soften. Sim == live. |
