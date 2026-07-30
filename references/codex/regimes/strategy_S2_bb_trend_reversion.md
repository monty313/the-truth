# Strategy S2 — Dual Bollinger Trend Reversion (entries + re-entry loop)
| Field | Value |
|---|---|
| Summary | Price escapes both volatility tunnels (extreme trend); buy the dip back INSIDE the fast tunnel |
| Buy | HTFs: close above BB100(0.5,+2) upper AND BB10(0.5,+2) upper. LTF: close > BB100 upper but closes BELOW BB10 upper (inside fast tunnel) = entry |
| Re-entry | Every touch of BB10 band while SMA50 > BB100 upper on LTF. SMA50 crossing below = stop reloading. Sell = mirror |
| Indicators | BB(100, dev .5, +2), BB(10, dev .5, +2) on price; SMA(50) |
| Interpretation | "A/shallow" confirmed by Monty's detailed breakdown 2026-07-18 |
| Telemetry | entry vs re-entry tagged separately |
