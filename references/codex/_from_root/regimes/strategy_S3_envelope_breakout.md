# Strategy S3 — Shifted Price Envelope Breakout
| Field | Value |
|---|---|
| Summary | Predictive high/low envelope; buy when price fully clears the top everywhere |
| Buy | HTFs: close above SMA(4,+4,High) AND SMA(4,+4,Low). LTF: close above SMA(4,+2,High) AND SMA(4,+2,Low). Sell = inverse |
| Indicators | SMA(4) on High and on Low; HTF shift +4, LTF shift +2 |
| Note | Same family as the forever masks (masks use +4 on M15/M30/H1 specifically) |
