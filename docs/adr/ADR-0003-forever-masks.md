# ADR-0003: Forever masks (Monty verbatim, 2026-07-18)
NO SELLS while price > SMA(4,shift+4,High) AND > SMA(4,shift+4,Low) on M15+M30+H1 at once.
NO BUYS while price < both those lines on M15+M30+H1 at once. Blocked side totally dead:
no market/pending orders, no re-entries, no adds, no probes, no bypass. Hard gate, forever,
identical in sim and live.
