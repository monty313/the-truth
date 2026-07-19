# ADR-0002: The Shell (hard, not learned)
Per-trade risk cap 0.25% equity (learn freely INSIDE it). Max 5 adds per winning stack.
400 trades/day (probes included) -> CLOSE-ONLY mode after. Wide broker-side stop on every
trade. Kill switch = freeze + close all. Crash -> reconnect + resume. Hedging allowed.
Heat guard (sum of open risk <= distance to floor) implemented, config-flagged ON —
engineering addition flagged to Monty (his ruling: brain decides within caps); flip in
configs/masks_shell.yaml if he vetoes. Forever masks: see ADR-0003 + codex.
