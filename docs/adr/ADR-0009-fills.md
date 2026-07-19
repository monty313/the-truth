# ADR-0009: Paranoid fills (Claude's call, Monty deflected)
Signals computed on bar t close. Entries fill at bar t+1 worst realistic price for the
direction + full recorded spread. Exits same worst-side logic. Stops assume worst intrabar
ordering. If it profits here, the money is real. Paranoia level configurable (paranoid |
mid | optimistic) — default paranoid; Gauntlet runs paranoid.
