# ADR-0001: Daily goal & floor are typed inputs
Decision (Monty, 2026-07-18): +X% daily goal and -X% daily floor are operator inputs,
equity-based, USD, day = 00:00-00:00 CEST, no midnight buffer. ONE brain must serve ANY X
(goal-conditioned training). Boot camp trains with goal=2.5, floor=4 (training values only).
Floor hit -> close all, stand down. Goal hit -> keep trading, ratchet: floor rises to goal,
then climbs with every new equity peak. Flat at reset, always, even monster trends.
