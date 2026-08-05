# CURRICULUM — adaptive_rl_brain_7_31_26

Real-price day list for first serious training (Phase B/C).

**Source:** `data/raw/XAUUSD_curriculum_2026.csv`

Selection rules:
- Min bars/day: 900
- Trend days: high |net| / path length (directional sessions)
- Mix days: weaker trend (more two-way / pullback-ish)
- Cap: modest set for first serious run

| Date | Role | Bars | Net move | Trend strength |
|------|------|-----:|---------:|---------------:|
| 2026-01-28 | trend_bull | 1380 | +243.5700 | 0.0821 |
| 2026-02-27 | pullback_mix | 1243 | +101.4700 | 0.0658 |
| 2026-03-18 | pullback_mix | 1380 | -167.0500 | 0.0728 |
| 2026-04-21 | trend_bear | 1320 | -142.2800 | 0.0882 |
| 2026-05-06 | trend_bull | 1320 | +135.0700 | 0.0822 |

Machine-readable: `checkpoints/curriculum_days.json`

PROVEN: not used. Checkpoints only under this lineage folder.
