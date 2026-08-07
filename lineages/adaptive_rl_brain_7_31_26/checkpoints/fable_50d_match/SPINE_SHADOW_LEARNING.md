# SPINE SHADOW LEARNING LOG

Goal: same_outcome 50/50 on held-out (new data), breach 0. Practice keep-floor 33.
Method: Day Spine compile → oracle green → miss-first shadow train → KEEP/REJECT → error card.

| Cycle | same | policy | mwt | breach | decision / top error / change |
|------:|-----:|-------:|----:|-------:|-------------------------------|

| spine-shadow 0 | 33 | 33 | 17 | 0 | **BASELINE** · false_fire · freeze_start_meters |
| spine-shadow 1 | 33 | 33 | 17 | 0 | **REJECT** · wrong_size_or_timing · boost_wait_false_fire |
| spine-shadow 2 | 33 | 33 | 17 | 0 | **REJECT** · wrong_size_or_timing · plan_path_size_timing |
| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** · wrong_size_or_timing · freeze_start_meters |
| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** · wrong_size_or_timing · spine_one_day_start |
| spine-shadow 1 | 34 | 34 | 16 | 0 | **REJECT** · wrong_size_or_timing · 2026-01-21:wrong_size_or_timing:plan_pat |
| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** · wrong_size_or_timing · dagger_timing_path |
| spine-shadow 1 | 27 | 27 | 23 | 0 | **REJECT** · wrong_size_or_timing · dagger_x3:2026-02-13 |
| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** · wrong_size_or_timing · dagger_timing_path |
| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** · unknown_no_spine · safe_one_day_works_recipe |
| spine-shadow 1 | 31 | 31 | 19 | 0 | **REJECT** · unknown_no_spine · safe_crater:2026-02-13 |
| spine-shadow 2 | 30 | 30 | 20 | 0 | **REJECT** · unknown_no_spine · safe_crater:2026-02-20 |
| spine-shadow 3 | 31 | 31 | 19 | 0 | **REJECT** · unknown_no_spine · safe_crater:2026-03-11 |
| spine-shadow 0 | 35 | 35 | 15 | 0 | **BASELINE** · unknown_no_spine · safe_one_day_works_recipe |
| spine-shadow 1 | 31 | 31 | 19 | 0 | **REJECT** · unknown_no_spine · safe_crater:2026-02-13 |
| spine-shadow 2 | 30 | 30 | 20 | 0 | **REJECT** · unknown_no_spine · safe_crater:2026-02-20 |
