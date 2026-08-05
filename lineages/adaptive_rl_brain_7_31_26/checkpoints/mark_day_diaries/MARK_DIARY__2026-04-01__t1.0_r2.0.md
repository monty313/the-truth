# Mark day diary — 2026-04-01

**Order:** I looked at the chart **before** the policy was trusted.
**Goal that day:** target **1.0%** · risk floor **−2.0%** (runtime inputs — no retrain).

## Principles I used (pt5 = basics only)
- pt5.1 HTF permission / gravity — LTF never votes side against HTF
- pt5.1 slingshot: pullback loads, resume with HTF releases
- pt5.2 breath vs launch — different playbooks
- pt5.3 regime: bull/bear/chop/flat rewrites what is allowed
- pt5.4 capital: floor and size before edge
- MARK SETS LAW: LTF=first (pullback/cont/add); HTF=last two (confirm); scan all 4

## Sets I scanned (MARK SETS LAW)
- Set 1: LTF **1m** (pullback/cont/add) · HTF **15m, 30m** (trend confirm)
- Set 2: LTF **5m** (pullback/cont/add) · HTF **30m, 1h** (trend confirm)
- Set 3: LTF **15m** (pullback/cont/add) · HTF **1h, 4h** (trend confirm)
- Set 4: LTF **30m** (pullback/cont/add) · HTF **4h, 1d** (trend confirm)

## What I would have done during the day

- **2026-04-01 12:00:00** — I would **HOLD** (eq≈0.0% · pos→FLAT · entries=0)
  - why: `law3_no_permission_or_trigger`
  - force=Direction.BEAR regime=Regime.FLAT
- **2026-04-01 12:25:00** — I would **HOLD** (eq≈0.0% · pos→FLAT · entries=0)
  - why: `law3_no_permission_or_trigger`
  - force= regime=Regime.FLAT
- **2026-04-01 12:50:00** — I would **HOLD** (eq≈0.0% · pos→FLAT · entries=0)
  - why: `law3_no_permission_or_trigger`
  - force= regime=Regime.FLAT
- **2026-04-01 13:15:00** — I would **BUY** (eq≈0.0% · pos→LONG · entries=1)
  - why: `law1_slingshot_release_long n_aligned=2 vel=2`
  - force=Direction.BULL regime=Regime.BULL
- **2026-04-01 13:40:00** — I would **HOLD** (eq≈0.8138% · pos→LONG · entries=1)
  - why: `law1_slingshot_release_long n_aligned=2 vel=2`
  - force=Direction.BULL regime=Regime.BULL
- **2026-04-01 14:05:00** — I would **SELL** (eq≈0.3843% · pos→SHORT · entries=2)
  - why: `law1_soft_single_set_scalp aligned_bear n=1 score=4.00>0.00`
  - force=Direction.BEAR regime=Regime.FLAT

## End of day (my score)
- entries: **2**
- pnl: **1.2958%** · min equity: **-0.0851%**
- banked: **True** · breached: **False**
- **award/clear: True** (hit target without floor — or banked clean)
- action mix: `{'HOLD': 4, 'BUY': 1, 'SELL': 1}`

_This diary is principles applied to this day's price path — not a claim of live discretionary Mark beyond the codified teacher._
