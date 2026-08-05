# Mark day diary — 2026-03-18

**Order:** I looked at the chart **before** the policy was trusted.
**Goal that day:** target **1.5%** · risk floor **−2.0%** (runtime inputs — no retrain).

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

- **2026-03-18 12:00:00** — I would **SELL** (eq≈0.0% · pos→SHORT · entries=1)
  - why: `law1_slingshot_release_short n_aligned=1 vel=1`
  - force=Direction.BEAR regime=Regime.BEAR
- **2026-03-18 12:25:00** — I would **SELL** (eq≈-0.2675% · pos→SHORT · entries=2)
  - why: `law1_slingshot_release_short n_aligned=1 vel=2`
  - force=Direction.BEAR regime=Regime.BEAR
- **2026-03-18 12:50:00** — I would **HOLD** (eq≈1.0004% · pos→SHORT · entries=2)
  - why: `law1_slingshot_release_short n_aligned=1 vel=1`
  - force=Direction.BEAR regime=Regime.BEAR

## End of day (my score)
- entries: **2**
- pnl: **1.6865%** · min equity: **-0.3472%**
- banked: **True** · breached: **False**
- **award/clear: True** (hit target without floor — or banked clean)
- action mix: `{'SELL': 2, 'HOLD': 1}`

_This diary is principles applied to this day's price path — not a claim of live discretionary Mark beyond the codified teacher._
