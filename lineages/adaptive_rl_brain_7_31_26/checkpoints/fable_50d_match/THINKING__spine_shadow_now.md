# What's on my mind (Spine Shadow · live)

Written so Mark can see the reasoning, not just meters.

## The picture (facts, not vibes)

1. **Mark's day program already works.** Soul plans clear 50/50 on practice. Gold spine re-exec is 50/50 after fallback fix. Oracle is green — we are *not* blocked on compiler/shell.

2. **Policy is not dumb; it is off the path.** At same=35, all 15 misses are MARK_WOULD_TAKE. Gap diagnosis:
   - policy award on those days: **0/15**
   - same policy + Mark size dials locked: **1/15**
   - gold plan execution: **15/15**  
   So the gap is almost entirely **when** it fires (timing/path under live eyes), not "no edge" and not mainly size.

3. **Why offline BC lies.** Plan-path BC walks Mark's trajectory and labels those states → dir_match ~0.95. At score time the policy walks *its own* trajectory → different states → never hits t1/t2. Classic **covariate shift**. That's why DAgger (label *policy* states with Mark/spine actions) is the correct primary lever now — not more reward crank, not 3 teachers, not denser bar CE.

4. **What already failed (don't repeat)**
   - Pack-wide BC using pack-dominant error class (boosted HOLD for everyone when only some thrash) → flat 33.
   - One-day focus that converts a day but slips the pack (35→34) without successful pack-repair KEEP.
   - Treating dir_match as day win.

5. **Data honesty for "100 days forward"**
   - Curriculum alone only has ~40 days after practice 50.
   - Price lives in **the-truth/data/raw/** (`XAUUSD_M1_full.csv` etc.). Full M1 → ~1461 loadable days.
   - Holdout design: 40 future after fit + 60 past before fit = **100 calendar days, ∩ fit = ∅**. That is real unseen, not fake 50/50.

## What I'm doing now

- **Running `spine_dagger_climb.py` (v2 after scare):** DAgger on policy states labeled by spine plan.
- **Lesson just learned:** round-1 DAgger with 3 MWT targets + low KL **collapsed pack 35→27**. Conscience REJECT worked.  
  Fix: **focus-only** DAgger, **high KL (~0.55)** to freeze awards, heavier award self-imitate, thrash HOLD weight up.
- Price only from **the-truth/data/raw**.
- **Not** promoting PROVEN. **Not** opening 3 LLM teachers.
- Logging every cycle into SPINE_SHADOW_LEARNING + error cards for KAG.

## What success looks like

| Gate | Bar |
|------|-----|
| Practice | same climbs from 35 with breach 0; KEEP only if not worse |
| Forward 100 | dual score, same high + breach 0, leakage audit empty |
| Method | spine compile + oracle green already done |

## Risks I'm watching

- DAgger can thrash (too many fires) → breach → REJECT (good conscience).
- Focus converts / pack dies → pack-repair then re-score; if still down, REJECT.
- 100d Mark oracle is expensive (soul search per day) — cache to disk, resume-friendly.
- Peer pipelines also touch BEST/embryo — we treat live score as truth each cycle.

## If DAgger stalls 3–4 flat rejects

Next creative move (in order):
1. **More DAgger mass on only false_hold / late_entry days** (from per-day fire times vs policy n_entries).
2. **Multi-year spine sample** from M1_full *excluding* the 100 holdout — teach timing structure, not 50-day memo.
3. HITL spine edit only if a day is chart-ambiguous (F7) — not reward JSON.

— Fable / implementer
