# FULL PHASE 1 SPECIFICATION
## Lineage: `adaptive_rl_brain_7_31_26`
**(Parallel only — never touch PROVEN)**

============================================================
1. OFFICIAL SETS + SUB-SETS
============================================================

Official Sets (full strength):
  Set 1: Entry=1m,  Confirmation=15m, 30m
  Set 2: Entry=5m,  Confirmation=30m, 1h
  Set 3: Entry=15m, Confirmation=1h, 4h
  Set 4: Entry=30m, Confirmation=4h, 1d

Sub-Sets (weaker / lower confidence):
  Sub A: 1m, 5m
  Sub B: 5m, 15m
  Sub C: 15m, 30m
  Sub D: 1h, 4h
  Sub E: 4h, 1d

Roles are relative: first TF = Entry, remaining = Confirmation.

============================================================
2. THREE CONFLUENCE GROUPS (Direction + Velocity)
============================================================

Evaluated on the TWO Confirmation (higher) TFs of a set only.

Group 1 – CCI:
  CCI 30 and CCI 100 vs their SMA(period=1, shift=+4)
  Both above on both higher TFs → Bull vote
  Both below on both higher TFs → Bear vote
  Otherwise → Neutral

Group 2 – RSI:
  RSI 5 and RSI 14 vs their SMA(period=1, shift=+4)
  Same logic as Group 1

Group 3 – Price Channel:
  SMA(4) of High shift=+2 and SMA(4) of Low shift=+2
  Price above both channel lines on both higher TFs → Bull vote
  Price below both channel lines on both higher TFs → Bear vote
  Otherwise → Neutral

Aggregation = Simple Majority:
  Direction = majority of non-neutral group votes
  Velocity strength:
    3 groups agree → Strong
    2 groups agree → Medium
    1 or conflict  → Weak / None

Mapping (locked for implementation):
  3 agree → STRONG
  2 agree → MEDIUM
  1 agree → WEAK
  0 or conflict → NONE

============================================================
3. PULLBACK + SCALE CONFLICT
============================================================

Pullback exists when:
  - Both higher TFs of a Major Set show the same clear Direction
    (guaranteed higher-TF trend)
  - The lower / Entry TF (or relevant Sub-Set) shows the opposite Direction

Scale Conflict exists when:
  - A Major Set and a Sub-Set (or smaller set) have opposite clear Directions

============================================================
4. FOUR TRADE TAGS + MINDLESS WALL
============================================================

Tags (priority order):

1. MINDLESS          → hard wall (mask or fixed massive penalty)
2. WITH_VECTOR       → Major + lower signal agree
3. QUALIFIED_MACRO   → Pullback (trade follows guaranteed higher-TF trend
                       while lower is opposite)
4. QUALIFIED_MICRO   → Trade follows lower / Sub-Set against higher-TF trend

MINDLESS 3-condition test
(A trade against the higher-set Vector is MINDLESS unless ALL three hold):

  a) The active lower-set Vector M has TURNED in the trade direction
  b) Lower-set velocity CONFIRMS the turn
  c) The higher sets show weakening or pullback (not fresh acceleration)

All three true → QUALIFIED (allowed, gets credit)
Any missing    → MINDLESS = wall
Even a lucky profitable mindless trade still receives the mindless penalty.

"Higher set" and "lower set" are relative to whatever sets are active at runtime.
No fixed timeframe pairs.

============================================================
5. FILE LAYOUT (Phase 1 only)
============================================================

lineages/
  adaptive_rl_brain_7_31_26/
    README.md
    SPEC_PHASE1.md
    __init__.py
    perception/
      __init__.py
      sets.py
      confluence.py
      structure.py
      classify.py
      types.py

tests/
  lineages/
    adaptive_rl_brain_7_31_26/
      test_sets.py
      test_confluence.py
      test_structure.py
      test_classify.py
      test_mindless_wall.py
      conftest.py

Isolation rules:
- Pure functions only
- No writes to models/, artifacts/checkpoints/, or any PROVEN path
- Phase 1 tests use hand-built synthetic snapshots only

============================================================
6. REQUIRED UNIT TESTS (must all pass before any training code)
============================================================

test_sets.py
- Official Sets 1–4 exist, each exactly 3 TFs, LTF first
- Sub-Sets A–E exist with correct pairs
- Objects are frozen / hashable

test_confluence.py
- Majority 2-of-3 bull / bear
- Split or all-zero → flat
- Velocity independent of Direction
- Velocity strength levels (Strong / Medium / Weak)

test_structure.py
- Pullback true only when higher TFs agree and lower opposes
- Pullback false on full continuation
- Scale Conflict true on opposite clear Directions
- Scale Conflict false if either side is flat

test_classify.py + test_mindless_wall.py
- MINDLESS when any of the 3 conditions fails
- MINDLESS does not fire when all 3 conditions hold
- WITH_VECTOR, QUALIFIED_MACRO, QUALIFIED_MICRO assigned correctly
- Exactly one primary tag per case
- Exhaustive small synthetic grid covering side × direction × velocity × macro

============================================================
7. WHAT NOT TO DO IN PHASE 1
============================================================

- No PPO / training / meta_tuner changes for this lineage
- No observation-dimension expansion on PROVEN
- No overwrite of any PROVEN checkpoint or champion docs
- No reward dials or Channel 2 plasticity yet
