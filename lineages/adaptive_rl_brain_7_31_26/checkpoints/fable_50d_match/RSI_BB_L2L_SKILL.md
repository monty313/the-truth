# RSI + BB on LTF under HTF mass — the real learn-to-learn skill

**When:** 2026-08-07  
**Status:** doctrine for next climb past teen 36  
**Code:** `strategies/rsi_bb_pullback_continuation.py` · `fable_kag_l2l.py` · (optional) `rsi_bb_l2l_skill.py`

---

## Step back (JSpace / KAG / harness intuition)

### What 35→36 actually taught

We did **not** yet teach “read RSI with BB applied.”  
We taught: **on many MWT days at once, when structure looks like fire_skill, do Mark’s fire** — with KL + award protect.

That is *partial* L2L (calendar-free fire family). It is **not** the full Mark instrument language.

### What Mark actually times with

```
HTF (confirmed strong trend / mass)
  = both set HTFs: price vs BB mid (bull above, bear below)

LTF (timing only — never redefines side)
  = RSI(5) with Bollinger applied ON THE RSI SERIES (not on price)
      period=10, dev=0.5, shift=+5

  Pullback load:   RSI outside the far band with the tide  → WAIT LOADED
  Continuation:    RSI crosses the release band with the tide → FIRE with HTF
```

Sets 1–4 always (MARK SETS LAW). Same construction on every stack.

### Why “just more fire_skill BC” plateaus

From teen 36: repeating the same family pack dropped (29–34).  
The model overfit **correlated** fire moments; it did not lock the **instrument geometry** that generalizes (RSI-BB load vs release under mass).

### How harness intelligence should climb next

| Layer | Role |
|-------|------|
| **Immutable base** | Teen/child act geometry; PROVEN untouched |
| **Skill teacher** | RSI-BB detector = named path laws (load / release) |
| **Mind teacher** | Mark plans = ground truth when present |
| **Concurrence** | Prefer bars where **Mark act ≡ RSI-BB principle** → learn principle, not day |
| **Multi-signal** | topology pullback/continuation + wait_loaded (learn≠copy) |
| **Fable gate** | full 50d KEEP only if same↑ breach 0 |

**Do not:** make RSI-BB the sole act teacher (proved pack crater when strategy-only).  
**Do:** make RSI-BB the **skill id** that *selects and names* which multi-day bars teach fire vs wait.

---

## Principle IDs (KAG)

- `htf_price_bb_mass` — side permission  
- `ltf_rsi_bb_geometry` — timing  
- `pullback_load_then_release` — wait then fire  
- `ltf_never_defines_side` — pt5  
- `learn_not_copy` — topology/wait with act  

---

## Instrument params (frozen)

| Piece | Spec |
|-------|------|
| LTF RSI | period **5** |
| LTF BB on RSI | period **10**, dev **0.5**, shift **+5** |
| HTF mass BB on price | period **100**, dev **0.5**, shift **+2** |
| BUY mass | close **>** mid on **both** HTFs |
| SELL mass | close **<** mid on **both** HTFs |
| BUY pullback | RSI **<** lower BB → HOLD / wait_loaded |
| BUY continuation | RSI **cross up** upper BB → BUY |
| SELL pullback | RSI **>** upper BB → HOLD / wait_loaded |
| SELL continuation | RSI **cross down** lower BB → SELL |

---

## Learn-to-learn loop (next)

```
for each practice day bar (scan sets 1–4):
  if HTF mass ok and LTF RSI-BB says load → skill=wait_skill / pullback_load
  if HTF mass ok and LTF RSI-BB says release → skill=fire_skill / continuation_release
  if Mark plan available:
     keep sample if Mark.act agrees with skill act  # concurrence
     else soft-skip or error class (do not day-memo force)
pool multi-day by skill family
BC + KL + award protect + HOLD floor
score 50d → KEEP only same↑
remember family quality in pattern memory
```

Obs still unpoisoned (no strategy flags forced into 168-d) — geometry selects **which bars teach**, Mark/concurrence selects **what label**, Fable selects **whether weights live**.

---

## Relation to 36 KEEP

| KEEP 36 | RSI-BB L2L |
|---------|------------|
| fire_skill multi-day Mark pool | **name** those fires as RSI-BB continuation under mass when concurrent |
| wait rejected alone | wait must be **pullback_load under mass**, not thrash spam |
| plateau after 36 | switch skill id to RSI-BB concurrence, smaller steps |

---

## Done criteria (skill, not score alone)

- [ ] Climb rounds attribute KEEP to `ltf_rsi_bb_geometry` + `htf_price_bb_mass`  
- [ ] Pullback samples carry wait_loaded / TOPO_PULLBACK when multi-head used  
- [ ] Continuation samples only when HTF mass ok  
- [ ] No strategy-only sole teacher  
- [ ] Pack same ≥ teen floor on KEEP  
