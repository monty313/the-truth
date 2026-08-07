# Self-climb L2L — how the policy learns to pass 35/50 on its own

**Status:** active recipe (2026-08-06) — KAG-indexed  
**Code:** `the-truth/lineages/adaptive_rl_brain_7_31_26/self_climb_l2l.py`  
**Goal:** `same_outcome > 35`, `breach == 0`, skill = path laws (not day memos)  
**Physics ADD:** `against_htf_hold`=PINN gravity · `entropy_hold`=chop/flat HOLD · `tension_req`=launch mass

---

## Why prior L2L could not climb alone

| Attempt | Result | Why stuck |
|---------|--------|-----------|
| Joint freeze trunk thrash | pack 35→32 | Act geometry moved without path skill |
| Two-phase path-only | path_class ~0.22–0.30 | Never unlocked phase B (≥0.55) |
| Phase B pathmask once | floor held 35 | No score raise |
| Act KL restore | 31–33 | Residual act mismatch kills awards |
| Struct-aug path head | still path_weak skip | **Path head never reaches score_policy** (`as_channel1` drops it) |

Root cause: training a path *classifier* does not change *actions* unless you either
(1) risk the child act head, or (2) wrap decode with a path skill layer.

---

## The way: Continual Harness + path skill decode

Borrow **prime-agent** Continual Harness + **prime-rl** hygiene:

```
IMMUTABLE BASE   = CHILD_STAGE embryo (SHA 9BDCEAAE…) — never trained
HARNESS (refine) = PathSkillDials + optional TinyPathGate
SKILLS           = thrash/pullback wait · against-HTF hold · continuation fire
META             = diagnose MWT structure laws → boost dial → rescore → KEEP/REJECT
VERIFY           = same ≥ floor · breach 0 · promote only if same > 35
ROLLBACK         = dials only (child weights never moved)
```

### Path skill at score time

`PathSkillPolicy.act(obs)`:

1. Child greedy action (frozen Channel1).
2. Read structure already in full_obs:
   - sets 1–4 dirs, pullback@27, doctrine force/play/regime @32–45
3. Apply MARK path laws:
   - **ltf_pullback / weak HTF / chop** → force HOLD (anti_thrash)
   - **fire against HTF** → HOLD
   - **continuation + launch/aligned under strong HTF** while child HOLDs → fire with HTF
4. Optional tiny neural gate on structure slice (trained offline; never touches child).

### Meta-learn (learn to learn)

1. Seed dial grid (thrash × cont × against × launch_req).
2. Score each candidate on practice 50d.
3. KEEP if `same > best` and `same ≥ 35` and `breach == 0`.
4. After grid: diagnose MWT days’ structure-law histogram → refine dials.
5. Loop until `same ≥ goal` (default 36) or budget.

No focus-day BC. No calendar memos. Skill dials are the only refinable state.

---

## Artifacts

| Path | Role |
|------|------|
| `SELF_CLIMB_HARNESS__latest.json` | best dials + cycles |
| `SELF_CLIMB_MEMORY.jsonl` | round evidence |
| `SELF_CLIMB__latest.md` | human report |
| `path_skill_self_climb_v1.pt` | promoted skill (only if same>35) |
| `BEST__latest.json` | updated only on promote past 35 |

**Does not overwrite** `mark_clone_full_obs_v1.pt` on promote — child stays sacred; scoring uses the adapter.

---

## Run

```powershell
cd $truth
$env:PYTHONPATH=".;code"
python -u lineages/adaptive_rl_brain_7_31_26/self_climb_l2l.py `
  --max-rounds 18 --keep-floor 35 --goal-same 36
```

Leave other terminal `climb_35_with_strategies.py` alone (writes CKPT only on KEEP).

---

## Done criteria

- [ ] `best_same > 35` with breach 0  
- [ ] `growth_method=self_climb_path_skill` in BEST  
- [ ] skill attributed to path dials / laws, not a single MWT day id  
- [ ] child SHA unchanged  

---

## Binding with L2L R1–R10

- R: child floor sacred → base never trained  
- R: learn ≠ copy → no day BC into act head  
- R: path laws > calendar → dials + structure from obs  
- R: MARK SETS LAW → HTF from regime/force/set3–4; LTF set1–2 + pullback flag  

---

## Does `physics.md` help past 35?

**Yes — only as path-skill *decode* laws. No as act-head BC / PINN loss retrain.**

| Pillar | Helps >35 now? | L2L safe? | How wired |
|--------|----------------|-----------|-----------|
| PINN HTF tide penalty | **Yes** | Decode only | `against_htf_hold` → force HOLD if fire opposes HTF |
| Entropy regime mask | **Yes** | Decode only | `entropy_hold` + thrash → HOLD on chop/flat/conflict |
| Kinematic tension / a_mass | **Partial** | Decode only | `tension_req` uses force + set3/4 mass proxy (no aux head retrain) |
| Dimensionless ATR tensors | Later | **No now** | Changes obs → retrain → child SHA / floor risk |
| PINN in `train_mark_clone_bc` CE | **No for this push** | **Violates R: base never trained** | REJECT until after >35 promote path is proven |

Master equation at score time (not Pattern X → Buy):

```
[High Tension + Low Entropy + HTF Mass OK] → Launch with tide
else → wait / HOLD
```

Physics Super Agent’s “inject PINN into BC loss” is **ADD later**, not the R1 climb lever. The climb lever is `self_climb_l2l.py` dials + frozen child.