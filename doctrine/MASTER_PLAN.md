# MASTER PLAN — Autonomous Self-Heal Execution

CHANGE LOG:
- 2026-07-24  Phase 1–3 infrastructure committed — WHY: Mind Probe CLI, Ghost Trades, IRAC diagnose, policy_skill; w_pullback unlocked in tuner plan.
- 2026-07-24  created — WHY: Monty authorized full autonomous execution.
# NEXT EDITOR: append dated status updates.

---

## Objective

Self-correcting RL scalper: dynamic target/floor, Gravity Framework, evolve only via reward shaping + policy adaptation, Diagnostic LLM (MRI + Ghost Trades + IRAC), RL must see chart patterns that produce consistent clears.

Baseline: `PROVEN_SPRINT_row04_clear24` — 24/90 cleared, row 4, zero breaches.

---

## Phase 0 — Doctrine Lock  ✅ DONE

- [x] Branch `fable5/self-heal-plan`
- [x] `doctrine/STANDING_LAWS.md`
- [x] Residual Law-0 / TF freeze documented

---

## Phase 1 — MRI Scanner / Mind Probe  ✅ DONE

- [x] `telemetry/mind_probe.py`
- [x] `scripts/mind_probe_day.py`
- [ ] Batch Perception scoreboard over 90 days (run on machine with data+brain)

```
python scripts/mind_probe_day.py PROVEN_SPRINT_row04_clear24_2026-07-20 42 3.0 3.5
```

---

## Phase 2 — Ghost Trades  ✅ DONE (module)

- [x] `telemetry/ghost_trades.py`
- [ ] Full DaySim counterfactual PnL (optional upgrade; pattern-alignment ghosts ship now)

---

## Phase 3 — Conversational Diagnostic Surface  ✅ DONE (entry)

- [x] `doctrine/policy_skill.md`
- [x] `scripts/diagnose_day.py` (Mind + Ghosts + IRAC proposal)
- [x] Bounded rewards_delta schema (not auto-applied)

```
python scripts/diagnose_day.py PROVEN_SPRINT_row04_clear24_2026-07-20 42 3.0 3.5
```

---

## Phase 4 — Consistency Self-Heal Loop  🔧 WIRED / RUN ON GPU HOST

- [x] meta_tuner already implements gated adopt (McNemar day-paired)
- [x] `w_pullback_with_htf` added to BOUNDS (so bread-and-butter can be tuned)
- [ ] Run meta_train / self_tune seeded from PROVEN_SPRINT on curriculum host
- [ ] Re-measure clear-rate + clean row after adopts
- [ ] Update policy_skill.md with measured evidence

```
python scripts/meta_train.py   # or scripts/self_tune.py — existing entry points
python scripts/prove_it.py <brain> <target%> <risk%>
```

---

## Phase 5 — Multi-symbol + MT5

Only after single-symbol consistency climbs. Existing `build_symbol_set` / `run_live.py` paths.

---

## Invariants

1. No core weight retrain from scratch
2. Obs space unchanged
3. Evolution = rewards / skill-doc + gated adopt
4. Never "impossible" without measured bound
5. Standing Laws govern IRAC

---

## How Monty knows this is finished (infrastructure)

Infrastructure phases 0–3 are on branch `fable5/self-heal-plan`.  
Phase 4 measured climb requires GPU/data host with PROVEN brains — run commands above.  
PR to main = review gate for merge.
