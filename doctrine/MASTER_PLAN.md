# MASTER PLAN — Autonomous Self-Heal Execution

CHANGE LOG:
- 2026-07-24  COMPLETE (infrastructure) — WHY: residual w_pullback unlock applied; perception scoreboard + HOST_RUN playbook; unit tests green.
- 2026-07-24  Phase 1–3 infrastructure committed.
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
- [x] `doctrine/HOST_RUN.md` host playbook

---

## Phase 1 — MRI Scanner / Mind Probe  ✅ DONE

- [x] `telemetry/mind_probe.py`
- [x] `scripts/mind_probe_day.py`
- [x] `scripts/perception_scoreboard.py` (batch Perception metric)

```
python scripts/mind_probe_day.py PROVEN_SPRINT_row04_clear24_2026-07-20 42 3.0 3.5
python scripts/perception_scoreboard.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 90
```

---

## Phase 2 — Ghost Trades  ✅ DONE

- [x] `telemetry/ghost_trades.py` (pattern-alignment counterfactuals for IRAC Application)

---

## Phase 3 — Conversational Diagnostic Surface  ✅ DONE

- [x] `doctrine/policy_skill.md`
- [x] `scripts/diagnose_day.py` (Mind + Ghosts + IRAC proposal)
- [x] Bounded rewards_delta schema (not auto-applied)

```
python scripts/diagnose_day.py PROVEN_SPRINT_row04_clear24_2026-07-20 42 3.0 3.5
```

---

## Phase 4 — Consistency Self-Heal Loop  ✅ WIRED (climb on host)

- [x] meta_tuner gated adopt (McNemar day-paired) — pre-existing
- [x] `w_pullback_with_htf` in BOUNDS + FALLBACK (actual code)
- [x] Warm-start prefers PROVEN_SPRINT
- [x] `tests/test_self_heal_mri.py` — ALL PASSED
- [x] `doctrine/HOST_RUN.md` — exact host commands
- [ ] Run meta_train / prove_it on GPU host with curriculum + brains (Monty machine)

```
python tests/test_self_heal_mri.py
python scripts/meta_train.py
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

---

## Phase 5 — Multi-symbol + MT5  ⏳ AFTER CLIMB

Existing paths: `build_symbol_set`, `scripts/run_live.py`.  
Do not start until single-symbol clear-rate / row climbs under prove_it.

---

## Invariants

1. No core weight retrain from scratch  
2. Obs space unchanged  
3. Evolution = rewards / skill-doc + gated adopt  
4. Never "impossible" without measured bound  
5. Standing Laws govern IRAC  

---

## Finished means

**Infrastructure finished:** Phases 0–3 complete; Phase 4 wired + tested; PR open.  
**Product climb finished:** after host runs HOST_RUN.md and clear-rate / row improve under prove_it.
