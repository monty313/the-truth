# MASTER PLAN — Autonomous Self-Heal Execution
<!-- 5W+I: WHO Fable 5 (Grok + team) under Monty’s authorization. WHAT full path from current PROVEN_SPRINT to consistent daily clears under dynamic target/floor.
WHEN started 2026-07-24 while Monty at work. WHERE github.com/monty313/the-truth branch fable5/self-heal-plan.
WHY Project Instructions + refined standing laws. INTERCONNECTED WITH: STANDING_LAWS.md, training/meta_tuner.py, telemetry/, evaluation/consistency.py, configs/rewards.yaml -->

CHANGE LOG:
- 2026-07-24  created — WHY: Monty authorized full autonomous execution of the self-healing plan without further permission; added Perception requirement (RL must see chart patterns that produce consistent clears).
# NEXT EDITOR: append dated status updates.

---

## Objective

Self-correcting RL scalper that, given dynamic inputs `target x%` / `risk x%`:
- clears the target day after day without breaching the floor
- trades multiple symbols via MT5 under the Gravity Framework
- evolves **only** through reward shaping + policy adaptation (core weights and obs space frozen)
- is overseen by a conversational Diagnostic LLM that uses MRI telemetry + IRAC to prescribe evidence-backed YAML cures

Current measured baseline: `PROVEN_SPRINT_row04_clear24` — 24/90 cleared, longest clean row 4, zero breaches.

---

## Phase 0 — Doctrine Lock  ✅ IN PROGRESS

- [x] Branch `fable5/self-heal-plan` created
- [x] `doctrine/STANDING_LAWS.md` written from 2026-07-24 session (Laws 0–2, TF hierarchy, bread-and-butter, Perception, SkillOpt, IRAC)
- [ ] Align `docs/LAWS.md` and flea-jar references to the new standing document
- [ ] Record residual Law-0 approximation and TF-set freeze decisions so no future editor “fixes” them by breaking obs

---

## Phase 1 — MRI Scanner / Mind Probe

Goal: make the policy answerable in conversation so Fable 5 can diagnose Perception vs Policy vs Generalization.

Deliverables:
1. Extended span / mind-dump that, for every decision bar on an eval day, records:
   - Active set / state flags (continuation, pullback, reversal, neutral) derived from existing features
   - Policy action probabilities (op distribution + size stats)
   - Chosen action vs Shell-effective action (separates neural intent from vetoes/cages)
   - Self-state snapshot (dist_to_goal, dist_to_floor, open_risk, streak, etc.)
2. Day-level JSON mind dump that can be loaded by a diagnostic script
3. Query helpers: “on day D, when both HTFs were trending and LTF showed pullback, what probability did you assign to the correct direction?”

Constraint: read-only with respect to weights and obs. Logging only.

Status: starting immediately after Phase 0 commit.

---

## Phase 2 — Ghost Trades

Goal: supply the hard evidence required by IRAC Application.

For selected decision bars (especially high-conviction or high-miss bars):
- Compute the counterfactual reward / contribution that alternative ops would have produced
- Store as Ghost Trade records attached to the mind dump

This proves “what the day would have looked like if you had taken the pullback long instead of holding.”

---

## Phase 3 — Conversational Diagnostic Surface (SkillOpt-style)

Goal: turn diagnosis into a living dialog + bounded skill edits.

1. `doctrine/policy_skill.md` — evolving natural-language skill document that describes current preferred patterns, known blind spots, and active reward emphases (the SkillOpt trainable artifact)
2. IRAC proposal schema that outputs only `rewards.yaml` key/value deltas + rationale grounded in mind-dump evidence
3. Diagnostic entry point (`scripts/diagnose_day.py` or notebook cell) that loads brain + day, reconstructs mind, and supports turn-by-turn reasoning

Perception focus: explicitly test whether the policy’s probability mass tracks the bread-and-butter pattern and other consistent-clear patterns. If the pattern is present in features but probability stays near zero → Perception issue → reward shaping that makes correct recognition pay.

---

## Phase 4 — Consistency Self-Heal Loop

Goal: climb clear-rate and clean-row from the current 24/90 + row-4 baseline.

1. Seed all work from `PROVEN_SPRINT_row04_clear24` (or later serial-stamped champions)
2. Use Mind Probe + Ghost Trade evidence to generate candidate reward mutations focused on:
   - Amplifying `w_pullback_with_htf` and related hierarchical signals
   - Strengthening the penalty for “did nothing” on days that contained clear bread-and-butter setups
   - Keeping reversal states available (do not re-introduce counter-trend fear)
3. Every candidate still passes through `meta_tuner.adopt_gate` (day-paired McNemar + two fresh confirms + non-backslide audit)
4. After each accepted change: re-measure full 90-day clear-rate, longest clean row, breach rate, average day
5. Update `policy_skill.md` with what was learned

Never claim a day is unwinnable without a measured bound (Antibody Law).

---

## Phase 5 — Multi-symbol + MT5 Hardening

Only after single-symbol consistency is demonstrably climbing.
- Fold additional symbols using existing `build_symbol_set` path
- Harden live MT5 bridge and session handling
- Re-validate Shell + Law 0 approximation under multi-symbol load

---

## Measurement Standard

Every claim of improvement is measured with:
```
python scripts/prove_it.py <brain> <target%> <risk%>
```
on the full curriculum, greedy, under the corrected ratchet law.
Serial numbers on every champion. Byte-verified where possible.

Primary metrics:
- Clear rate (days that hit target with zero breach)
- Longest clean row
- Breach rate (must stay 0 or near-0)
- Average day PnL
- Green-day fraction

Secondary: evidence that the policy’s action probabilities now track bread-and-butter and other consistent patterns (Perception score from Mind Probe).

---

## Invariants (checked on every commit)

1. Core weights not retrained from scratch; only adaptation on top of proven seeds
2. Observation space dimension and feature definitions unchanged
3. All evolution is a rewards.yaml (or skill-doc) change that must pass the consistency gate
4. Shell / mask changes do not alter obs
5. Never “impossible” — only measured bounds
6. Standing Laws govern every IRAC conclusion

---

*Execution is autonomous under Monty’s 2026-07-24 authorization. Status updates will be appended here and in HANDOFF notes as phases complete.*
