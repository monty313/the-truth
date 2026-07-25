# Policy Skill Document — Momentum One
<!-- SkillOpt trainable state. Frozen LLM + frozen RL weights.
     Only evidence-gated edits. Validation = prove_it clear% up, breach 0%. -->

CHANGE LOG:
- 2026-07-25  agreement obs slots 80–83 — WHY: high-precision suggestions; hesitation under them is Policy.
- 2026-07-24  SkillOpt memory sections + Monty thinking anchors — WHY: wire skill memory for diagnostic LLM.
- 2026-07-24  seeded — WHY: Phase 3 SkillOpt surface.
# NEXT EDITOR: gated edits only; prove_it must improve or stay flat with breach 0.

---

## Standing Gravity rules

1. **Bread-and-butter** — LTF pullback while both HTFs strongly trending (pull flags + HTF bias).
2. **Continuation** — LTF aligned with HTF (cont flags).
3. **Reversal** — first-class when structure flips; not banned.
4. Open set — new combinations only with Ghost + prove_it evidence.

## How Monty thinks (do not dilute)

- Consistency > hero days.
- Never mark a swing-bound day impossible.
- Hesitation when setup is visible is a **Policy** disease (fix incentives), not a market veto.
- Multi-TF composition always; no lone indicator.

## Active reward emphases

- `w_pullback_with_htf` — bread-and-butter (raised toward 0.25 after policy_hold IRAC).
- `w_did_nothing` — large negative (must engage to hit goal).
- `w_day_goal_hit` / `w_streak_per_day` — consistency.
- `w_death_penalty` — floor is sacred.

## Known failure modes (from telemetry)

- High `policy_hold` on firm HTF + LTF pull/cont → Policy class → raise pullback weight / frontier train.
- After TF-set realign, old brains need re-prove + sprint (semantic obs shift).
- Mask veto is rare relative to policy_hold — do not blame masks first.

## Agreement suggestions (obs slots 80–83)

Full evidence: `PERFORMANCE_IS_POSSIBLE_PART4.md`. Code: `signals/agree.py`.

| Slot | Meaning |
|------|---------|
| 80 `agree_seA_r2A` | stoch_ema_A and rsi2_ema_A same side (~75% @10) |
| 81 `agree_seB_r2B_epB` | 2-of Set B stoch/rsi2/ema_pull (~70–72%) |
| 82 `agree_2of_top4` | 2-of top four families (~76/71% @10/20) |
| 83 `agree_seA_r2A_atr` | 80 + ATR active (~78–81%, rarer) |

**Skill rule:** Non-zero agreement under firm HTF Gravity is a **high-value engage**
context. Holding while these fire is the same disease class as holding a visible
bread-and-butter pull — fix incentives, do not invent a new lid.


## SkillOpt memory rules

- Edit this file only via gated path (`scripts/skillopt_gate.py` or manual IRAC with prove_it).
- Rejected edits → `artifacts/skills/rejected/`.
- Last accepted snapshot → `artifacts/skills/best_skill.md`.
- Trajectories → `artifacts/llm_curriculum/`.
