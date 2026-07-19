# ADR-0012: Architecture fork RESOLVED — meta-RL confirmed (Monty, 2026-07-19)

**Context:** Cross-checking Monty's source docs (see claude/intent-drift-check-2026-07-19)
surfaced three lineages: Camillion (RL), the Autonomous/ATI-FTMO spec v2.2
(classical-ML + rules, which EXPLICITLY rejected the RL architecture), and
Momentum One (meta-RL, what we built). Momentum One re-introduced the RL approach
v2.2 had set aside — a genuine conflict. Per Monty's own conflict rule (v2.2 §12.2:
never auto-resolve, ask), we asked instead of guessing.

**Monty's ruling (2026-07-19):**
1. **Keep meta-RL (Momentum One).** RL is the current, chosen architecture. The v2.2
   classical-ML-ranking + rules approach is SET ASIDE (superseded). Do not pivot the
   brain to gradient boosting; the goal-conditioned recurrent RL policy stands.
2. **Personal account first.** Keep the adjustable goal/floor + keep-trading ratchet
   (ADR-0001). FTMO becomes a switchable MODE later (challenge mode = stop-at-target,
   real FTMO walls 2.5%/-5%/-10%/4%-trailing), NOT the near-term target.
3. **Both the daily target AND the daily risk are X% inputs.** Already implemented:
   configs/goals.yaml goal_pct + floor_pct, goal-conditioned into the observation so
   ONE brain serves any typed pair. Per-trade cap (0.25%) and heat guard stay as-is
   (Monty did not adopt v2.2's 0.2% / 1.5-2% numbers).

**What transfers unchanged (already aligned with his specs):** data/features, 4-Set
matrix, the 4 entry strategies, Shell mechanics + intrabar floor law, paranoid fills +
no-look-ahead, runtime-editable risk via percentages-in-obs, Bot-A/Bot-B verification +
tests, 5W headers + change log, MT5 + demo-first + Jarvis HUD.

**Candidate enhancements from v2.2 — NOT adopted without Monty's OK (parked, not lost):**
- Extra strategies (Gold ORB "SC", RSI/BB Slingshot Divergence, Price/BB Mean Reversion)
  entering a Trust Ladder at the bottom with no assumed edge.
- Promote BB-on-CCI ("Dimension Jump") from observation-only to a core momentum/velocity
  state per Gravity_Framework_Rules.
- Auto kill-switch triggers (2 red days each < -1.5%, abnormal single loss, predicted-vs-
  real divergence).
- Learn-from-demo-only (exclude live/challenge trades from training data).
- FTMO consistency-rule tracking (~50% single-day cap) for challenge mode.
- Early-flatten buffer (-3.9% before -4%) for order-send/fill lag, when FTMO mode is built.

**Why recorded:** this fork cost real analysis to find; writing the ruling here (and in
the change log) means no future session re-opens a decision Monty already made.
