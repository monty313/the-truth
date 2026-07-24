# Policy Skill Document — Momentum One
<!-- SkillOpt-style trainable artifact. Evolves only via evidence-gated edits.
WHO: Fable 5 + Monty. WHAT: current preferred patterns, known blind spots, active reward emphases.
WHEN: seeded 2026-07-24. WHERE: judged against STANDING_LAWS.md. WHY: emergence without core retrain. -->

CHANGE LOG:
- 2026-07-24  seeded — WHY: Phase 3 SkillOpt surface; baseline from standing doctrine + PROVEN_SPRINT state.
# NEXT EDITOR: append dated evidence-backed edits only.

---

## Preferred patterns (priority order)

1. **Bread-and-butter (primary consistency pattern)**  
   LTF pullback while **both** higher TFs remain strongly trending.  
   Enter in the direction of HTF gravity on the LTF pull.  
   Flags in obs: `set{k}::pull_buy` / `set{k}::pull_sell` with HTF cont present.

2. **Continuation** when LTF and HTFs are aligned (high-velocity momentum states).

3. **Reversal states** (Law 1 first-class): Bearish Reversal / Bullish Reversal when the 6-set is perfectly aligned. Not exceptions.

4. Additional trade types may emerge via evidence; regime matrix is an open starting set.

---

## Known constraints

- Observation space frozen (no new indicators).
- Core weights not retrained from scratch; adaptation only via reward shaping + short PPO probes under meta_tuner.
- Shell: 0.25% risk cap, floor, ratchet, 400 trades/day, flat at 00:00 CEST.
- Law 0 dual-TF SMA gate currently approximated in forever masks (documented residual).

---

## Active reward emphases (starting)

From `configs/rewards.yaml` (champion may diverge via meta_tuner):
- `w_net_profit` leads
- `w_did_nothing` is large negative (must act to hit goal)
- `w_pullback_with_htf` is small positive (nudge) — **candidate to raise** when Mind Probe shows high-miss pull
- `w_day_goal_hit` / `w_streak_per_day` reward consistency

---

## Known / suspected blind spots (to verify with Mind Probe)

- Policy may sit flat on days with clear pull flags → Perception or Policy (IRAC).
- Reversal states may be under-represented in action mass (Law 1 must remain legal).
- Hierarchical HTF vs LTF composition must stay relational (Law 2) — no lone-indicator behavior.

---

## How this document evolves

1. Run `scripts/diagnose_day.py` on miss / flat / broken-row days.
2. IRAC proposal cites Mind Probe + Ghost Trades.
3. Bounded `rewards.yaml` delta only.
4. Accept only if `meta_tuner.adopt_gate` + non-backslide audit pass.
5. Append a CHANGE LOG line with the measured evidence (clear-rate / row delta).

Never claim impossible without a measured bound. Leverage 1:100 — ask how we make it happen.
