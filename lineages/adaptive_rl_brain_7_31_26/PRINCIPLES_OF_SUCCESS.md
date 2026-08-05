# Principles of Success — student & tutor

**Student:** me (learning how the winning multi-pair bot thinks).  
**Tutor:** the **policy stack that actually produced the results** — not a chat model, not pure RL folklore.

**Tutor’s body:**

| Piece | Where |
|-------|--------|
| Direction (eyes) | `GoalEquityDay.recommended_action` + `DayRunner` structure (higher TF → lower fallback) |
| Hands (risk shell) | `GoalEquityDay` bank / heat / size / stop / every-bar marks |
| Personality dials | `checkpoints/multi_pair_dials.json` + ckpt |
| Proof | `checkpoints/ten_pair_score_all.json` · `score_ten_pairs.py --mode all` |

**Claim the tutor earned (all 90 days, same brain, no retrain per pair):**  
**10 / 10** pairs · each **≥ 30 clear days** · **0% breach** · checkpoint `multi_pair_consistent_v1.pt` · decode **`heuristic`**.

**Mission language (GOAL.md):** climb **clear%**, keep **breach% = 0**, **any typed target%/risk% without retrain**.

---

## Student: “Tutor, who are you?”

**Tutor:** I am not “argmax of a frozen MLP.” I am a **closed loop**:

1. **See** structure direction the same way flat and in-trade.  
2. **Act** only if heat allows and I am not banked/dead.  
3. **Size** from **runtime risk%** and distance to the floor.  
4. **Mark every M1 bar** for stop and breach (not only decision bars).  
5. **Bank** when equity% hits **your** target%.  
6. **Die honestly** if equity% hits **your** floor%.

Same weights and dials for 1.0/2.0 and 3.0/3.5. The day inputs change; I do not retrain.

---

## Part A — Decision logic (what the tutor does each bar)

### A1. Runtime goal and floor (not baked into weights)

**Tutor:** Target% and risk% enter the day as constructor inputs. Observation also carries **progress_to_goal** and **danger** so any learner sees how close we are to bank or floor.

**Student takeaway:** Without runtime target/risk in the episode, “any pair without retrain” is a lie.

**Evidence:** `GoalEquityDay.__init__(target_pct=..., risk_pct=...)`; `observe()` writes progress/danger into Channel1 slots.

---

### A2. One direction signal — flat and in-trade

**Tutor:** I force **flat perception** for direction (clear `runner.position` briefly), then:

- **Flat:** follow structure (BUY / SELL / HOLD).  
- **In trade:** reverse **only** if signal is the **opposite** side; else HOLD (manage).

I do **not** use a weaker “in-trade only” eye. That froze high-target days.

**Student takeaway:** Same eyes open and managed. Reverse on flip, not on noise.

**Evidence:** `recommended_action()` — IRAC-03 KEEP. Clear climb on high pairs after this change (claim pairs 9–10: 48 and 40 clears).

---

### A3. Heat and refuse open

**Tutor:** Before open, distance to floor = equity − (−risk). I only risk a **fraction** of that distance (`risk_use_frac`, dialed **0.35**), scaled by how tight the floor is (`floor_scale` from runtime risk%). Cap per trade (`per_trade_cap_pct` **0.25** of equity). If residual heat is ~0 → **no open**.

**Student takeaway:** Entries that would kiss the floor are refused. Breach protection is **before** the trade, not after the funeral.

**Evidence:** `_try_open()` heat math; multi-pair claim **0 breaches** on all 10 pairs.

---

### A4. Floor-scale size and stop

**Tutor:** Tight risk% → smaller size and slightly tighter ATR stop. Wide risk% → more room. **Same code path** for every pair.

**Student takeaway:** Switching risk% does not need a new network. Physics scales.

**Evidence:** `floor_scale = clip(risk/2.5, …)` in `_try_open`; dials in `multi_pair_dials.json`.

---

### A5. Every-bar marks (honest path)

**Tutor:** Decisions may fire every 25 M1 bars, but **between** decisions I still walk every bar: stop hit, worst-case equity, bank, breach. Gaps cannot walk through the floor unseen.

**Student takeaway:** Clear/breach that only check on decision bars are dishonest. The claim used honest marks.

**Evidence:** `run()` loop → `_mark_bar` for `bt in range(prev_t, t)`; IRAC-02 KEEP (8/10 → later 10/10 after signal unity).

---

### A6. Bank at target

**Tutor:** When equity% ≥ target%, flatten and **bank**. After bank I HOLD. I stop digging for “more.”

**Student takeaway:** Clear is a finish line. Overtrading after clear invites floor hits.

**Evidence:** `_check_breach_and_bank`; result field `banked=true` on many clear days in `ten_pair_score_all.json`.

---

### A7. Breach is death for the day

**Tutor:** If worst equity% (intrabar on open risk) ≤ −risk%, I mark **breached**, flatten, dead for the day. That day cannot be a clear.

**Student takeaway:** Breach is binary under GOAL. One floor touch ruins the day.

**Evidence:** `cleared = goal_hit AND NOT breached` logic in result assembly; claim `breached: 0` every pair.

---

### A8. Decode that won the bar

**Tutor:** For the multi-pair claim, action = **`recommended_action` (heuristic)**. Tiny Channel1 weights may be BC’d to imitate that, but **winning decode is structure + shell**, not pure greedy RL alone (that path freezes / fails the claim).

**Student takeaway:** Do not confuse “MLP exists” with “MLP is the tutor.” The tutor is the **loop**.

**Evidence:** `multi_pair_dials.json` → `"decode": "heuristic"`; score path uses heuristic when configured; IRAC REJECT pure greedy alone.

---

## Part B — Principles of success (KEEP)

Each principle is tied to **clear / breach / runtime pairs / no retrain**.

| # | Principle (tutor’s law) | GOAL outcome | What taught it |
|---|-------------------------|--------------|----------------|
| **P1** | **Speak equity %.** Clear and breach are percent of starting equity vs **typed** target and floor. | Same score language as GOAL / prove_it | Equity day engine |
| **P2** | **Target and risk are runtime inputs.** Never bake one pair into weights. | Any pair without retrain | Constructor + obs progress/danger |
| **P3** | **Size from remaining floor heat.** Refuse open when heat is gone. | Breach → 0 | Heat + risk_use_frac |
| **P4** | **Scale size (and stop) with runtime risk%.** Tight floor → smaller book. | Same brain, many risks | floor_scale in `_try_open` |
| **P5** | **Mark every bar for stop/breach/bank.** Decision stride is not the universe. | Honest breach 0 | IRAC-02; `_mark_bar` |
| **P6** | **One signal flat and in-trade.** Reverse only on opposite structure. | Higher clear on hard targets | IRAC-03; 10/10 claim |
| **P7** | **Bank at target.** Stop after clear. | Clear days stick | banked path |
| **P8** | **Dials over dogma.** Search `risk_use_frac`, `stop_atr_mult`, `per_trade_cap_pct`; freeze winners in ckpt. | Reproducible multi-pair personality | dials JSON + IRAC recipe |
| **P9** | **Prove on the real scorer.** Claim = all days, 10 pairs, ≥30 clear, 0 breach; re-run twice. | No “felt good” wins | `score_ten_pairs --mode all` |
| **P10** | **Protect floor first.** If clear and breach fight, protect breach. | Breach sacred | IRAC-01 mindset |

---

## Part C — Anti-principles (REJECT)

| # | Anti-principle | What happened | GOAL link |
|---|----------------|---------------|-----------|
| **R1** | **Trail stop + big cushion + scale-in as a package** | Multi-pair pass **6/10 → 0/10** (breaches everywhere) | Failed breach 0 |
| **R2** | **Stops only on decision bars** | Floor walk-through between strides | Fake “safe” days |
| **R3** | **Different weaker signal while in trade** | High targets stuck ~18–23 clears | Clear stuck |
| **R4** | **Pure greedy RL alone as the multi-pair solver** | Freeze / no claim win | Wrong win condition |
| **R5** | **Huge dial grids on full rebuilds** | Too slow; thrash without learning | Wasted practice |
| **R6** | **Promote on “entries up” without clear/breach** | Not GOAL | Skip |

**Tutor’s hard lesson (R1):** clever exits and scale-ins that feel “more active” can **destroy** multi-pair breach. Revert fast.

---

## Part D — How a student uses the tutor tomorrow

1. Before any change: re-read **P3, P5, P6, P10**.  
2. One change only → run `score_ten_pairs.py --mode all` (or `prove_it` at **your** pair).  
3. Keep only if **clear↑ or equal** and **breach still 0** on the pairs that matter.  
4. If trail/scale “helpers” reappear as a bundle → recall **R1**.  
5. Channel1 HOLD/regret experiments stay **sandbox** until they drive this shell and raise prove_it clear without breach.

---

## Part E — Anchors (do not invent)

| Anchor | Value |
|--------|--------|
| Checkpoint | `lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_consistent_v1.pt` |
| Dials | `risk_use_frac=0.35`, `stop_atr_mult=2.0`, `per_trade_cap_pct=0.25`, `decode=heuristic` |
| Score (claim) | `ten_pair_score_all.json` — **10/10 pass**, 0 breach |
| Example clears (all 90d) | pair1 76 · pair5 70 · pair8 60 · pair10 40 |
| Process log | `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md` |
| GOAL extract | `references/plans/GOAL_FROM_TEN_PAIR_IRAC.md` |
| Engine | `lineages/adaptive_rl_brain_7_31_26/equity_day.py` |

---

## Closing dialogue

**Student:** So success was not “smarter HOLD logits”?  

**Tutor:** Correct. Success was **honest equity physics**, **runtime target/risk**, **heat-aware size**, **every-bar marks**, and **one structure eye** — scored as **clear days and zero breaches on ten typed pairs without retrain**. Learn that loop; then attention (RL) may serve it. Never replace it with folklore.
