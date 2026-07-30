# STANDING LAWS — Momentum One
<!-- 5W+I: WHO Monty + Fable 5. WHAT permanent trading + evolution laws.
WHEN locked 2026-07-24 from live doctrine session. WHERE every module, every reward, every diagnosis.
WHY consistency under dynamic target/floor without catastrophic forgetting.
INTERCONNECTED WITH: configs/*, training/meta_tuner.py, telemetry/*, evaluation/consistency.py,
features/*, doctrine/flea-jar/*, MASTER_PLAN.md -->

CHANGE LOG (newest first):
- 2026-07-24  created — WHY: Monty refined Laws 0–2, TF sets, bread-and-butter, Perception requirement, and SkillOpt emergence while authorizing autonomous self-heal execution.
# NEXT EDITOR: append dated WHY; keep this line.

---

## Optimization Constraint (never suspended)

All behavioral evolution happens through **dynamic reward-shaping and policy adaptation only**.

- Do **not** retrain core neural network weights from scratch.
- Do **not** alter the observation space (indicators / feature dimensions).
- Proven baselines (PROVEN_SPRINT, PROVEN_LIFT, etc.) are preserved; new behavior is gated by the day-paired consistency test.
- The only trainable surface for emergence is the reward configuration + the evolving skill document.

---

## Law 0 — Dual-TF SMA Gate (+ CCI dual Shell)

### Price SMA gate (live 2026-07-30)
- **No buys** when close **below** SMA(4, shift +4) on **both** **1m and 15m**.
- **No sells** when close **above** SMA(4, shift +4) on **both** **1m and 15m**.
- Opens/adds/probes only; hold/close free.

### CCI dual regime gate (live 2026-07-30)
- **No sells** when **both** CCI(30) and CCI(100) are **> 0** and **each above its applied SMA(2)+2**, on **5m OR 30m** (either TF).
- **No buys** when **both** CCI(30) and CCI(100) are **< 0** and **each below its applied SMA**, on **5m OR 30m**.
- Both CCI periods required; one period alone does not activate.

### Legacy envelope
- Still OR’d: 15m/30m/1h env high/low forever-masks (fail-closed on warmup).

**Obs dim unchanged** (`mask_buy_blocked` / `mask_sell_blocked` only). Semantic change → re-prove brains; wipe GPU feature caches after deploy.

---

## Law 1 — Regime Direction (corrected)

Counter-trend and reversal trades are **legal and first-class**.

The 6-set Momentum-Velocity matrix defines four absolute states:
1. High Velocity Bull Momentum (continuation)
2. High Velocity Bear Momentum (continuation)
3. Bearish Reversal (still Bull Territory, velocity flipped South)
4. Bullish Reversal (still Bear Territory, velocity flipped North)

States 3 and 4 are not exceptions. They are intended scalper setups when the full 6-set is perfectly aligned.
Neutral / incomplete alignment → no new momentum entries.

---

## Law 2 — Relational Composition (no lone indicators)

Every indicator has exactly one primary job (gravity filter, volatility/movement filter, entry timing, or exit/management).
That job is one component of the larger multi-timeframe Gravity / Momentum-Velocity picture.

**No indicator works alone.**
- Fast is always relative to Slow (same timeframe).
- Lower TF is always relative to the same indicator family on the higher timeframes.

Isolation is illegal. Composition and hierarchy are mandatory. This is why the 6-set exists.

---

## Multi-Timeframe Architecture

### Target sets (Monty 2026-07-24)
All sets have exactly three timeframes:

1. **1m · 15m · 30m**
2. **5m · 1h · 4h**
3. **15m · 4h · 1d**

Inside every set:
- **First TF = lowest** → execution, timing, risk (scalper entry)
- **Last two = higher timeframes** → dominant regime, bias, structure

### Current feature pipeline (frozen)
`configs/timeframes.yaml` still contains the original 4-set interlocking matrix (set1 1m/15m/30m, set2 5m/30m/1h, set3 15m/1h/4h, set4 30m/4h/1d + extras). Changing those TFs would alter obs_dim and invalidate frozen PROVEN brains → **forbidden** under the Optimization Constraint.

**Rule:** Target architecture lives in doctrine. Behavior is driven toward the hierarchical HTF-trend + LTF-pullback pattern by reward shaping and diagnosis, not by rewriting the feature engine.

---

## Bread-and-Butter Setup (primary consistency pattern)

**Pullbacks on the lower timeframe while both higher timeframes remain strongly trending.**

In every set:
- The two higher TFs define strong directional gravity (high-velocity continuation).
- The lowest TF supplies the pullback against that gravity.
- The scalper enters in the direction of the higher-TF trend on that lower-TF pullback.

This is the pattern the bot must prioritize for day-after-day clears. When the policy sits flat on a clear bread-and-butter day, that is a diagnostic event (Perception, Policy, or both).

---

## Perception Requirement (RL model must see the chart)

The RL policy must learn to **see the chart** from the existing observation vector and recognize the patterns that correlate with consistently passing the daily target / floor inputs.

Specifically it must be able to distinguish, from the 1820-dim obs:
- HTF trend strength and direction (both higher TFs)
- LTF pullback / tension relative to that HTF gravity
- Perfect 6-set alignment vs incomplete alignment
- Continuation vs Reversal states
- Volatility regime (Nothing Happening / Tradable / Great Movement)

Recognition is not hard-coded. It emerges through reward shaping that pays for correct action on the recognized pattern and through the Diagnostic LLM verifying (via Mind Probe) that the policy’s internal probabilities actually track those patterns. Failure to recognize a clear bread-and-butter setup is classified as a **Perception** issue under IRAC.

---

## Regime Matrix — Open Starting Set

The Gravity × Volatility combinations defined so far are a **baseline**, not a closed universe.
The bot must be free to discover and validate additional trade types through evidence (SkillOpt-style reflection + gated acceptance). Emergence is required.

---

## Diagnostic Method (IRAC)

When consistency fails, the Diagnostic LLM (Fable 5) acts as Chief Medical Officer:

1. **Issue** — what failed (missed clear, breach, flat on rich day, broken row)
2. **Rule** — which standing law / Gravity principle / bread-and-butter rule applies
3. **Application** — hard evidence from Mind Probe (action probs, chosen vs effective, Ghost Trades, set flags)
4. **Conclusion** — evidence-backed, bounded reward-shaping tweak (YAML only)

Diagnosis classes:
- **Perception** — blindness to the setup (policy never assigned high probability to the correct action when the pattern was present)
- **Policy** — fear / bad incentives (pattern was seen but reward shape made the bot sit out or size wrong)
- **Generalization** — overfitting to one regime (works on some days, fails on structurally similar others)

---

## SkillOpt Emergence Rule

Treat the reward surface + evolving skill document as the trainable parameters of a frozen agent (exactly as SkillOpt treats a skill Markdown file).

- Roll out / evaluate under the current rewards
- Reflect with Mind Probe + Ghost Trade evidence
- Propose bounded edits
- Accept only if the day-paired consistency gate + non-backslide audit pass
- Never claim “impossible” without a measured bound. We have 1:100 leverage. The question is always “how do we make this happen?”

---

## Communication Rules (Fable 5)

- Speak in multi-timeframe roles: HTF = regime/bias, Mid = structure/acceptance, LTF = execution/timing/risk
- Prefer dialog over data dumps
- Reason with Monty turn-by-turn about what the policy saw, thought, and did
- Scalper language: quick in/out, tight risk, high opportunity density relative to the target

---

## Shell Constants (unchanged hard law)

- Daily floor (dynamic input)
- 0.25% per-trade risk cap
- 400 trades/day then close-only
- Flat at 00:00 CEST
- Win ratchet (lock line = goal + flatten cost)
- Kill switch file
- Forever masks (current approximation of Law 0)

---

*These laws are standing. Every reward proposal, every mind-probe diagnosis, and every self-tuner generation is judged against them.*
