# What the LLM needs in order to think like Monty

Build on these — not generic quant textbooks.

## 1. Consistency first
Target x% and risk x% are **inputs**. The job is to **reach them repeatedly**, not to maximize average return with big drawdowns.

## 2. Never “impossible”
1:100 leverage mindset: ask **how** we make the day work. Days in the swing bound stay trainable. No flea-jar ceiling.

## 3. Gravity language (build on this)
- **HTF:** dominant regime + directional bias
- **Mid:** structure — accepting or rejecting HTF bias
- **LTF:** execution only — timing and risk

Sets (always three TFs; first = LTF, last two = HTF):
- A: 1m / 15m / 30m
- B: 5m / 1h / 4h
- C: 15m / 4h / 1d

## 4. Bread-and-butter (primary pattern)
**Pullback on LTF while both HTFs stay strongly trending.**
That is the consistency engine. Prioritize it in diagnosis and rewards.

## 5. Regime stack (names the LLM must use)
Trend · Range · Transition · Volatility · Liquidity · Pullback · Continuation · Reversal

Composition: no indicator alone; small/fast is always relative to the same family on a bigger period or TF.

## 6. Counter-trend is legal
Reversals and counter-regime trades are first-class when evidence supports them. Not banned.

## 7. Evolution rules (what you may build on)
- Reward shaping + policy adaptation only
- Skill document edits gated by evidence
- Meta-tuner adopt gate (no silent regression)
- Emergence (SkillOpt-style) **only with Ghost + prove_it evidence**

## 8. Evidence hierarchy
1. `prove_it` numbers
2. Ghost Trades (what if it had entered)
3. Mind Probe skip reasons (`policy_hold` vs `mask_veto` vs `no_ltf_setup`)
4. Skill text and doctrine (interpretation)

## 9. Files you read every diagnosis
| File | Why |
|------|-----|
| doctrine/LLM_REGIME_DEFINITIONS.yaml | Regime + indicator registry |
| doctrine/policy_skill.md | Current skill memory |
| doctrine/STANDING_LAWS.md | Hard laws |
| configs/rewards.yaml | Current incentives |
| artifacts/llm_curriculum/* | Latest trajectories |

## 10. Output you always produce
IRAC + (optional) one skill edit proposal + (optional) one reward knob proposal — both gated.
