# GOAL knowledge from TEN-PAIR IRAC

**Mission (GOAL.md only):** one bot that solves **any typed target% / risk% without retrain** — climb **clear%**, keep **breach% = 0**.

**Source process log:** [TEN_PAIR_CONSISTENCY_IRAC.md](TEN_PAIR_CONSISTENCY_IRAC.md)  
**Student–tutor principles (winning policy logic):**  
→ [lineages/adaptive_rl_brain_7_31_26/PRINCIPLES_OF_SUCCESS.md](../../lineages/adaptive_rl_brain_7_31_26/PRINCIPLES_OF_SUCCESS.md)  
**This file:** only knowledge that helps that mission. No parallel goals.

**Lineage sandbox:** `lineages/adaptive_rl_brain_7_31_26/`  
**Do not touch:** `models/PROVEN_*.pt` until `prove_it` beats champion (higher clear, breach still 0).

---

## 1. Words (do not invent new ones)

| GOAL word | Meaning |
|-----------|---------|
| **Target %** | Runtime daily profit goal (input, not baked into weights) |
| **Risk %** | Runtime daily floor / max loss (input) |
| **Clear** | Hit **that** target% and **never** hit **that** floor that day |
| **Breach** | Hit the floor that day → **must stay 0%** under test |
| **No retrain** | Change the two numbers → same brain (weights + dials + decode) |
| **Score** | Champion: `scripts/prove_it.py` at **your** pair. Multi-pair claim: `score_ten_pairs.py --mode all` |

If a change does not raise clear% or protect breach 0% **at the pair under test** → **skip it**.

---

## 2. KEEP — mechanics that implement “any pair without retrain”

These passed **10 pairs × ≥30 clear days × 0% breach** on real XAUUSD in the lineage (heuristic decode + equity shell).

| # | Mechanism | Why GOAL needs it |
|---|-----------|-------------------|
| **A** | **Equity-% day engine** (bank at target, heat vs floor) | Clear/breach in **% of equity**, not raw price luck |
| **B** | **Mark every M1 bar** for stop / breach / bank | Honest path — floor cannot walk through between decisions |
| **C** | **Floor-scale sizing** from **runtime risk%** | Tight risk → smaller size; **no retrain** to switch risk |
| **D** | **Heat / refuse open** if new risk would break floor | Breach protection before entry |
| **E** | **Bank** when equity% ≥ target% | Stop after clear; protect floor after a win |
| **F** | **Same direction signal flat and in-trade** | One set of eyes; reverse only on **opposite** structure |
| **G** | **Dials in checkpoint** (not one frozen pair) | Searchable; same weights for every (target, risk) |
| **H** | **Frozen pairs + seed + claim recipe** | Reproducible multi-pair proof |

### Lineage claim recipe (reproducible)

| Item | Value |
|------|--------|
| Data | `data/raw/XAUUSD_curriculum_2026.csv` |
| Pairs | `lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` |
| Seed | `42` |
| Checkpoint | `lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_consistent_v1.pt` |
| Decode | **heuristic** (perception + equity shell; dials in ckpt) |
| Dials | `risk_use_frac=0.35`, `stop_atr_mult=2.0`, `per_trade_cap_pct=0.25` |
| Claim score | `python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all` |
| PYTHONPATH | `.;code` from repo root |

Run claim **twice**; clear counts and breach must match.

### Code map

| What | Path |
|------|------|
| Equity engine | `lineages/adaptive_rl_brain_7_31_26/equity_day.py` |
| Scorer | `lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py` |
| Pairs | `lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` |
| Multi-pair train / BC | `lineages/adaptive_rl_brain_7_31_26/train_multi_pair.py` |
| Full IRAC log | [TEN_PAIR_CONSISTENCY_IRAC.md](TEN_PAIR_CONSISTENCY_IRAC.md) |
| Champion prove | `scripts/prove_it.py` / `USE/1_prove.bat` |

---

## 3. REJECT — do not reintroduce

| Change | Result | GOAL reason |
|--------|--------|-------------|
| **Trail stop + big cushion + scale-in** as a package | **6/10 → 0/10** multi-pair pass | Broke breach 0 and clear |
| **Stops only every N bars** | Silent floor walk | Dishonest breach |
| **Different signal in-trade vs flat** | High targets stuck | Same brain must not freeze on a dead side |
| **Huge dial grid on full DayRunner rebuilds** | Too slow | Small probe → then prove |
| **Pure greedy RL alone as multi-pair solver** | Freezes; did not win claim | Win = clear/breach at runtime pairs |

**Rule (IRAC-01):** protect **floor first**; only keep what improves the multi-pair / prove_it bar.

---

## 4. Process = GOAL improve order

| Step | Action |
|------|--------|
| 1 Diagnose | Mind probe / one IRAC hypothesis |
| 2 Dials | Search; do not hardcode forever |
| 3 Practice | Vary goal/floor when training allows |
| 4 Prove | `prove_it` at **your** numbers and/or multi-pair claim |
| 5 Keep or reject | Higher clear + breach still 0 → keep; else revert |

**Promote to `models/PROVEN_*` only if** clear **beats** current champion and breach still **0**. Then update `models/00_CHAMPION.md`, GOAL scoreboard, `references/doctrine/SUCCESS_LEDGER.md`.

---

## 5. Two tracks (do not mix win conditions)

| Track | Brain | GOAL role |
|-------|--------|-----------|
| **Champion (PROVEN)** | `models/PROVEN_SPRINT_row04_clear24_2026-07-20.pt` | Official yardstick; score with `prove_it` at any pair |
| **Multi-pair lineage** | `multi_pair_consistent_v1.pt` + heuristic + equity shell | Proof that runtime target/risk + shell can do **many pairs, 0 breach** |
| **Channel1 RL (v1–v3)** | Tiny MLP + HOLD/regret shaping | Research only until it **drives the equity shell** and raises prove_it clear without breach |

**Steal for champion climb:** shell physics (A–F), signal unity, every-bar honesty, runtime sizing.  
**Do not steal:** “entries went up” without clear/breach; trail+scale-in package; promoting RL freeze-fixes as GOAL wins.

---

## 6. Pre-change checklist (use every time)

1. Are **target%** and **risk%** runtime inputs the episode sees?  
2. Does **sizing scale with risk%** (floor residual / heat)?  
3. Are stop/breach checked on an **honest path** (every M1 bar or equivalent)?  
4. Is **direction the same** flat and in-trade? Reverse only on opposite?  
5. **Bank at target?** Refuse open that would break heat?  
6. Score **≥2 pairs** without retrain (e.g. 3.0/3.5 and 1.5/2.0)?  
7. **Breach still 0%?** If no → reject.  
8. **Clear higher** at the pair under test? If no → skip.  
9. Did we reintroduce **trail + cushion + scale-in** as a bundle? → **no** unless multi-pair / prove_it still pass.

---

## 7. Commands (GOAL gates)

```powershell
$env:PYTHONPATH = ".;code"

# Champion — YOUR numbers (no retrain)
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 1.5 2.0

# Lineage multi-pair claim (same brain, 10 pairs)
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all

# Forward window only (inspection; not the full claim alone)
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode forward
```

Or: **`USE/1_prove.bat`** (change the two numbers only).

---

## 8. Not the GOAL priority (unless Monty asks)

- Channel1 HOLD rate / second-best regret / BUY–SELL logit balance alone  
- Meta dials for regret or side asymmetry (note for later; not the gate)  
- New UIs, parallel frameworks, essays without a prove_it / multi-pair score  

---

## Bottom line

**IRAC already showed the GOAL shape in the lineage:** same brain · many target/risk pairs · no retrain · **0% breach** · high clear — via **equity shell + honest bar path + runtime sizing + one structure signal**, not pure greedy RL.

**To improve the bot under GOAL.md:** keep those shell rules, never reintroduce the rejected trail stack, score only clear/breach at **typed** pairs, promote only when **`prove_it` (or multi-pair claim) improves clear with breach still 0**.
