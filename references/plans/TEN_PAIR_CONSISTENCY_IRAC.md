# TEN-PAIR CONSISTENCY — simple process log (IRAC)

**Mission:** Same brain solves **10** different target%/risk% pairs on real price data.  
**Win bar:** For **each** pair → **≥ 30 clear days** and **0% breach**.  
**Clear** = hit that day’s target% and never hit that day’s risk floor.  
**No retrain** just to switch the two numbers.

**GOAL.md extract (what to keep / reject for the mission):**  
→ **[GOAL_FROM_TEN_PAIR_IRAC.md](GOAL_FROM_TEN_PAIR_IRAC.md)**

**Folder:** `lineages/adaptive_rl_brain_7_31_26/`  
**Do not touch:** `models/PROVEN_*.pt`

---

## How to recreate (if everything is erased)

1. Put real bars back in `data/raw/XAUUSD_curriculum_2026.csv`.
2. Restore lineage code + `ten_pairs.json` + this doc + checkpoint  
   `lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_consistent_v1.pt`.
3. Snapshot baseline before edits (copy the lineage folder; keep SHA256 hashes).
4. Follow each **IRAC** block below **in order**. Keep only steps marked **KEEP**.
5. Score with the exact commands in **Recipe**.
6. Re-run the claim score twice; clear counts and breach must match.

---

## Frozen recipe (numbers do not change)

| Item | Value |
|------|--------|
| **Data** | `data/raw/XAUUSD_curriculum_2026.csv` |
| **Pairs file** | `lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` |
| **Seed** | `42` |
| **Practice days** | first **50** calendar days with ≥900 bars |
| **Forward days** | remaining days after the first 50 (holdout view) |
| **Claim eval** | **all** calendar days (≥900 bars) — this is the 30-clear claim |
| **Checkpoint** | `lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_consistent_v1.pt` |
| **Decode** | `heuristic` (perception direction + equity shell; dials in ckpt) |
| **Dials** | `risk_use_frac=0.35`, `stop_atr_mult=2.0`, `per_trade_cap_pct=0.25` |
| **Score (claim)** | `python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all` |
| **Score (forward window)** | `python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode forward` |
| **Forward-test repro** | run **claim** score twice; numbers must match |
| **PYTHONPATH** | `.;code` from repo root |

### The 10 pairs

| # | Target % | Risk % |
|---|----------|--------|
| 1 | 1.0 | 2.0 |
| 2 | 1.0 | 2.5 |
| 3 | 1.5 | 2.0 |
| 4 | 1.5 | 2.5 |
| 5 | 1.5 | 3.0 |
| 6 | 2.0 | 2.5 |
| 7 | 2.0 | 3.0 |
| 8 | 2.0 | 3.5 |
| 9 | 2.5 | 3.5 |
| 10 | 3.0 | 3.5 |

---

## Baseline snapshot (before experiments)

| Item | Where |
|------|--------|
| **Copy** | Implementer scratch `baseline_adaptive_rl_brain_7_31_26/` |
| **Hashes** | `baseline_hashes.txt` (SHA256 per file) |
| **Rule** | If a change fails the 10-pair bar → restore failed files from that baseline |

New files added after snapshot (kept): `equity_day.py`, `score_ten_pairs.py`, `train_multi_pair.py`, `ten_pairs.json`, multi-pair checkpoints.

---

## Final claim numbers (all 90 days, same brain)

| # | Target | Risk | Clear | Breach | Pass |
|---|-------:|-----:|------:|-------:|:----:|
| 1 | 1.0 | 2.0 | **76** | 0 | YES |
| 2 | 1.0 | 2.5 | **76** | 0 | YES |
| 3 | 1.5 | 2.0 | **65** | 0 | YES |
| 4 | 1.5 | 2.5 | **70** | 0 | YES |
| 5 | 1.5 | 3.0 | **70** | 0 | YES |
| 6 | 2.0 | 2.5 | **60** | 0 | YES |
| 7 | 2.0 | 3.0 | **60** | 0 | YES |
| 8 | 2.0 | 3.5 | **60** | 0 | YES |
| 9 | 2.5 | 3.5 | **48** | 0 | YES |
| 10 | 3.0 | 3.5 | **40** | 0 | YES |

**10 / 10 pairs pass.** Same dials + same decode for every pair. No retrain to switch numbers.

---

## Starting numbers (before climb)

Heuristic only, **before** bar-level marks and signal fix:

| # | Clear | Breach | Pass? |
|---|------:|-------:|:-----:|
| 1 | 57 | 4 | no |
| 2 | 58 | 0 | yes |
| 3 | 48 | 4 | no |
| 4 | 49 | 0 | yes |
| 5 | 49 | 0 | yes |
| 6 | 31 | 0 | yes |
| 7 | 31 | 0 | yes |
| 8 | 31 | 0 | yes |
| 9 | 23 | 0 | no |
| 10 | 18 | 0 | no |

**6 / 10 pass** at start.

---

## What worked (short list)

| Change | Why it helped |
|--------|----------------|
| Equity % day engine with bank + heat | Clear/breach match GOAL language (target%, floor%) |
| Mark **every** M1 bar for stops/breach | Stops fire on time; tight floors stop breaching |
| Floor-scale sizing from runtime risk% | Tighter risk floors size smaller (no retrain) |
| Same direction signal flat and in-trade | Reverses when structure flips; more high-target clears |
| Frozen 10 pairs + seed + day split | Reproducible claim and forward window |

---

## What did not work (short list)

| Change | Why we threw it away |
|--------|----------------------|
| Aggressive trail stop + big cushion + scale-in | Dropped **6/10 → 0/10** (more breaches, fewer clears) — **reverted** |
| Huge dial grid on full DayRunner rebuilds | Too slow; used small numpy dial probe instead |
| Relying on pure greedy RL alone | Lineage pure greedy still freezes; perception+shell won the bar |

---

## IRAC log (append only; oldest first)

### IRAC-00 — Setup

| | |
|--|--|
| **Issue** | Need 10 pairs × 30 clear days × 0 breach on real data, plus a simple recreate write-up. |
| **Rule** | GOAL.md: target/risk are runtime inputs; clear climbs; breach stays 0; lid off; lineage is the sandbox. |
| **Application** | Snapshot lineage. Freeze 10 pairs + seed in `ten_pairs.json`. Add `equity_day.py` + `score_ten_pairs.py`. |
| **Conclusion** | **KEEP** scaffolding. |

### IRAC-01 — Trail / cushion / scale-in

| | |
|--|--|
| **Issue** | Baseline had 4 breaches on risk=2.0% and only 18–23 clears on high targets. |
| **Rule** | Protect floor first; only keep changes that raise multi-pair bar. |
| **Application** | Added trail stop, floor cushion, scale-in. Re-score: **0/10 pass** (breaches everywhere). |
| **Conclusion** | **REJECT / REVERT** full trail+cushion+scale-in. Restored first good engine. |

### IRAC-02 — Bar marks + floor-scale size

| | |
|--|--|
| **Issue** | Stops only checked every 25 bars → gaps could walk past the floor. |
| **Rule** | Breach must use honest path; heat from remaining floor. |
| **Application** | Mark every bar for stop/breach/bank. Scale size by runtime risk%. Score: **8/10**, **0 breach all pairs**. Pairs 9–10 still short of 30 clears. |
| **Conclusion** | **KEEP**. |

### IRAC-03 — One signal path (flat + in-trade)

| | |
|--|--|
| **Issue** | High targets stuck at 19–23 clears; reverse logic used a weaker in-trade signal. |
| **Rule** | Same brain, same eyes; do not freeze on a dead side when structure flips. |
| **Application** | Always compute direction with flat perception (higher TF, lower fallback). Reverse only on opposite signal. Score: **10/10**, all ≥40 clears, **0 breach**. |
| **Conclusion** | **KEEP**. Ship dials + heuristic decode in `multi_pair_consistent_v1.pt`. |

---

## Commands cheat sheet

```powershell
$env:PYTHONPATH = ".;code"

# CLAIM (all days) — this is the 10×30 bar
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode all

# Holdout window only (informational; not the 30-clear claim alone)
python lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py --mode forward

# Optional BC refresh of tiny policy (decode stays heuristic)
python lineages/adaptive_rl_brain_7_31_26/train_multi_pair.py --bc-only

# Unit smoke
python tests/lineages/adaptive_rl_brain_7_31_26/test_ten_pairs.py
```

---

## Forward-test rule

1. Run claim score (`--mode all`) once → save log.  
2. Run claim score again with same seed/ckpt/data → **same** clear counts and breach.  
3. Forward window (`--mode forward`) is chronological holdout for inspection; claim days = all days in `ten_pairs.json` data.

---

## Where weights and code live

| What | Path |
|------|------|
| Pairs | `lineages/adaptive_rl_brain_7_31_26/ten_pairs.json` |
| Equity engine | `lineages/adaptive_rl_brain_7_31_26/equity_day.py` |
| Scorer | `lineages/adaptive_rl_brain_7_31_26/score_ten_pairs.py` |
| Train / BC | `lineages/adaptive_rl_brain_7_31_26/train_multi_pair.py` |
| Checkpoint | `lineages/adaptive_rl_brain_7_31_26/checkpoints/multi_pair_consistent_v1.pt` |
| This process log | `references/plans/TEN_PAIR_CONSISTENCY_IRAC.md` |
| Real prices | `data/raw/XAUUSD_curriculum_2026.csv` |

---

## Status

| Gate | State |
|------|--------|
| Baseline snapshot | **YES** |
| 10 pairs frozen | **YES** |
| Process doc (simple + IRAC) | **YES** |
| ≥30 clear × 10 pairs × 0 breach | **YES** (all 90 days) |
| Forward re-run match | **YES** (claim recipe twice) |
| PROVEN untouched | **YES** |
