# MARK HERE clone → policy: why the bot trades like this

**Date:** 2026-08-04  
**Repo:** `the-truth` · lineage only (`adaptive_rl_brain_7_31_26`)  
**Not:** PROVEN champion · Channel1 pure-greedy sandbox  
**Voice:** Mark clone (MARK HERE) diagnosing the multi-pair policy transplant  
**North star:** bot moves the way **we** would — multi-TF structure + heat shell + bank at target — not thrash, not freeze, not raw signal-agent dump.

---

## 0) Where we left off

| Item | State |
|------|--------|
| Multi-pair **heuristic + equity shell** | Claim winner: 10/10 pairs ≥30 clear, **0 breach** on 90 XAUUSD days |
| Label Contract V1 logger | Done (5+5 replay PASS) |
| Channel1 pure greedy RL | Still freezes / not claim |
| **Mark clone inside meta-learn as policy** | **Incomplete** — persona exists; **eyes thrash** not cured; meta plasticity (G14–G17) missing |
| PROVEN | Untouched (correct) |

---

## 1) What “clone as policy” means (no ambiguity)

```
MARK HERE personality  →  how we decide (want more, evidence, no thrash)
Multi-pair tutor stack →  closed loop that already clears pairs with 0 breach
Meta-learn RL          →  attention / trust rewiring when past windows prove eyes lied
Shell laws             →  LOCKED (heat, floor-scale, every-bar marks, bank, breach death)
```

**Winning claim “I” today** = `recommended_action` (structure eye) + shell hands.  
**Not yet** = MLP that *is* Mark under pure argmax.  
**Transplant goal** = BC + rewards + attention gates so the **policy path** moves like the tutor *and* like Mark (fewer flips, wait for structure, bank and stop digging).

---

## 2) Chart / day-walk evidence (why it trades this way)

### 2.1 Standalone signal “charts” (repo equity PNGs)

| Artifact | Behavior | Mark read |
|----------|----------|-----------|
| `outputs/artifacts/bb_rsi_sma/equity_EURUSD.png` | All set variants **bleed to ~0** equity over years | Raw BB/RSI/SMA agents are **not** a tradeable policy |
| `outputs/artifacts/momentum_vector/mv_equity_EURUSD.png` | Crash to floor early, then flat dead | Vector alone = suicide without shell + selection |
| Multi-pair tutor equity path (day scores) | Soft targets clear; hard targets thrash | Shell saves floor; **eyes** still over-trade |

**Issue I-0 — Fake edge from agents alone**  
Putting signal agents *as* the policy (no heat/bank/select) is not how Mark trades. Agents are **sensors**, not the soul.

### 2.2 Hard miss day walk — `2026-04-02` · target 3.0 / risk 3.5

Measured: **12 entries**, pnl **−0.65%**, min_eq **−1.32%**, **not banked**, **not breached**.

Pattern (every ~25 M1 bars):

| Time | Structure eye | What “I” did | Equity% |
|------|---------------|--------------|---------|
| 12:00 | BUY | OPEN BUY | ~0 |
| 12:25 | SELL | OPEN SELL (prior stopped) | −0.27 |
| 12:50 | BUY | OPEN BUY | −0.56 |
| 13:40 | SELL | OPEN SELL | −0.87 |
| 14:05 | BUY | REVERSE BUY | −0.94 |
| … | flip / stop / re-open | … | … |
| EOD | — | 12 entries, no bank | −0.65 |

**What this looks like on a chart:** chop / transition day. Higher-stack collapse (official set 2 only) flips side every decision window. Stops fire *between* decisions → flat → immediately re-open on next opposite tick. That is **not** Mark scalping HTF gravity; that is **whipsaw following**.

### 2.3 Soft clear day walk — `2026-04-01` · target 1.0 / risk 2.0

| Metric | Value |
|--------|------:|
| Entries | **2** |
| Banked | **yes** |
| PnL | **+1.29%** |
| min_eq | −0.44% |

**Mark-like day:** few entries, bank early, stop hunting. Same eyes, **softer target** → less need to thrash for 3%.

### 2.4 Forward hard pair aggregate (3.0 / 3.5, 40 forward days)

| Group | n | Mean entries | Mean pnl |
|-------|--:|-------------:|---------:|
| **Misses** | 28 | **10.5** | +0.82 |
| **Clears** | 12 | **5.4** | +3.16 |

Quiet clears often **2–4 entries** and bank.  
Thrash misses **12–16 entries**; many still **positive PnL** but **never reach 3%** because churn + stops eat the move.

**Issue I-1 — Death by thrash (hard targets)**  
High entry count co-occurs with hard-target miss. Day walk **shows mechanism**: reverse-on-single-flip + stop-and-reflip. Not yet fully labeled by setup_type, but **causal enough to gate attention** (confirm bars / post-stop cooldown / max entries).

### 2.5 Soft vs hard (same eyes, same shell)

| Pair | Forward clear | Miss mean entries |
|------|--------------:|------------------:|
| 1.0 / 2.0 | 87.5% | 13.0 (few misses) |
| 3.0 / 3.5 | **30%** | **10.5** |

Soft targets bank before thrash compounds. Hard targets **require a clean swing**; thrash burns the swing.

---

## 3) Root-cause issue list (defined)

| ID | Issue | Mechanism (code) | Symptom on chart / day | Mark would… |
|----|--------|------------------|------------------------|-------------|
| **I-0** | Agents-as-policy | Signal slots / raw BB·MV alone | Equity charts wipe | Use agents as **vote sensors**, not size/entry law |
| **I-1** | Single-bar reverse thrash | `recommended_action`: reverse on **one** opposite sig | BUY/SELL flip every 25m | Require **confirm** (N same opposite bars) before reverse |
| **I-2** | Stop → instant re-entry opposite | `_maybe_stop` flattens; next decide opens new side | Sawtooth equity, many small losses | **Cooldown** after stop before new open |
| **I-3** | Eyes ignore pullback / conflict | `structure.pullback` computed, **not used** by claim heuristic | Entries into LTF against HTF | Prefer HOLD / wait on pullback unless continuation rules fire |
| **I-4** | Single higher collapse | `higher = official set 2 only` (legacy) | One flaky stack = full flip | **FIXED path:** `eyes_mode=mark_all_sets` scans Sets 1–4 (1m/15m/30m; 5m/30m/1h; 15m/1h/4h; 30m/4h/1d). LTF=first, HTF=last two. Aligned only. |
| **I-5** | Hard-target over-hunting | No max-entries / opportunity gate | 13 entries, +2% never banks 3% | Cap entries or refuse open when remaining opportunity thin |
| **I-6** | Pure RL ≠ Mark | Channel1 greedy → 100% HOLD on real | No trades | BC to **Mark-gated heuristic**, not raw freeze |
| **I-7** | Meta missing | G14–G17: no sensor trust / plasticity | Same thrash forever | Meta only rewires **attention dials**, never shell |
| **I-8** | Claim dial search leakage | Historical dials may have seen all days | Forward hard drop | Practice-only dial search (already documented) |
| **I-9** | Persona not wired into train loop | Tutor skill is chat, not reward/BC | Bot doesn’t “feel” like Mark | Encode Mark rules as **attention gates + rewards** |

---

## 4) What is *not* broken (do not “fix”)

| Keep | Why |
|------|-----|
| Heat / refuse-open | 0 breach on claim |
| Floor-scale size + ATR stop | Same brain any risk% |
| Every-bar marks | Honest floor |
| Bank at target | Soft clear days work |
| One signal path flat + in-trade | IRAC-03 KEEP (vs freeze) |
| Trail + cushion + scale-in | **Banned** — killed multi-pair 6/10 → 0/10 |

Shell stays sacred. Fixes are **attention** (eyes + when to act), not shell physics.

---

## 5) Transplant plan (Mark clone → meta-learn policy)

### Layer A — Attention gates (ship first, reversible)

Defaults **OFF** so multi-pair claim path is unchanged. Enable with `mark_clone: true` dials.

| Gate | Dial | Default claim | Mark-on suggestion |
|------|------|---------------|--------------------|
| Signal confirm before open/reverse | `sig_confirm_decisions` | 1 | 2–3 |
| Post-stop open cooldown | `post_stop_cooldown_decisions` | 0 | 2–4 |
| Max entries / day | `max_entries_day` | 0 (off) | 8–10 hard targets |
| Optional pullback refuse-open | `refuse_open_on_pullback` | false | true (experiment) |

### Layer B — BC / reward (policy becomes Mark)

1. Teacher = heuristic **after** Layer A gates (Mark-gated structure).  
2. Train Channel1 (or later larger policy) with CE to teacher + existing anti-hold stack.  
3. Decode for claim stays heuristic until pure greedy matches teacher ≥ threshold.

### Layer C — Meta (learn to learn)

From `UNSEEN_CONSISTENCY_RECIPE.md` G14–G17:

- Sensor trust tables (tag → clear rate practice vs forward)  
- Plasticity controller: breach↑ → freeze shell; clear↓ breach=0 → retune **attention only**  
- Never retune bank/heat/every-bar marks from meta

### Layer D — Product

- Mark persona skill already: `.grok/skills/multi-pair-tutor`  
- Wire day walk + issues into MARK HERE / Army so “fix the-truth” starts here

---

## 6) Immediate verification commands

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# Baseline thrash day (claim eyes)
python lineages/adaptive_rl_brain_7_31_26/tutor_day_walk.py --date 2026-04-02 --target 3.0 --risk 3.5

# Soft bank day
python lineages/adaptive_rl_brain_7_31_26/tutor_day_walk.py --date 2026-04-01 --target 1.0 --risk 2.0

# A/B Mark-clone attention (when script present)
python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 3.0 3.5 --mode forward
```

---

## 7) Success criteria (when transplant is real)

| Meter | Pass |
|-------|------|
| Soft pairs forward clear | No regression vs claim baseline |
| Hard pair 3.0/3.5 forward clear | **Up** vs 30% without raising breach |
| Mean entries on hard misses | **Down** (target &lt; 8) |
| Breach% | Still **0** on claim + forward |
| Pure greedy vs teacher | Match rate rising (not all-hold) |
| PROVEN | Still untouched |

---

## 8) One-line diagnosis for Monty

> The shell is Mark-safe (0 breach). The **eyes thrash**: one flaky HTF collapse flips every 25 minutes, stop-outs re-enter opposite immediately, and hard targets never bank because churn eats the swing. Fix attention (confirm + cooldown + entry cap + later BC/meta) — not the shell, not more raw agents.
