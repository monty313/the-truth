# Mark doctrine — five laws (how the clone must think)

**Owner:** Mark Montgomery Jr. / MARK HERE  
**Voice:** ENTJ · fast logical scalping · Fable translator of lab truth  
**Code:** `perception/mark_doctrine.py` · teacher for new Channel1 policy  
**Sets:** LTF first, HTF last two — `1m|15m,30m` · `5m|30m,1h` · `15m|1h,4h` · `30m|4h,1d`

This is **not chat philosophy**. The policy becomes what these laws **measure**.

---

## Stack (control chain)

```
FORCE (HTF) → REGIME → allowed playbook + size multiplier
     ↓
VELOCITY (LTF) → pullback (breather) vs launch classification
     ↓
ENTRY only if side(force) == side(setup) and Risk$ ≤ RemainingDaily
     ↓
Regime shift → rewrite rules / flatten / m→0
```

---

## Law 1 — Dominant trends (MTF alignment)

| Layer | Job | May do | May not |
|-------|-----|--------|---------|
| HTF force (last two of each set) | Permission / gravity | Long-only, short-only, flat | Trigger the scalp |
| LTF velocity (first of set) | Slingshot timing | Enter pullback resume / reclaim with tide | Reverse against permission |

**Chain:** classify tide → hard filter opposite LTF setups → slingshot only with tide → fire on LTF resume with HTF → kill if tide breaks.

**One sentence:** HTF is the binary gate on side; LTF only chooses *when* within that gate.

---

## Law 2 — Acceleration (momentum vs mean reversion)

| State | Signature | Play |
|-------|-----------|------|
| Breather | Fast dips against force; slow/HTF unbroken | Wait pullback; do not fade force |
| Launch | Fast + slow same side; closes expand with tide | Ride; do not mean-revert fade |

**Rule:** If only fast snapped and slow unbroken → pullback toolkit. If slow structure breaks on HTF → momentum toolkit; fades forbidden.

---

## Law 3 — Environmental survival (regime)

| Regime | System must |
|--------|-------------|
| Bull trend | Long-only playbooks |
| Bear trend | Short-only mirror |
| Chop / conflict | Flat or mean-reversion only; no breakout chase |
| Undefined | **No trade** (m = 0) |

Regime shifts on HTF structure + hold, not one noisy bar. Open positions that violate new regime reduce/close.

---

## Law 4 — Capital preservation

```
Risk$ = E × r_base × m_conf × m_regime
Size  = Risk$ / StopDistance$
```

- Scale up only when Laws 1–3 green + daily budget remaining.  
- HTF flat/conflict → m → 0.  
- Daily floor hit → flat rest of day (shell bank/breach already encode goal/floor).  
- **Edge optional; survival not.**

Lab shell: heat, floor-scale size, every-bar marks, bank, breach death — **locked**.

---

## Law 5 — Speed vs weight

| | Meaning | Lives on |
|--|---------|----------|
| Velocity | Fast CCI/RSI; LTF rate of change | LTF entry TF |
| Force | Slow structure; HTF envelope/CCI mass | HTF confirm TFs |

**Slingshot identity:** force with tide + velocity dips against → breath → velocity re-aligns → entry. Force flip invalidates velocity.

---

## Implementation status

| Law | Code path | Status |
|-----|-----------|--------|
| 1 | `mark_doctrine.permission_and_trigger` | Implemented |
| 2 | `classify_breath_vs_launch` | Implemented (velocity vs force) |
| 3 | `regime_from_sets` | Implemented (bull/bear/chop/flat) |
| 4 | `equity_day` shell + heat | Existing; doctrine does not break shell |
| 5 | Force HTF / velocity LTF split | Implemented |

Teacher action = doctrine decision. BC trains Channel1 to imitate teacher. Claim multi-pair heuristic remains `legacy_set2` unless dials say otherwise.
