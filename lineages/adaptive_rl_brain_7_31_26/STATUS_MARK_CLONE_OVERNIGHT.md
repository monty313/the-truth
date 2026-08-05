# STATUS — Mark clone overnight (2026-08-04 sleep)

**Mission:** Keep iterating until the bot is **our clone** (ENTJ · five laws · 4 official sets).  
**PROVEN:** not touched.

---

## Doctrine locked (how an LLM / policy must think)

```
FORCE (HTF last two of each set) → REGIME → playbook + m_regime
     ↓
VELOCITY (LTF first) → breath vs launch / aligned
     ↓
ENTRY only if side(force)==side(setup) and heat/daily risk allows
     ↓
Regime shift → HOLD / rewrite / m→0
```

| Law | One line |
|-----|----------|
| 1 Dominant trends | HTF permission; LTF only times; never side against HTF |
| 2 Acceleration | Breath = wait slingshot; launch = ride; no fade force |
| 3 Regime | bull/bear/chop/flat; chop & flat → no breakout chase |
| 4 Capital | shell heat/floor/bank/breach sacred; m_conf/m_regime hints |
| 5 Speed vs weight | velocity=LTF fast; force=HTF mass |

**Sets:** `1m|15m,30m` · `5m|30m,1h` · `15m|1h,4h` · `30m|4h,1d`

---

## Code shipped this session

| Path | Role |
|------|------|
| `MARK_DOCTRINE_FIVE_LAWS.md` | Human doctrine |
| `perception/mark_doctrine.py` | Teacher |
| `perception/mark_sets_opportunity.py` | Multi-set scan |
| `day_runner.py` `eyes_mode=mark_doctrine` | Wired |
| `equity_day.py` | Mark clone → doctrine eyes |
| `train_mark_clone_bc.py` | New brain BC |
| `compare_mark_clone_attention.py --eyes-mode mark_doctrine` | A/B |

---

## Latest measured (post BC 40 ep / hidden 96 / 50 practice days)

| Meter | Value |
|-------|------:|
| Teacher action mix | HOLD 818 / BUY 103 / SELL 96 (n=1017) |
| Train match | **0.70** (need ≥0.75 for policy_ready) |
| Train **dir_match** | **0.95** |
| Forward label match | **0.71** / dir **0.93** |
| Teacher forward clear @2/3 | **40%** · breach **0** · entries ~4.6 |
| Policy greedy forward | **45%** · breach **0** · entries ~4.9 |
| Doctrine thrash day entries | **12 → 5** vs legacy |
| `clone_ready_heuristic` | **True** |
| `clone_ready_policy` | **False** (match 0.70 &lt; 0.75) |

**Read:** directional clone is real (dir_match 95%). HOLD still under-cloned (policy slightly busier). Not “I would always do the same” until match ≥0.75 and day-walk reasons read as Mark.

## What “clone ready” means (morning check)

1. Open `checkpoints/mark_clone_bc_report.json` if present.  
2. Teacher decode (heuristic doctrine): breach **0**, thrash entries **down**, clear% not collapsed to ~0.  
3. Policy greedy: train **match ≥ 0.75**, dir_match **≥ 0.55**, breach **0**.  
4. Day walk `2026-04-02` and a clean clear day — Mark would say *I would have done the same* on force/trigger/hold reasons.

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"
python lineages/adaptive_rl_brain_7_31_26/tutor_day_walk.py --date 2026-04-02 --target 3.0 --risk 3.5
# (day walk still prints structure; doctrine reasons via compare / train report)
python lineages/adaptive_rl_brain_7_31_26/compare_mark_clone_attention.py --pair 3.0 3.5 --mode forward --eyes-only --eyes-mode mark_doctrine
Get-Content lineages/adaptive_rl_brain_7_31_26/checkpoints/mark_clone_bc_report.json
```

---

## Still not “us” until

- Policy greedy **matches teacher** on real bars (not freeze all-HOLD).  
- Clear% on soft pairs competitive with legacy without thrash.  
- Reverse only when **force/regime** flips — never LTF noise.  
- Optional: score_ten_pairs with `eyes_mode=mark_doctrine` full table.

**Do not** promote over PROVEN. **Do not** reintroduce trail+scale-in.
