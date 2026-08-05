# Mark-clone iterative update protocol (for `/loop` fires)

**Voice:** MARK HERE using Fable as translator into policy weights.  
**Repo root:** `C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth`  
**Lineage:** `lineages/adaptive_rl_brain_7_31_26/`  
**Never:** touch `models/PROVEN*.pt` · trail/cushion/scale-in · fit dials on forward

---

## One fire does (no chat memory required)

```powershell
cd C:\Users\user\Fable5_Foundation\MOMENTUM_ONE\the-truth
$env:PYTHONPATH = ".;code"

# 1) unit gate
python -m pytest tests/lineages/adaptive_rl_brain_7_31_26/test_mark_sets_opportunity.py tests/lineages/adaptive_rl_brain_7_31_26/test_mark_clone_policy.py -q

# 2) retrain / refresh labels (practice only) — five-law doctrine teacher
python lineages/adaptive_rl_brain_7_31_26/train_mark_clone_bc.py --epochs 24 --practice-n 50

# 3) thrash + bank day probe is inside train script; read report:
#    checkpoints/mark_clone_bc_report.json
#    checkpoints/mark_clone_doctrine_v1.pt
```

**Status line to report:**  
`dir_match=… match_rate=… thrash_day_entries=… soft_cleared=… breach=… next=…`

---

## Meters (“same way we do”)

| Meter | Pass |
|-------|------|
| Teacher labels not all HOLD | required |
| Pure greedy not all HOLD | required |
| Overall match rate | ≥ 0.70 |
| Directional dir_match | ≥ 0.85 |
| Thrash day `2026-04-02` @ 3.0/3.5 entries | **≤ 6** (legacy ~12) |
| Soft day `2026-04-01` @ 1.0/2.0 | cleared, breach 0 |
| Breach on eval windows | **0** |
| Soft clear not collapsed | soft clear drop ≤ 20pp vs claim baseline (see `mark_clone_policy_ab_hard_soft.json`) |
| Hard thrash improved | policy mean_entries &lt; baseline |
| PROVEN mtime | unchanged |

---

## If Mark would disagree on a day

1. Run day walk: `tutor_day_walk` or `train_mark_clone.run_policy_day`  
2. Diff policy action vs Mark teacher (`eyes_mode=mark_doctrine`)  
3. Apply **one** intelligent change only:
   - more practice labels / epochs, **or**
   - doctrine scoring tweak (regime / force thresholds), **or**
   - BC class weights — **not** shell  
4. Re-measure thrash + soft days once  
5. Stop if meters pass; else leave `next=` in report for following fire

---

## Stop / exit loop

When **all** meters pass for two consecutive fires **or** dir_match ≥ 0.90 and thrash entries ≤ 6 and soft clears and breach 0:  
report SUCCESS and call `scheduler_delete` on the loop task_id.

Auto-expires after 7 days even if not perfect.

---

## Doctrine map (pt5 → code)

| pt5 law | Code |
|---------|------|
| HTF permission / gravity | Set 4 macro gate in `mark_sets_opportunity.scan_mark_opportunities` |
| LTF timing only | aligned LTF+HTF score; pullback score 0 |
| All four sets | `OFFICIAL_SETS` 1m/15m/30m · 5m/30m/1h · 15m/1h/4h · 30m/4h/1d |
| Agents ≠ soul | Channel1 structure packs only; shell risk separate |
| Capital preservation | `equity_day` heat / bank / breach death (locked) |

---

## Artifacts

| Path | Role |
|------|------|
| `checkpoints/mark_clone_channel1_v1.pt` | New same-obs brain |
| `checkpoints/mark_teacher_labels_practice.json` | Teacher dump |
| `checkpoints/mark_clone_train_report.json` | Last train meters |
| `MARK_CLONE_POLICY.md` / `MARK_CLONE_AS_POLICY_ISSUES.md` | Spec |
| `train_mark_clone.py` | Entry |
