# IRAC — Bread-and-butter under HTF trend (2026-07-24)

## Issue
Clear rate stuck ~27% while Mind Probe shows LTF pullback/continuation flags under HTF trend with high `policy_hold` (setup visible, policy holds). Not pure Perception blindness.

## Rule
Standing Laws: bread-and-butter = LTF pullback while both HTFs strong-trend. Regime language: `trend_*` + `pullback`/`continuation` should be acted unless mask veto or risk floor.

## Application
Multi-day Mind Probe (PROVEN_SPRINT, curriculum sample):
- Pull/cont setups present across days
- Dominant `skip_reason=policy_hold` when HTF trending and LTF setup on
- `mask_veto` near zero on sample
- Prior reward `w_pullback_with_htf=0.02` under-weighted the primary scalper pattern

## Conclusion
**Class: Policy** (incentives), not Perception.
**Cure:** raise `w_pullback_with_htf` 0.02 → **0.25** in `configs/rewards.yaml` (meta_tuner BOUNDS keeps it tunable).
Re-measure with `prove_it` / consistency_sprint after host train; adopt only if clear rate / row improves without breach regression.
