# Doctor's Prescriptive Patch: The S1 Constraint Fix
**Target:** Enable `S1` constraint decode to convert MWT days to target hits.
**Patient:** Channel1 (`mark_clone_full_obs_v1.pt`)
**Rule:** Child is frozen. Patch runs in `apply_geometry_decode`.

Channel1 is mathematically blind at action time. The sensors are on the board, but the `KEEP` dials have disabled `S1` constraint checking (`constraint_gate=0`). To reach 36, we must turn S1 on, but we must wire it correctly so it doesn't blanket-HOLD valid trades or force-fill trash.

## The Logical Forms to Wire

### 1. Goal Starvation (Regime vs Target)
If ATR is flat, the market cannot fund the day's target. Refuse open.
```python
is_starved := regime_target_ratio < 0.35
# Action: If is_starved ∧ ¬in_trade, HOLD
```

### 2. The Bank Wall (Near Target)
If we are almost at the finish line, do not take new risk unless it is a perfect HQ setup.
```python
goal_banked_near := goal_pressure ≤ 0.12
# Action: If goal_banked_near ∧ ¬in_trade ∧ path_class ≠ continuation_fire_hq, HOLD
```

### 3. S1 Pressure Fire (MWT Conversion)
On MWT days, the trunk is hesitating on valid setups because of score-fear. We will force a fill *only* if all constraints pass and geometry is HQ.
```python
pressure_fire_ok := 
    permission ∧ ltf_with_tide
  ∧ path_class ∈ {fire_window, continuation_fire_hq}
  ∧ (goal_pressure ≥ 0.50)      # Must actually need the money
  ∧ (survival_margin ≥ 0.55)    # Must have equity room vs daily floor
  ∧ (regime_target_ratio ≥ 0.55) # Market must be moving
  ∧ (entropy_chop < 0.65)       # Not in chop
```

## The Prescription (Action Items for the Lab)

1. Set `constraint_gate = 1.0` in the `KEEP_dials` to activate S1.
2. Ensure `is_starved` and `goal_banked_near` are wired into `apply_geometry_decode` as hard HOLDs for new entries.
3. Allow `pressure_fire_ok` to override the child's greedy HOLD (force a BUY/SELL aligned with tide).
4. Run the 50d. Target is 36. Breach must remain 0.