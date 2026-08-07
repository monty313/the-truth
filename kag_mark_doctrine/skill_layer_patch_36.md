# Skill Layer Schema: 35 → 36 Patch
# Target: mark_clone_full_obs_v1.pt (Child frozen, weights locked)
# Core: Momentum Conservation & KAG Logical Forms

## 1. Roles (Novel Sensor Mask)
# Uncoupling the Macro Rails (Set 4)
4h_solo_dir, 1d_solo_dir     → inertia / force (TIDE support, not sole tide)
4h_solo_mass, 1d_solo_mass   → inertia magnitude (efficiency / mass gate)

# Velocity & Tension
ltf_entry[1..4]              → velocity (ACT timing only)
tension_scalar               → equilibrium ↔ expansion (BREATH vs LAUNCH feel)
htf_permission_age           → regime_gate / path clock

## 2. Logical Forms
# Macro Tide & Permission
tide_macro := sign_agree(4h_solo_dir, 1d_solo_dir) if both |dir|=1 else 0
mass_alive := (4h_solo_mass ≥ m_min) ∧ (1d_solo_mass ≥ m_min)
rails_conflict := (4h_solo_dir * 1d_solo_dir < 0)
permission := (tide_macro ≠ 0) ∧ mass_alive ∧ ¬rails_conflict

# Tension & Launch States
is_chop      := tension_low_entropy ∧ ¬mass_building
is_loaded    := permission ∧ tension_cocking ∧ ltf_against_or_soft
is_launch    := permission ∧ ltf_with_tide ∧ (age ≥ 0)

# Path Clock (Deriving Classes without memos)
path_class:
  premature     := ltf_fire_edge ∧ ¬permission
  fire_window   := is_launch ∧ age ∈ [1, A_young]
  continuation  := is_launch ∧ age ≥ A_cont ∧ mass_alive
  wait_loaded   := is_loaded
  wait_no_trade := is_chop ∨ ¬permission
  pullback_hold := in_trade ∧ permission ∧ ltf_against_or_soft ∧ ¬collapse

## 3. Skill Decode Overlay
# Path laws override child_greedy safely
child_a := child.greedy(full_obs_168)
skill_a := map(path_class → HOLD|BUY|SELL)  

# Decode Laws:
- if path_class == premature|anti_thrash → HOLD
- if path_class == wait_loaded → HOLD (cocking slingshot)
- if path_class == fire_window|continuation → child_a (permitted fire)
- if in_trade ∧ pullback_hold → HOLD (do not exit on tension decay)
- if in_trade ∧ is_collapse|force_flip → KILL

final := skill_override(child_a, skill_a) 

## 4. Normalization (Dimensionless variables)
• mass: clip(ADX/50, 0, 1) or gate at 20 → 0–1
• tension: combine (1 - bandwidth_pctile) with signed FI/ATR
• age: min(age_bars / age_cap, 1)

## 5. Build Contract
1. Child frozen — mark_clone_full_obs_v1.pt / SHA 9BDCEAAE… never trained.
2. Additive skill board only — skill_slice or harness dials; do not explode 168 into a retrain of trunk.
3. One primary lever per cycle (R10) — ship geometry sensors first; dial age/mass thresholds second.
4. KEEP: same ≥ 35, breach = 0, learn ≠ copy (class attribution).
5. REJECT: pack drop or breach → restore prior skill dials only.
6. Novel mask: new channels never override known HTF force at low conf.
7. Exit law: collapse/force flip owns kill; tension owns wait, not panic close.