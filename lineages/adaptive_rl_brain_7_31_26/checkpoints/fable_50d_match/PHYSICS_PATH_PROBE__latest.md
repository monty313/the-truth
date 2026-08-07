# Physics path-skill probe (2026-08-06)

Child SHA: `9BDCEAAE3B282DA1548F6C58E55F5935AED5ECF5720EC95C4913CE17F06FD555`  
Method: PathSkillPolicy decode only (no act BC, no weight train)

| dial note | same | mwt | breach | clear | law interventions |
|-----------|------|-----|--------|-------|-------------------|
| pure | 35 | 15 | 0 | 35 | (none) |
| pinn_only | 35 | 15 | 0 | 35 | pinn_against_htf x2 |
| thrash_grav | 35 | 15 | 0 | 35 | anti_thrash x4 |
| grav_ent_only | **30** | 20 | 0 | 30 | entropy_hold x61 |
| strict_launch | **30** | 20 | 0 | 30 | phys_launch x64 + entropy x49 |
| loose phys_eq (earlier) | **30** | 20 | 0 | 30 | phys_launch over-fire |

## Conclusion vs physics.md

- **Does not get past 35** as global decode laws on this embryo.
- Entropy regime mask and cont-fill are **too coarse** relative to award fragility.
- L2L-safe path remains: keep child floor; refine only if a dial set scores **same > 35**.
- Physics Super BC-PINN recommendation stays **deferred** (base never trained).

## Done criteria status

- [ ] best_same > 35 breach 0
- [ ] growth_method=self_climb_path_skill
- [ ] skill = path dials/laws not day id
- [x] child SHA unchanged
