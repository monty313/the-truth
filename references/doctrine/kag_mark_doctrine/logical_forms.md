# Logical Forms — Multi-Hop Questions KAG Must Answer

Agents and teacher labelers call these forms. Answers must name **set, roles, topology, act** — never bare indicator slogans.

## Tide / permission

1. `tide(?set)` → {long_only, short_only, flat}  
   Given force sensors on support TFs of `?set`, what is allowed side?

2. `ltf_may_fire(?set, ?side)` → bool  
   True only if `tide(?set) == ?side` and regime tradeable.

## Dual period / tension

3. `is_max_tension(?set, ?side)` → bool  
   Slow/inertia with tide AND fast/velocity against relative to own baselines AND macro G holds.

4. `is_launch(?set, ?side)` → bool  
   Fast and slow co-aligned through thresholds same side and holding (not one-bar spike).

5. `is_collapse(?set)` → bool  
   Macro force/tunnel G flipped or zeroed.

## Forward shift / tunnel

6. `macro_tunnel_state(?tf)` → {+1, -1, 0}  
   Full-body outside both rails vs inside/mixed.

7. `micro_phase(?ltf, ?G)` → {inside_reload, full_break_with_G, noise_without_G}

8. `slingshot_ready(?set)` → bool  
   Macro G fixed + micro re-achieves full break with G after load.

## Composition

9. `roles_of(?instance)` → Role  
   Map family+period+TF+shift → force|inertia|velocity|…

10. `composition_valid(?setup)` → bool  
    At least one force role and one velocity role cited; no lone indicator.

11. `efficiency_allows(?ltf)` → bool  
    Path efficiency / ADX gate — false ⇒ mask LTF fires.

## Decision chain

12. `next_act(?bar_context)` → Action  
    Run chain in order; return fire_buy|fire_sell|wait_loaded|wait_no_trade|kill|…

13. `why_wait(?bar_context)` → {loaded, no_trade, heat_blocked}  
    Must distinguish loaded vs freeze.

14. `still_mark_at_close(?day_plan)` → bool  
    Side, wait/go pattern, enough/too-much feel consistent with finish line.

## Multi-set

15. `force_consensus(?sets)` → {agree_long, agree_short, conflict, incomplete}  
16. `conviction(?sets, ?topology)` → {low, med, high} for size multiplier under heat.

## Kill / invalidate

17. `velocity_valid_given_force(?v_signal, ?G)` → bool  
    False if force flipped — fast screams invalid.

18. `regime_rewrite(?old, ?new)` → {flatten, m0, swap_playbook}

## Teacher labels for RL

For each bar pack, emit:

```json
{
  "sets": {"1": {"tide": "+1", "topology": "slingshot_load", "act": "wait_loaded"}, ...},
  "consensus": "agree_long",
  "global_act": "wait_loaded",
  "why": "G+ on set2/3; V dipped I intact on 5m; efficiency OK",
  "size_feel": "hold_dry_powder"
}
```

Pass criteria vs Mark HITL: same allowed side; same wait vs go; same enough/too-much/stop feel; true end of day.

---

## v1.1 — Teaching, novel sensors, learn≠copy (additive)

19. `assign_role(?sensor, ?set)` → Role + confidence  
    Never-seen indicator: TF slot + shape + period relativity + relation to known force.

20. `lesson(?bar)` → principle_application JSON  
    Must include principle_ids, relations, topology, act, novel_flags. Reject if only act.

21. `is_copying(?student_aux)` → bool  
    True if act match high but topology/role_map ~ chance → fail learn gate.

22. `held_out_transfer(?family_a, ?family_b)` → agreement  
    Same topology labels when velocity family swapped (CCI↔RSI↔Stoch).

23. `forward_day_ok(?day)` → bool  
    Goal path + breach=0 + wait subtypes coherent + one mind semantics.

24. `preserve_prior(?merge_diff)` → bool  
    No deleted Law/Set/DecisionStep; only additive graph edges.
