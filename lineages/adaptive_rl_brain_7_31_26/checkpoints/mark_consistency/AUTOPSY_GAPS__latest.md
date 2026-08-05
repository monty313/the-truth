# Streak gap autopsy — no-opp vs Mark-would-take

**When:** 2026-08-05 10:19 UTC

## Headline

Of 11 non-award days: 10 Mark-would-take (learnable), 1 no-opportunity (do not thrash).

- Max award streak (this window): **8**
- Learnable fraction of (Mark-take + no-opp): **0.909**
- Counts: `{"AWARD": 29, "MARK_WOULD_TAKE": 10, "NO_OPPORTUNITY": 1}`

## Rule for rewards (only)

| Gap class | Reward / penalty action |
|-----------|-------------------------|
| **NO_OPPORTUNITY** | Reward patient HOLD; **cut** inactivity / majority-idle tax when force neutral. Do **not** force entries for awards. |
| **MARK_WOULD_TAKE** | **Penalize** misread; boost soul-side entry; EOD **streak-break** penalty. BC labels from Mark soul plan. |
| **BOTH_MISS** | Neutral research; no thrash for score. |
| **POLICY_BREACH** | Keep floor walls; never loosen shell. |
| **AWARD** | EOD **streak award** bonus (longer streak → larger bonus). |

## Day-by-day

| Date | T/R | Policy PnL / n | Mark plan? / PnL / n | Class | Sub |
|------|-----|----------------|----------------------|-------|-----|
| 2026-03-31 | 1.0/2.0 | 1.2587% / 1 | Y / 1.0442% / 1 | **AWARD** | clear |
| 2026-04-01 | 2.0/3.5 | 2.1721% / 3 | Y / 2.0458% / 2 | **AWARD** | clear |
| 2026-04-02 | 2.0/3.0 | 2.0356% / 2 | Y / 2.0065% / 1 | **AWARD** | clear |
| 2026-04-06 | 1.5/3.0 | -1.6273% / 3 | Y / 1.5214% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-07 | 1.5/3.0 | 1.815% / 2 | Y / 1.8049% / 2 | **AWARD** | clear |
| 2026-04-08 | 2.5/3.5 | 2.908% / 2 | Y / 2.5699% / 2 | **AWARD** | clear |
| 2026-04-09 | 1.0/2.0 | 1.0242% / 4 | Y / 1.1732% / 2 | **AWARD** | clear |
| 2026-04-10 | 2.0/3.0 | -1.366% / 2 | Y / 2.0229% / 1 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-13 | 1.5/2.0 | -1.0246% / 3 | Y / 1.5639% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-14 | 1.0/2.0 | -1.0252% / 3 | Y / 1.0672% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-15 | 2.0/2.5 | -1.125% / 3 | Y / 2.0387% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-16 | 3.0/3.5 | 3.7483% / 1 | Y / 3.4589% / 2 | **AWARD** | clear |
| 2026-04-17 | 2.0/3.5 | 2.0711% / 1 | Y / 2.3027% / 1 | **AWARD** | clear |
| 2026-04-20 | 2.0/3.5 | 2.3113% / 2 | Y / 2.0072% / 2 | **AWARD** | clear |
| 2026-04-21 | 2.0/3.5 | -2.2346% / 2 | Y / 2.1606% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-22 | 2.0/3.5 | -1.6093% / 2 | Y / 2.0276% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-04-23 | 2.0/2.5 | 2.1402% / 2 | Y / 2.0587% / 2 | **AWARD** | clear |
| 2026-04-24 | 1.0/2.5 | 1.0265% / 1 | Y / 1.2245% / 2 | **AWARD** | clear |
| 2026-04-27 | 2.5/3.5 | 2.5342% / 2 | Y / 2.5121% / 2 | **AWARD** | clear |
| 2026-04-28 | 1.5/3.0 | 1.7603% / 2 | Y / 1.7488% / 2 | **AWARD** | clear |
| 2026-04-29 | 2.0/2.5 | 2.4239% / 3 | Y / 2.0523% / 2 | **AWARD** | clear |
| 2026-04-30 | 1.5/2.5 | 1.6567% / 2 | Y / 1.5705% / 1 | **AWARD** | clear |
| 2026-05-01 | 1.0/2.5 | 1.0456% / 1 | Y / 1.0405% / 1 | **AWARD** | clear |
| 2026-05-04 | 3.0/3.5 | 3.2646% / 1 | Y / 3.0285% / 1 | **AWARD** | clear |
| 2026-05-05 | 2.0/3.5 | -1.6192% / 2 | Y / 2.163% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-05-06 | 2.0/3.0 | 2.192% / 1 | Y / 2.2557% / 2 | **AWARD** | clear |
| 2026-05-07 | 1.5/3.0 | 1.5765% / 2 | Y / 1.7004% / 1 | **AWARD** | clear |
| 2026-05-08 | 2.5/3.5 | 2.7547% / 2 | Y / 2.5653% / 2 | **AWARD** | clear |
| 2026-05-11 | 2.0/2.5 | 2.4504% / 1 | Y / 2.0815% / 1 | **AWARD** | clear |
| 2026-05-12 | 1.5/3.0 | 2.0551% / 2 | Y / 1.5342% / 2 | **AWARD** | clear |
| 2026-05-13 | 1.5/3.0 | 1.643% / 1 | Y / 1.53% / 1 | **AWARD** | clear |
| 2026-05-14 | 1.5/2.0 | -0.9354% / 2 | Y / 1.5117% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-05-15 | 1.0/2.0 | 1.0042% / 1 | Y / 1.1188% / 2 | **AWARD** | clear |
| 2026-05-18 | 2.0/2.5 | 2.0291% / 4 | Y / 2.1966% / 2 | **AWARD** | clear |
| 2026-05-19 | 2.5/3.5 | 3.4789% / 1 | Y / 2.7334% / 1 | **AWARD** | clear |
| 2026-05-20 | 1.0/2.0 | 1.0205% / 1 | Y / 1.0103% / 1 | **AWARD** | clear |
| 2026-05-21 | 2.5/3.5 | 2.5203% / 3 | Y / 2.5456% / 2 | **AWARD** | clear |
| 2026-05-22 | 2.5/3.5 | -0.7461% / 2 | N / -0.7461% / 2 | **NO_OPPORTUNITY** | hard_target_no_force_path |
| 2026-05-25 | 1.5/2.0 | -1.2732% / 3 | Y / 1.5796% / 2 | **MARK_WOULD_TAKE** | policy_wrong_size_or_timing |
| 2026-05-26 | 2.0/3.0 | 2.4216% / 1 | Y / 2.0595% / 2 | **AWARD** | clear |

## Streak reward dials (updated from this autopsy)

```json
{
  "streak_award_base": 5.0,
  "streak_award_per_prior": 1.5,
  "streak_break_penalty": -7.5,
  "mark_would_take_eod_penalty": -10.0,
  "no_opp_hold_bonus": 2.0,
  "no_opp_inactivity_scale": 0.35,
  "soul_side_entry_bonus": 3.0,
  "soul_side_misread_penalty": -4.0
}
```

**Forbidden:** shell heat/bank/breach · trail package · PROVEN overwrite · entry-rule thrash for awards.
