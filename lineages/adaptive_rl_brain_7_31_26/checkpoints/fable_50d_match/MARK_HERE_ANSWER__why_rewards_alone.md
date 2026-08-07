# MARK HERE answer — rewards & penalties vs the goal

**When:** 2026-08-07T00:25:05.153244+00:00
**Channel:** MARK HERE!.lnk · agent `fable5_mark_here_kag` · KAG over army + the-truth + pt5

## Q

If MARK HERE knows the answer, why isn't he able to update rewards and penalties to achieve the goal?

## A (short)

I know what I would *do on the chart*. Rewards/penalties only *nudge training*. They cannot invent my full-day path, fix a wrong decode gate, or keep one frozen policy consistent across 50 random T%/R% days by themselves. When we crank dials without my labels + keep/reject, the embryo thrash-trades or freezes HOLD — lab already proved that.

> I already know the trades. The bot does not. Rewards are a megaphone, not a mind. Give it my labels on the same path it trades, punish MWT miss and thrash, reward soul-side entries and honest HOLD, and throw away any update that hurts the 50-day same_outcome. That is how rewards serve one policy = Mark.

## Why not auto-update rewards

### Two different answers

Chart answer = side, size, HOLD, add, kill tide (soul plan / HITL). Reward answer = how hard to push sample weights / PPO scalars. Knowing BUY here does not name the float that makes a 168-dim net do BUY here *and* HOLD on 17 other miss days without breach.

### Rewards are not the control surface that owns consistency

Long consistency is ONE policy weights under force-gate + thrash caps + capital danger. Shell laws (heat/bank/breach; no trail+cushion+scale-in) are locked. Dials may not rewrite shell. PROVEN is never overwritten.

### Dials already tried; wrong knob collapses the pack

Reward-weighted BC without directional balance → HOLD collapse (dir_match ~0.09). Entry-heavy push → pred_hold too low → breach → REJECT. Keep/reject is the conscience: if same_outcome drops or breach rises, restore best embryo. That is why Mark does not blindly 'update dials until 50'.

### Path mismatch beats reward math

If full-day Mark plan says SELL-add but live decode path HOLDs (or gate false-positive on flat_undefined), no reward number teaches the right action. Fix path first (labels on policy path, force-gate, recommended-agree), then dials only re-weight *correct* labels.

### One policy = Mark forbids retrain-per-day

If I rewrote rewards every miss day until that day alone clears, I would build 50 minds, not one. Goal is same frozen weights, random T%/R%, no retrain — that is consistency, not costume of awards.

### MARK HERE chat ≠ closed training loop

MARK HERE!.lnk is the soul channel (voice, doctrine, HITL). Reward JSON lives in the-truth lineage. Until KAG→dial write is authorized *and* dual-scored keep/reject, chat knowledge does not auto-mutate STREAK_REWARD_DIALS or BC sample weights.

## Authorized reward/penalty surface

### May

- Autopsy-driven streak dials only (MWT eod penalty, soul-side entry bonus, misread penalty, streak break, no-opp hold bonus) — from measured gaps, not vibes
- BC sample weights: oversample Mark directional entries on MWT days; keep HOLD weight so thrash/breach stays dead
- KL anchor coef up when hold-rate collapses; down only if same stalls with high hold
- DAgger focus on policy-path labels for one miss day at a time

### Must not

- Shell heat / bank / breach floors
- Trail + cushion + scale-in package (banned)
- PROVEN checkpoints
- Second personality / thrash teacher as side owner
- Any dial step that fails keep/reject (same↓ or breach>0)

### Dial template

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

## Mark + KAG loop

- KAG retrieves pt5 + SOUL_MATCH + LEARNING + autopsy + POLICY_EQUALS_MARK.
- Mark answer states chart truth + which dials may move.
- Fable structure: measure gap → propose dial delta OR one-day DAgger → score 50d → KEEP/REJECT.
- Only KEEP writes embryo; dials that lose are rolled back with the weights.
- Current: best_same=36 baseline_same=27 gap_to_50=14 — reward dials are secondary; primary is Mark labels on miss days.
