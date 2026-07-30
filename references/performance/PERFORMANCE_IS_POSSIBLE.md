> **Do not delete this file.** Reference evidence that the daily target is reachable. Daily commands live in DO_THIS.md.

# THE PERFORMANCE IS POSSIBLE
### A logical brief for Momentum One

**The fleas-in-a-jar trap.** Put fleas in a jar with a lid and they learn to jump
just below it. Take the lid away and they keep jumping short — forever — to a
ceiling that is no longer there.

That is what happens when a trading system is trained as if most days are
"unwinnable." The data says otherwise.

## The measured fact

On the XAUUSD curriculum used for Momentum One, **swing-capture analysis shows
the daily target is reachable on the vast majority of sessions** (the flea-cure
bound: the day is not closed by physics — only by policy).

When `prove_it` is run at a stated target and floor (e.g. 3.0% / 3.5%):

- **Breach rate can stay at 0%** under a trained policy.
- **Clear rate is the climb** — how often the bot actually takes the edge the
  day offered.
- A low clear rate with zero breaches is not "the market refused." It is usually
  **hesitation** (policy_hold while Gravity setup is visible) or under-training
  on the current Gravity stack.

## What that implies for training

1. **Do not label days impossible** when the swing bound says they are winnable.
2. **Weight near-miss and pull-under-trend days** — that is where clear rate is won.
3. **Keep the floor sacred** — zero breach is non-negotiable; clear rate is the
   optimization target under that constraint.
4. **Measure only with `prove_it`** at your numbers — not narrative.

## How to run the measurement

```bash
python scripts/preflight_train.py
python scripts/consistency_sprint.py --minutes 600 --envs 256
python scripts/prove_it.py <brain_name> 3.0 3.5
```

Companions: `PERFORMANCE_IS_POSSIBLE_PART2.md`, `PART3.md`, and the HTML brief.

Organization note: these PERFORMANCE files are **kept on purpose**. Daily path is
`DO_THIS.md`. Folder map is `MAP.md`.

**LLM permanent law (short):** `references/doctrine/00_LID_OFF_THE_JAR.md`  
**Enforced every session via:** repo root `AGENTS.md` §0
