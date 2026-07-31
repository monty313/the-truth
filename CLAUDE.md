# Claude / multi-tool bridge

**Mission:** [GOAL.md](GOAL.md)  
**Full law:** [AGENTS.md](AGENTS.md)

## GOAL (forever)

**One bot that solves for any target % and risk % Monty types in — without having to retrain.**

1. Hit the **daily profit target %** he set  
2. Do not breach the **daily risk floor %** he set  

Same `.pt`. Change the two numbers anytime. **No retrain** just to switch pairs.  
3.0/3.5 is a yardstick, not the whole goal.

| Meter (at his numbers) | Rule |
|------------------------|------|
| **Clear %** | Climb |
| **Breach %** | Stay **0** |
| **Streak** | Climb |

**Score:** `python scripts/prove_it.py <brain> <target%> <risk%>`  
If it does not help clear % / breach 0 at the pair under test → **skip it**.

## Also

- **Lid off the jar** — never “impossible”; measure with `prove_it`  
- **Organization** — follow `AGENTS.md` (no root doc dumps)

Detail: `AGENTS.md` GOAL + §0 · `references/doctrine/00_LID_OFF_THE_JAR.md`
