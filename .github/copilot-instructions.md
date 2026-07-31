# GitHub Copilot — Momentum One

**Mission SSOT:** `GOAL.md`  
**Full law:** `AGENTS.md`

## GOAL (forever)

**One bot that solves for any target % and risk % the user types in — without having to retrain.**

- Hit that daily **target %**
- Never breach that daily **risk %**
- Climb **clear %**, keep **breach = 0**, climb **streak** — at **those** numbers
- Target/risk are **runtime inputs** — same brain when numbers change
- **Forbidden:** retrain only to switch pairs
- Score: `python scripts/prove_it.py <brain> <target> <risk>`
- Skip work that does not move clear % or protect breach 0 at the pair under test

## Hard rules

- Lid off the jar: never call a winnable day/setup impossible without measurement.
- Low clear + 0 breach = policy/hesitation, not “market refused.”
- Keep root clean. Long notes → `references/`. Brains → `models/`. Data → `data/raw/`.
- Packages at repo root. No parallel `src/` tree.
