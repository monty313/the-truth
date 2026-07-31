# START HERE

**You have dyslexia. This repo is built so you only look at a few places.**

---

## Open these 4 (in order)

| # | File | What it is |
|---|------|------------|
| **1** | [GOAL.md](GOAL.md) | The only mission |
| **2** | [DO_THIS.md](DO_THIS.md) | Commands to type |
| **3** | [USE/](USE/) | One-click daily buttons |
| **4** | [models/00_CHAMPION.md](models/00_CHAMPION.md) | Which brain is active |

Everything else is background.

---

## The mission (one line)

**One bot that works for whatever target % and risk % you type in — without retrain.**  
Change the numbers → same brain → score again. Not only 3.0 / 3.5.

## The score (at YOUR numbers)

```text
python scripts/prove_it.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5
```

Change the last two numbers anytime (e.g. `2.5 2.5`). Same brain.  
Or double-click: **`USE/1_prove.bat`**

| Word | Meaning |
|------|---------|
| **Clear %** | Good days (hit **your** target, no floor hit) → climb this |
| **Breach %** | Floor hits on **your** risk → must stay **0** |

---

## If you feel lost

1. Come back to **this file**
2. Read **GOAL.md** (one page)
3. Run **USE/1_prove.bat**
4. Ignore every other folder

---

## For AI / Grok sessions

**You do not have to remind Grok of the goal.**  
In this folder, Grok Build **auto-loads**:

| File | What it burns in |
|------|------------------|
| **[GOAL.md](GOAL.md)** | Mission (via AGENTS + rules) |
| **[AGENTS.md](AGENTS.md)** | Full law — starts with **GOAL (forever)** |
| **`.grok/rules/00_goal.md`** | Goal only — always on |
| **`.grok/rules/00_lid_off.md`** | Lid-off mind |

Check: terminal in this folder → `grok inspect` → **Project Instructions**.

Other tools: `CLAUDE.md`, `.github/copilot-instructions.md` (same goal).

Law card: **[references/doctrine/00_LID_OFF_THE_JAR.md](references/doctrine/00_LID_OFF_THE_JAR.md)**

Keep the repo organized. Do not dump files on the root.
