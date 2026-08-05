# AGENTS.md — rules for every Grok / LLM / coder session

> **AUTO-LOADED by Grok Build.**  
> You do **not** need to paste this or say “read AGENTS.md.”  
> When the session opens in this repo, Grok injects this file into its context.  
> Confirm anytime: `grok inspect` → **Project Instructions** should list this file.

**Owner:** Monty (dyslexic). Organization is not optional.  
**Canonical goal file:** [GOAL.md](GOAL.md) — SSOT; never invent a different mission.  
**House map (anti-confusion):** [00_MAP_OF_THE_HOUSE.md](00_MAP_OF_THE_HOUSE.md) — tracks T1–T4, work order, freeze list.  
**Current handoff:** [HANDOFF_2026-08-05.md](HANDOFF_2026-08-05.md)  
**Mindset law:** [references/doctrine/00_LID_OFF_THE_JAR.md](references/doctrine/00_LID_OFF_THE_JAR.md)  
**Second Brain (Mark):** [MARK HERE!.lnk](MARK%20HERE!.lnk) · soul bridge [SOUL_MATCH.md](SOUL_MATCH.md) · thinking laws [references/doctrine/llm_basic_thinking/00_INDEX.md](references/doctrine/llm_basic_thinking/00_INDEX.md) — **same MarkOS as ARMY**, not a second personality.  
**pt5 → code map:** [references/doctrine/llm_basic_thinking/HOW_PT5_MAPS_HERE.md](references/doctrine/llm_basic_thinking/HOW_PT5_MAPS_HERE.md)  
**Agents/obs after soul:** [KEEP_AFTER_SOUL.md](KEEP_AFTER_SOUL.md) — **keep all; do not wire until Mark soul is full in policy.**

**Also auto-loaded:** `.grok/rules/00_goal.md` · `.grok/rules/00_lid_off.md` · `.grok/rules/00_map_house.md`  
**Other tools:** `CLAUDE.md` + `.github/copilot-instructions.md` point here.

If you do not know where something goes, **stop and put it in the right place**.  
Do not leave new mess on the root.  
**Always name the track (T1–T4)** before changing code.

---

## GOAL (forever — never confuse this)

**This is the only mission. Permanent. Every session. Every agent.**

### What we are building

**One bot that solves for any target % and risk % Monty types in — without having to retrain.**

Not “the 3% bot.”  
Not “change numbers → retrain.”  
Same brain; **whatever pair** goes in the **target** and **risk** places:

1. Hit that **daily profit target %**  
2. Without breaching that **daily risk floor %**

| | |
|--|--|
| **Job** | Goal-conditioned scalper: **any** runtime target + risk pair, **no retrain to switch pairs** |
| **Win meter** | `prove_it <brain> <target> <risk>` only |
| **Climb** | **Clear %** at the pair under test |
| **Sacred** | **Breach % = 0** at that pair |
| **Also climb** | Streak at that pair |
| **Filter** | If it does not help clear % or breach 0% **for the numbers Monty cares about** → **skip it** |
| **Target / risk** | **Inputs you type** at runtime — same `.pt` |
| **Forbidden** | Retrain **only** because Monty changed 2.5 / 3.0 / 3.5 (or any other pair) |
| **Yardstick** | 3.0 / 3.5 is a common scoreboard — still test other pairs without retrain |
| **Champion** | `models/00_CHAMPION.md` + `GOAL.md` |
| **Score** | `python scripts/prove_it.py <brain> <target%> <risk%>` or `USE/1_prove.bat` |

**Full text:** root `GOAL.md`. If this section and `GOAL.md` disagree, **`GOAL.md` wins** — then update this section.

**Improve order:** diagnose → dials/masks → practice (vary goal/floor when possible) → prove_it → keep or reject.

**Out of scope unless Monty asks:** new UIs, parallel frameworks, essays without `prove_it`, or a single-pair-only bot.

---

## 0. LID OFF THE JAR (non-negotiable thinking)

**This is permanent. Every model, every session, in this folder.**

Fleas in a jar learn a lid. **The lid is gone.**  
Performance is possible. Low clear rate is not “the market said no.”

### Oath (load this before you touch code)

1. Nothing the swing-bound / ledger showed possible is **impossible**.  
2. We only figure out **how to make it consistent** — clear % **up**, breach **0%**, longer rows.  
3. Only **measurement** may say no (`prove_it`, serials, bounds) — never narrative fear.  
4. The bot becomes what you **measure**, not what you **mean**. Fix the scoreboard before blaming the policy.

### Forbidden conclusions (reject on sight)

| Forbidden | Replace with |
|-----------|----------------|
| “Day / setup is impossible” | Find the **finite lid** (law bug, fee, weight=0 day, flinch seed, thin single edge) |
| “Market refused” when breach=0 and clear is low | **Policy / perception / hesitation / under-training** |
| “Range too small → unwinnable” | Quiet day → tight stop → **size** still pays path; do not zero-weight by range |
| “70% signal is a fairy tale” | Measure **agreement** of independent families (PART4, slots 80–83) |
| “Good enough forever at ~55%” | That was the lid — compose, practice, prove |

### Four ticks already cured — do not reintroduce

1. **Pay-cliff** (lock/fee so banked &lt; goal → unpaid wins)  
2. **Counterfeit applause** (noise as achievement)  
3. **Forbidden classroom** (zero-weight “unwinnable” days by assumption)  
4. **Inherited flinch** (warm-start from flat ancestors)

### Evidence chain (when you need depth)

| Part | File | One sentence |
|------|------|--------------|
| 1 | `references/performance/PERFORMANCE_IS_POSSIBLE.md` | Money / bound exists; clear is the climb |
| 2 | `references/performance/PERFORMANCE_IS_POSSIBLE_PART2.html` | Lid was often **our law**; lift is real |
| 3 | `references/performance/PERFORMANCE_IS_POSSIBLE_PART3.md` | Diagnosis: ticks in feedback; body (breach=0) was fine |
| 4 | `references/performance/PERFORMANCE_IS_POSSIBLE_PART4.md` | Agreement lifts the signal lid (70–81% band) |
| Law card | `references/doctrine/00_LID_OFF_THE_JAR.md` | Short permanent summary |
| Audit | `references/doctrine/flea-jar/THE_FLEA_CURE.md` | Full pathology of assumed ceiling |

### Cure order when the patient is sick

1. Rewards / skill / **honest scoreboard**  
2. Practice map (never zero days by assumption)  
3. Indicator logic **last** (with a measured case)

**Standing order (Monty’s founding words):** never say impossible again — let only measurements say no.

---

## 1. Read order (every session)

1. **GOAL section above** (and root `GOAL.md` if anything is unclear)
2. `00_START_HERE.md` (human path)
3. **This file** — **GOAL** + **§0 LID OFF THE JAR**
4. `references/doctrine/00_LID_OFF_THE_JAR.md` (if doing diagnosis / training / signals)
5. `DO_THIS.md` (if running commands)
6. `references/handoffs/HANDOFF.md` (only if you need full history)
7. Then code

---

## 2. Root is sacred (keep it empty)

**Allowed at repo root:**

| Name | Why |
|------|-----|
| `00_START_HERE.md` | First file humans/AI open |
| `GOAL.md` | Mission |
| `DO_THIS.md` | Daily commands |
| `MAP.md` | Folder map |
| `AGENTS.md` | This file |
| `README.md` | Short overview |
| `USE/` | Monty’s daily buttons |
| `configs/` `data/` `models/` `scripts/` `training/` … | Real code/data |
| `Makefile` `requirements.txt` `pyproject.toml` `.vscode/` | Tooling |

**Never create at root:**

- New `*.md` essays, handoffs, plans, “notes”
- Random `test_*.py`, scratch scripts
- New top-level packages without asking Monty

**Put long writing here instead:**

| Kind | Put in |
|------|--------|
| Session handoff | `references/handoffs/` |
| Doctrine / laws / wins | `references/doctrine/` |
| Performance essays | `references/performance/` |
| Plans / how-to | `references/plans/` |
| Prompts | `references/prompts/` |
| Charts to keep | `reports/figures/` |
| Temp outputs | `artifacts/` |
| Dead experiments | `_archive/` |

---

## 3. Naming convention (easy to notice)

Use **prefixes** so important things sort to the top:

| Prefix | Meaning | Examples |
|--------|---------|----------|
| `00_` | Look first | `00_START_HERE.md`, `models/00_CHAMPION.md`, `scripts/00_DAILY.md` |
| `1_` `2_` `3_` | Daily order in `USE/` | `USE/1_prove.bat` |
| `PROVEN_` | Frozen proof brains | `models/PROVEN_*.pt` |
| `_` | Internal / ignore | `_archive/`, `scripts/_fix_*.py` |

**Daily human zone = `USE/`**  
If Monty needs it often, put a **button** in `USE/`, do not make him hunt in `scripts/`.

---

## 4. Goal filter (before any change)

Ask: **Does this raise clear % or protect breach 0% on `prove_it`?**

- **Yes** → do the smallest change  
- **No** → do not build it  
- **Also ask:** Am I reinstalling a jar lid (impossible-day labels, fake ceilings, unpaid wins)?

Target % and risk % are **runtime inputs**. Do not retrain only to change 2.5 / 3.0 / 3.5.

---

## 5. Where code lives (one home each)

| Job | Folder |
|-----|--------|
| Numbers | `configs/*.yaml` |
| Price data | `data/raw/` |
| Brains | `models/` |
| Commands | `scripts/` (document daily ones in `scripts/00_DAILY.md`) |
| Learning | `training/` |
| Features | `features/` |
| Signals | `signals/` |
| Load brain | `inference/` |
| MT5 | `execution_bridge/` |
| Tests | `tests/` |

**Do not** create a second parallel tree (`src/` was archived because it duplicated packages).  
Working packages live at **repo root** (`training/`, `core/`, …).

---

## 6. After you finish work

1. No new junk on root  
2. Update `GOAL.md` scoreboard if clear/breach changed (keep AGENTS **GOAL** section in sync if the mission wording changes)  
3. If champion brain changed → update `models/00_CHAMPION.md`  
4. If new daily command → add to `USE/` + `scripts/00_DAILY.md` + `DO_THIS.md`  
5. Long notes → `references/handoffs/` not root  
6. Run or mention: `python scripts/preflight_train.py` if you touched paths  
7. Never redefine the mission — only `GOAL.md` may change the goal  

---

## 7. Dyslexia style (writing)

- Short sentences  
- Tables over paragraphs  
- Bold the **one** action  
- Prefer `00_` files over buried README walls  
- Never say “see docs” without a full path  

---

## 8. Forbidden without Monty asking

- New frameworks / second pipelines  
- Root markdown dumps  
- Deleting `PROVEN_*.pt` or success ledger  
- “Cleaning” by scattering files into random folders  
- Declaring days/setups **impossible** without a measured bound  
- Zero-weighting curriculum days by high-low range assumption  
- Warm-starting from known-flat ancestors when a banking seed exists  

---

## Quick “where does this go?” 

```text
Daily button for Monty     → USE/
Command script             → scripts/  (+ list in scripts/00_DAILY.md if daily)
Config number              → configs/
Trained brain              → models/
Keep chart                 → reports/figures/
Temp cache                 → artifacts/
Session story              → references/handoffs/
Law / win / doctrine       → references/doctrine/
Unsure                     → ask Monty OR put in _archive/inbox/ with a note
```
