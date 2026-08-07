# prime-agent → our L2L + KAG map

**Source:** [PrimeIntellect-ai/prime-agent](https://github.com/PrimeIntellect-ai/prime-agent)  
**Local clone:** `data/knowledge/external/prime-agent`  
**Tagline:** Self-improving **RLM agent** + **Continual Harness** for long-running work.

**Not a drop-in** for Channel1 trading policy. We borrow **how knowledge improves** and how long-running agent systems learn without erasing the base.

**Our skill:** LTF pullback vs continuation while HTF strong bull|bear on MARK SETS 1–4; child floor 35; KAG mentors.

---

## Core abstractions (prime-agent)

### 1. RLM — Recursive Language Model
- Context as **variables** (prompt-as-a-variable)
- Tools / subagents as **function calls** in a persistent REPL
- Parent keeps focused context; children get only needed context

### 2. Continual Harness
- Stores **supplemental** prompts, memories, skill descriptions, subagent specs
- `/refine` applies **small, evidence-backed** updates
- **Never rewrites the immutable base system prompt**
- Snapshots support **rollback**

### 3. Skills
- Executable packages (`SKILL.md` + optional Python)
- Recurring workflows become reusable skills

### 4. Long-running continuity
- Daemon sessions, goals, heartbeats, schedules, autonomous budgets
- Quality gates: pass only proves what the gate checks

---

## Map → ARMY / the-truth L2L + KAG

| prime-agent | Our system | Lab use |
|-------------|------------|---------|
| **Immutable base prompt** | Child embryo SHA + PROVEN + pt5 / Mark sets law | Never overwrite child / PROVEN |
| **Continual harness refine** | KAG notes + `L2L_PATH_MEMORY` + WHAT_WORKS + charter | Only evidence-backed law updates |
| **Snapshot / rollback** | KEEP/REJECT restore child backup | REJECT → restore embryo |
| **Skills (executable)** | Path laws: pullback, continuation, anti_thrash… | Train skill classes, not day memos |
| **rlm() subagents** | Peer · Super Mentor · Physics · 15m loop | Parallel perspectives, one primary each |
| **Persistent goal** | same>35 breach=0 L2L skill | Goal until DONE |
| **Heartbeat / schedule** | scheduler 15m L2L fire | Idle climb, no double-train |
| **Compaction** | Error cards + class memory (not full transcript) | Keep structure, drop noise |
| **Quality gates** | learn≠copy, path_class threshold, pack same≥35 | Gate before KEEP |
| **Agent messaging** | Dialogue jsonl + mentor talk CLIs | Cross-agent evidence |
| **Python skills** | `learn_to_learn_path.py`, spine scripts | Executable lab skills |

---

## L2L recipe (prime-agent informed)

```
BASE (immutable)     = CHILD_STAGE embryo + MARK SETS LAW + pt5
HARNESS (refinable)  = path laws, class boosts, KAG notes, WHAT_WORKS
REFINE               = only after REJECT/KEEP evidence this cycle
SUBAGENTS            = dual peer + super mentor + physics (+ prime-rl map)
SKILL TARGET         = ltf_pullback_htf_strong | ltf_continuation_htf_strong
VERIFY GATE          = same≥35 · breach=0 · learn≠copy · path_class strong
ROLLBACK             = restore child SHA on FAIL
```

### Refine rules (from Continual Harness)

1. **Never edit base** — child weights / PROVEN / sets law stacks  
2. **Small updates only** — one law boost, one recipe tag, one note line  
3. **Evidence required** — metrics from this round (path_class, POST same, labels)  
4. **Rollback** — REJECT restores embryo; snapshots in BEST + memory jsonl  
5. **Skills > chat** — encode wins as path laws / scripts, not long prose  

---

## What we will NOT do

| prime-agent | Why not wholesale |
|-------------|-------------------|
| Install daemon / full TUI on lab | Optional later; we already have ARMY agents + scheduler |
| Let model rewrite base system | Violates child floor / PROVEN |
| Unbounded autonomous mode | Floor + budgets required |

---

## Immediate KAG actions (this session)

1. Index prime-agent README + docs into Super Mentor / knowledge  
2. Treat `L2L_PATH_MEMORY` + `KAG_L2L_HTF_LTF__latest.md` as **harness state**  
3. Continue struct-path climb: skill = pullback/continuation under HTF  
4. Mentors = rlm-style subagents with ONE primary each  

---

## Links

- Upstream: https://github.com/PrimeIntellect-ai/prime-agent  
- Continual Harness paper: https://arxiv.org/abs/2605.09998  
- Related: prime-rl map `PRIME_RL_L2L_MAP.md`  
