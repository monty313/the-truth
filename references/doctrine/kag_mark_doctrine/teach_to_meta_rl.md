# How LLM KAG Agents Teach the Meta-Learning RL Bot

Agents are not only sensors. They are **tutors**.  
The meta-RL policy is the **student**.  
KAG is the **shared textbook that supports multi-hop why**.

---

## Teaching loop (always on)

```
Market obs (multi-set, multi-indicator)
        ↓
KAG agents: bind instances → roles → relations (logical forms)
        ↓
Teacher lesson: principle_application JSON (see PRINCIPLES_LEARNING.md)
        ↓
Channels into student (any/all, gated):
  A. Behavior cloning targets (act + topology + wait subtype)
  B. Reward shaping terms (pay correct topology; punish copy-thrash)
  C. Attention / meta hints (which roles mattered this bar)
  D. HITL shadow (MARK HERE corrections become gold lessons)
        ↓
Meta-learner updates attention / trust / role-mapper head
        ↓
Policy act: fire | wait_loaded | wait_no_trade | kill | manage
        ↓
Forward outcome → consistency score → accept/reject meta update
```

---

## What teachers must teach (curriculum order)

1. **Permission first** — tide from HTF force; LTF never sets opposite side.
2. **Regime gate** — playbook is a function of state; unknown → m=0.
3. **Breath vs launch** — two markets, two rules; measure, don’t feel.
4. **Dual clocks** — fast vs slow period; tension vs co-aligned launch.
5. **Shift / tunnel** — full-body mass vs wick; micro phase under fixed G.
6. **Composition** — no lone indicator; every fire cites force+velocity roles.
7. **Wait as skill** — loaded ≠ freeze ≠ kill.
8. **Finish line** — goal/heat/floor; still Mark at close.
9. **Novel sensor protocol** — role inference without name memory.
10. **Forward consistency** — same story tomorrow under distribution shift.

Do not start teaching at “entry signal.” Start at tide.

---

## Teacher roles inside ARMY

| Agent role | Teaches | Must not |
|------------|---------|----------|
| Force tutor | HTF mass / G / tide | Time entries |
| Velocity tutor | LTF phase / load depth | Override tide |
| Regime tutor | chop / trend / vol shock masks | Invent sets |
| Composition tutor | multi-indicator purpose | Lone-indicator fires |
| Novel-sensor tutor | role assignment for unseen families | Guess side from unknown alone |
| Soul tutor | goal/heat/bank/finish | Bypass shell |
| Consistency tutor | day-level story / thrash detection | Optimize single-bar reward only |

Shared constitution: `agent_constitution.md`. Shared graph: KAG. Shared soul: decision chain — not majority vote of tutors.

---

## Interfaces into the bot (implementation contract)

### 1) Lesson bus (recommended)

- Topic: `mark.teacher.lesson.v1`
- Payload: principle_application JSON
- Consumer: BC buffer + reward wrapper + optional meta hint buffer
- Dedupe by `(day, bar_index, topology, principle_ids)`

### 2) Aux heads on policy (student must predict)

| Head | Target | Why |
|------|--------|-----|
| `act` | fire_buy/sell, wait_*, kill, manage | Behavior |
| `topology` | slingshot_load/release, launch, collapse, chop | Principle class |
| `tide` | +1/0/−1 per set | Permission |
| `wait_subtype` | loaded / no_trade / heat | Wait skill |
| `role_map` | soft assignment novel→role | Zero-shot |
| `goal_pressure` | lagging / on_pace / bankable | Finish line |

Training loss = policy RL + aux CE on teacher labels (weighted).  
**Copy failure mode:** only `act` head trained → student memorizes actions without principles.  
**Learn mode:** topology + role_map + wait_subtype required.

### 3) Reward terms (shaped, gated)

| Term | Pay when | Punish when |
|------|----------|-------------|
| `w_topology_match` | student topology = teacher | mismatch on clear bread-and-butter |
| `w_wait_loaded` | HOLD while slingshot_load | freeze on clear load with later miss |
| `w_novel_role` | correct role on held-out sensor | random role / ignore all novel |
| `w_forward_consistency` | day clear without breach | thrash after stop, side flip noise |
| `w_setup_skip` | — | sit out clear B&B |
| `w_copy_thrash` | — | single-bar reverse spam |

Meta may search weights of these terms inside floors; may not delete shell laws.

---

## Teaching without erasing prior knowledge

- New lessons **add** edges in KAG; they do not delete Laws.
- BC datasets **union** old Mark days + new forward shadow days.
- When a teacher is wrong, HITL corrects; wrong lesson is down-weighted, not “wipe curriculum.”
- PROVEN checkpoint remains the yardstick until Mark promotes.

See `MERGE_AND_PRESERVE.md`.

---

## Success criteria (student learned, not copied)

1. **Held-out indicator family:** swap CCI velocity for RSI/Stoch/WPR with same topology labels → act agreement stays high.
2. **Held-out days (forward):** goal-hit rate and breach=0 comparable to validated window under same shell.
3. **Mind probe:** when asked why, policy/aux cites topology + roles, not only last reward.
4. **Wait quality:** loaded waits convert or correctly kill; not random HOLD.
5. **One mind:** train and live action semantics match.

If act match is high but topology/role_map are chance → **copying**. Fail the gate.

---

## Minimal pseudo-code (tutor step)

```text
ctx = pack_all_official_sets(obs)
for set in sets:
  tide[set] = kag.query("tide", set)
  roles = assign_roles(ctx.sensors, set)   # includes novel protocol
  rel = relations(roles)                   # with/against, G, efficiency
  topo[set] = classify_topology(rel)
act, why = decision_chain(tide, topo, heat, goal)
lesson = {principle_ids, roles, rel, topo, act, why, novel_flags}
emit(lesson)
train_student(lesson)  # BC + aux + shaped r; meta on attention only
```

Teachers run this every bar (or every decision pack). Students never skip the chain because a sensor name is new.
