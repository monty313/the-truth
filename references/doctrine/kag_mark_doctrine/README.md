# Mark Doctrine KAG Seed Pack

**Version:** 1.1.0 (additive — never wipe prior doctrine when merging)  
**For:** ARMY LLM-KAG agents · the-truth tutors · meta-learning RL Mark clone · Grok CLI / VS Code agents

---

## Mission (read this first)

This pack is **not** a cheat sheet of answers to copy into the bot.

It is the **principles + learning process** Mark’s brain uses so that:

1. **LLM KAG agents teach** the meta-learning RL trading bot (tutor → student).
2. The bot **identifies its own observations** and assigns **meaning to each indicator relative to others**.
3. The bot knows what to do with indicators it has **never seen or used before** — by role and relation, not by memorized name.
4. Learning serves **long-term consistent daily goal hits** in **forward testing**, not backtest parroting.
5. Mark’s brain **learns**; it does not only **copy** direct answers. The clone must do the same.

```
PRIOR KNOWLEDGE (keep forever)
  five laws · sets · decision chain · composition · dual period · shift · tensor
        ↓
KAG AGENTS (teachers)
  map novel sensors → roles → relations → topology → act
  explain WHY in Mark language; never only “buy because RSI”
        ↓
META-LEARNING RL (student)
  learns the mapping function + attention, not a frozen indicator cookbook
  generalizes under forward distribution shift
        ↓
FORWARD TEST CONSISTENCY
  same mind train/live · goal/floor · wait as skill · one story all day
```

---

## Files (do not delete when updating)

| File | Role | Erase? |
|------|------|--------|
| `README.md` | Mission + merge rules | No — edit in place |
| `MERGE_AND_PRESERVE.md` | **Grok CLI / any agent:** how to update without wiping knowledge | No |
| `PRINCIPLES_LEARNING.md` | Learn ≠ copy; generalization law | No |
| `teach_to_meta_rl.md` | How KAG agents teach the RL bot | No |
| `novel_indicator_protocol.md` | Never-seen indicator → role → act | No |
| `schema.yaml` | Entity/relation types (additive only) | Append types only |
| `seed_triples.jsonl` | Doctrine SPO seeds | **Append only** |
| `logical_forms.md` | Multi-hop questions | Append forms only |
| `obs_feel_spec.md` | What clone must perceive | Append sections only |
| `agent_constitution.md` | Shared ARMY constitution | Append rules only |
| `CHANGELOG.md` | What each update added | Always append |

---

## Install sketch (KAG)

1. Create Mark domain project in [OpenSPG/KAG](https://github.com/OpenSPG/KAG).
2. Load `schema.yaml` (merge new types; do not drop old).
3. Ingest wiki + this pack + existing ARMY/the-truth doctrine (union, not replace).
4. Index with schema-constrained extractor.
5. Expose logical-form QA to **teacher agents**.
6. Teacher agents emit **principle lessons + topology labels** into the meta-RL reward/BC channel (see `teach_to_meta_rl.md`).
7. Meta may retune **attention / sensor trust**; never overwrite PROVEN weights, shell floor, or Mark sets without human order.

---

## Hard locks (never search away)

- Official sets only: `1m|15m,30m` · `5m|30m,1h` · `15m|1h,4h` · `30m|4h,1d`
- Decision order: tide → regime → breath/launch → fire/wait/kill → finish
- Meta may retune attention only; never overwrite PROVEN / shell floor / sets without human order
- Agents = sensors **and teachers**; policy soul = Mark decision chain + learned relational mapper
- **Learn principles; do not only copy answers**
- Novel indicator = assign role via relation; never “unknown → freeze forever”

---

## Hand-off to Grok CLI (VS Code)

1. Open this folder as context.
2. Read `MERGE_AND_PRESERVE.md` before any edit.
3. Apply updates **additively** into ARMY / the-truth.
4. Run doctrine tests / `assert_mark_sets_law` if present.
5. Do **not** delete prior basic-knowledge packs, STANDING_LAWS, or PROVEN paths.

---

## Owner

Mark Montgomery Jr — monty313  
Repos: [ARMY](https://github.com/monty313/ARMY) · [the-truth](https://github.com/monty313/the-truth) · [OpenSPG/KAG](https://github.com/OpenSPG/KAG)
