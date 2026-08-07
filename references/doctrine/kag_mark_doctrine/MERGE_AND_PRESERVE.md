# MERGE AND PRESERVE — Mandatory for Grok CLI / Any Updater

**Priority:** Prior knowledge is sacred. New content is **additive**. Erasing doctrine is a regression.

## Before you touch anything

1. List existing paths you will edit (ARMY, the-truth, KAG corpus, this pack).
2. Confirm you will **not** delete or overwrite:
   - `all llm's have to know this is the most basic knowledge` pt1–pt5 (any copy)
   - `references/doctrine/STANDING_LAWS.md` and sibling doctrine files
   - `lineages/adaptive_rl_brain_7_31_26/*` Mark clone pins
   - PROVEN weights, configs, and scoreboard artifacts
   - Official sets in `perception/sets.py` / `assert_mark_sets_law`
   - Existing KAG graph nodes already ingested (merge graph, don’t rebuild empty)
3. Prefer: new files, new sections, appended triples, new schema types.
4. Forbidden: “clean rewrite” of a whole doctrine file unless Mark ordered a full replace and you kept a backup branch.

## Merge algorithm

```
for each target_file:
  if missing → create from pack
  if exists →
    keep all prior sections
    append new sections under clear headings with date
    resolve conflicts by KEEPING both + flagging CONFLICT for Mark
    never drop a Law, Set, or DecisionStep
```

### schema.yaml
- Add new `entity_types` / `relation_types` only.
- Do not rename existing IDs (`force`, `velocity`, `TimeframeSet`, …).
- Bump `version` minor (1.1 → 1.2) on additive change; major only if Mark orders breaking change.

### seed_triples.jsonl
- **Append-only.** Never rewrite the whole file from scratch in place without backup.
- Duplicate SPO is OK; dedupe at ingest time.

### Wiki / markdown doctrine
- Add `##` sections; do not replace entire page bodies.
- Keep prior citations and examples.

## What “teach the bot” updates must preserve

| Keep | Why |
|------|-----|
| Five laws stack map | Control chain |
| Sets law | Immutable clocks |
| Wait = loaded | Action semantics |
| One mind train/live | Forward-test identity |
| Relational composition | No lone indicators |
| Dual-period tension | Slingshot topology |
| Forward-shift tunnel | Mass vs phase |
| Shell floor / heat / bank | Survival |
| PROVEN as yardstick | No silent overwrite |

## What new teaching layer adds (does not replace)

- Principle-learning protocol (learn ≠ copy)
- Teacher agent → meta-RL lesson channel
- Novel-indicator zero-shot role assignment
- Forward-test consistency gates
- Generalization rewards (same topology, new sensors)

## Conflict resolution

If pack text disagrees with live PROVEN behavior:

1. Do **not** silently change PROVEN.
2. File a CONFLICT note: `docs/conflicts/YYYY-MM-DD_topic.md`
3. Mark decides; meta does not.

## Checklist before commit

- [ ] `git status` shows no mass deletions of doctrine
- [ ] Sets strings unchanged
- [ ] Decision chain order unchanged
- [ ] New files listed in README table
- [ ] CHANGELOG.md appended
- [ ] Tests / asserts still pass (if repo has them)
- [ ] No “temporary” wipe of graph for convenience

## One sentence for the agent

**Update like a brain adding a skill — not like a disk format that erases the drive.**
