# adaptive_rl_brain_7_31_26

**Parallel lineage only.** Never overwrite PROVEN checkpoints, living champion
artifacts, or promote this lineage over the current champion without an explicit
human gate.

## Purpose

A clean adaptive RL brain lineage that classifies structure with Official Sets,
three-group confluence (Direction + Velocity), pullback / scale-conflict rulers,
and a hard **MINDLESS** wall — so teaching and rewards can target named trade
classes instead of noise.

## Phase 1 (current)

Pure perception modules under `perception/`:

| Module | Role |
|--------|------|
| `types.py` | Enums / dataclasses |
| `sets.py` | Official Sets 1–4 + Sub-Sets A–E |
| `confluence.py` | 3 groups + simple majority Direction/Velocity |
| `structure.py` | Pullback + Scale-Conflict |
| `classify.py` | 4 tags + MINDLESS wall |
| `live_indicators.py` | Phase 2 Slice 1: real CCI/RSI/channel → confluence flags |
| `pipeline.py` | Phase 2 Slice 2: live structure + classify end-to-end |
| `observation.py` | Phase 2 Slice 3: Channel 1 obs block (dim 32, lineage-local) |
| `../rewards.py` | Phase 2 Slice 4: dials + credit formula |
| `data/mtf.py` | Phase 2 Slice 5: M1 → multi-TF pack (reuses data_io.loader) |
| `day_runner.py` | Phase 2 Slice 5: day loop + rewards/inactivity |
| `policy_stub.py` | Phase 2 Slice 5: tiny Channel1 policy (not PROVEN) |
| `train_stub.py` | Phase 2 Slice 5: rollout + REINFORCE smoke |

**Phase 1–2 through Slice 5 complete.** Still parallel-only: no PROVEN writes / no champion promotion.

Full rules: [`SPEC_PHASE1.md`](SPEC_PHASE1.md).

## Isolation

- Pure functions only in Phase 1.
- No writes to `models/`, `artifacts/checkpoints/`, or any PROVEN path.
- Tests use hand-built synthetic snapshots only (no curriculum required).

## Package path

```text
lineages.adaptive_rl_brain_7_31_26.perception.*
```

## Tests

```bash
python tests/lineages/adaptive_rl_brain_7_31_26/test_sets.py
```

Do **not** start Phase 2 (training / rewards / obs expansion) until every Phase 1
test is green.
