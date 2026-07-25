# scripts/

**Only place for commands you run.**

## How to update
- New runnable command → add `verb_noun.py` here.
- Put real logic in `training/`, `telemetry/`, etc.; script should be thin.
- Add one row to the table below.
- Do not leave a second copy of the same script under another name.

## Before train
| Script | Does |
|--------|------|
| preflight_train.py | Check stack is ready |
| restore_meta_tuner.py | Fix missing meta_tuner |
| align_tf_sets.py | Lock TF sets |

## Train
| Script | Does |
|--------|------|
| consistency_sprint.py | Climb clear rate |
| meta_train.py | Meta-tuner (rewards + hparams) |

## Measure
| Script | Does |
|--------|------|
| prove_it.py | Clear % / row / breach |
| give_llm_what_it_needs.py | Probe + ghosts + IRAC pack |
| mind_probe_day.py / diagnose_day.py | One-day look |

See **DO_THIS.md** at repo root for the order to run them.
