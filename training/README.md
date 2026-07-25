# training/

Brain + learning + meta_tuner.

## How to update
1. Edit the real module (not a copy)
2. Append CHANGE LOG: date + WHY
3. If meta_tuner.py is missing: `python scripts/restore_meta_tuner.py`

| File | What |
|------|------|
| meta_tuner.py | Adjusts rewards and lr (wrapper) |
| policy.py | Network |
| fastsim.py | Fast simulator |
