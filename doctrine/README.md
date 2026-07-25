# doctrine/

Laws and definitions the bot and LLM follow.

## How to update
- **Regimes / indicators:** edit `LLM_REGIME_DEFINITIONS.yaml` only (append to registries).
- **Standing laws:** edit `STANDING_LAWS.md` with a dated CHANGE LOG line.
- Do not copy doctrine text into root README or TRAINING.md — link here instead.
- Old IRAC one-offs can stay as dated files (`IRAC_*`); do not spawn duplicates.

| File | Role |
|------|------|
| LLM_REGIME_DEFINITIONS.yaml | SSOT for regimes + obs indicator registry |
| STANDING_LAWS.md | Hard laws |
| HOST_RUN.md | Longer host playbook |
| REGIME_LANGUAGE.md | Human notes (points at the YAML SSOT) |
