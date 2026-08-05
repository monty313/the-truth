# telemetry/

How we see what the bot is thinking.

## How to update
- New diagnostic field → `mind_probe.py` + short note in CHANGE LOG.
- Regime labels / skip reasons → keep in sync with `doctrine/LLM_REGIME_DEFINITIONS.yaml`.
- Ghosts stay read-only (no weight changes).
- Do not dump large JSON into git; write under `artifacts/`.

| File | Role |
|------|------|
| mind_probe.py | MRI of decisions |
| ghost_trades.py | Counterfactuals for IRAC |
| regime_language.py | Names regimes from flags |
