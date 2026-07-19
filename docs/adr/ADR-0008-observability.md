# ADR-0008: Observability-first (Fable 5 doctrine)
Telemetry before strategy code. Spans: ingestion, features, regime/state calc, mask checks,
inference, action, reward, order submit, fill, state update, checkpoint, errors, recovery.
Run-card metadata: code/config/reward/mask/registry/model versions, window, symbols, TFs,
seed, broker+execution assumptions, lineage, metrics, summary. MLflow optional; local JSONL
run-cards + spans always on (Langfuse-equivalent spec; vendor swappable — Monty picks at
Phase-2 gate; overnight default = local, documented).
