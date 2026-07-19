# Agent Instructions (for any LLM working in this repo)
<!-- 5W+I: WHO future VS Code agents + Claude sessions. WHAT ground rules.
WHEN 2026-07-18. WHERE repo-wide. WHY Fable 5 doctrine: constrained autonomy.
INTERCONNECTED WITH docs/LAWS.md, docs/adr/, codex/. -->
- Read docs/LAWS.md and docs/adr/ FIRST. The codex/ is ground truth for trading logic.
- NEVER change: Shell rules, forever masks, reward structure, caps, bars — without
  an explicit approval from Monty recorded as a new ADR.
- Every edit: keep the 5W+I header truthful AND append a dated WHY line to the file's
  CHANGE LOG block (ADR-0011). Keep the 'NEXT EDITOR' line. Run scripts/check_changelog.py.
- No hidden constants. New numbers go to configs/ with a comment.
- Telemetry: any new pipeline stage must emit a span (telemetry/tracer.py).
Roles (mission | may touch | must never touch):
- Architect: structure, ADRs | docs, architecture | trading logic
- Codex: regime docs | codex/ | code
- Data Pipeline: data_io, features | those + tests | rewards, Shell
- RL Training: training/ | training, experiments | Shell, masks
- Evaluation: evaluation/, reports | those | live bridge
- Execution Safety: execution_bridge | bridge + its tests | reward, policy
- Observability: telemetry, dashboards | those | trading logic
- Documentation: docs everywhere | *.md | code behavior
