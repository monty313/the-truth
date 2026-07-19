# notebooks/ — research & inspection space
<!-- 5W+I: WHO Monty/Claude. WHAT ad-hoc Jupyter analysis of run cards, spans,
trophy case, feature distributions. WHEN as needed. WHERE local only (git-
ignored outputs). WHY exploration precedes ADRs. INTERCONNECTED: artifacts/,
experiments/tracker run cards, logs/spans.jsonl. -->
Load a run card: `json.load(open('artifacts/runs/<id>/run_card.json'))`.
Load spans:      `[json.loads(l) for l in open('logs/spans.jsonl')]`.
Nothing here is on the critical path; it is a workbench.
