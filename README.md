# the-truth

## Project mission
Build and maintain the the truth project with reliable setup, execution, and delivery workflows.

## What this repository currently contains
Top-level structure:
- `.github` (dir)
- `.gitignore` (file)
- `.grok` (dir)
- `00_MAP_OF_THE_HOUSE.md` (file)
- `00_START_HERE.md` (file)
- `00_TEACH_META_AND_RL_EVERY_DAY.md` (file)
- `_archive` (dir)
- `AGENTS.md` (file)
- `CLAUDE.md` (file)
- `code` (dir)
- `configs` (dir)
- `data` (dir)
- `DO_THIS.md` (file)
- `docs` (dir)
- `FURTHEST_WEAVE__POLICY_EQUALS_MARK_ON_CHART.md` (file)
- `GOAL.md` (file)
- `HANDOFF_2026-07-31.md` (file)
- `HANDOFF_2026-08-05.md` (file)
- `HANDOFF_IRAC.md` (file)
- `kag_mark_doctrine` (dir)

Primary languages/tooling detected: Python (333), Markdown (289), YAML (18), Jupyter Notebook (3)

Likely key implementation files:
- `README.md`

## Quickstart (operator + LLM friendly)
Use this sequence first, then adapt to repo-specific details.

1. Clone and enter repo:
   ```bash
   git clone https://github.com/monty313/the-truth.git
   cd the-truth
   ```
2. Setup environment/dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   make setup  # if defined
   ```
3. Run locally:
   ```bash
   python main.py  # adjust to project entrypoint
   make run  # if defined
   ```
4. Validate changes:
   ```bash
   pytest -q
   make test  # if defined
   ```

## Operating intent (CEO-mode guide for future LLMs)
When continuing this project, optimize for business outcome first:

- Preserve or improve strategy quality, reliability, and reproducibility.
- Prefer small, testable, reversible changes over broad rewrites.
- Keep docs synchronized with code behavior.
- Add explicit assumptions, constraints, and success criteria in every major change.

## Delivery checklist
- [ ] Environment setup is deterministic.
- [ ] Primary run path is documented and works.
- [ ] Validation/test path is documented and works.
- [ ] Key risks and TODOs are tracked in the handoff file.

## Branch and release notes
- Default branch: `main`
- Keep PRs focused and attach proof-of-run or proof-of-test for behavior changes.
