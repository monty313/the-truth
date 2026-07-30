# MAP

**Rule:** if it is not in the “Daily” table, you usually ignore it.

---

## Daily (notice these)

| Name | What |
|------|------|
| **00_START_HERE.md** | First open |
| **GOAL.md** | Mission |
| **DO_THIS.md** | Commands |
| **USE/** | One-click buttons (`1_` `2_` `3_`…) |
| **models/00_CHAMPION.md** | Active brain |
| **scripts/00_DAILY.md** | Daily scripts only |
| **configs/** | Numbers (goals, rewards, masks) |
| **AGENTS.md** | AI rules + **lid-off jar mind** |

---

## Whole tree (simple)

```text
00_START_HERE.md     ← start
GOAL.md              ← mission
DO_THIS.md           ← commands
MAP.md               ← this file
AGENTS.md            ← AI organization + lid-off mind
USE/                 ← your buttons (1_ 2_ 3_)

configs/             numbers
data/raw/            price CSVs
models/              brains  (open 00_CHAMPION.md first)
scripts/             commands (open 00_DAILY.md first)

training/            learning code
features/            indicators
signals/             signal agents
inference/           load brain
execution_bridge/    MT5
telemetry/           mind probe
backtesting/         day sim
evaluation/          consistency helpers
core/                config door
tests/               pytest

references/          long reads (not daily)
  doctrine/          laws + SUCCESS_LEDGER
    00_LID_OFF_THE_JAR.md  ← permanent “lid is off” law
  handoffs/          session notes
  performance/       PERFORMANCE series (parts 1–4)
  plans/             training plans

reports/figures/     keep charts
artifacts/           temp caches (safe delete *.npz)
_archive/            old duplicates (do not use)
docs/                ADRs
.vscode/             editor tasks
```

---

## Paths you type often

| Thing | Path |
|-------|------|
| Champion | `models/PROVEN_SPRINT_row04_clear24_2026-07-20.pt` |
| Gold CSV | `data/raw/XAUUSD_curriculum_2026.csv` |
| Goals | `configs/goals.yaml` |
| Masks | `configs/masks_shell.yaml` |
| Rewards | `configs/rewards.yaml` |

---

## Naming so things pop

| Prefix | Meaning |
|--------|---------|
| `00_` | Look first |
| `1_` `2_` `3_` | Daily order in USE |
| `PROVEN_` | Proof brain — do not delete |
| `_archive` | Old junk — ignore |
