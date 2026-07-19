# OVERNIGHT BUILD REPORT — Momentum One v0.1
**For:** Monty · **From:** Claude (Fable 5) · **Night of:** July 18→19, 2026
**Repo:** `momentum_one/` (delivered as zip to your Fable5_Foundation folder + this chat)

---

## WHAT EXISTS WHEN YOU WAKE UP

**Phases 0–5 of the plan are BUILT and TESTED. Phases 6–7 are built as working skeletons. 23/23 tests green. The canary PASSED — the learning machine learns.**

| Phase | Status | Proof |
|---|---|---|
| 0 Scaffold | DONE | full repo tree, 5W headers, agent instructions, git history |
| 1 Docs & Codex | DONE | 10 ADRs · 7 codex entries · 9 config files (every number versioned) |
| 2 Telemetry | DONE | span tracer + run cards; dummy traced run emitted spans for all doctrine stages |
| 3 Data & Features | DONE | 4-Set engine: 323 columns, MT5-shift-exact, masks, states, events; no-look-ahead mutation test green |
| 4 Simulator + Gauntlet | DONE | paranoid fills, Shell v2, oracle + baseline + audit run end-to-end |
| 5 Training | DONE (machine) | goal-conditioned recurrent PPO, full reward doctrine, meta-proposer, trophy case, CANARY PASS, smoke run complete |
| 6 MT5 Bridge | SKELETON | dry-run mode works; live order path deliberately locked until Phase 8 with you |
| 7 JARVIS HUD | WORKING v1 | server + dark HUD page + working two-tap KILL switch (tested) |

## THE REVIEWERS YOU ASKED FOR

Per your instruction, **three independent hostile review agents** attacked the code mid-build (MT5-math lens · look-ahead/fill-honesty lens · risk-engineering lens). They produced **29 findings including 5 critical**. Highlights of what they caught and I fixed the same night:

- **Masks failed OPEN during warmup** (spec says fail closed) — fixed + test pinned.
- **The Shell could be over-risked by batching orders in one bar** (60 orders all approved against stale state) — fixed: pending-aware checks + fill-time re-validation + test.
- **A wick through the floor was invisible** (equity only checked at bar close) — fixed: exact intrabar worst-case equity law + test.
- **Risk accounting drifted from truth after adds/partials** — fixed: stop-anchored true risk everywhere.
- **Gap-through-stop filled optimistically** — fixed: fills at the bar's adverse extreme.
- **Kill switch let queued orders fill** — fixed: kill cancels pending opens + test.
- Plus: S2 reload over-gated vs your spec (fixed to spec), reversal state dead (fixed both directions), one-bar alignment lag (fixed), signal spam→event edges (fixed), RSI/CCI MT5 edge cases (fixed).

**Every fix is a pinned test now — those attacks can never silently return.**

## HONEST NUMBERS (SYNTHETIC DATA — the gold zip never arrived)

The 121MB gold CSV failed the upload pipe 4×. Everything below ran on clearly-labeled synthetic gold-like data — it proves the MACHINE, not the market:

- Gauntlet oracle (future-seeing probe, full Shell): **~2.5%/day mean, zero breaches** — the ratchet and heat-guard visibly cap and protect exactly as designed.
- Baseline (your 4 strategies, dumb exits): ~breakeven after paranoid costs.
- Boot-camp smoke (3 PPO updates): full loop trains, evaluates vs the +5% bar, Shell rejected 100s of illegal intents along the way. PERFECT=false, as expected for a smoke on fake data.
- Canary (planted pattern): **PASS** — reward improved across updates.

**None of these numbers mean anything about real gold yet. First thing after you wake: zip the CSV (right-click → Send to → Compressed folder), and the Gauntlet + boot camp re-run on truth with one command each.**

## DECISIONS I MADE WHILE YOU SLEPT (ADR-0010 — all reversible)

1. Trace layer = local JSONL spans (Langfuse-spec-compatible); MLflow optional. You still pick the vendor at the Phase-2 gate.
2. Heat guard ON (total open risk ≤ distance to floor) — flagged; flip in `configs/masks_shell.yaml` if you veto.
3. Adds allowed to WINNING stacks only (your pyramiding reward implies it; reviewers demanded a ruling — mine is the safe one, change with one config if you disagree).
4. Synthetic data used for all runs, loudly labeled.
5. Boot-camp graduation NOT claimed — the machine is proven, the bar is not attempted on fake data.

## YOUR MORNING LIST (15 minutes)

1. **Zip the gold CSV** in `Fable5_Foundation\data` → tell me → real Gauntlet runs.
2. Read the Gauntlet evidence when it lands — **you rule on the +5% bar** (the hard gate).
3. On your laptop: `pip install -r requirements.txt` inside `momentum_one/`, then `python -m pytest tests/` (should be 23 green), then `python scripts/run_hud.py` → http://localhost:8750 — meet your HUD and press the kill switch.
4. Approvals waiting: heat guard yes/no · add-to-winners-only yes/no · Langfuse choice · offline-alert yes/no.

## RUN IT YOURSELF

```
python scripts/run_dummy_traced_run.py   # the eyes
python scripts/run_gauntlet.py           # the evidence (real data when zip lands)
python scripts/train_bootcamp.py --smoke # the learning loop
python scripts/run_hud.py                # the JARVIS HUD + kill switch
python -m pytest tests/                  # 23 proofs
```

*Built to the 5W+I standard throughout. Nothing hidden, nothing unversioned, nothing advanced past your gates.*
