# MOMENTUM ONE — Meta-Learning RL Trading System
<!-- 5W+I ============================================================
WHO:   Built by Claude (Fable 5) for Monty (operator/owner).
WHAT:  Repo root of the Momentum One trading system: a goal-conditioned,
       meta-learning RL bot for MT5 (all symbols), governed by hard risk
       Shell rules and the Fable 5 observability-first doctrine.
WHEN:  Scaffolded 2026-07-18 (overnight autonomous build).
WHERE: Runs on Monty's Windows laptop (live) + cloud bursts (training).
WHY:   Hit the daily goal (+X%) without breaching the daily floor (-X%),
       on any symbol, one brain for life, always improving.
INTERCONNECTED WITH: every subfolder; governing docs in Claude project:
       momentum-one-handoff.md, momentum-one-plan.md, fable5-master-prompt.md.
==================================================================== -->

One meta-learning brain. Hits the goal you type, never breaches the floor
you type, on every symbol you own — built once, correctly, for life.

## Map
| Folder | Purpose |
|---|---|
| docs/adr | Architecture Decision Records (why everything is the way it is) |
| codex/regimes | The Codex: strategies, states, masks, matrix — machine-readable law |
| configs | EVERY number lives here, versioned. No hidden thresholds. |
| telemetry | Span tracer + logging standards (the Eyes) |
| experiments | Run cards + tracker (the Memory) |
| data_io | M1 CSV loading, resampling, calendar (CEST day) |
| features | MT5-exact indicators, 4-Set matrix, states, observation builder |
| backtesting | Paranoid-fill simulator + Shell + Feasibility Gauntlet |
| training | Gym env, rewards, recurrent PPO, meta-optimizer, trophy case |
| execution_bridge | MT5 live bridge (Windows), dry-run mode |
| dashboards/hud | JARVIS HUD server + page + kill switch |
| alerts | Push notifications + weekly retrain reminder |
| tests | pytest suite — golden, shell, no-look-ahead, reward |
| scripts | Entry points (run_gauntlet, train_bootcamp, run_live, run_hud) |

## Quick start (Monty's laptop)
```
pip install -r requirements.txt
python scripts/run_dummy_traced_run.py      # prove the eyes work
python scripts/run_gauntlet.py              # evidence before training
python scripts/train_bootcamp.py            # boot camp (approval-gated tuning)
python scripts/run_hud.py                   # JARVIS HUD on http://localhost:8750
```
Laws: see docs/LAWS.md. Nothing trades without the Shell. Ever.
