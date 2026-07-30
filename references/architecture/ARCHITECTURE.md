# Momentum One — Architecture (the Nerves, mapped)
<!-- 5W+I: WHO Claude/Monty. WHAT the module map + data flow, source of truth
for how the pieces connect. WHEN 2026-07-19. WHERE repo-wide reference.
WHY the Fable 5 doctrine: an LLM must reason over the whole system later.
INTERCONNECTED WITH: every module; docs/CODEBASE_MAP.md is the file-level index. -->

## Data flow (one cycle)
```
data_io/loader.py   read_mt5_m1 / synthetic_m1  -> M1 DataFrame (day-clock converted)
        |                                            trading_days() -> day episodes
        v
features/engine.py  build_features()  -> 4-Set signals, states, FOREVER MASKS, obs cols
        |            (features/indicators.py = MT5-exact math)
        v
training/env.py     TradingEnv.step(op,size)  -- wraps -->  backtesting/simulator.py DaySim
        |                                                     (THE SHELL: masks, caps, heat,
        |                                                      400->close-only, ratchet, floor,
        v                                                      paranoid fills — one physics)
training/policy.py  Brain (GRU + op head + Beta size head + value)
        |
training/ppo.py     PPO.update()  (GAE, clipped, joint op+size log-prob)
        |            training/rewards.py  RewardEngine (closed-only pay, ADR-0005)
        v
artifacts/checkpoints/*.pt  --load--> inference/loader.py --> execution_bridge/mt5_bridge.py
                                                               (dry-run/live, SAME DaySim physics)
                                                               -> artifacts/hud_state.json
                                                               -> dashboards/hud (JARVIS HUD)
```

## The four pillars (Fable 5 living machine)
- **Brain**: training/policy.py, training/ppo.py, training/rewards.py, training/meta_optimizer.py
- **Nerves**: data_io/, features/, backtesting/simulator.py (Shell), execution_bridge/
- **Memory**: experiments/tracker.py (run cards), artifacts/, configs/, codex/regimes/, docs/adr/
- **Eyes**: telemetry/tracer.py (spans), logs/, dashboards/hud/, evaluation/champion.py

## The one-physics guarantee
Sim, training, and live all step the SAME `backtesting.simulator.DaySim`. The Shell
(masks incl. on adds/probes, 0.25% cap, heat guard, 400->close-only, midnight flat,
ratchet, floor, kill switch) is therefore identical everywhere — the brain can never
learn or execute an illegal move. See ADR-0002/0003.

## Config is the single source of numbers
Every threshold lives in configs/*.yaml, loaded ONLY through core/configs.py. No module
hardcodes a ruled number (LAWS #3). Changing a config changes the machine.
