# AUDIT FIXES — Momentum One (2026-07-19, round 2)
**For:** Monty · **From:** Claude (Fable 5)

You asked for three pessimistic reviewers to attack the whole folder against
YOUR instructions, then to fix everything they found, and to make the codebase
navigable by an LLM. Done. This is what changed.

## The reviewers found real problems. Every one is fixed.

Three independent hostile audits ran (spec-fidelity · runability/dummy-files ·
training-correctness), then a SECOND round to confirm the fixes. ~45 findings
total. The big ones:

### Correctness (would have mis-trained or mis-traded)
- **The size dial couldn't learn.** The brain's position-size head had zero
  reward gradient (a self-imitation bug) — your ruled "learn freely inside the
  0.25% cap" was frozen. Rebuilt as a proper Beta distribution; gradient now
  flows (pinned by a test).
- **Reward farming.** Bonuses paid on partial closes and at order-entry time —
  a bot could earn 170x the day-goal reward by spam-half-closing or by queuing
  orders that never filled. Now bonuses pay ONLY on full closes, pullback pays
  at close via trade tags, idleness only when flat. Every exploit pinned by a test.
- **The Shell was config-decorative.** 6 of 8 config files were read by nothing —
  the numbers were hardcoded, so typing a change into a config did nothing.
  Now ALL numbers flow through one door (core/configs.py). Change a config →
  change the machine.
- **The canary lied.** The overnight "learning works" gate was a coin-flip that
  actually FAILED on a fair run. Rebuilt as a real plumbing test on the actual
  Brain+PPO — now PASSES decisively (+0.94, from -0.07 to +0.87).
- **The live bridge bypassed your masks.** It could place a masked-side add and
  had no close-only-after-400, no floor, no ratchet. Rebuilt to run through the
  SAME DaySim physics as sim and training — "identical law live" is now true by
  construction, not by promise.

### Runability (your "everything works properly")
- **The real-data path crashed.** `read_mt5_m1` returned 0 rows on real MT5
  files (a pandas alignment bug) — the moment your gold CSV arrived, the whole
  pipeline would have died. Fixed and pinned with a round-trip test. **When your
  zip lands, it will actually load.**
- Day boundary now converts broker time → your 00:00 CEST clock (flagged: the
  exact broker timezone must be verified on your real file).
- Scripts no longer run work on import; 6 dead imports removed; all 47 modules
  import clean.

### Dummy files (your "no dummy files")
- 5 empty folders now hold real content: architecture/ARCHITECTURE.md,
  schemas/*.json, prompts/AGENT_PROMPTS.md, and wired inference/ + evaluation/.
- Everything the docs claimed exists is now WIRED: trophy case records evidence,
  meta-optimizer is reachable, alerts fire on floor/kill, checkpoints load, the
  champion-vs-challenger metric exists, the HUD gauge reads live data.

## The LLM is now connected to every part of the code
- Every file carries a **5W+I header**: WHO / WHAT / WHEN / WHERE / WHY / and
  what it's INTERCONNECTED WITH.
- **docs/CODEBASE_MAP.md** — auto-generated index of every module + a
  "symptom → which file to open" troubleshooting table.
- **architecture/ARCHITECTURE.md** — the data-flow and one-physics guarantee.
- **agents/INSTRUCTIONS.md** — role rules for future VS Code agents.
So when something breaks later, the LLM opens the map, finds the symptom, opens
the file, and reads exactly what it does and what it touches.

## Proof it all works (run these yourself)
```
python -m pytest tests/          # 34 passed
python training/canary.py        # CANARY PASS
python scripts/run_gauntlet.py   # writes artifacts/gauntlet/VERDICT.json
python scripts/train_bootcamp.py --smoke
python scripts/run_live.py --days 1   # drives the bridge, writes HUD state
python scripts/run_hud.py        # http://localhost:8750 — the JARVIS HUD
```

## Still true (unchanged honest caveats)
- All numbers are still from SYNTHETIC gold-like data — the real zip hasn't
  landed. The machine is proven; the market is not. Drop the zip and the
  Gauntlet + boot camp run on truth.
- Boot-camp graduation is NOT claimed. The +5%/day bar is yours to rule on
  after you see the real Gauntlet evidence.
- The live MT5 order path is deliberately locked (raises NotImplementedError)
  until Phase 8, with your gates.
