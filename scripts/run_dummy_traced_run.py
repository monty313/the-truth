"""Phase-2 proof: the eyes and memory work before any strategy code.
5W+I: WHO Claude/ADR-0008. WHAT dummy end-to-end traced run producing a full
run card + spans for every doctrine stage. WHEN 2026-07-19. WHY doctrine gate.
INTERCONNECTED: telemetry/tracer, experiments/tracker.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from telemetry import tracer
from telemetry.logging_setup import setup
from experiments.tracker import Run

def main():
    log = setup("dummy_run")
    run = Run("dummy_proof", symbols=["SYNTH"], timeframes=["1min"],
              data_window="none", seed=1,
              assumptions={"fills": "n/a", "broker": "n/a"})
    with tracer.span("data_ingestion", rows=100):
        pass
    with tracer.span("feature_generation", features=20):
        with tracer.span("state_classification", sets=4):
            pass
    with tracer.span("mask_check", side="buy", blocked=False):
        pass
    with tracer.span("policy_inference"):
        action = random.choice(["hold", "buy"])
    with tracer.span("action_selection", action=action):
        pass
    with tracer.span("reward_computation", reward=0.0):
        pass
    with tracer.span("checkpoint", path="none"):
        pass
    run.log(steps=1, ok=1)
    card = run.finish("dummy proof: spans + run card emitted", model_version=None)
    log.info("run card at %s", card)
    print("PHASE2_PROOF_OK", card)


if __name__ == "__main__":
    main()
