"""Logging standards — module loggers with uniform format.
5W+I: WHO Claude/ADR-0008. WHAT stdlib logging config (console + logs/app.log,
INFO default, DEBUG via MOMENTUM_DEBUG=1). WHEN 2026-07-19. WHERE imported by
scripts. WHY uniform, greppable logs. INTERCONNECTED: logs/app.log, tracer.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import logging, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def setup(name="momentum_one"):
    lvl = logging.DEBUG if os.environ.get("MOMENTUM_DEBUG") else logging.INFO
    logging.basicConfig(
        level=lvl,
        format="%(asctime)s %(levelname)-7s %(name)s :: %(message)s",
        handlers=[logging.StreamHandler(),
                  logging.FileHandler(os.path.join(ROOT, "logs", "app.log"))])
    return logging.getLogger(name)
