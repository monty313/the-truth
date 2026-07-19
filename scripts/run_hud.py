"""Start the JARVIS HUD. 5W+I: see dashboards/hud/server.py.
Usage: python scripts/run_hud.py  ->  http://localhost:8750 (+ phone on LAN).
CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboards.hud.server import app, _weekly_reminder_thread

if __name__ == "__main__":
    _weekly_reminder_thread()
    app.run(host="0.0.0.0", port=8750)
