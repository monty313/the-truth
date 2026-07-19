"""Start the JARVIS HUD. 5W+I: see dashboards/hud/server.py.
Usage: python scripts/run_hud.py  ->  http://localhost:8750 (+ phone on LAN)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboards.hud.server import app, _weekly_reminder_thread

if __name__ == "__main__":
    _weekly_reminder_thread()
    app.run(host="0.0.0.0", port=8750)
