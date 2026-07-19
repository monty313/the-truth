"""Start the JARVIS HUD. 5W+I: see dashboards/hud/server.py.
Usage: python scripts/run_hud.py  ->  http://localhost:8750 (+ phone on LAN)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dashboards.hud.server import app
app.run(host="0.0.0.0", port=8750)
