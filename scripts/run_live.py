"""Run the bridge loop (dry-run by default). 5W+I: see execution_bridge/.
WINDOWS + MT5 required for demo/live; dry-run works anywhere with data.
"""
import sys, os, time, yaml
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from execution_bridge.mt5_bridge import Bridge
from telemetry.logging_setup import setup
log = setup("live")
cfg = yaml.safe_load(open("configs/execution.yaml"))
b = Bridge(cfg)
if not b.connect():
    raise SystemExit("MT5 connect failed — check configs/execution.yaml")
b.resume()
log.info("bridge up: mode=%s (frozen champion not attached yet — Phase 7)", b.mode)
# heartbeat loop skeleton: wire feature stream + champion at Phase 7 gate.
