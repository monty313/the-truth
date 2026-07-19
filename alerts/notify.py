"""Alerts — phone pushes + the weekly retrain reminder (Monty's ritual).
5W+I: WHO Claude for Monty. WHAT pluggable push (Pushover/Telegram config in
configs/execution.yaml -> env vars), events: floor_hit, trading_stopped,
weekly_retrain_reminder, (offline_heartbeat pending Monty OK). WHEN 2026-07-19.
WHY hands-off operator must hear the two sounds that matter. INTERCONNECTED:
bridge, HUD server (scheduler thread), telemetry events.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
import json, os, time, urllib.parse, urllib.request
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOGP = os.path.join(ROOT, "logs", "alerts.jsonl")

def push(title: str, message: str) -> bool:
    """Pushover if configured; always journaled locally."""
    rec = {"ts": time.time(), "title": title, "message": message}
    with open(LOGP, "a") as f:
        f.write(json.dumps(rec) + "\n")
    tok, usr = os.environ.get("PUSHOVER_TOKEN"), os.environ.get("PUSHOVER_USER")
    if tok and usr:
        try:
            data = urllib.parse.urlencode({"token": tok, "user": usr,
                                           "title": title, "message": message}).encode()
            urllib.request.urlopen("https://api.pushover.net/1/messages.json",
                                   data=data, timeout=10)
            return True
        except Exception:
            return False
    return False

def weekly_retrain_reminder():
    push("Momentum One", "Weekly ritual: retrain the challengers. "
         "Champion stays frozen until beaten on consistency winning rate.")
