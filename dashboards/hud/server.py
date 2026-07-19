"""JARVIS HUD server — Flask, serves the page + state + kill switch.
5W+I: WHO Claude for Monty. WHAT / -> HUD page; /state -> bridge's
hud_state.json + latest run card + trophy ladder; POST /kill -> touch the
KILL file (freeze + close all); POST /unkill -> remove it. WHEN 2026-07-19.
WHERE laptop, http://localhost:8750 (+ phone on same network). WHY the
command center + the one red button. INTERCONNECTED: bridge (state file),
artifacts/runs, trophy_case.jsonl, alerts.
"""
import glob, json, os
from flask import Flask, jsonify, send_from_directory
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ART = os.path.join(ROOT, "artifacts")
app = Flask(__name__, static_folder=os.path.dirname(os.path.abspath(__file__)))

@app.get("/")
def index():
    return send_from_directory(app.static_folder, "hud.html")

@app.get("/state")
def state():
    out = {"bridge": None, "latest_run": None, "trophies": [], "killed":
           os.path.exists(os.path.join(ART, "KILL"))}
    p = os.path.join(ART, "hud_state.json")
    if os.path.exists(p):
        out["bridge"] = json.load(open(p))
    cards = sorted(glob.glob(os.path.join(ART, "runs", "*", "run_card.json")))
    if cards:
        c = json.load(open(cards[-1]))
        out["latest_run"] = {"run_id": c["run_id"], "kind": c["kind"],
                             "summary": c["summary"], "metrics_tail":
                             dict(list(c["metrics"].items())[-6:])}
    tp = os.path.join(ART, "trophy_case.jsonl")
    if os.path.exists(tp):
        out["trophies"] = [json.loads(l) for l in open(tp).readlines()[-5:]]
    return jsonify(out)

@app.post("/kill")
def kill():
    open(os.path.join(ART, "KILL"), "w").write("operator")
    return jsonify({"killed": True})

@app.post("/unkill")
def unkill():
    p = os.path.join(ART, "KILL")
    os.path.exists(p) and os.remove(p)
    return jsonify({"killed": False})

def _weekly_reminder_thread():
    """Fire the weekly retrain reminder (audit R9: alerts were disconnected)."""
    import threading, time as _t
    from alerts import notify
    def loop():
        while True:
            _t.sleep(7 * 24 * 3600)
            notify.weekly_retrain_reminder()
    threading.Thread(target=loop, daemon=True).start()


if __name__ == "__main__":
    _weekly_reminder_thread()
    app.run(host="0.0.0.0", port=8750)
