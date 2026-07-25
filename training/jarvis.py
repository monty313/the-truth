"""Jarvis mid-train channel — Iron Man style.
artifacts/jarvis/status.json, inbox.md, outbox.md
Commands: RELOAD_REWARDS | SET w_key=val | NOTE text
"""
from __future__ import annotations
import json, os, re
from datetime import datetime, timezone
from core.configs import path as rpath

def _dir():
    d = rpath("artifacts", "jarvis"); os.makedirs(d, exist_ok=True); return d

def write_status(**kwargs):
    path = os.path.join(_dir(), "status.json")
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    with open(path + ".tmp", "w", encoding="utf-8") as f: json.dump(payload, f, indent=2)
    os.replace(path + ".tmp", path); return path

def read_inbox():
    path = os.path.join(_dir(), "inbox.md")
    if not os.path.exists(path): return ""
    try: return open(path, encoding="utf-8").read()
    except Exception: return ""

def clear_inbox():
    open(os.path.join(_dir(), "inbox.md"), "w", encoding="utf-8").write("")

def write_outbox(text):
    with open(os.path.join(_dir(), "outbox.md"), "w", encoding="utf-8") as f:
        f.write("# Jarvis outbox\n\n" + text + "\n")

def apply_inbox_to_sim(sim):
    text = read_inbox().strip()
    if not text: return []
    logs = []
    for m in re.finditer(r"(?im)^\s*SET\s+([A-Za-z0-9_]+)\s*=\s*([-+0-9.eE]+)\s*$", text):
        key, val = m.group(1), float(m.group(2))
        if not hasattr(sim, "w"): continue
        sim.w[key] = val
        logs.append("SET %s=%s" % (key, val))
        try:
            import yaml
            rp = rpath("configs", "rewards.yaml")
            data = yaml.safe_load(open(rp, encoding="utf-8")) or {}
            data[key] = val
            with open(rp, "w", encoding="utf-8") as f: yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            logs.append("persisted %s" % key)
        except Exception as e:
            logs.append("persist skip: %s" % e)
    if re.search(r"(?im)^\s*RELOAD_REWARDS\s*$", text):
        if hasattr(sim, "reload_rewards"):
            w = sim.reload_rewards(); logs.append("RELOAD_REWARDS ok keys=%d" % len(w))
        else:
            logs.append("RELOAD_REWARDS unavailable")
    notes = [ln for ln in text.splitlines() if ln.strip().upper().startswith("NOTE ")]
    if notes: logs.append("notes: " + " | ".join(n[5:].strip() for n in notes)[:500])
    write_outbox("Applied at %s\n\n" % datetime.now(timezone.utc).isoformat() + "\n".join("- " + x for x in logs))
    clear_inbox(); return logs
