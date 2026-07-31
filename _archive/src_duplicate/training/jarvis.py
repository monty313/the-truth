"""Jarvis mid-train channel — Iron Man style.

Files (no restart needed):
  artifacts/jarvis/status.json   <- train writes live state for Jarvis to read
  artifacts/jarvis/inbox.md      <- you/LLM write commands/notes; train reads each N updates
  artifacts/jarvis/outbox.md     <- train acknowledges applied changes
  doctrine/cmo_inbox/*.md        <- standing think notes (also scanned)

Inbox commands (one per line, optional):
  RELOAD_REWARDS
  SET w_pullback_with_htf=0.35
  NOTE your free text for the CMO log

CHANGE LOG:
- 2026-07-25  created — WHY: Monty wants Jarvis talk + hot updates while Colab trains.
"""
from __future__ import annotations
import json
import os
import re
from datetime import datetime, timezone

from core.configs import path as rpath, load as load_cfg


def _dir():
    d = rpath("artifacts", "jarvis")
    os.makedirs(d, exist_ok=True)
    return d


def write_status(**kwargs):
    d = _dir()
    path = os.path.join(d, "status.json")
    payload = {"updated_at": datetime.now(timezone.utc).isoformat(), **kwargs}
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp, path)
    return path


def read_inbox() -> str:
    path = os.path.join(_dir(), "inbox.md")
    if not os.path.exists(path):
        return ""
    try:
        return open(path, encoding="utf-8").read()
    except Exception:
        return ""


def clear_inbox():
    path = os.path.join(_dir(), "inbox.md")
    open(path, "w", encoding="utf-8").write("")


def write_outbox(text: str):
    path = os.path.join(_dir(), "outbox.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Jarvis outbox\n\n")
        f.write(text)
        f.write("\n")


def apply_inbox_to_sim(sim) -> list[str]:
    """Parse inbox, hot-reload rewards / set weights. Returns log lines."""
    text = read_inbox().strip()
    if not text:
        return []
    logs = []
    # SET key=value
    for m in re.finditer(r"(?im)^\s*SET\s+([A-Za-z0-9_]+)\s*=\s*([-+0-9.eE]+)\s*$", text):
        key, val = m.group(1), float(m.group(2))
        if not hasattr(sim, "w"):
            continue
        if key not in sim.w and not key.startswith("w_"):
            logs.append("unknown key %s" % key)
            continue
        sim.w[key] = val
        logs.append("SET %s=%s" % (key, val))
        # also persist to rewards.yaml if key exists there
        try:
            import yaml
            rp = rpath("configs", "rewards.yaml")
            data = yaml.safe_load(open(rp, encoding="utf-8")) or {}
            data[key] = val
            with open(rp, "w", encoding="utf-8") as f:
                yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
            logs.append("persisted %s to rewards.yaml" % key)
        except Exception as e:
            logs.append("persist skip: %s" % e)

    if re.search(r"(?im)^\s*RELOAD_REWARDS\s*$", text):
        if hasattr(sim, "reload_rewards"):
            w = sim.reload_rewards()
            logs.append("RELOAD_REWARDS ok keys=%d" % len(w))
        else:
            logs.append("RELOAD_REWARDS unavailable")

    notes = [ln for ln in text.splitlines() if ln.strip().upper().startswith("NOTE ")]
    if notes:
        logs.append("notes: " + " | ".join(n[5:].strip() for n in notes)[:500])

    write_outbox(
        "Applied at %s\n\n" % datetime.now(timezone.utc).isoformat()
        + "\n".join("- " + x for x in logs)
        + "\n\nInbox cleared after apply.\n"
    )
    clear_inbox()
    return logs
