"""Span tracer — the Eyes (Langfuse-spec-compatible local JSONL).

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0008, ADR-0010: local vendor by default).
WHAT:  Minimal span system: every pipeline stage emits a span with name,
       start/end, attributes, parent — appended to logs/spans.jsonl.
WHEN:  2026-07-19 overnight build.
WHERE: Imported by every module that does work (data, features, sim,
       training, bridge, HUD).
WHY:   Doctrine law: telemetry before strategy code; silent failure is
       the enemy. Vendor (Langfuse cloud) swappable behind this API.
INTERCONNECTED WITH: logs/spans.jsonl (out), telemetry/logging_setup.py,
       experiments/tracker.py (run_id joins spans to runs).
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  created/last-major  — WHY: v0.1 build + v0.2 audit fixes (see docs/AUDIT_FIXES_2026-07-19.md).
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import json, os, time, threading, uuid
from contextlib import contextmanager

_LOCK = threading.Lock()
_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(_DIR, exist_ok=True)
_PATH = os.path.join(_DIR, "spans.jsonl")
_CTX = threading.local()

# Doctrine span names (ADR-0008) — use these, don't invent drift:
STAGES = ("data_ingestion", "feature_generation", "state_classification",
          "mask_check", "policy_inference", "action_selection",
          "reward_computation", "order_submission", "fill_handling",
          "state_update", "checkpoint", "ppo_update", "error", "recovery")


def _write(rec: dict) -> None:
    with _LOCK:
        with open(_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def set_run(run_id: str) -> None:
    """Bind subsequent spans in this thread to an experiment run."""
    _CTX.run_id = run_id


@contextmanager
def span(name: str, **attrs):
    """Trace one unit of work. Usage: with span('mask_check', side='buy'): ...
    Names outside STAGES are recorded but tagged nonstandard=True so drift
    from the doctrine vocabulary is visible (audit R15)."""
    if name not in STAGES:
        attrs = {**attrs, "nonstandard": True}
    sid = uuid.uuid4().hex[:12]
    parent = getattr(_CTX, "stack", [])
    rec = {"span_id": sid, "name": name, "parent": parent[-1] if parent else None,
           "run_id": getattr(_CTX, "run_id", None), "t0": time.time(), "attrs": attrs}
    _CTX.stack = parent + [sid]
    try:
        yield rec
        rec["status"] = "ok"
    except Exception as e:  # error spans are mandatory, then re-raise
        rec["status"] = "error"
        rec["error"] = repr(e)
        raise
    finally:
        rec["t1"] = time.time()
        rec["ms"] = round((rec["t1"] - rec["t0"]) * 1000, 3)
        _CTX.stack = _CTX.stack[:-1]
        _write(rec)


def event(name: str, **attrs) -> None:
    """Zero-duration span for point events (mask flip, kill switch, alert)."""
    _write({"span_id": uuid.uuid4().hex[:12], "name": name, "parent": None,
            "run_id": getattr(_CTX, "run_id", None), "t0": time.time(),
            "t1": time.time(), "ms": 0.0, "status": "event", "attrs": attrs})
