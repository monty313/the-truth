#!/usr/bin/env python3
"""Align features/engine.py SETS to Monty lock: 1m/15m/30m; 5m/1h/4h; 15m/4h/1d.

Run: python scripts/align_tf_sets.py
SEMANTIC obs shift for set2/set3 — re-prove frozen brains after align + cache rebuild.
"""
from pathlib import Path

p = Path("features/engine.py")
t = p.read_text()
old = '''SETS = {  # ADR-0004 (extra-confidence TF listed last, weighted not gating)
    "set1": {"ltf": "1min", "htfs": ["15min", "30min"], "extra": "1h"},
    "set2": {"ltf": "5min", "htfs": ["30min", "1h"], "extra": "4h"},
    "set3": {"ltf": "15min", "htfs": ["1h", "4h"], "extra": "1d"},
    "set4": {"ltf": "30min", "htfs": ["4h", "1d"], "extra": "1w"},
}'''
new = '''SETS = {  # Monty lock 2026-07-24: A=1m/15m/30m B=5m/1h/4h C=15m/4h/1d
    # SEMANTIC obs shift for set2/set3 — re-prove frozen brains.
    "set1": {"ltf": "1min", "htfs": ["15min", "30min"], "extra": "1h"},
    "set2": {"ltf": "5min", "htfs": ["1h", "4h"], "extra": "1d"},
    "set3": {"ltf": "15min", "htfs": ["4h", "1d"], "extra": "1w"},
    "set4": {"ltf": "30min", "htfs": ["4h", "1d"], "extra": "1w"},
}'''
if '"htfs": ["30min", "1h"]' not in t and '"htfs": ["1h", "4h"]' in t:
    print("already aligned")
elif old not in t:
    raise SystemExit("SETS block shape changed — manual edit required")
else:
    p.write_text(t.replace(old, new, 1))
    print("aligned SETS in features/engine.py")
