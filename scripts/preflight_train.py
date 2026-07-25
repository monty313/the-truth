#!/usr/bin/env python3
"""Preflight before any training run — fail loud if the stack is not ready.

Usage: python scripts/preflight_train.py
Exit 0 = ready to train. Non-zero = fix listed items first.
"""
from __future__ import annotations
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

errors, warns = [], []

def ok(msg): print(f"  OK  {msg}")
def bad(msg): errors.append(msg); print(f" FAIL {msg}")
def warn(msg): warns.append(msg); print(f" WARN {msg}")

print("=== PREFLIGHT TRAIN ===\n[1] Core modules")
try:
    from training.meta_tuner import BOUNDS, run, adopt_gate, base_config
    if "w_pullback_with_htf" not in BOUNDS:
        bad("meta_tuner BOUNDS missing w_pullback_with_htf — run: python scripts/restore_meta_tuner.py")
    else:
        ok(f"meta_tuner (pullback bounds={BOUNDS['w_pullback_with_htf']})")
except Exception as e:
    bad(f"meta_tuner import: {e} — run: python scripts/restore_meta_tuner.py")

print("\n[2] Gravity SETS (Monty lock)")
try:
    from features.engine import SETS
    s2, s3 = SETS["set2"]["htfs"], SETS["set3"]["htfs"]
    if s2 != ["1h", "4h"] or s3 != ["4h", "1d"]:
        bad(f"SETS not aligned: set2={s2} set3={s3} — run: python scripts/align_tf_sets.py")
    else:
        ok(f"SETS A/B/C locked: set2={s2} set3={s3}")
except Exception as e:
    bad(f"features.engine: {e}")

print("\n[3] Rewards / goals")
try:
    from core.configs import load, goals_cfg
    rw, g = load("rewards"), goals_cfg()
    pb = float(rw.get("w_pullback_with_htf", 0))
    if pb < 0.1:
        warn(f"w_pullback_with_htf={pb} (IRAC recommended 0.25 for bread-and-butter)")
    else:
        ok(f"w_pullback_with_htf={pb}")
    ok(f"goal={g.get('goal_pct')}% floor={g.get('floor_pct')}%")
except Exception as e:
    bad(f"configs: {e}")

print("\n[4] Telemetry / regime language")
try:
    from telemetry.regime_language import document_decision, CODE_SETS
    from telemetry.mind_probe import probe_day
    from telemetry.ghost_trades import build_ghosts
    ok("regime_language + mind_probe + ghost_trades")
    if CODE_SETS.get("set2", {}).get("htfs") != ["1h", "4h"]:
        warn("regime_language CODE_SETS may lag engine SETS")
except Exception as e:
    bad(f"telemetry: {e}")

print("\n[5] Doctrine SSOT")
reg = os.path.join(ROOT, "doctrine", "LLM_REGIME_DEFINITIONS.yaml")
if os.path.isfile(reg):
    ok("doctrine/LLM_REGIME_DEFINITIONS.yaml")
else:
    bad("missing doctrine/LLM_REGIME_DEFINITIONS.yaml")

print("\n[6] Data + checkpoints")
cur = os.path.join(ROOT, "data", "XAUUSD_curriculum_2026.csv")
if os.path.isfile(cur):
    ok("data/XAUUSD_curriculum_2026.csv")
else:
    bad("missing curriculum CSV")
ckpt = os.path.join(ROOT, "artifacts", "checkpoints")
proven = os.path.join(ckpt, "PROVEN_SPRINT_row04_clear24_2026-07-20.pt")
if os.path.isfile(proven):
    ok("PROVEN_SPRINT checkpoint")
else:
    warn("PROVEN_SPRINT checkpoint missing — warm-start will fall back")

print("\n[7] Entry scripts")
for s in ("consistency_sprint.py", "meta_train.py", "prove_it.py",
          "give_llm_what_it_needs.py", "restore_meta_tuner.py"):
    path = os.path.join(ROOT, "scripts", s)
    if os.path.isfile(path):
        ok(f"scripts/{s}")
    else:
        bad(f"missing scripts/{s}")

print()
if errors:
    print(f"PREFLIGHT FAILED ({len(errors)} errors)")
    for e in errors:
        print("  -", e)
    sys.exit(1)
print("PREFLIGHT PASSED — ready to train")
if warns:
    print("Warnings:")
    for w in warns:
        print("  -", w)
print("\nNext:")
print("  python scripts/give_llm_what_it_needs.py PROVEN_SPRINT_row04_clear24_2026-07-20 3.0 3.5 --days 10")
print("  python scripts/consistency_sprint.py --minutes 600 --envs 256")
print("  python scripts/meta_train.py --minutes 600")
sys.exit(0)
