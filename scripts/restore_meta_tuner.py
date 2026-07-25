#!/usr/bin/env python3
"""Ensure training/meta_tuner.py exists on the host.

The self-tuner is part of the repo. This script only restores from the committed
file if someone deleted it locally. Target% and risk% are NEVER baked into the
network weights — they are runtime inputs (obs self-state + goals.yaml + CLI).

Usage (repo root):
  python scripts/restore_meta_tuner.py
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "training" / "meta_tuner.py"


def main() -> int:
    if OUT.is_file() and OUT.stat().st_size > 1000:
        text = OUT.read_text(encoding="utf-8")
        if "PLACEHOLDER" in text and len(text) < 100:
            print("PLACEHOLDER detected — refusing. Restore from git:", file=sys.stderr)
            print("  git checkout HEAD -- training/meta_tuner.py", file=sys.stderr)
            return 1
        print(f"OK: {OUT} present ({OUT.stat().st_size} bytes)")
        print("Target/floor remain runtime inputs (goals.yaml + prove_it CLI + obs self-state).")
        return 0
    print("MISSING training/meta_tuner.py — restore from git:", file=sys.stderr)
    print("  git checkout HEAD -- training/meta_tuner.py", file=sys.stderr)
    print("  # or: git pull origin main", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
