#!/usr/bin/env python3
"""Score signal slots at horizons 5/10/20 (SIGON accuracy board).

Usage:
  python scripts/score_signal_accuracy.py
  python scripts/score_signal_accuracy.py --csv data/XAUUSD_curriculum_2026.csv
  python scripts/score_signal_accuracy.py --slots 80,81,82,83
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from data_io.loader import read_mt5_m1
from features.engine import build_features
from signals.encode import load_filled_slots, compute_slot
from training.signal_accuracy import score_slots_on_close, write_accuracy_report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/XAUUSD_curriculum_2026.csv")
    ap.add_argument("--horizons", default="5,10,20")
    ap.add_argument("--slots", default="", help="comma indices; empty = all enabled")
    ap.add_argument("--min-fires", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]

    print("Loading %s ..." % args.csv, flush=True)
    m1 = read_mt5_m1(args.csv, max_rows=args.max_rows)
    print("  bars=%d" % len(m1), flush=True)
    print("Building features ...", flush=True)
    # Ensure signal columns exist even if features.yaml flag is off
    from signals.encode import append_signal_obs
    F = build_features(m1)
    # If slots not in F, compute via encode
    filled = load_filled_slots(only_enabled=True)
    if args.slots.strip():
        want = {int(x) for x in args.slots.split(",") if x.strip()}
        filled = {k: v for k, v in filled.items() if k in want}

    print("  scoring %d slots @ %s" % (len(filled), horizons), flush=True)
    close = F["close"].to_numpy(dtype=np.float64)
    signals = {}
    for idx, spec in filled.items():
        try:
            signals[int(idx)] = compute_slot(F, spec).to_numpy(dtype=np.float32)
        except Exception as e:
            print("  skip slot %s: %s" % (idx, e), flush=True)

    report = score_slots_on_close(close, signals, horizons=horizons, min_fires=args.min_fires)
    report["csv"] = args.csv
    report["n_slots_scored"] = len(signals)
    path = write_accuracy_report(report, "latest.json")
    # Also write compact table for 80-83
    hi = [r for r in report["rows"] if r["slot"] in (80, 81, 82, 83)]
    if hi:
        write_accuracy_report({"updated_at": report["updated_at"], "rows": hi, "horizons": horizons},
                              "agree_80_83.json")

    print("\n=== SIGNAL ACCURACY (5/10/20) ===\n")
    for r in sorted(report["rows"], key=lambda x: (-(x.get("hit_10") or 0), -x["n_fires"])):
        if r["n_fires"] < args.min_fires:
            continue
        print(
            "slot %3d | n=%5d | hit5=%s hit10=%s hit20=%s"
            % (
                r["slot"],
                r["n_fires"],
                r.get("hit_5"),
                r.get("hit_10"),
                r.get("hit_20"),
            )
        )
    print("\nWrote %s" % path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
