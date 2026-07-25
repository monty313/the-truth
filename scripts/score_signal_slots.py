#!/usr/bin/env python3
"""Score signal slots: when slot fires +1/-1, does forward price agree?

Usage:
  python scripts/score_signal_slots.py
  python scripts/score_signal_slots.py --csv data/XAUUSD_curriculum_2026.csv --horizons 15,60,240
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_io.loader import read_mt5_m1
from features.engine import build_features
from signals.encode import load_filled_slots, compute_slot


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data/XAUUSD_M1_drill.csv")
    ap.add_argument("--horizons", default="15,60,240")
    ap.add_argument("--min-fires", type=int, default=10)
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    horizons = [int(x) for x in args.horizons.split(",") if x.strip()]
    h = horizons[min(1, len(horizons) - 1)]

    print(f"Loading {args.csv} ...", flush=True)
    m1 = read_mt5_m1(args.csv, max_rows=args.max_rows)
    print(f"  bars={len(m1)}", flush=True)
    print("Building features ...", flush=True)
    F = build_features(m1)
    close = F["close"].to_numpy(dtype=np.float64)
    filled = load_filled_slots(only_enabled=True)
    print(f"  enabled={len(filled)}  horizon={h} bars", flush=True)

    rows = []
    for idx, spec in sorted(filled.items()):
        sig = compute_slot(F, spec).to_numpy(dtype=np.float32)
        fires = np.where(sig != 0)[0]
        n = len(fires)
        if n < args.min_fires:
            rows.append(dict(slot=idx, name=spec.get("name", "?"), family=spec.get("family", ""),
                             n_fires=n, hit_rate=np.nan, avg_bps=np.nan, score=np.nan))
            continue
        hits, rets = [], []
        for i in fires:
            j = i + h
            if j >= len(close) or not np.isfinite(close[i]) or close[i] == 0:
                continue
            r = (close[j] - close[i]) / close[i]
            hits.append(1.0 if sig[i] * r > 0 else 0.0)
            rets.append(sig[i] * r)
        if not hits:
            rows.append(dict(slot=idx, name=spec.get("name", "?"), family=spec.get("family", ""),
                             n_fires=n, hit_rate=np.nan, avg_bps=np.nan, score=np.nan))
            continue
        hit = float(np.mean(hits))
        avg_bps = float(np.mean(rets) * 1e4)
        score = (hit - 0.5) * 2.0 * np.log10(max(n, 10)) + avg_bps / 100.0
        rows.append(dict(slot=idx, name=spec.get("name", "?"), family=spec.get("family", ""),
                         n_fires=n, hit_rate=round(hit, 4), avg_bps=round(avg_bps, 2),
                         score=round(float(score), 3)))

    df = pd.DataFrame(rows)
    ranked = df.dropna(subset=["score"]).sort_values("score", ascending=False)
    print("\n=== SIGNAL SCOREBOARD ===\n")
    cols = ["slot", "name", "family", "n_fires", "hit_rate", "avg_bps", "score"]
    print(ranked[cols].to_string(index=False))
    out = ROOT / "artifacts" / "signal_scoreboard.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out, index=False)
    print(f"\nWrote {out}")
    print(f"hit_rate>52%: {int((ranked.hit_rate > 0.52).sum())} / {len(ranked)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
