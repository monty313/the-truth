"""Signal-agent scoreboard for BB/RSI-BB/SMA sets.

Question: after a long/short signal, how far does price go (path), not full P&L.

For each set + symbol:
  - forward return at H bars (LTF bars): 1, 2, 3, 5, 8, 10, 15, 20
  - directional hit rate (long: close[t+H] > entry; short: opposite)
  - mean / median forward return (signed with trade direction)
  - mean MFE / MAE over horizon window (max favorable / adverse excursion %)
  - fire count

Usage:
  python scripts/score_bb_rsi_sma_signal.py --symbol EURUSD,US30
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ROOT)

from features.bb_rsi_sma_sets import SETS, build_set_frame  # noqa: E402

import importlib.util

_p = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("bt", _p)
_bt = importlib.util.module_from_spec(_spec)
sys.modules["bt"] = _bt
_spec.loader.exec_module(_bt)
load_m1_fast = _bt.load_m1_fast
resolve_csv = _bt.resolve_csv

HORIZONS = (1, 2, 3, 5, 8, 10, 15, 20)


def score_signals(df: pd.DataFrame, horizons=HORIZONS) -> list[dict]:
    """Entry at next bar open (honest); path from that fill through H LTF bars."""
    c = df["close"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    long_sig = df["long_entry"].fillna(False).to_numpy(bool)
    short_sig = df["short_entry"].fillna(False).to_numpy(bool)
    n = len(df)
    rows = []

    for side_name, sig, side in (("long", long_sig, 1), ("short", short_sig, -1), ("any", None, 0)):
        if side_name == "any":
            idxs = np.where(long_sig | short_sig)[0]
        else:
            idxs = np.where(sig)[0]
        if len(idxs) == 0:
            continue

        for H in horizons:
            hits = []
            rets = []
            mfes = []
            maes = []
            for i in idxs:
                # signal at bar i close -> fill at i+1 open
                j = i + 1
                end = j + H
                if end >= n or j >= n:
                    continue
                s = side if side != 0 else (1 if long_sig[i] else -1)
                entry = o[j]
                if not np.isfinite(entry) or entry == 0:
                    continue
                # path bars j .. j+H-1 inclusive for excursion; return uses close at j+H-1
                # horizon H = H bars later close relative to entry
                exit_px = c[j + H - 1] if (j + H - 1) < n else np.nan
                if not np.isfinite(exit_px):
                    continue
                ret = s * (exit_px - entry) / entry * 100.0
                rets.append(ret)
                hits.append(1.0 if ret > 0 else 0.0)

                # MFE/MAE over bars from fill through horizon
                hi = np.nanmax(h[j:j + H])
                lo = np.nanmin(l[j:j + H])
                if s > 0:
                    mfe = (hi - entry) / entry * 100.0
                    mae = (entry - lo) / entry * 100.0
                else:
                    mfe = (entry - lo) / entry * 100.0
                    mae = (hi - entry) / entry * 100.0
                mfes.append(mfe)
                maes.append(mae)

            if not rets:
                continue
            rets = np.array(rets)
            hits = np.array(hits)
            mfes = np.array(mfes)
            maes = np.array(maes)
            rows.append(dict(
                side=side_name,
                horizon=H,
                n=len(rets),
                hit_rate=100.0 * hits.mean(),
                mean_ret=float(rets.mean()),
                median_ret=float(np.median(rets)),
                mean_mfe=float(mfes.mean()),
                mean_mae=float(maes.mean()),
                mfe_mae_ratio=float(mfes.mean() / maes.mean()) if maes.mean() > 1e-12 else np.nan,
                p25=float(np.percentile(rets, 25)),
                p75=float(np.percentile(rets, 75)),
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD,US30")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "bb_rsi_sma"))
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    all_rows = []

    print("=" * 72)
    print("BB/RSI-BB/SMA — SIGNAL PATH SCOREBOARD (how far after signal)")
    print("Fill: next bar open | horizons in LTF bars | MFE/MAE on path")
    print("=" * 72)

    for sym in [s.strip().upper() for s in args.symbol.split(",") if s.strip()]:
        path = resolve_csv(args.data_dir, sym)
        if not path:
            print(f"[{sym}] no data")
            continue
        print(f"\n### {sym} loading ...", flush=True)
        m1 = load_m1_fast(path, max_rows=args.max_rows)
        print(f"    M1={len(m1):,}", flush=True)

        for set_name, spec in SETS.items():
            print(f"  {set_name} (LTF={spec['ltf']}) ...", flush=True)
            fr = build_set_frame(m1, set_name)
            fr = fr.dropna(subset=["rsi", "sma_low"])
            n_l = int(fr["long_entry"].sum())
            n_s = int(fr["short_entry"].sum())
            print(f"    signals long={n_l} short={n_s}", flush=True)
            for r in score_signals(fr):
                r["symbol"] = sym
                r["set"] = set_name
                r["ltf"] = spec["ltf"]
                all_rows.append(r)

            # print compact table for any-side at key horizons
            sub = [x for x in all_rows if x["symbol"] == sym and x["set"] == set_name and x["side"] == "any"]
            if sub:
                print(f"    {'H':>3} {'n':>6} {'hit%':>7} {'mean%':>8} {'med%':>8} {'MFE%':>7} {'MAE%':>7} {'MFE/MAE':>8}")
                for x in sub:
                    if x["horizon"] in (1, 3, 5, 10, 20):
                        print(
                            f"    {x['horizon']:3d} {x['n']:6d} {x['hit_rate']:6.1f}% "
                            f"{x['mean_ret']:+7.3f}% {x['median_ret']:+7.3f}% "
                            f"{x['mean_mfe']:6.3f}% {x['mean_mae']:6.3f}% "
                            f"{x['mfe_mae_ratio']:7.2f}"
                        )

    df = pd.DataFrame(all_rows)
    out = os.path.join(args.out_dir, "signal_path_scoreboard.csv")
    df.to_csv(out, index=False)

    # highlight best cells
    print("\n" + "=" * 72)
    print("BEST CELLS (any side, hit rate @ H=10, n>=50)")
    print("=" * 72)
    g = df[(df.side == "any") & (df.horizon == 10) & (df.n >= 50)].sort_values("hit_rate", ascending=False)
    if len(g):
        print(g[["symbol", "set", "n", "hit_rate", "mean_ret", "median_ret", "mean_mfe", "mean_mae", "mfe_mae_ratio"]]
              .to_string(index=False))
    print(f"\nFull -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
