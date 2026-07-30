"""First-touch race after a signal (agent path quality).

For each long/short signal (next-bar open fill), walk forward bar-by-bar:
  Does price touch +T (favorable) BEFORE -T (adverse)?

Thresholds T in {5, 10, 20} measured in symbol pips/points
  (EURUSD: 0.0001; US30: 1.0 point).

Also reports max look-ahead bars used and median bars-to-touch.

Usage:
  python scripts/score_signal_first_touch.py --symbol EURUSD,US30
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
pip_size = _bt.pip_size

THRESHOLDS = (5, 10, 20)
MAX_BARS = 200  # give up if neither side hit


def first_touch_race(
    entry: float,
    side: int,
    highs: np.ndarray,
    lows: np.ndarray,
    thr_price: float,
) -> tuple[str, int]:
    """Return ('favor'|'adverse'|'none', bars_to_event)."""
    fav = entry + side * thr_price
    adv = entry - side * thr_price
    for k in range(len(highs)):
        hi, lo = highs[k], lows[k]
        if not (np.isfinite(hi) and np.isfinite(lo)):
            continue
        hit_f = hit_a = False
        if side > 0:
            hit_f = hi >= fav
            hit_a = lo <= adv
        else:
            hit_f = lo <= fav  # fav is lower for shorts
            hit_a = hi >= adv
        # same bar both: adverse first (conservative)
        if hit_f and hit_a:
            return "adverse", k + 1
        if hit_a:
            return "adverse", k + 1
        if hit_f:
            return "favor", k + 1
    return "none", len(highs)


def score_frame(df: pd.DataFrame, pip: float, thresholds=THRESHOLDS) -> list[dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    long_sig = df["long_entry"].fillna(False).to_numpy(bool)
    short_sig = df["short_entry"].fillna(False).to_numpy(bool)
    n = len(df)
    rows = []

    events = []
    for i in range(n - 2):
        if long_sig[i]:
            events.append((i, 1))
        if short_sig[i]:
            events.append((i, -1))

    for thr in thresholds:
        thr_px = thr * pip
        for side_name, side_filter in (("long", 1), ("short", -1), ("any", 0)):
            fav_n = adv_n = none_n = 0
            bars_f = []
            bars_a = []
            for i, side in events:
                if side_filter and side != side_filter:
                    continue
                j = i + 1  # fill next open
                if j >= n:
                    continue
                entry = o[j]
                if not np.isfinite(entry):
                    continue
                end = min(j + MAX_BARS, n)
                outcome, b = first_touch_race(entry, side, h[j:end], l[j:end], thr_px)
                if outcome == "favor":
                    fav_n += 1
                    bars_f.append(b)
                elif outcome == "adverse":
                    adv_n += 1
                    bars_a.append(b)
                else:
                    none_n += 1
            total = fav_n + adv_n + none_n
            decided = fav_n + adv_n
            rows.append(dict(
                side=side_name,
                threshold=thr,
                n=total,
                n_decided=decided,
                favor_n=fav_n,
                adverse_n=adv_n,
                none_n=none_n,
                favor_pct=100.0 * fav_n / total if total else np.nan,
                adverse_pct=100.0 * adv_n / total if total else np.nan,
                # among races that hit either side first
                favor_given_touch=100.0 * fav_n / decided if decided else np.nan,
                median_bars_favor=float(np.median(bars_f)) if bars_f else np.nan,
                median_bars_adverse=float(np.median(bars_a)) if bars_a else np.nan,
                thr_price=thr_px,
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
    print("FIRST-TOUCH RACE: +T before -T after signal")
    print("T = 5, 10, 20 pips/points | fill = next open | same-bar both => adverse")
    print("=" * 72)

    for sym in [s.strip().upper() for s in args.symbol.split(",") if s.strip()]:
        path = resolve_csv(args.data_dir, sym)
        if not path:
            print(f"[{sym}] no data")
            continue
        pip = pip_size(sym)
        print(f"\n### {sym}  pip/point={pip}", flush=True)
        m1 = load_m1_fast(path, max_rows=args.max_rows)
        for set_name, spec in SETS.items():
            print(f"  {set_name} ...", flush=True)
            fr = build_set_frame(m1, set_name).dropna(subset=["rsi", "sma_low"])
            for r in score_frame(fr, pip):
                r["symbol"] = sym
                r["set"] = set_name
                r["ltf"] = spec["ltf"]
                all_rows.append(r)
            # print any-side summary
            sub = [x for x in all_rows if x["symbol"] == sym and x["set"] == set_name and x["side"] == "any"]
            print(f"    {'T':>3} {'n':>6} {'+first%':>8} {'-first%':>8} {'+given':>8} {'medB+':>6} {'medB-':>6} {'none':>5}")
            for x in sub:
                print(
                    f"    {x['threshold']:3d} {x['n']:6d} "
                    f"{x['favor_pct']:7.1f}% {x['adverse_pct']:7.1f}% "
                    f"{x['favor_given_touch']:7.1f}% "
                    f"{x['median_bars_favor']:6.1f} {x['median_bars_adverse']:6.1f} "
                    f"{x['none_n']:5d}"
                )

    df = pd.DataFrame(all_rows)
    out = os.path.join(args.out_dir, "signal_first_touch.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("LONG ONLY @ T=10 (agent-relevant)")
    print("=" * 72)
    g = df[(df.side == "long") & (df.threshold == 10)].sort_values("favor_given_touch", ascending=False)
    print(g[["symbol", "set", "n", "favor_pct", "adverse_pct", "favor_given_touch",
             "median_bars_favor", "median_bars_adverse", "none_n"]].to_string(index=False))

    print("\n" + "=" * 72)
    print("ANY SIDE summary table")
    print("=" * 72)
    g2 = df[df.side == "any"][["symbol", "set", "threshold", "n", "favor_pct", "adverse_pct",
                                 "favor_given_touch", "median_bars_favor", "median_bars_adverse"]]
    print(g2.to_string(index=False))
    print(f"\nFull -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
