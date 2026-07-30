"""Path quality in BARS from entry (not pips).

After signal -> fill next open (bar 0). Then for H in {5, 10, 20}:

1) AT bar H (close H bars after fill):
   - hit: close in signal direction vs entry
   - mean/median signed return %

2) FIRST SIGN race within bars 1..H:
   - walk closes bar-by-bar
   - first time signed_ret > 0  = favor
   - first time signed_ret < 0  = adverse
   - which happens first? (same bar: adverse if both, rare at close)

3) FIRST EXTREME race within bars 1..H (high/low):
   - long: first touch of high>entry vs low<entry
   - short: opposite
   - same-bar both => adverse (conservative)

Usage:
  python scripts/score_signal_bars_from_entry.py --symbol EURUSD,US30
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

HORIZONS = (5, 10, 20)


def analyze_event(entry: float, side: int, o, h, l, c, H: int) -> dict:
    """Path uses bars [0..H-1] after fill; bar H close is c[H-1] if H bars later."""
    # arrays o,h,l,c start at fill bar (index 0 = fill bar)
    n = len(c)
    if H > n or H < 1:
        return {}

    # --- at exactly H bars: use close of bar index H-1 ---
    exit_c = c[H - 1]
    ret_h = side * (exit_c - entry) / entry * 100.0 if entry else np.nan
    hit_h = ret_h > 0

    # MFE/MAE over first H bars (using highs/lows)
    if side > 0:
        mfe = (np.nanmax(h[:H]) - entry) / entry * 100.0
        mae = (entry - np.nanmin(l[:H])) / entry * 100.0
    else:
        mfe = (entry - np.nanmin(l[:H])) / entry * 100.0
        mae = (np.nanmax(h[:H]) - entry) / entry * 100.0

    # --- first sign of close vs entry within 1..H ---
    first_close = "none"
    first_close_bar = np.nan
    for k in range(H):
        r = side * (c[k] - entry)
        if r > 0:
            first_close, first_close_bar = "favor", k + 1
            break
        if r < 0:
            first_close, first_close_bar = "adverse", k + 1
            break

    # --- first extreme touch (price beyond entry) within 1..H ---
    first_ext = "none"
    first_ext_bar = np.nan
    for k in range(H):
        if side > 0:
            hit_f = h[k] > entry
            hit_a = l[k] < entry
        else:
            hit_f = l[k] < entry
            hit_a = h[k] > entry
        if hit_f and hit_a:
            first_ext, first_ext_bar = "adverse", k + 1
            break
        if hit_a:
            first_ext, first_ext_bar = "adverse", k + 1
            break
        if hit_f:
            first_ext, first_ext_bar = "favor", k + 1
            break

    return dict(
        ret_at_H=ret_h,
        hit_at_H=hit_h,
        mfe_H=mfe,
        mae_H=mae,
        first_close=first_close,
        first_close_bar=first_close_bar,
        first_ext=first_ext,
        first_ext_bar=first_ext_bar,
    )


def score_frame(df: pd.DataFrame) -> list[dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    long_sig = df["long_entry"].fillna(False).to_numpy(bool)
    short_sig = df["short_entry"].fillna(False).to_numpy(bool)
    n = len(df)

    events = []
    for i in range(n):
        if long_sig[i]:
            events.append((i, 1))
        if short_sig[i]:
            events.append((i, -1))

    rows = []
    for H in HORIZONS:
        for side_name, side_f in (("long", 1), ("short", -1), ("any", 0)):
            recs = []
            for i, side in events:
                if side_f and side != side_f:
                    continue
                j = i + 1
                if j + H > n:
                    continue
                entry = o[j]
                if not np.isfinite(entry) or entry == 0:
                    continue
                r = analyze_event(
                    entry, side,
                    o[j:j + H], h[j:j + H], l[j:j + H], c[j:j + H],
                    H,
                )
                if r:
                    recs.append(r)
            if not recs:
                continue
            hit = np.mean([x["hit_at_H"] for x in recs]) * 100.0
            rets = np.array([x["ret_at_H"] for x in recs], float)
            mfe = np.array([x["mfe_H"] for x in recs], float)
            mae = np.array([x["mae_H"] for x in recs], float)

            def pct(key, val):
                return 100.0 * np.mean([x[key] == val for x in recs])

            rows.append(dict(
                side=side_name,
                bars_from_entry=H,
                n=len(recs),
                # at exactly H bars
                hit_pct_at_H=hit,
                mean_ret_at_H=float(np.mean(rets)),
                median_ret_at_H=float(np.median(rets)),
                mean_mfe_in_H=float(np.mean(mfe)),
                mean_mae_in_H=float(np.mean(mae)),
                # first close sign within H bars
                first_close_favor_pct=pct("first_close", "favor"),
                first_close_adverse_pct=pct("first_close", "adverse"),
                first_close_none_pct=pct("first_close", "none"),
                # first extreme beyond entry within H bars
                first_ext_favor_pct=pct("first_ext", "favor"),
                first_ext_adverse_pct=pct("first_ext", "adverse"),
                first_ext_none_pct=pct("first_ext", "none"),
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

    print("=" * 72)
    print("PATH IN BARS FROM ENTRY (not pips)")
    print("H = 5, 10, 20 LTF bars after fill (next open)")
    print("  hit@H     = close at bar H still in signal direction")
    print("  firstClose= which sign of close-vs-entry appears first in 1..H")
    print("  firstExt  = which extreme (beyond entry) appears first in 1..H")
    print("=" * 72)

    all_rows = []
    for sym in [s.strip().upper() for s in args.symbol.split(",") if s.strip()]:
        path = resolve_csv(args.data_dir, sym)
        if not path:
            continue
        print(f"\n### {sym}", flush=True)
        m1 = load_m1_fast(path, max_rows=args.max_rows)
        for set_name, spec in SETS.items():
            fr = build_set_frame(m1, set_name).dropna(subset=["rsi", "sma_low"])
            print(f"  {set_name} (LTF={spec['ltf']}) signals "
                  f"L={int(fr.long_entry.sum())} S={int(fr.short_entry.sum())}", flush=True)
            for r in score_frame(fr):
                r["symbol"] = sym
                r["set"] = set_name
                r["ltf"] = spec["ltf"]
                all_rows.append(r)
            sub = [x for x in all_rows if x["symbol"] == sym and x["set"] == set_name and x["side"] == "any"]
            print(f"    {'H':>3} {'n':>6} {'hit@H':>7} {'mean%':>8} {'1stCl+':>7} {'1stCl-':>7} {'1stEx+':>7} {'1stEx-':>7} {'MFE':>6} {'MAE':>6}")
            for x in sub:
                print(
                    f"    {x['bars_from_entry']:3d} {x['n']:6d} "
                    f"{x['hit_pct_at_H']:6.1f}% {x['mean_ret_at_H']:+7.3f}% "
                    f"{x['first_close_favor_pct']:6.1f}% {x['first_close_adverse_pct']:6.1f}% "
                    f"{x['first_ext_favor_pct']:6.1f}% {x['first_ext_adverse_pct']:6.1f}% "
                    f"{x['mean_mfe_in_H']:5.3f} {x['mean_mae_in_H']:5.3f}"
                )

    df = pd.DataFrame(all_rows)
    out = os.path.join(args.out_dir, "signal_bars_from_entry.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("LONG ONLY — hit rate at H bars from entry")
    print("=" * 72)
    g = df[df.side == "long"][["symbol", "set", "bars_from_entry", "n", "hit_pct_at_H",
                                 "mean_ret_at_H", "first_close_favor_pct", "first_ext_favor_pct"]]
    # pivot-ish print
    for (sym, sname), gg in g.groupby(["symbol", "set"]):
        print(f"\n{sym} {sname}")
        print(gg[["bars_from_entry", "n", "hit_pct_at_H", "mean_ret_at_H",
                  "first_close_favor_pct", "first_ext_favor_pct"]].to_string(index=False))

    print(f"\nFull -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
