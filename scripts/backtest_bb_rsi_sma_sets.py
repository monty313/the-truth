"""Backtest BB / RSI-BB / SMA multi-set strategy.

Sets:
  A: 5m + 30m/1h
  B: 15m + 1h/4h
  C: 30m + 4h/1d

Buy: any HTF (close > BB_up & close > SMA_low) + LTF (RSI cross up lower RSI-BB & close > SMA_low)
Sell: opposite (HTF below BB_lo & < SMA_high; LTF RSI cross down upper RSI-BB & < SMA_high)

Exit: opposite entry signal, or SL 1.5 ATR / TP 2.5 ATR (risk 1%).

Usage:
  python scripts/backtest_bb_rsi_sma_sets.py --symbol EURUSD
  python scripts/backtest_bb_rsi_sma_sets.py --symbol US30
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
from features.indicators import atr as atr_ind  # noqa: E402

import importlib.util

_p = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("bt", _p)
_bt = importlib.util.module_from_spec(_spec)
sys.modules["bt"] = _bt
_spec.loader.exec_module(_bt)
load_m1_fast = _bt.load_m1_fast
resolve_csv = _bt.resolve_csv
pip_size = _bt.pip_size


def run_bt(
    df: pd.DataFrame,
    *,
    init: float = 100_000.0,
    risk_pct: float = 0.01,
    sl_atr: float = 1.5,
    tp_atr: float = 2.5,
    spread_pips: float = 0.8,
    commission_pips: float = 0.4,
    pip: float = 0.0001,
) -> tuple[list[dict], pd.Series, dict]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr_ind(df, 14).to_numpy(float)
    le = df["long_entry"].fillna(False).to_numpy(bool)
    se = df["short_entry"].fillna(False).to_numpy(bool)
    idx = df.index
    cost = (spread_pips + commission_pips) * pip

    equity = float(init)
    floor = init * 0.05
    eq = np.full(len(df), np.nan)
    trades = []
    side = 0
    entry = stop = target = risk_dist = qty = 0.0
    entry_i = -1
    pending = 0
    pending_atr = 0.0

    for i in range(len(df)):
        if equity < floor:
            eq[i] = equity
            continue
        if pending and side == 0 and i > 0:
            fill = o[i]
            aa = pending_atr if np.isfinite(pending_atr) and pending_atr > 0 else a[i - 1]
            if np.isfinite(fill) and np.isfinite(aa) and aa > 0:
                if pending > 0:
                    fill = fill + 0.5 * cost
                    stop = fill - sl_atr * aa
                    target = fill + tp_atr * aa
                else:
                    fill = fill - 0.5 * cost
                    stop = fill + sl_atr * aa
                    target = fill - tp_atr * aa
                risk_dist = abs(fill - stop)
                if risk_dist > 0:
                    qty = (equity * risk_pct) / risk_dist
                    side = pending
                    entry = fill
                    entry_i = i
            pending = 0

        if side != 0:
            exit_px = reason = None
            if side > 0:
                if l[i] <= stop:
                    exit_px, reason = stop, "sl"
                elif h[i] >= target:
                    exit_px, reason = target, "tp"
                elif se[i]:
                    exit_px, reason = c[i] - 0.5 * cost, "flip"
            else:
                if h[i] >= stop:
                    exit_px, reason = stop, "sl"
                elif l[i] <= target:
                    exit_px, reason = target, "tp"
                elif le[i]:
                    exit_px, reason = c[i] + 0.5 * cost, "flip"
            if exit_px is not None:
                pnl = qty * (exit_px - entry) * side
                equity += pnl
                trades.append(dict(
                    side=side, entry_time=idx[entry_i], exit_time=idx[i],
                    entry=entry, exit=exit_px, reason=reason, pnl=pnl,
                    bars=i - entry_i,
                ))
                side = 0
                qty = 0.0

        if side == 0 and pending == 0:
            if le[i] and np.isfinite(a[i]) and a[i] > 0:
                pending, pending_atr = 1, a[i]
            elif se[i] and np.isfinite(a[i]) and a[i] > 0:
                pending, pending_atr = -1, a[i]
        eq[i] = equity

    if side != 0:
        pnl = qty * (c[-1] - entry) * side
        equity += pnl
        trades.append(dict(
            side=side, entry_time=idx[entry_i], exit_time=idx[-1],
            entry=entry, exit=c[-1], reason="eod", pnl=pnl,
            bars=len(df) - 1 - entry_i,
        ))
        eq[-1] = equity

    eq_s = pd.Series(eq, index=idx).ffill().fillna(init)
    final = float(eq_s.iloc[-1])
    ret = (final / init - 1.0) * 100.0
    dd = float((eq_s / eq_s.cummax() - 1.0).min() * 100.0)
    c0 = float(c[np.isfinite(c)][0])
    c1 = float(c[np.isfinite(c)][-1])
    bh = (c1 / c0 - 1.0) * 100.0 if c0 else np.nan
    n = len(trades)
    if n == 0:
        stats = dict(n=0, wr=np.nan, pf=np.nan, ret=ret, bh=bh, dd=dd, sharpe=np.nan, avg_bars=np.nan)
    else:
        pnls = np.array([t["pnl"] for t in trades])
        wr = 100.0 * (pnls > 0).mean()
        gw, gl = pnls[pnls > 0].sum(), -pnls[pnls <= 0].sum()
        pf = (gw / gl) if gl > 0 else (np.inf if gw > 0 else 0.0)
        try:
            d = eq_s.resample("1D").last().dropna().pct_change().dropna()
            sh = float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 5 and d.std() > 0 else np.nan
        except Exception:
            sh = np.nan
        stats = dict(
            n=n, wr=wr, pf=float(pf), ret=ret, bh=bh, dd=dd, sharpe=sh,
            avg_bars=float(np.mean([t["bars"] for t in trades])),
            avg_win=float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0.0,
            avg_loss=float(pnls[pnls <= 0].mean()) if (pnls <= 0).any() else 0.0,
        )
    return trades, eq_s, stats


def plot_curves(curves: dict, path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    for name, eq in curves.items():
        norm = eq / float(eq.iloc[0]) * 100.0
        axes[0].plot(norm.index, norm.values, label=name, lw=1.2)
        dd = (norm / norm.cummax() - 1.0) * 100.0
        axes[1].plot(dd.index, dd.values, label=name, lw=1.0)
    axes[0].set_title(title)
    axes[0].set_ylabel("Equity (100=start)")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].set_ylabel("DD %")
    axes[1].grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "bb_rsi_sma"))
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()

    path = resolve_csv(args.data_dir, args.symbol)
    if not path:
        raise SystemExit(f"no data for {args.symbol}")

    print("=" * 72)
    print(f"BB / RSI-BB / SMA SETS  |  {args.symbol}")
    print("HTF: BB(100,shift2,dev0.5) + SMA10 hi/lo")
    print("LTF: RSI(5)+BB(10,shift5,dev1) + SMA10 hi/lo")
    print("Buy: any HTF above BB_up & >SMA_low; LTF RSI x-up lower BB & >SMA_low")
    print("Sell: opposite")
    print("=" * 72)

    m1 = load_m1_fast(path, max_rows=args.max_rows)
    print(f"M1={len(m1):,}  {m1.index[0]} -> {m1.index[-1]}", flush=True)
    pip = pip_size(args.symbol)
    spread = 2.0 if args.symbol.upper() in ("US30", "NAS100") else 0.8

    rows = []
    curves = {}
    all_tr = []

    for set_name in SETS:
        print(f"\n=== {set_name}  LTF={SETS[set_name]['ltf']} HTFs={SETS[set_name]['htfs']} ===", flush=True)
        fr = build_set_frame(m1, set_name)
        fr = fr.dropna(subset=["rsi", "sma_low", "sma_high"])
        n_long = int(fr["long_entry"].sum())
        n_short = int(fr["short_entry"].sum())
        print(f"  bars={len(fr):,}  long_signals={n_long}  short_signals={n_short}", flush=True)
        trades, eq, stats = run_bt(fr, pip=pip, spread_pips=spread)
        stats["set"] = set_name
        stats["symbol"] = args.symbol
        stats["ltf"] = SETS[set_name]["ltf"]
        rows.append(stats)
        curves[set_name] = eq
        for t in trades:
            all_tr.append({**t, "set": set_name, "side_s": "long" if t["side"] > 0 else "short"})
        print(
            f"  n={stats['n']}  WR={stats['wr']:.1f}%  PF={stats['pf']:.2f}  "
            f"ret={stats['ret']:+.1f}%  BH={stats['bh']:+.1f}%  DD={stats['dd']:.1f}%  "
            f"Sh={stats['sharpe']:.2f}  avgBars={stats['avg_bars']:.1f}",
            flush=True,
        )

    os.makedirs(args.out_dir, exist_ok=True)
    rep = pd.DataFrame(rows)
    rpath = os.path.join(args.out_dir, f"report_{args.symbol}.csv")
    rep.to_csv(rpath, index=False)
    if all_tr:
        pd.DataFrame(all_tr).to_csv(os.path.join(args.out_dir, f"trades_{args.symbol}.csv"), index=False)
    ppath = os.path.join(args.out_dir, f"equity_{args.symbol}.png")
    plot_curves(curves, ppath, f"BB/RSI-BB/SMA sets — {args.symbol}")

    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE")
    print(rep.to_string(index=False))
    print(f"\nReport -> {rpath}")
    print(f"Plot   -> {ppath}")
    print("Done.")


if __name__ == "__main__":
    main()
