"""Backtest: Momentum Vector M multi-timeframe strategy.

M = |CCI| × tanh((CCI−SMA4)/40) × Strength(RSI BB) × sign(CCI)

Configs:
  1) 5m timing + 30m HTF filter
  2) 5m timing + 1h HTF filter
Thresholds: entry ±18 (and ±25 sensitivity), exit ±4 / flip

Usage:
  python scripts/backtest_momentum_vector.py --symbol EURUSD
  python scripts/backtest_momentum_vector.py --symbol US30
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_io.loader import resample  # noqa: E402
from features.momentum_vector import build_mtf_momentum  # noqa: E402

# reuse fast load helpers from DVMR backtest
import importlib.util

_bt_path = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("backtest_dvmr_mtf", _bt_path)
_bt = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["backtest_dvmr_mtf"] = _bt
_spec.loader.exec_module(_bt)
load_m1_fast = _bt.load_m1_fast
resolve_csv = _bt.resolve_csv
pip_size = _bt.pip_size

COMBOS = [
    {"name": "5m+30m", "base": "5min", "htf": "30min"},
    {"name": "5m+1h", "base": "5min", "htf": "1h"},
]


@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    reason: str
    pnl_pct: float
    bars_held: int


def run_strategy(
    df: pd.DataFrame,
    *,
    entry_thr: float = 18.0,
    exit_thr: float = 4.0,
    initial_equity: float = 100_000.0,
    risk_pct: float = 0.01,
    spread_pips: float = 0.8,
    commission_pips: float = 0.4,
    pip: float = 0.0001,
    atr_sl_mult: float = 1.5,
) -> tuple[list[Trade], pd.Series, dict]:
    """Signal on close t -> fill open t+1. Optional soft ATR stop for risk sizing.
    Position size: risk 1% of equity on 1.5×ATR(14) stop distance (stop not hard exit
    unless hit; primary exits are score rules).
    """
    from features.indicators import atr as atr_ind

    M = df["M"].to_numpy(float)
    htf_M = df["htf_M"].to_numpy(float)
    htf_dir = df["htf_direction"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr_ind(df, 14).to_numpy(float)
    idx = df.index

    # Edge trigger: cross above/below entry threshold (avoids re-firing every bar
    # while M stays extreme after a quick ±4 exit).
    M_prev = np.roll(M, 1)
    M_prev[0] = np.nan
    long_level = (M > entry_thr) & (htf_dir > 0) & (htf_M > 0)
    short_level = (M < -entry_thr) & (htf_dir < 0) & (htf_M < 0)
    long_entry = long_level & (M_prev <= entry_thr)
    short_entry = short_level & (M_prev >= -entry_thr)
    long_entry = np.nan_to_num(long_entry.astype(float), nan=0.0).astype(bool)
    short_entry = np.nan_to_num(short_entry.astype(float), nan=0.0).astype(bool)
    # flip detection uses level (not edge)
    long_flip = np.nan_to_num(short_level.astype(float), nan=0.0).astype(bool)
    short_flip = np.nan_to_num(long_level.astype(float), nan=0.0).astype(bool)

    cost = (spread_pips + commission_pips) * pip
    equity = float(initial_equity)
    init_floor = initial_equity * 0.05  # stop trading if account crushed
    eq = np.full(len(df), np.nan)
    trades: list[Trade] = []

    side = 0
    entry = stop = qty = 0.0
    entry_i = -1
    pending = 0
    pending_atr = 0.0

    for i in range(len(df)):
        if equity < init_floor:
            eq[i] = equity
            continue

        # fill pending
        if pending != 0 and side == 0 and i > 0:
            fill = o[i]
            aa = pending_atr if np.isfinite(pending_atr) and pending_atr > 0 else (
                a[i - 1] if np.isfinite(a[i - 1]) else np.nan
            )
            if np.isfinite(fill) and np.isfinite(aa) and aa > 0:
                if pending > 0:
                    fill = fill + 0.5 * cost
                    stop = fill - atr_sl_mult * aa
                else:
                    fill = fill - 0.5 * cost
                    stop = fill + atr_sl_mult * aa
                risk_dist = abs(fill - stop)
                if risk_dist > 0:
                    qty = (equity * risk_pct) / risk_dist
                    side = pending
                    entry = fill
                    entry_i = i
            pending = 0

        if side != 0:
            exit_px = reason = None
            # hard ATR stop
            if side > 0 and l[i] <= stop:
                exit_px, reason = stop, "sl"
            elif side < 0 and h[i] >= stop:
                exit_px, reason = stop, "sl"
            else:
                # score exits
                if side > 0:
                    flip = long_flip[i]
                    soft = np.isfinite(M[i]) and M[i] < exit_thr
                    if flip or soft:
                        exit_px = c[i] - 0.5 * cost
                        reason = "flip" if flip else "score"
                else:
                    flip = short_flip[i]
                    soft = np.isfinite(M[i]) and M[i] > -exit_thr
                    if flip or soft:
                        exit_px = c[i] + 0.5 * cost
                        reason = "flip" if flip else "score"

            if exit_px is not None:
                pnl = qty * (exit_px - entry) * side
                equity += pnl
                trades.append(Trade(
                    side=side, entry_time=idx[entry_i], exit_time=idx[i],
                    entry=entry, exit=exit_px, reason=reason or "exit",
                    pnl_pct=pnl / initial_equity * 100.0,
                    bars_held=i - entry_i,
                ))
                side = 0
                qty = 0.0

        if side == 0 and pending == 0:
            if long_entry[i] and np.isfinite(a[i]) and a[i] > 0:
                pending, pending_atr = 1, a[i]
            elif short_entry[i] and np.isfinite(a[i]) and a[i] > 0:
                pending, pending_atr = -1, a[i]

        eq[i] = equity

    if side != 0:
        exit_px = c[-1]
        pnl = qty * (exit_px - entry) * side
        equity += pnl
        trades.append(Trade(
            side=side, entry_time=idx[entry_i], exit_time=idx[-1],
            entry=entry, exit=exit_px, reason="eod",
            pnl_pct=pnl / initial_equity * 100.0,
            bars_held=len(df) - 1 - entry_i,
        ))
        eq[-1] = equity

    equity_s = pd.Series(eq, index=idx, name="equity").ffill().fillna(initial_equity)

    # buy & hold on same window
    c0, c1 = float(c[np.isfinite(c)][0]), float(c[np.isfinite(c)][-1])
    bh = (c1 / c0 - 1.0) * 100.0 if c0 > 0 else float("nan")

    stats = summarize(trades, equity_s, initial_equity, bh)
    return trades, equity_s, stats


def summarize(trades: list[Trade], equity: pd.Series, init: float, bh_pct: float) -> dict:
    final = float(equity.iloc[-1])
    total_ret = (final / init - 1.0) * 100.0
    peak = equity.cummax()
    dd = (equity / peak - 1.0) * 100.0
    max_dd = float(dd.min())

    t0, t1 = equity.index[0], equity.index[-1]
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    try:
        daily = equity.resample("1D").last().dropna().pct_change().dropna()
        sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if len(daily) > 2 and daily.std() > 0 else float("nan")
    except Exception:
        sharpe = float("nan")

    n = len(trades)
    if n == 0:
        return dict(
            n_trades=0, win_rate=np.nan, avg_win=np.nan, avg_loss=np.nan,
            profit_factor=np.nan, total_return_pct=total_ret, buy_hold_pct=bh_pct,
            max_dd_pct=max_dd, sharpe=sharpe, avg_bars=np.nan, final_equity=final,
        )

    pnls = np.array([t.pnl_pct for t in trades])  # in % of initial
    # use dollar pnl from equity path: reconstruct from pct of initial
    # better: store was pnl_pct of initial — use for WR
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    wr = 100.0 * (pnls > 0).mean()
    avg_win = float(wins.mean()) if len(wins) else 0.0
    avg_loss = float(losses.mean()) if len(losses) else 0.0
    gw, gl = wins.sum(), -losses.sum()
    pf = (gw / gl) if gl > 0 else (np.inf if gw > 0 else np.nan)
    avg_bars = float(np.mean([t.bars_held for t in trades]))

    return dict(
        n_trades=n,
        win_rate=wr,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=float(pf) if np.isfinite(pf) else pf,
        total_return_pct=total_ret,
        buy_hold_pct=bh_pct,
        max_dd_pct=max_dd,
        sharpe=sharpe,
        avg_bars=avg_bars,
        final_equity=final,
    )


def plot_curves(curves: dict[str, pd.Series], out_path: str, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    for name, eq in curves.items():
        norm = eq / float(eq.iloc[0]) * 100.0
        ax0.plot(norm.index, norm.values, label=name, lw=1.2)
        dd = (norm / norm.cummax() - 1.0) * 100.0
        ax1.plot(dd.index, dd.values, label=name, lw=1.0)
    ax0.set_ylabel("Equity (start=100)")
    ax0.set_title(title)
    ax0.legend(fontsize=8, loc="best")
    ax0.grid(True, alpha=0.3)
    ax1.set_ylabel("Drawdown %")
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--spread-pips", type=float, default=0.8)
    ap.add_argument("--commission-pips", type=float, default=0.4)
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "momentum_vector"))
    args = ap.parse_args()

    path = resolve_csv(args.data_dir, args.symbol)
    if not path:
        raise SystemExit(f"No data for {args.symbol}")

    print("=" * 72)
    print(f"MOMENTUM VECTOR M  |  {args.symbol}")
    print(f"data: {os.path.basename(path)}")
    print("M = |CCI| × tanh(vel/40) × Strength(RSI-BB) × sign(CCI)")
    print("Entry ±thr + HTF agree | Exit ±4 or flip | risk 1% / 1.5 ATR stop")
    print("=" * 72)

    print("Loading M1 ...", flush=True)
    m1 = load_m1_fast(path, max_rows=args.max_rows)
    print(f"M1 bars={len(m1):,}  {m1.index[0]} -> {m1.index[-1]}", flush=True)
    pip = pip_size(args.symbol)
    # US30 / gold: wider spread defaults
    spread = args.spread_pips
    if args.symbol.upper() in ("US30", "NAS100"):
        spread = max(spread, 2.0)
    if args.symbol.upper().startswith("XAU"):
        spread = max(spread, 1.5)

    thresholds = [18.0, 25.0]
    rows = []
    curves = {}
    all_trades = []

    base_5 = resample(m1, "5min")
    print(f"5m bars={len(base_5):,}", flush=True)

    for combo in COMBOS:
        htf = resample(m1, combo["htf"])
        print(f"\nBuilding features {combo['name']} (htf bars={len(htf):,}) ...", flush=True)
        frame = build_mtf_momentum(base_5, htf, combo["htf"])
        frame = frame.dropna(subset=["M", "htf_M", "htf_direction"])
        print(f"  usable={len(frame):,}", flush=True)

        for thr in thresholds:
            label = f"{combo['name']}|thr{int(thr)}"
            print(f"  run {label} ...", flush=True)
            trades, eq, stats = run_strategy(
                frame,
                entry_thr=thr,
                exit_thr=4.0,
                initial_equity=args.equity,
                spread_pips=spread,
                commission_pips=args.commission_pips,
                pip=pip,
            )
            stats["label"] = label
            stats["combo"] = combo["name"]
            stats["entry_thr"] = thr
            stats["symbol"] = args.symbol
            rows.append(stats)
            curves[label] = eq
            for t in trades:
                all_trades.append(dict(
                    label=label, side="long" if t.side > 0 else "short",
                    entry_time=t.entry_time, exit_time=t.exit_time,
                    entry=t.entry, exit=t.exit, reason=t.reason,
                    pnl_pct=t.pnl_pct, bars_held=t.bars_held,
                ))
            print(
                f"    n={stats['n_trades']:4d}  WR={stats['win_rate']:.1f}%  "
                f"PF={stats['profit_factor']:.2f}  ret={stats['total_return_pct']:+.1f}%  "
                f"BH={stats['buy_hold_pct']:+.1f}%  DD={stats['max_dd_pct']:.1f}%  "
                f"Sh={stats['sharpe']:.2f}  avgBars={stats['avg_bars']:.1f}",
                flush=True,
            )

    os.makedirs(args.out_dir, exist_ok=True)
    report = pd.DataFrame(rows)
    rpath = os.path.join(args.out_dir, f"mv_report_{args.symbol}.csv")
    report.to_csv(rpath, index=False)
    if all_trades:
        pd.DataFrame(all_trades).to_csv(
            os.path.join(args.out_dir, f"mv_trades_{args.symbol}.csv"), index=False
        )
    ppath = os.path.join(args.out_dir, f"mv_equity_{args.symbol}.png")
    plot_curves(curves, ppath, f"Momentum Vector M — {args.symbol}")

    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE")
    print("=" * 72)
    cols = ["label", "n_trades", "win_rate", "avg_win", "avg_loss", "profit_factor",
            "total_return_pct", "buy_hold_pct", "max_dd_pct", "sharpe", "avg_bars"]
    with pd.option_context("display.float_format", lambda x: f"{x: .3f}", "display.width", 160):
        print(report[cols].to_string(index=False))

    # short ranking
    print("\nRANK by total return:")
    for _, r in report.sort_values("total_return_pct", ascending=False).iterrows():
        print(f"  {r['label']:16s}  ret={r['total_return_pct']:+7.1f}%  PF={r['profit_factor']:.2f}  "
              f"WR={r['win_rate']:.1f}%  DD={r['max_dd_pct']:.1f}%")

    print(f"\nReport -> {rpath}")
    print(f"Plot   -> {ppath}")
    print("Done.")


if __name__ == "__main__":
    main()
