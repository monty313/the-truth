"""Backtest: Triple EMA (9/21/50) + RSI(14) Scalp.

Strategy rules (as specified):
  Timeframe: 1-min or 5-min
  Indicators: EMA9 (fast), EMA21 (medium), EMA50 (trend), RSI(14)

  LONG:
    - Price (close) above EMA50
    - EMA9 crosses above EMA21
    - RSI > 50 and < 70
    - Enter on signal bar close (honest fill: next bar open)

  SHORT:
    - Price below EMA50
    - EMA9 crosses below EMA21
    - RSI < 50 and > 30
    - Enter on signal bar close (honest fill: next bar open)

  Exits:
    - Stop: below/above recent swing (lookback N), clamped to [sl_pips_min, sl_pips_max]
    - Take profit: 1.5R
    - Time stop: close after time_stop_bars if not hit
    - Trailing: move stop to breakeven once +be_pips in profit

Honest fills: signal at bar t close -> enter at bar t+1 open.
Intrabar: if both SL and TP touchable same bar, assume SL first (conservative).
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from data_io.loader import read_mt5_m1, resample  # noqa: E402
from features.indicators import rsi  # noqa: E402

# Pip size by symbol family (price distance of 1 pip)
PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "AUDUSD": 0.0001,
    "NZDUSD": 0.0001,
    "USDCAD": 0.0001,
    "USDCHF": 0.0001,
    "USDJPY": 0.01,
    "EURJPY": 0.01,
    "GBPJPY": 0.01,
    "XAUUSD": 0.1,   # gold: 0.1 = 1 pip (common retail convention)
    "US30": 1.0,     # index points as "pips"
    "NAS100": 1.0,
}


def pip_size(symbol: str) -> float:
    s = symbol.upper().replace(".", "").replace("m", "")
    if s in PIP_SIZE:
        return PIP_SIZE[s]
    if s.endswith("JPY"):
        return 0.01
    if s.startswith("XAU") or s.startswith("GOLD"):
        return 0.1
    if any(x in s for x in ("US30", "DJ", "NAS", "SPX", "DE40", "UK100")):
        return 1.0
    return 0.0001


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def swing_low(low: pd.Series, n: int = 5) -> pd.Series:
    return low.rolling(n, min_periods=n).min()


def swing_high(high: pd.Series, n: int = 5) -> pd.Series:
    return high.rolling(n, min_periods=n).max()


def in_session(ts: pd.Timestamp, sessions: str) -> bool:
    """Broker/local hour filter. sessions: all | london | ny | london_ny."""
    if sessions == "all":
        return True
    h = int(ts.hour)
    # Approximate broker-time windows (CEST-family common on EU brokers)
    london = 8 <= h < 17
    ny = 13 <= h < 22
    if sessions == "london":
        return london
    if sessions == "ny":
        return ny
    if sessions in ("london_ny", "london+ny"):
        return london or ny
    return True


def build_frame(m1: pd.DataFrame, tf: str, swing_n: int = 5) -> pd.DataFrame:
    """Resample + indicators. Cross detected on closed bars only."""
    if tf in ("1min", "1m", "M1"):
        d = m1.copy()
    else:
        rule = "5min" if tf in ("5min", "5m", "M5") else tf
        d = resample(m1, rule).copy()

    d["ema9"] = ema(d["close"], 9)
    d["ema21"] = ema(d["close"], 21)
    d["ema50"] = ema(d["close"], 50)
    d["rsi"] = rsi(d["close"], 14)
    d["sw_lo"] = swing_low(d["low"], swing_n)
    d["sw_hi"] = swing_high(d["high"], swing_n)
    # previous EMAs for cross detection (no look-ahead)
    d["ema9_prev"] = d["ema9"].shift(1)
    d["ema21_prev"] = d["ema21"].shift(1)
    return d.dropna()


@dataclass
class Trade:
    symbol: str
    side: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    stop_init: float
    target: float
    exit: float
    reason: str
    r_multiple: float
    pnl_pips: float
    bars_held: int
    risk_pips: float


def long_signal(row) -> bool:
    """Price > EMA50, EMA9 cross above EMA21, RSI in (50, 70)."""
    if not (row.close > row.ema50):
        return False
    cross_up = (row.ema9_prev <= row.ema21_prev) and (row.ema9 > row.ema21)
    if not cross_up:
        return False
    if not (50.0 < row.rsi < 70.0):
        return False
    return True


def short_signal(row) -> bool:
    """Price < EMA50, EMA9 cross below EMA21, RSI in (30, 50)."""
    if not (row.close < row.ema50):
        return False
    cross_dn = (row.ema9_prev >= row.ema21_prev) and (row.ema9 < row.ema21)
    if not cross_dn:
        return False
    if not (30.0 < row.rsi < 50.0):
        return False
    return True


def compute_stop(
    side: int,
    entry: float,
    row,
    pip: float,
    sl_pips_min: float = 8.0,
    sl_pips_max: float = 12.0,
    swing_buffer_pips: float = 1.0,
) -> float:
    """SL beyond recent swing, clamped into [min, max] pips from entry."""
    buf = swing_buffer_pips * pip
    if side > 0:
        raw = float(row.sw_lo) - buf
        dist = entry - raw
        dist_pips = dist / pip
        if dist_pips < sl_pips_min:
            dist = sl_pips_min * pip
        elif dist_pips > sl_pips_max:
            dist = sl_pips_max * pip
        return entry - dist
    else:
        raw = float(row.sw_hi) + buf
        dist = raw - entry
        dist_pips = dist / pip
        if dist_pips < sl_pips_min:
            dist = sl_pips_min * pip
        elif dist_pips > sl_pips_max:
            dist = sl_pips_max * pip
        return entry + dist


def run_backtest(
    df: pd.DataFrame,
    symbol: str,
    sl_pips_min: float = 8.0,
    sl_pips_max: float = 12.0,
    rr: float = 1.5,
    time_stop_bars: int = 20,
    be_pips: float = 5.0,
    sessions: str = "all",
    spread_pips: float = 0.0,
) -> list[Trade]:
    """Walk bars; one position at a time; no pyramiding."""
    pip = pip_size(symbol)
    spread = spread_pips * pip
    trades: list[Trade] = []
    i = 0
    n = len(df)

    while i < n - 2:
        row = df.iloc[i]
        ts = df.index[i]
        if not in_session(ts, sessions):
            i += 1
            continue

        long_ok = long_signal(row)
        short_ok = short_signal(row)
        if long_ok == short_ok:  # none or both
            i += 1
            continue
        side = 1 if long_ok else -1

        # fill next bar open (signal known only at close of bar i)
        entry_i = i + 1
        entry_row = df.iloc[entry_i]
        raw_open = float(entry_row.open)
        # adverse spread: long pays ask, short sells bid
        entry = raw_open + side * (spread / 2.0) if spread > 0 else raw_open

        stop = compute_stop(side, entry, row, pip, sl_pips_min, sl_pips_max)
        risk_dist = abs(entry - stop)
        if risk_dist <= 0 or not np.isfinite(risk_dist):
            i += 1
            continue
        risk_pips = risk_dist / pip
        target = entry + side * rr * risk_dist
        stop_init = stop
        be_moved = False
        be_trigger = be_pips * pip

        exit_price = None
        reason = None
        exit_i = entry_i

        for j in range(entry_i, min(entry_i + time_stop_bars + 1, n)):
            b = df.iloc[j]
            hi, lo, cl = float(b.high), float(b.low), float(b.close)

            # trail to breakeven once favorable by be_pips (use prior bar extreme
            # for trigger so we don't use same-bar future information poorly;
            # on entry bar, allow extreme vs entry)
            if not be_moved:
                fav = (hi - entry) if side > 0 else (entry - lo)
                if fav >= be_trigger:
                    stop = entry  # breakeven
                    be_moved = True

            if side > 0:
                hit_sl = lo <= stop
                hit_tp = hi >= target
            else:
                hit_sl = hi >= stop
                hit_tp = lo <= target

            if hit_sl and hit_tp:
                exit_price, reason, exit_i = stop, "stop_both", j
                break
            if hit_sl:
                exit_price, reason, exit_i = stop, ("breakeven" if be_moved and abs(stop - entry) < 1e-12 else "stop"), j
                break
            if hit_tp:
                exit_price, reason, exit_i = target, "target", j
                break
            if j == entry_i + time_stop_bars:
                exit_price, reason, exit_i = cl, "time", j
                break

        if exit_price is None:
            last_i = min(entry_i + time_stop_bars, n - 1)
            exit_price = float(df.iloc[last_i].close)
            reason = "eod"
            exit_i = last_i

        r_mult = side * (exit_price - entry) / risk_dist
        pnl_pips = side * (exit_price - entry) / pip

        trades.append(Trade(
            symbol=symbol,
            side=side,
            entry_time=df.index[entry_i],
            exit_time=df.index[exit_i],
            entry=entry,
            stop_init=stop_init,
            target=target,
            exit=exit_price,
            reason=reason or "?",
            r_multiple=float(r_mult),
            pnl_pips=float(pnl_pips),
            bars_held=int(exit_i - entry_i + 1),
            risk_pips=float(risk_pips),
        ))
        i = exit_i + 1

    return trades


def summarize(trades: list[Trade], label: str) -> dict:
    if not trades:
        return {
            "label": label, "n": 0, "win_rate": None, "avg_R": None,
            "profit_factor": None, "expectancy_R": None, "total_R": 0.0,
            "max_dd_R": 0.0, "total_pips": 0.0, "avg_pips": None,
            "target_hits": 0, "stop_hits": 0, "be_hits": 0, "time_exits": 0,
            "long_n": 0, "short_n": 0, "long_wr": None, "short_wr": None,
            "avg_risk_pips": None, "avg_bars": None,
        }
    r = np.array([t.r_multiple for t in trades], dtype=float)
    pips = np.array([t.pnl_pips for t in trades], dtype=float)
    wins = r > 0
    losses = r < 0
    gp = r[wins].sum() if wins.any() else 0.0
    gl = -r[losses].sum() if losses.any() else 0.0
    pf = (gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else 0.0)
    eq = np.cumsum(r)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    reasons = [t.reason for t in trades]
    longs = [t for t in trades if t.side > 0]
    shorts = [t for t in trades if t.side < 0]
    long_wr = (np.mean([t.r_multiple > 0 for t in longs]) * 100) if longs else None
    short_wr = (np.mean([t.r_multiple > 0 for t in shorts]) * 100) if shorts else None
    return {
        "label": label,
        "n": len(trades),
        "win_rate": float(wins.mean() * 100),
        "avg_R": float(r.mean()),
        "median_R": float(np.median(r)),
        "profit_factor": float(pf) if np.isfinite(pf) else None,
        "expectancy_R": float(r.mean()),
        "total_R": float(r.sum()),
        "max_dd_R": float(dd.max()) if len(dd) else 0.0,
        "total_pips": float(pips.sum()),
        "avg_pips": float(pips.mean()),
        "target_hits": sum(1 for x in reasons if x == "target"),
        "stop_hits": sum(1 for x in reasons if x in ("stop", "stop_both")),
        "be_hits": sum(1 for x in reasons if x == "breakeven"),
        "time_exits": sum(1 for x in reasons if x == "time"),
        "long_n": len(longs),
        "short_n": len(shorts),
        "long_wr": long_wr,
        "short_wr": short_wr,
        "avg_risk_pips": float(np.mean([t.risk_pips for t in trades])),
        "avg_bars": float(np.mean([t.bars_held for t in trades])),
    }


def resolve_csv(data_dir: str, symbol: str) -> Optional[str]:
    preferred = [
        f"{symbol}_M1_full.csv",
        f"{symbol}_M1_curriculum.csv",
        f"{symbol}_curriculum_2026.csv",
        f"{symbol}_M1_drill.csv",
    ]
    for name in preferred:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return p
    for fn in os.listdir(data_dir):
        if fn.upper().startswith(symbol.upper()) and fn.lower().endswith(".csv"):
            return os.path.join(data_dir, fn)
    return None


def _fmt(v, fmt: str = ".2f", na: str = "n/a") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return na
    return format(v, fmt)


def main():
    ap = argparse.ArgumentParser(description="Triple EMA + RSI Scalp backtest")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--symbols", default="EURUSD,GBPUSD")
    ap.add_argument("--tf", default="5min", choices=["1min", "5min"],
                    help="Chart timeframe")
    ap.add_argument("--sessions", default="all",
                    choices=["all", "london", "ny", "london_ny"],
                    help="Session filter (hour windows, broker time)")
    ap.add_argument("--sl-min", type=float, default=8.0, help="Min stop pips")
    ap.add_argument("--sl-max", type=float, default=12.0, help="Max stop pips")
    ap.add_argument("--rr", type=float, default=1.5, help="Take profit R multiple")
    ap.add_argument("--time-stop", type=int, default=None,
                    help="Time stop in bars (default: 20 on M1, 4 on M5 = ~20 min)")
    ap.add_argument("--be-pips", type=float, default=5.0,
                    help="Move stop to BE after this many pips profit")
    ap.add_argument("--spread-pips", type=float, default=0.8,
                    help="Round-trip-ish half-spread applied adversely on entry")
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Optional M1 row cap for faster smoke runs")
    ap.add_argument("--out", default=os.path.join(_ROOT, "artifacts", "ema_rsi_scalp_report.csv"))
    ap.add_argument("--trades-out", default=None,
                    help="Optional path to write every trade CSV")
    args = ap.parse_args()

    # ~15-20 minutes: M1 -> 20 bars, M5 -> 4 bars
    if args.time_stop is None:
        time_stop = 20 if args.tf == "1min" else 4
    else:
        time_stop = args.time_stop

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    all_trades: list[Trade] = []
    rows = []

    print("=" * 78)
    print("TRIPLE EMA (9/21/50) + RSI(14) SCALP  |  honest next-open fill")
    print(f"TF={args.tf}  SL=[{args.sl_min},{args.sl_max}] pips  RR={args.rr}  "
          f"time_stop={time_stop} bars  BE@{args.be_pips}p  session={args.sessions}  "
          f"spread~{args.spread_pips}p")
    print("LONG:  close>EMA50, EMA9 x above EMA21, 50<RSI<70")
    print("SHORT: close<EMA50, EMA9 x below EMA21, 30<RSI<50")
    print("Conservative: same-bar SL+TP -> SL first")
    print("=" * 78)

    # Load each symbol ONCE (reloading ~120MB CSVs caused MemoryError on GBPUSD).
    frames: dict[str, dict[str, pd.DataFrame]] = {}
    for sym in symbols:
        path = resolve_csv(args.data_dir, sym)
        if not path:
            print(f"\n[{sym}] NO CSV found in {args.data_dir}")
            continue
        print(f"\n[{sym}] loading {os.path.basename(path)} ...", flush=True)
        m1 = read_mt5_m1(path, max_rows=args.max_rows)
        print(f"       M1 bars={len(m1):,}  range={m1.index[0]} -> {m1.index[-1]}", flush=True)
        d_tf = build_frame(m1, args.tf)
        print(f"       {args.tf} bars={len(d_tf):,}  pip={pip_size(sym)}", flush=True)
        d_m1 = None
        if args.tf == "5min":
            d_m1 = build_frame(m1, "1min")
            print(f"       1min bars={len(d_m1):,} (cached for comparison)", flush=True)
        del m1  # free raw M1 before heavy multi-pass runs
        frames[sym] = {"tf": d_tf, "m1": d_m1}

        tr = run_backtest(
            d_tf, sym,
            sl_pips_min=args.sl_min,
            sl_pips_max=args.sl_max,
            rr=args.rr,
            time_stop_bars=time_stop,
            be_pips=args.be_pips,
            sessions=args.sessions,
            spread_pips=args.spread_pips,
        )
        all_trades.extend(tr)
        s = summarize(tr, f"{sym}|{args.tf}|{args.sessions}")
        rows.append(s)
        print(
            f"  n={s['n']:5d}  win={_fmt(s['win_rate'], '.1f')}%  "
            f"avgR={_fmt(s['avg_R'], '+.3f')}  PF={_fmt(s['profit_factor'], '.2f')}  "
            f"totalR={_fmt(s['total_R'], '+.1f')}  maxDD_R={_fmt(s['max_dd_R'], '.1f')}  "
            f"pips={_fmt(s['total_pips'], '+.1f')}  "
            f"TP={s['target_hits']} SL={s['stop_hits']} BE={s['be_hits']} time={s['time_exits']}",
            flush=True,
        )
        print(
            f"       long={s['long_n']} (wr={_fmt(s['long_wr'], '.1f')}%)  "
            f"short={s['short_n']} (wr={_fmt(s['short_wr'], '.1f')}%)  "
            f"avg_risk={_fmt(s['avg_risk_pips'], '.1f')}p  avg_bars={_fmt(s['avg_bars'], '.1f')}",
            flush=True,
        )

    # Also run session-filtered variants for FX majors if session=all
    if args.sessions == "all" and any(s in ("EURUSD", "GBPUSD", "USDJPY") for s in symbols):
        print("\n--- Session filter variants (same data) ---")
        for sess in ("london_ny", "london", "ny"):
            for sym, fr in frames.items():
                tr = run_backtest(
                    fr["tf"], sym,
                    sl_pips_min=args.sl_min,
                    sl_pips_max=args.sl_max,
                    rr=args.rr,
                    time_stop_bars=time_stop,
                    be_pips=args.be_pips,
                    sessions=sess,
                    spread_pips=args.spread_pips,
                )
                s = summarize(tr, f"{sym}|{args.tf}|{sess}")
                rows.append(s)
                print(
                    f"  {sym:7s} {sess:10s}  n={s['n']:5d}  win={_fmt(s['win_rate'], '.1f')}%  "
                    f"avgR={_fmt(s['avg_R'], '+.3f')}  PF={_fmt(s['profit_factor'], '.2f')}  "
                    f"totalR={_fmt(s['total_R'], '+.1f')}  pips={_fmt(s['total_pips'], '+.1f')}",
                    flush=True,
                )

    # Optional M1 comparison if we ran M5
    if args.tf == "5min":
        print("\n--- M1 comparison (same rules, 20-bar time stop) ---")
        for sym, fr in frames.items():
            if fr["m1"] is None:
                continue
            tr = run_backtest(
                fr["m1"], sym,
                sl_pips_min=args.sl_min,
                sl_pips_max=args.sl_max,
                rr=args.rr,
                time_stop_bars=20,
                be_pips=args.be_pips,
                sessions=args.sessions,
                spread_pips=args.spread_pips,
            )
            s = summarize(tr, f"{sym}|1min|{args.sessions}")
            rows.append(s)
            print(
                f"  {sym:7s} 1min       n={s['n']:5d}  win={_fmt(s['win_rate'], '.1f')}%  "
                f"avgR={_fmt(s['avg_R'], '+.3f')}  PF={_fmt(s['profit_factor'], '.2f')}  "
                f"totalR={_fmt(s['total_R'], '+.1f')}  pips={_fmt(s['total_pips'], '+.1f')}  "
                f"TP={s['target_hits']} SL={s['stop_hits']} BE={s['be_hits']} time={s['time_exits']}",
                flush=True,
            )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)

    if args.trades_out and all_trades:
        tdf = pd.DataFrame([{
            "symbol": t.symbol,
            "side": "long" if t.side > 0 else "short",
            "entry_time": t.entry_time,
            "exit_time": t.exit_time,
            "entry": t.entry,
            "stop_init": t.stop_init,
            "target": t.target,
            "exit": t.exit,
            "reason": t.reason,
            "r_multiple": t.r_multiple,
            "pnl_pips": t.pnl_pips,
            "risk_pips": t.risk_pips,
            "bars_held": t.bars_held,
        } for t in all_trades])
        tdf.to_csv(args.trades_out, index=False)
        print(f"\nTrades -> {args.trades_out}")

    print("\n" + "=" * 78)
    print("HOW TO READ: win rate alone is not enough.")
    print("  Need avg_R > 0 and profit_factor > 1.0 for edge after costs.")
    print(f"Report -> {args.out}")
    print("Done.")


if __name__ == "__main__":
    main()
