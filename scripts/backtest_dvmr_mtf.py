"""Multi-timeframe DVMR strategy backtest.

Four combinations:
  1) 5min  + 30min HTF
  2) 15min + 1h HTF
  3) 30min + 4h HTF
  4) 1h    + 1d HTF

Long:  base DVMR x above 0, HTF DVMR > +0.35, regime in {1,2}
Exit:  base DVMR x below 0 OR base DVMR < -0.5
Short: mirror with HTF < -0.35 and regime == -1
SL=1.5*ATR, TP=2.5*ATR, risk 1% equity, spread+commission

Usage:
  python scripts/backtest_dvmr_mtf.py --symbol EURUSD
  python scripts/backtest_dvmr_mtf.py --symbol EURUSD --max-rows 400000
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

from data_io.loader import TF_DELTA, read_mt5_m1, resample  # noqa: E402
from features.dvmr import (  # noqa: E402
    DEFAULT_M,
    DEFAULT_N,
    build_mtf_feature_frame,
)

# ---- TF combos ----
COMBOS = [
    {"name": "5m+30m", "base": "5min", "htf": "30min"},
    {"name": "15m+1h", "base": "15min", "htf": "1h"},
    {"name": "30m+4h", "base": "30min", "htf": "4h"},
    {"name": "1h+1d", "base": "1h", "htf": "1d"},
]

PIP_SIZE = {
    "EURUSD": 0.0001,
    "GBPUSD": 0.0001,
    "USDJPY": 0.01,
    "XAUUSD": 0.1,
    "US30": 1.0,
}


def pip_size(symbol: str) -> float:
    s = symbol.upper().replace(".", "")
    if s in PIP_SIZE:
        return PIP_SIZE[s]
    if s.endswith("JPY"):
        return 0.01
    if s.startswith("XAU") or s.startswith("GOLD"):
        return 0.1
    return 0.0001


def load_m1_fast(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """Faster MT5 M1 load (tab/C engine) with fallback to project loader."""
    try:
        df = pd.read_csv(path, sep="\t", nrows=max_rows)
        df.columns = [str(c).strip().strip("<>").upper() for c in df.columns]
        if "DATE" not in df.columns or "OPEN" not in df.columns:
            raise ValueError(f"unexpected columns: {list(df.columns)[:8]}")
        ts = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            format="mixed", errors="coerce",
        )
        vol_col = "TICKVOL" if "TICKVOL" in df.columns else ("VOL" if "VOL" in df.columns else None)
        out = pd.DataFrame({
            "open": pd.to_numeric(df["OPEN"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(df["HIGH"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(df["LOW"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(df["CLOSE"], errors="coerce").to_numpy(),
            "vol": (
                pd.to_numeric(df[vol_col], errors="coerce").to_numpy()
                if vol_col else np.ones(len(df))
            ),
        }, index=ts).dropna(subset=["open", "high", "low", "close"])
        out = out[~out.index.isna()].sort_index()
        if len(out) == 0:
            raise ValueError("empty after parse")
        return out
    except Exception as e:
        print(f"  fast load failed ({e}); falling back to read_mt5_m1", flush=True)
        return read_mt5_m1(path, max_rows=max_rows)


def resolve_csv(data_dir: str, symbol: str) -> Optional[str]:
    sym = symbol.upper()
    candidates = [
        f"{sym}_M1_curriculum.csv",
        f"{sym}_M1_full.csv",
        f"{sym}_curriculum_2026.csv",
        f"{sym}_M1_drill.csv",
    ]
    for name in candidates:
        p = os.path.join(data_dir, name)
        if os.path.isfile(p):
            return p
    # fuzzy
    for fn in os.listdir(data_dir):
        if fn.upper().startswith(sym) and fn.lower().endswith(".csv"):
            return os.path.join(data_dir, fn)
    return None


@dataclass
class Trade:
    side: int
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry: float
    exit: float
    stop: float
    target: float
    reason: str
    pnl: float
    r_multiple: float
    bars_held: int
    equity_after: float


def signals(df: pd.DataFrame, *, mode: str = "baseline", htf_thr: float = 0.35) -> pd.DataFrame:
    """Entry/exit flags at bar close (no look-ahead on indicators).

    mode:
      baseline   — original rules (zero-cross exit OR DVMR<-0.5)
      improved   — HTF thr 0.55 + soft exit (no zero-cross kill)
      hard       — HTF thr 0.55 + SL/TP only
      v2         — best EURUSD research: HTF thr 0.50 + soft exit + min 4 bars
      v2_robust  — cross-pair compromise: HTF thr 0.45 + hard SL/TP (1h+1d best)
    """
    d = df["dvmr"]
    d_prev = d.shift(1)
    htf = df["htf_dvmr"]
    reg = df["regime"]

    cross_up = (d_prev <= 0) & (d > 0)
    cross_dn = (d_prev >= 0) & (d < 0)

    if mode == "v2":
        thr = 0.50
    elif mode == "v2_robust":
        thr = 0.45
    elif mode in ("improved", "hard"):
        thr = max(htf_thr if htf_thr != 0.35 else 0.55, 0.55)
    else:
        thr = htf_thr

    long_entry = cross_up & (htf > thr) & reg.isin([1, 2])
    short_entry = cross_dn & (htf < -thr) & (reg == -1)

    if mode in ("hard", "v2_robust"):
        long_exit = pd.Series(False, index=df.index)
        short_exit = pd.Series(False, index=df.index)
    elif mode in ("improved", "v2"):
        long_exit = d < -0.5
        short_exit = d > 0.5
    else:
        long_exit = cross_dn | (d < -0.5)
        short_exit = cross_up | (d > 0.5)

    out = df.copy()
    out["long_entry"] = long_entry.fillna(False)
    out["short_entry"] = short_entry.fillna(False)
    out["long_exit"] = long_exit.fillna(False)
    out["short_exit"] = short_exit.fillna(False)
    return out


def run_backtest(
    df: pd.DataFrame,
    *,
    initial_equity: float = 100_000.0,
    risk_pct: float = 0.01,
    sl_atr: float = 1.5,
    tp_atr: float = 2.5,
    spread_pips: float = 0.8,
    commission_pips: float = 0.4,
    pip: float = 0.0001,
    mode: str = "baseline",
) -> tuple[list[Trade], pd.Series]:
    """Bar-by-bar sim. Signal on close t -> enter at open t+1 (honest fill).
    Intrabar: if SL and TP both touchable, assume SL first (conservative).
    """
    d = signals(df, mode=mode)
    o = d["open"].to_numpy(float)
    h = d["high"].to_numpy(float)
    l = d["low"].to_numpy(float)
    c = d["close"].to_numpy(float)
    atr = d["atr"].to_numpy(float)
    le = d["long_entry"].to_numpy(bool)
    se = d["short_entry"].to_numpy(bool)
    lx = d["long_exit"].to_numpy(bool)
    sx = d["short_exit"].to_numpy(bool)
    idx = d.index
    # v2: don't allow soft signal exit until trade has lived a few bars
    min_bars_signal = 4 if mode == "v2" else 0

    cost_per_unit = (spread_pips + commission_pips) * pip  # adverse on round-trip split on entry

    equity = float(initial_equity)
    equity_curve = np.full(len(d), np.nan, dtype=float)
    trades: list[Trade] = []

    side = 0
    entry = stop = target = risk_dist = 0.0
    qty = 0.0
    entry_i = -1
    pending_side = 0  # fill next open
    pending_atr = 0.0

    for i in range(len(d)):
        # --- fill pending entry at this open ---
        if pending_side != 0 and side == 0 and i > 0:
            fill = o[i]
            a = pending_atr if np.isfinite(pending_atr) and pending_atr > 0 else atr[i - 1]
            if not (np.isfinite(a) and a > 0 and np.isfinite(fill)):
                pending_side = 0
            else:
                # half spread adverse on entry
                if pending_side > 0:
                    fill = fill + 0.5 * cost_per_unit
                    stop = fill - sl_atr * a
                    target = fill + tp_atr * a
                else:
                    fill = fill - 0.5 * cost_per_unit
                    stop = fill + sl_atr * a
                    target = fill - tp_atr * a
                risk_dist = abs(fill - stop)
                if risk_dist <= 0:
                    pending_side = 0
                else:
                    risk_cash = equity * risk_pct
                    qty = risk_cash / risk_dist
                    side = pending_side
                    entry = fill
                    entry_i = i
                    pending_side = 0

        # --- manage open position on this bar ---
        if side != 0:
            hit_sl = hit_tp = False
            exit_px = None
            reason = None
            if side > 0:
                if l[i] <= stop:
                    hit_sl = True
                if h[i] >= target:
                    hit_tp = True
                if hit_sl and hit_tp:
                    exit_px, reason = stop, "sl"  # conservative
                elif hit_sl:
                    exit_px, reason = stop, "sl"
                elif hit_tp:
                    exit_px, reason = target, "tp"
                elif lx[i] and (i - entry_i) >= min_bars_signal:
                    exit_px, reason = c[i] - 0.5 * cost_per_unit, "signal"
            else:
                if h[i] >= stop:
                    hit_sl = True
                if l[i] <= target:
                    hit_tp = True
                if hit_sl and hit_tp:
                    exit_px, reason = stop, "sl"
                elif hit_sl:
                    exit_px, reason = stop, "sl"
                elif hit_tp:
                    exit_px, reason = target, "tp"
                elif sx[i] and (i - entry_i) >= min_bars_signal:
                    exit_px, reason = c[i] + 0.5 * cost_per_unit, "signal"

            if exit_px is not None:
                pnl = qty * (exit_px - entry) * side
                equity += pnl
                r_mult = (exit_px - entry) * side / risk_dist if risk_dist > 0 else 0.0
                trades.append(
                    Trade(
                        side=side,
                        entry_time=idx[entry_i],
                        exit_time=idx[i],
                        entry=entry,
                        exit=exit_px,
                        stop=stop,
                        target=target,
                        reason=reason or "exit",
                        pnl=pnl,
                        r_multiple=r_mult,
                        bars_held=i - entry_i,
                        equity_after=equity,
                    )
                )
                side = 0
                qty = 0.0

        # --- queue new entry at close (fill next open) ---
        if side == 0 and pending_side == 0:
            if le[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending_side = 1
                pending_atr = atr[i]
            elif se[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending_side = -1
                pending_atr = atr[i]

        equity_curve[i] = equity

    # force flat last bar
    if side != 0:
        exit_px = c[-1]
        pnl = qty * (exit_px - entry) * side
        equity += pnl
        r_mult = (exit_px - entry) * side / risk_dist if risk_dist > 0 else 0.0
        trades.append(
            Trade(
                side=side,
                entry_time=idx[entry_i],
                exit_time=idx[-1],
                entry=entry,
                exit=exit_px,
                stop=stop,
                target=target,
                reason="eod",
                pnl=pnl,
                r_multiple=r_mult,
                bars_held=len(d) - 1 - entry_i,
                equity_after=equity,
            )
        )
        equity_curve[-1] = equity

    eq = pd.Series(equity_curve, index=idx, name="equity").ffill()
    if eq.isna().all():
        eq = pd.Series(initial_equity, index=idx, name="equity")
    else:
        eq = eq.fillna(initial_equity)
    return trades, eq


def metrics(
    trades: list[Trade],
    equity: pd.Series,
    initial_equity: float,
    bar_minutes: float,
) -> dict:
    final = float(equity.iloc[-1]) if len(equity) else initial_equity
    total_return = final / initial_equity - 1.0

    # CAGR from calendar span
    t0, t1 = equity.index[0], equity.index[-1]
    years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
    cagr = (final / initial_equity) ** (1.0 / years) - 1.0 if final > 0 else -1.0

    # Max drawdown
    peak = equity.cummax()
    dd = equity / peak - 1.0
    max_dd = float(dd.min()) if len(dd) else 0.0

    # Daily-ish Sharpe from equity returns resampled to 1D
    try:
        daily = equity.resample("1D").last().dropna().pct_change().dropna()
        if len(daily) > 2 and daily.std() > 0:
            sharpe = float(daily.mean() / daily.std() * np.sqrt(252))
        else:
            sharpe = float("nan")
    except Exception:
        sharpe = float("nan")

    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0,
            "total_return_pct": total_return * 100,
            "cagr_pct": cagr * 100,
            "max_dd_pct": max_dd * 100,
            "sharpe": sharpe,
            "win_rate_pct": float("nan"),
            "profit_factor": float("nan"),
            "avg_bars": float("nan"),
            "avg_duration_min": float("nan"),
            "final_equity": final,
        }

    pnls = np.array([t.pnl for t in trades], dtype=float)
    wins = pnls[pnls > 0]
    losses = pnls[pnls <= 0]
    gross_win = wins.sum() if len(wins) else 0.0
    gross_loss = -losses.sum() if len(losses) else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else (np.inf if gross_win > 0 else float("nan"))
    wr = 100.0 * (pnls > 0).mean()
    avg_bars = float(np.mean([t.bars_held for t in trades]))
    avg_dur = avg_bars * bar_minutes

    return {
        "n_trades": n,
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "max_dd_pct": max_dd * 100,
        "sharpe": sharpe,
        "win_rate_pct": wr,
        "profit_factor": float(pf) if np.isfinite(pf) else pf,
        "avg_bars": avg_bars,
        "avg_duration_min": avg_dur,
        "final_equity": final,
    }


BAR_MINUTES = {
    "5min": 5,
    "15min": 15,
    "30min": 30,
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def run_combo(
    m1: pd.DataFrame,
    combo: dict,
    *,
    n: int,
    m: int,
    initial_equity: float,
    spread_pips: float,
    commission_pips: float,
    pip: float,
    mode: str = "baseline",
) -> tuple[dict, list[Trade], pd.Series]:
    base_tf, htf_tf = combo["base"], combo["htf"]
    print(f"\n=== {combo['name']}: base={base_tf}  htf={htf_tf}  mode={mode} ===", flush=True)
    base = resample(m1, base_tf)
    htf = resample(m1, htf_tf)
    print(f"  bars base={len(base):,}  htf={len(htf):,}", flush=True)

    frame = build_mtf_feature_frame(
        base, htf, htf_tf, TF_DELTA[htf_tf], n=n, m=m,
    )
    # need warm-up: drop until base + htf features ready
    frame = frame.dropna(subset=["dvmr", "htf_dvmr", "atr"])
    print(f"  usable bars after warm-up={len(frame):,}", flush=True)

    trades, equity = run_backtest(
        frame,
        initial_equity=initial_equity,
        spread_pips=spread_pips,
        commission_pips=commission_pips,
        pip=pip,
        mode=mode,
    )
    met = metrics(trades, equity, initial_equity, BAR_MINUTES[base_tf])
    met["combo"] = combo["name"]
    met["base_tf"] = base_tf
    met["htf_tf"] = htf_tf
    print(
        f"  trades={met['n_trades']}  ret={met['total_return_pct']:+.1f}%  "
        f"CAGR={met['cagr_pct']:+.1f}%  MaxDD={met['max_dd_pct']:.1f}%  "
        f"Sharpe={met['sharpe']:.2f}  WR={met['win_rate_pct']:.1f}%  "
        f"PF={met['profit_factor']:.2f}",
        flush=True,
    )
    return met, trades, equity


def plot_equity_dd(
    curves: dict[str, pd.Series],
    out_path: str,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    for name, eq in curves.items():
        # normalize to 100
        norm = eq / float(eq.iloc[0]) * 100.0
        ax0.plot(norm.index, norm.values, label=name, linewidth=1.2)
        peak = norm.cummax()
        dd = (norm / peak - 1.0) * 100.0
        ax1.plot(dd.index, dd.values, label=name, linewidth=1.0)
    ax0.set_ylabel("Equity (start=100)")
    ax0.set_title(title)
    ax0.legend(loc="best", fontsize=9)
    ax0.grid(True, alpha=0.3)
    ax1.set_ylabel("Drawdown %")
    ax1.set_xlabel("Time")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Plot -> {out_path}", flush=True)


def rank_for_rl(rows: list[dict]) -> list[str]:
    """Simple score: PF, Sharpe, return, low DD, enough trades."""
    lines = []
    scored = []
    for r in rows:
        n = r["n_trades"]
        pf = r["profit_factor"] if np.isfinite(r["profit_factor"]) else 0.0
        sh = r["sharpe"] if np.isfinite(r["sharpe"]) else 0.0
        ret = r["total_return_pct"]
        dd = abs(r["max_dd_pct"])
        # hard filters for RL agent candidacy
        ok = (n >= 40) and (pf >= 1.15) and (sh >= 0.5) and (ret > 0) and (dd < 35)
        score = (
            2.0 * min(pf, 3.0)
            + 1.5 * min(max(sh, -1), 3.0)
            + 0.01 * min(ret, 200)
            - 0.03 * dd
            + 0.002 * min(n, 500)
        )
        scored.append((score, ok, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    lines.append("RANKING (best first)")
    for i, (score, ok, r) in enumerate(scored, 1):
        flag = "PASS -> RL signal-agent candidate" if ok else "FAIL gates (not ready as standalone agent)"
        lines.append(
            f"  {i}. {r['combo']:8s}  score={score:+.2f}  PF={r['profit_factor']:.2f}  "
            f"Sharpe={r['sharpe']:.2f}  ret={r['total_return_pct']:+.1f}%  "
            f"MaxDD={r['max_dd_pct']:.1f}%  n={r['n_trades']}  | {flag}"
        )
    strong = [r["combo"] for _, ok, r in scored if ok]
    if strong:
        lines.append(
            f"\nRECOMMENDATION: Promote to independent RL agents: {', '.join(strong)}. "
            "Use dvmr, dvmr_star, regime, d_dvmr, atr, htf_dvmr as obs channels."
        )
    else:
        best = scored[0][2] if scored else None
        if best:
            lines.append(
                f"\nRECOMMENDATION: None clear the agent gates yet. Best research lead: "
                f"{best['combo']} (tune thresholds / costs / n,m before promoting). "
                "Still export DVMR features into obs for the policy to learn around them."
            )
        else:
            lines.append("\nRECOMMENDATION: No trades generated — check data / params.")
    return lines


def main():
    ap = argparse.ArgumentParser(description="DVMR multi-TF backtest")
    ap.add_argument("--data-dir", default=os.path.join(_ROOT, "data"))
    ap.add_argument("--symbol", default="EURUSD")
    ap.add_argument("--n", type=int, default=DEFAULT_N, help="velocity lookback")
    ap.add_argument("--m", type=int, default=DEFAULT_M, help="momentum lookback (must > n)")
    ap.add_argument("--max-rows", type=int, default=None)
    ap.add_argument("--equity", type=float, default=100_000.0)
    ap.add_argument("--spread-pips", type=float, default=0.8)
    ap.add_argument("--commission-pips", type=float, default=0.4)
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "dvmr"))
    ap.add_argument(
        "--mode", default="baseline",
        choices=["baseline", "improved", "hard", "v2", "v2_robust"],
        help="baseline | improved (HTF0.55 soft) | hard | v2 (EURUSD best) | v2_robust",
    )
    args = ap.parse_args()

    if args.n >= args.m:
        raise SystemExit(f"Require n < m (got n={args.n}, m={args.m})")

    path = resolve_csv(args.data_dir, args.symbol)
    if not path:
        raise SystemExit(f"No CSV for {args.symbol} in {args.data_dir}")

    print("=" * 72)
    print(f"DVMR multi-TF backtest  |  {args.symbol}  n={args.n} m={args.m}  mode={args.mode}")
    print(f"data: {os.path.basename(path)}")
    print(f"cost: spread={args.spread_pips}p + commission={args.commission_pips}p")
    print("=" * 72)

    print("Loading M1 ...", flush=True)
    m1 = load_m1_fast(path, max_rows=args.max_rows)
    print(f"M1 bars={len(m1):,}  {m1.index[0]} -> {m1.index[-1]}", flush=True)
    pip = pip_size(args.symbol)

    rows = []
    curves: dict[str, pd.Series] = {}
    all_trades = []

    for combo in COMBOS:
        met, trades, equity = run_combo(
            m1,
            combo,
            n=args.n,
            m=args.m,
            initial_equity=args.equity,
            spread_pips=args.spread_pips,
            commission_pips=args.commission_pips,
            pip=pip,
            mode=args.mode,
        )
        rows.append(met)
        curves[combo["name"]] = equity
        for t in trades:
            all_trades.append({
                "combo": combo["name"],
                "side": "long" if t.side > 0 else "short",
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry": t.entry,
                "exit": t.exit,
                "reason": t.reason,
                "pnl": t.pnl,
                "r_multiple": t.r_multiple,
                "bars_held": t.bars_held,
            })

    os.makedirs(args.out_dir, exist_ok=True)
    report = pd.DataFrame(rows)
    # nice column order
    cols = [
        "combo", "base_tf", "htf_tf", "n_trades", "total_return_pct", "cagr_pct",
        "max_dd_pct", "sharpe", "win_rate_pct", "profit_factor",
        "avg_bars", "avg_duration_min", "final_equity",
    ]
    report = report[[c for c in cols if c in report.columns]]
    report_path = os.path.join(args.out_dir, f"dvmr_mtf_report_{args.symbol}.csv")
    report.to_csv(report_path, index=False)

    trades_path = os.path.join(args.out_dir, f"dvmr_mtf_trades_{args.symbol}.csv")
    if all_trades:
        pd.DataFrame(all_trades).to_csv(trades_path, index=False)

    plot_path = os.path.join(args.out_dir, f"dvmr_mtf_equity_{args.symbol}.png")
    plot_equity_dd(
        curves,
        plot_path,
        title=f"DVMR multi-TF equity & drawdown — {args.symbol} (n={args.n}, m={args.m})",
    )

    print("\n" + "=" * 72)
    print("SIDE-BY-SIDE COMPARISON")
    print("=" * 72)
    with pd.option_context("display.float_format", lambda x: f"{x: .3f}", "display.width", 140):
        print(report.to_string(index=False))

    print("\n" + "\n".join(rank_for_rl(rows)))
    print(f"\nReport -> {report_path}")
    if all_trades:
        print(f"Trades -> {trades_path}")
    print(f"Plot   -> {plot_path}")
    print("Done.")


if __name__ == "__main__":
    main()
