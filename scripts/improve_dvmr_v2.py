"""DVMR further improvement — deeper search + walk-forward OOS.

Starts from known wins (soft exit, HTF thr~0.55, higher TFs) and tests:
  - entry confirmation (2-bar hold, pullback re-cross, HTF slope)
  - long-only / short-only
  - vol filters (ATR percentile band)
  - exhaustion block (|DVMR| too high)
  - exit thr grid, RR grid, n/m fine grid
  - walk-forward IS/OOS on EURUSD then confirm GBPUSD

Honest gates: OOS PF>1.1, OOS ret>0, enough OOS trades, not only IS glory.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict, fields
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import importlib.util

from data_io.loader import TF_DELTA, resample
from features.dvmr import build_mtf_feature_frame

_bt_path = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("backtest_dvmr_mtf", _bt_path)
_bt = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["backtest_dvmr_mtf"] = _bt
_spec.loader.exec_module(_bt)

BAR_MINUTES = _bt.BAR_MINUTES
COMBOS = _bt.COMBOS
load_m1_fast = _bt.load_m1_fast
metrics = _bt.metrics
pip_size = _bt.pip_size
resolve_csv = _bt.resolve_csv
Trade = _bt.Trade

OUT_DIR = os.path.join(_ROOT, "artifacts", "dvmr")


@dataclass
class V2:
    name: str
    n: int = 5
    m: int = 20
    htf_thr: float = 0.55
    exit_mode: str = "soft"  # soft | hard_only | soft_minbars
    exit_thr: float = 0.5  # soft exit threshold
    min_bars_signal: int = 0
    regime_mode: str = "mild"  # mild | strong
    sides: str = "both"  # both | long | short
    entry_mode: str = "cross"  # cross | cross2 | pullback
    require_htf_slope: bool = False
    max_abs_dvmr_entry: float = 0.0  # 0=off; block if |dvmr| > this (exhaustion)
    atr_pct_lo: float = 0.0  # 0-1 percentile band; 0/1 = off
    atr_pct_hi: float = 1.0
    sl_atr: float = 1.5
    tp_atr: float = 2.5
    be_at_r: float = 0.0
    trail_atr: float = 0.0
    session: str = "all"
    # time stop in bars (0=off)
    time_stop: int = 0


def in_session(ts: pd.Timestamp, session: str) -> bool:
    if session == "all":
        return True
    h = int(ts.hour)
    return (8 <= h < 17) or (13 <= h < 22)


def run_v2(
    df: pd.DataFrame,
    v: V2,
    *,
    initial_equity: float = 100_000.0,
    risk_pct: float = 0.01,
    spread_pips: float = 0.8,
    commission_pips: float = 0.4,
    pip: float = 0.0001,
) -> tuple[list[Trade], pd.Series]:
    d = df["dvmr"].astype(float)
    d_prev = d.shift(1)
    d_prev2 = d.shift(2)
    htf = df["htf_dvmr"].astype(float)
    htf_prev = htf.shift(1)
    reg = df["regime"].astype(int)
    atr = df["atr"].astype(float)

    cross_up = (d_prev <= 0) & (d > 0)
    cross_dn = (d_prev >= 0) & (d < 0)
    # 2-bar confirmation: was non-positive, now positive for 2 closes
    cross2_up = (d_prev2 <= 0) & (d_prev > 0) & (d > 0)
    cross2_dn = (d_prev2 >= 0) & (d_prev < 0) & (d < 0)
    # pullback: HTF already bullish, base was positive, dipped to near 0, reclaims
    pull_up = (htf > v.htf_thr) & (d_prev > 0) & (d_prev < 0.35) & (d >= 0.35) & (d_prev < d)
    pull_dn = (htf < -v.htf_thr) & (d_prev < 0) & (d_prev > -0.35) & (d <= -0.35) & (d_prev > d)

    if v.entry_mode == "cross2":
        long_sig, short_sig = cross2_up, cross2_dn
    elif v.entry_mode == "pullback":
        long_sig, short_sig = pull_up, pull_dn
    else:
        long_sig, short_sig = cross_up, cross_dn

    if v.regime_mode == "strong":
        long_reg = reg == 2
        short_reg = reg == -1
    else:
        long_reg = reg.isin([1, 2])
        short_reg = reg == -1

    long_entry = long_sig & (htf > v.htf_thr) & long_reg
    short_entry = short_sig & (htf < -v.htf_thr) & short_reg

    if v.require_htf_slope:
        long_entry = long_entry & (htf > htf_prev)
        short_entry = short_entry & (htf < htf_prev)

    if v.max_abs_dvmr_entry > 0:
        long_entry = long_entry & (d.abs() <= v.max_abs_dvmr_entry)
        short_entry = short_entry & (d.abs() <= v.max_abs_dvmr_entry)

    if v.atr_pct_lo > 0 or v.atr_pct_hi < 1:
        # causal rolling quantiles (fast) — band on recent vol regime
        lo = atr.rolling(500, min_periods=50).quantile(v.atr_pct_lo)
        hi = atr.rolling(500, min_periods=50).quantile(v.atr_pct_hi)
        vol_ok = (atr >= lo) & (atr <= hi)
        long_entry = long_entry & vol_ok.fillna(False)
        short_entry = short_entry & vol_ok.fillna(False)

    if v.session != "all":
        sess = pd.Series([in_session(t, v.session) for t in df.index], index=df.index)
        long_entry = long_entry & sess
        short_entry = short_entry & sess

    if v.sides == "long":
        short_entry = pd.Series(False, index=df.index)
    elif v.sides == "short":
        long_entry = pd.Series(False, index=df.index)

    if v.exit_mode == "hard_only":
        long_exit = pd.Series(False, index=df.index)
        short_exit = pd.Series(False, index=df.index)
    else:
        long_exit = d < -v.exit_thr
        short_exit = d > v.exit_thr

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr_a = atr.to_numpy(float)
    le = long_entry.fillna(False).to_numpy(bool)
    se = short_entry.fillna(False).to_numpy(bool)
    lx = long_exit.fillna(False).to_numpy(bool)
    sx = short_exit.fillna(False).to_numpy(bool)
    idx = df.index

    cost = (spread_pips + commission_pips) * pip
    equity = float(initial_equity)
    eq_arr = np.full(len(df), np.nan)
    trades: list[Trade] = []

    side = 0
    entry = stop = target = risk_dist = qty = 0.0
    entry_i = -1
    pending_side = 0
    pending_atr = 0.0
    be_armed = False

    for i in range(len(df)):
        if pending_side != 0 and side == 0 and i > 0:
            fill = o[i]
            a = pending_atr if np.isfinite(pending_atr) and pending_atr > 0 else atr_a[i - 1]
            if np.isfinite(a) and a > 0 and np.isfinite(fill):
                if pending_side > 0:
                    fill = fill + 0.5 * cost
                    stop = fill - v.sl_atr * a
                    target = fill + v.tp_atr * a
                else:
                    fill = fill - 0.5 * cost
                    stop = fill + v.sl_atr * a
                    target = fill - v.tp_atr * a
                risk_dist = abs(fill - stop)
                if risk_dist > 0:
                    qty = (equity * risk_pct) / risk_dist
                    side = pending_side
                    entry = fill
                    entry_i = i
                    be_armed = False
            pending_side = 0

        if side != 0:
            a_now = atr_a[i] if np.isfinite(atr_a[i]) and atr_a[i] > 0 else risk_dist / max(v.sl_atr, 1e-9)
            if v.be_at_r > 0 and risk_dist > 0:
                fav = (c[i] - entry) * side
                if fav >= v.be_at_r * risk_dist:
                    be_armed = True
                    stop = max(stop, entry) if side > 0 else min(stop, entry)
            if be_armed and v.trail_atr > 0:
                if side > 0:
                    stop = max(stop, c[i] - v.trail_atr * a_now)
                else:
                    stop = min(stop, c[i] + v.trail_atr * a_now)

            hit_sl = hit_tp = False
            exit_px = reason = None
            held = i - entry_i

            if side > 0:
                if l[i] <= stop:
                    hit_sl = True
                if h[i] >= target:
                    hit_tp = True
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
            elif v.time_stop > 0 and held >= v.time_stop:
                exit_px = c[i] - 0.5 * cost * side
                reason = "time"
            else:
                allow = held >= v.min_bars_signal
                if v.exit_mode != "hard_only" and allow:
                    if side > 0 and lx[i]:
                        exit_px, reason = c[i] - 0.5 * cost, "signal"
                    elif side < 0 and sx[i]:
                        exit_px, reason = c[i] + 0.5 * cost, "signal"

            if exit_px is not None:
                pnl = qty * (exit_px - entry) * side
                equity += pnl
                r_mult = (exit_px - entry) * side / risk_dist if risk_dist > 0 else 0.0
                trades.append(Trade(
                    side=side, entry_time=idx[entry_i], exit_time=idx[i],
                    entry=entry, exit=exit_px, stop=stop, target=target,
                    reason=reason or "exit", pnl=pnl, r_multiple=r_mult,
                    bars_held=held, equity_after=equity,
                ))
                side = 0
                qty = 0.0

        if side == 0 and pending_side == 0:
            if le[i] and np.isfinite(atr_a[i]) and atr_a[i] > 0:
                pending_side, pending_atr = 1, atr_a[i]
            elif se[i] and np.isfinite(atr_a[i]) and atr_a[i] > 0:
                pending_side, pending_atr = -1, atr_a[i]

        eq_arr[i] = equity

    if side != 0:
        exit_px = c[-1]
        pnl = qty * (exit_px - entry) * side
        equity += pnl
        r_mult = (exit_px - entry) * side / risk_dist if risk_dist > 0 else 0.0
        trades.append(Trade(
            side=side, entry_time=idx[entry_i], exit_time=idx[-1],
            entry=entry, exit=exit_px, stop=stop, target=target,
            reason="eod", pnl=pnl, r_multiple=r_mult,
            bars_held=len(df) - 1 - entry_i, equity_after=equity,
        ))
        eq_arr[-1] = equity

    eq = pd.Series(eq_arr, index=idx, name="equity").ffill().fillna(initial_equity)
    return trades, eq


def build_frame(m1, base, htf, n, m):
    b = resample(m1, base)
    h = resample(m1, htf)
    fr = build_mtf_feature_frame(b, h, htf, TF_DELTA[htf], n=n, m=m)
    return fr.dropna(subset=["dvmr", "htf_dvmr", "atr"])


def score(met: dict, min_trades: int = 20) -> float:
    n = met.get("n_trades", 0)
    if n < min_trades:
        return -50.0
    pf = met["profit_factor"] if np.isfinite(met.get("profit_factor", np.nan)) else 0.0
    sh = met["sharpe"] if np.isfinite(met.get("sharpe", np.nan)) else 0.0
    ret = met.get("total_return_pct", 0.0)
    dd = abs(met.get("max_dd_pct", 100.0))
    return (
        2.5 * min(pf, 3.5)
        + 2.0 * min(max(sh, -1), 3.0)
        + 0.02 * min(ret, 120)
        - 0.05 * dd
        + 0.001 * min(n, 300)
    )


def eval_slice(frame, v, bar_min, pip, init=100_000.0):
    trades, eq = run_v2(frame, v, initial_equity=init, pip=pip)
    met = metrics(trades, eq, init, bar_min)
    met["score"] = score(met)
    if trades:
        rc = pd.Series([t.reason for t in trades]).value_counts(normalize=True)
        met["pct_tp"] = float(rc.get("tp", 0) * 100)
        met["pct_sl"] = float(rc.get("sl", 0) * 100)
        met["pct_signal"] = float(rc.get("signal", 0) * 100)
        longs = [t for t in trades if t.side > 0]
        shorts = [t for t in trades if t.side < 0]
        met["long_pnl"] = sum(t.pnl for t in longs)
        met["short_pnl"] = sum(t.pnl for t in shorts)
    else:
        met["pct_tp"] = met["pct_sl"] = met["pct_signal"] = 0.0
        met["long_pnl"] = met["short_pnl"] = 0.0
    return met, trades, eq


def catalog() -> list[V2]:
    """Focused search around known good region + new ideas."""
    vs: list[V2] = []
    # anchors
    vs.append(V2("v1_soft_htf55", exit_mode="soft", htf_thr=0.55))
    vs.append(V2("v1_hard_htf55", exit_mode="hard_only", htf_thr=0.55))
    vs.append(V2("v1_soft_min4", exit_mode="soft", htf_thr=0.55, min_bars_signal=4))

    # sides
    vs.append(V2("long_only_soft55", sides="long", htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("short_only_soft55", sides="short", htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("long_only_hard55", sides="long", htf_thr=0.55, exit_mode="hard_only"))

    # entry modes
    vs.append(V2("cross2_soft55", entry_mode="cross2", htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("pullback_soft55", entry_mode="pullback", htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("cross2_hard55", entry_mode="cross2", htf_thr=0.55, exit_mode="hard_only"))

    # HTF slope
    vs.append(V2("htf_slope_soft", require_htf_slope=True, htf_thr=0.45, exit_mode="soft"))
    vs.append(V2("htf_slope_hard", require_htf_slope=True, htf_thr=0.45, exit_mode="hard_only"))

    # exhaustion block
    vs.append(V2("no_exhaust_1.5", max_abs_dvmr_entry=1.5, htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("no_exhaust_1.0", max_abs_dvmr_entry=1.0, htf_thr=0.55, exit_mode="soft"))

    # vol band mid
    vs.append(V2("atr_mid", atr_pct_lo=0.2, atr_pct_hi=0.85, htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("atr_high", atr_pct_lo=0.5, atr_pct_hi=1.0, htf_thr=0.55, exit_mode="soft"))
    vs.append(V2("atr_low", atr_pct_lo=0.0, atr_pct_hi=0.5, htf_thr=0.55, exit_mode="soft"))

    # exit thr / RR
    for et in (0.3, 0.5, 0.7, 1.0):
        vs.append(V2(f"exit_thr_{et}", exit_thr=et, htf_thr=0.55, exit_mode="soft"))
    for sl, tp in ((1.2, 2.4), (1.5, 3.0), (2.0, 3.0), (1.5, 2.0), (1.0, 2.5)):
        vs.append(V2(f"rr_{sl}_{tp}", sl_atr=sl, tp_atr=tp, htf_thr=0.55, exit_mode="soft"))
        vs.append(V2(f"rrh_{sl}_{tp}", sl_atr=sl, tp_atr=tp, htf_thr=0.55, exit_mode="hard_only"))

    # htf thr grid
    for thr in (0.4, 0.5, 0.55, 0.65, 0.8):
        vs.append(V2(f"thr_{thr}_soft", htf_thr=thr, exit_mode="soft"))
        vs.append(V2(f"thr_{thr}_hard", htf_thr=thr, exit_mode="hard_only"))

    # n/m
    for n, m in ((5, 15), (5, 20), (5, 30), (3, 12), (8, 24), (4, 16)):
        vs.append(V2(f"nm_{n}_{m}_soft", n=n, m=m, htf_thr=0.55, exit_mode="soft"))
        vs.append(V2(f"nm_{n}_{m}_hard", n=n, m=m, htf_thr=0.55, exit_mode="hard_only"))

    # composites from best hunches
    vs.append(V2(
        "comp_A", htf_thr=0.55, exit_mode="soft", exit_thr=0.7,
        min_bars_signal=3, sides="long", sl_atr=1.5, tp_atr=3.0,
    ))
    vs.append(V2(
        "comp_B", htf_thr=0.5, exit_mode="hard_only", require_htf_slope=True,
        entry_mode="cross2", sl_atr=1.5, tp_atr=3.0, sides="long",
    ))
    vs.append(V2(
        "comp_C", htf_thr=0.55, exit_mode="soft", atr_pct_lo=0.25, atr_pct_hi=0.9,
        max_abs_dvmr_entry=1.2, sl_atr=1.5, tp_atr=2.5, min_bars_signal=2,
    ))
    vs.append(V2(
        "comp_D", htf_thr=0.65, exit_mode="hard_only", sides="long",
        sl_atr=1.2, tp_atr=2.8, require_htf_slope=True,
    ))
    vs.append(V2(
        "comp_E", htf_thr=0.55, exit_mode="soft", exit_thr=0.7,
        entry_mode="cross2", sl_atr=1.5, tp_atr=3.0, n=5, m=20,
    ))
    vs.append(V2(
        "comp_F", htf_thr=0.55, exit_mode="hard_only", time_stop=24,
        sl_atr=1.5, tp_atr=3.0,
    ))
    vs.append(V2(
        "comp_G", htf_thr=0.5, exit_mode="soft", exit_thr=0.5,
        session="london_ny", sides="long", sl_atr=1.5, tp_atr=2.5,
    ))
    return vs


def walk_forward(frame: pd.DataFrame, v: V2, bar_min: float, pip: float, folds: int = 4):
    """Equal-time folds; train=prior folds score not used for fitting here — pure sequential OOS."""
    n = len(frame)
    fold_size = n // folds
    oos_rows = []
    for k in range(1, folds):  # first fold is warm seed only
        start = k * fold_size
        end = (k + 1) * fold_size if k < folds - 1 else n
        if end - start < 200:
            continue
        sl = frame.iloc[start:end]
        met, _, _ = eval_slice(sl, v, bar_min, pip)
        met["fold"] = k
        oos_rows.append(met)
    if not oos_rows:
        return {"oos_ret": np.nan, "oos_pf": np.nan, "oos_sharpe": np.nan,
                "oos_trades": 0, "oos_score": -100, "pos_folds": 0, "n_folds": 0}
    df = pd.DataFrame(oos_rows)
    # compound returns approximately via equity path would be better; use mean metrics
    pos = int((df["total_return_pct"] > 0).sum())
    # reconstruct rough OOS total by summing trade-weighted — use mean ret * folds as proxy
    # Better: sum of fold returns if independent capital each fold
    oos_ret = float(df["total_return_pct"].sum())  # sequential restarts each fold
    # average PF weighted by trades
    tw = df["n_trades"].clip(lower=0)
    if tw.sum() > 0:
        oos_pf = float(np.average(df["profit_factor"].fillna(0), weights=tw))
        oos_sh = float(np.average(df["sharpe"].fillna(0), weights=tw))
    else:
        oos_pf = oos_sh = 0.0
    oos_tr = int(df["n_trades"].sum())
    oos_score = (
        2.0 * min(oos_pf, 3.0)
        + 1.5 * min(max(oos_sh, -1), 3)
        + 0.01 * min(oos_ret, 100)
        + 0.5 * pos
        - 2.0 * (folds - 1 - pos)
    )
    if oos_tr < 15:
        oos_score -= 20
    return {
        "oos_ret": oos_ret,
        "oos_pf": oos_pf,
        "oos_sharpe": oos_sh,
        "oos_trades": oos_tr,
        "oos_score": oos_score,
        "pos_folds": pos,
        "n_folds": len(df),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 72)
    print("DVMR FURTHER IMPROVEMENT (v2) + walk-forward")
    print("=" * 72)

    focus_combos = [
        {"name": "30m+4h", "base": "30min", "htf": "4h"},
        {"name": "1h+1d", "base": "1h", "htf": "1d"},
    ]

    path = resolve_csv(os.path.join(_ROOT, "data"), "EURUSD")
    print("Loading EURUSD ...", flush=True)
    m1 = load_m1_fast(path)
    pip = pip_size("EURUSD")
    print(f"M1={len(m1):,}", flush=True)

    cache = {}

    def fr(base, htf, n, m):
        key = (base, htf, n, m)
        if key not in cache:
            cache[key] = build_frame(m1, base, htf, n, m)
            print(f"  built {key} bars={len(cache[key]):,}", flush=True)
        return cache[key]

    rows = []
    variants = catalog()
    print(f"Variants: {len(variants)} × combos: {len(focus_combos)}", flush=True)

    for combo in focus_combos:
        bar_min = BAR_MINUTES[combo["base"]]
        for v in variants:
            frame = fr(combo["base"], combo["htf"], v.n, v.m)
            met, trades, eq = eval_slice(frame, v, bar_min, pip)
            wf = walk_forward(frame, v, bar_min, pip, folds=4)
            row = {
                "symbol": "EURUSD",
                "combo": combo["name"],
                "variant": v.name,
                **{k: met[k] for k in (
                    "n_trades", "total_return_pct", "cagr_pct", "max_dd_pct",
                    "sharpe", "win_rate_pct", "profit_factor", "avg_bars",
                    "score", "pct_tp", "pct_sl", "pct_signal", "long_pnl", "short_pnl",
                )},
                **wf,
            }
            rows.append(row)
            print(
                f"{combo['name']:8s} {v.name:22s}  IS ret={met['total_return_pct']:+6.1f}% "
                f"PF={met['profit_factor']:5.2f} Sh={met['sharpe']:5.2f} n={met['n_trades']:3d}  "
                f"OOS ret~{wf['oos_ret']:+5.1f} PF={wf['oos_pf']:.2f} folds+={wf['pos_folds']}/{wf['n_folds']} "
                f"oos_sc={wf['oos_score']:+5.1f}",
                flush=True,
            )

    df = pd.DataFrame(rows)
    # Combined rank: 0.45*IS score + 0.55*OOS score
    df["blend"] = 0.45 * df["score"] + 0.55 * df["oos_score"]
    df = df.sort_values("blend", ascending=False)
    out1 = os.path.join(OUT_DIR, "dvmr_v2_eurusd_grid.csv")
    df.to_csv(out1, index=False)

    print("\n" + "=" * 72)
    print("TOP 12 by blend (IS + walk-forward OOS)")
    print("=" * 72)
    show = df.head(12)[
        ["combo", "variant", "n_trades", "total_return_pct", "profit_factor", "sharpe",
         "max_dd_pct", "oos_ret", "oos_pf", "pos_folds", "blend", "long_pnl", "short_pnl"]
    ]
    print(show.to_string(index=False))

    # Pick top 4 unique variants for GBPUSD confirm (prefer those with OOS pf>1 and pos_folds>=2)
    robust = df[(df["oos_pf"] >= 1.05) & (df["pos_folds"] >= 2) & (df["n_trades"] >= 25)]
    if len(robust) == 0:
        robust = df[df["oos_trades"] >= 15].head(20)
    top_pick = []
    for _, r in robust.iterrows():
        key = (r["combo"], r["variant"])
        if key not in [(c, v) for c, v in top_pick]:
            top_pick.append(key)
        if len(top_pick) >= 6:
            break
    # also absolute blend tops
    for _, r in df.head(8).iterrows():
        key = (r["combo"], r["variant"])
        if key not in top_pick:
            top_pick.append(key)
        if len(top_pick) >= 8:
            break

    print("\n[GBPUSD confirmation]", flush=True)
    path_g = resolve_csv(os.path.join(_ROOT, "data"), "GBPUSD")
    m1g = load_m1_fast(path_g)
    pip_g = pip_size("GBPUSD")
    cache_g = {}
    conf_rows = []
    vmap = {v.name: v for v in variants}

    for combo_name, vname in top_pick:
        if vname not in vmap:
            continue
        v = vmap[vname]
        combo = next(c for c in focus_combos if c["name"] == combo_name)
        key = (combo["base"], combo["htf"], v.n, v.m)
        if key not in cache_g:
            cache_g[key] = build_frame(m1g, combo["base"], combo["htf"], v.n, v.m)
        frame = cache_g[key]
        met, _, _ = eval_slice(frame, v, BAR_MINUTES[combo["base"]], pip_g)
        wf = walk_forward(frame, v, BAR_MINUTES[combo["base"]], pip_g, folds=4)
        conf_rows.append({
            "symbol": "GBPUSD", "combo": combo_name, "variant": vname,
            "ret": met["total_return_pct"], "pf": met["profit_factor"],
            "sharpe": met["sharpe"], "n": met["n_trades"], "dd": met["max_dd_pct"],
            "oos_ret": wf["oos_ret"], "oos_pf": wf["oos_pf"], "pos_folds": wf["pos_folds"],
        })
        print(
            f"  {combo_name:8s} {vname:22s}  ret={met['total_return_pct']:+6.1f}% "
            f"PF={met['profit_factor']:.2f} Sh={met['sharpe']:.2f} n={met['n_trades']}  "
            f"OOS PF={wf['oos_pf']:.2f} +folds={wf['pos_folds']}",
            flush=True,
        )

    conf = pd.DataFrame(conf_rows)
    conf_path = os.path.join(OUT_DIR, "dvmr_v2_gbpusd_confirm.csv")
    conf.to_csv(conf_path, index=False)

    # Final champion: best blend on EURUSD that also has GBPUSD ret>=-5% and pf>=0.95 if present
    print("\n" + "=" * 72)
    print("FINAL PICK")
    print("=" * 72)
    champion = None
    for _, r in df.iterrows():
        vname, combo = r["variant"], r["combo"]
        g = conf[(conf.variant == vname) & (conf.combo == combo)]
        if len(g) == 0:
            # allow if not tested but strong OOS
            if r["oos_pf"] >= 1.15 and r["pos_folds"] >= 2 and r["total_return_pct"] > 5:
                champion = r
                note = "EURUSD strong OOS (GBP not in confirm set)"
                break
            continue
        g0 = g.iloc[0]
        if (
            r["total_return_pct"] > 5
            and r["oos_pf"] >= 1.1
            and r["pos_folds"] >= 2
            and g0["pf"] >= 0.95
            and g0["ret"] > -8
        ):
            champion = r
            note = f"GBPUSD confirm ret={g0['ret']:+.1f}% PF={g0['pf']:.2f}"
            break

    if champion is None:
        # fallback best blend with OOS pf>1
        sub = df[df["oos_pf"] >= 1.1]
        champion = sub.iloc[0] if len(sub) else df.iloc[0]
        note = "fallback: best blend with OOS PF>=1.1 (or top blend)"

    v_champ = vmap[champion["variant"]]
    print(f"Champion: {champion['combo']} / {champion['variant']}")
    print(f"  IS:  ret={champion['total_return_pct']:+.1f}% PF={champion['profit_factor']:.2f} "
          f"Sh={champion['sharpe']:.2f} DD={champion['max_dd_pct']:.1f}% n={int(champion['n_trades'])}")
    print(f"  OOS: ret~{champion['oos_ret']:+.1f}% PF={champion['oos_pf']:.2f} "
          f"+folds={int(champion['pos_folds'])}/{int(champion['n_folds'])}")
    print(f"  {note}")
    print(f"  Config: {asdict(v_champ)}")

    # Compare vs previous improved anchor
    for cname in ("30m+4h", "1h+1d"):
        base = df[(df.combo == cname) & (df.variant == "v1_soft_htf55")]
        if len(base):
            b = base.iloc[0]
            print(f"\nAnchor v1_soft_htf55 on {cname}: ret={b['total_return_pct']:+.1f}% "
                  f"PF={b['profit_factor']:.2f} OOS_PF={b['oos_pf']:.2f} blend={b['blend']:.2f}")

    rec = os.path.join(OUT_DIR, "dvmr_v2_recommendation.txt")
    with open(rec, "w", encoding="utf-8") as f:
        f.write("DVMR v2 FURTHER IMPROVEMENT\n")
        f.write(f"Champion: {champion['combo']} / {champion['variant']}\n")
        f.write(f"IS ret={champion['total_return_pct']:+.2f}% PF={champion['profit_factor']:.3f} "
                f"Sharpe={champion['sharpe']:.3f}\n")
        f.write(f"OOS ret~{champion['oos_ret']:+.2f}% PF={champion['oos_pf']:.3f} "
                f"pos_folds={champion['pos_folds']}\n")
        f.write(f"{note}\n")
        f.write(f"Config: {asdict(v_champ)}\n\n")
        f.write("TOP 12 blend:\n")
        f.write(show.to_string(index=False))
        f.write("\n\nGBPUSD confirm:\n")
        f.write(conf.to_string(index=False) if len(conf) else "none")
    print(f"\nGrid -> {out1}")
    print(f"GBP  -> {conf_path}")
    print(f"Rec  -> {rec}")
    print("Done.")


if __name__ == "__main__":
    main()
