"""Profit-first search for Momentum Vector M.

User rule: any rule changes OK if the result is profitable.
Sweep TF pairs, entry/exit, trail, HTF strength, sides, symbols.
Report only configs that clear profit gates, then promote best.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from itertools import product

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ROOT)

from data_io.loader import TF_DELTA, resample  # noqa: E402
from features.momentum_vector import build_mtf_momentum  # noqa: E402
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

OUT = os.path.join(_ROOT, "artifacts", "momentum_vector")


@dataclass
class Cfg:
    name: str
    base: str = "15min"
    htf: str = "1h"
    entry_thr: float = 18.0
    exit_mode: str = "zero"  # four | zero | neg | trail | hard
    exit_thr: float = 0.0  # for four/neg modes
    htf_m_min: float = 0.0  # require |htf M| >= this
    sl_atr: float = 1.5
    tp_atr: float = 0.0  # 0 = off
    trail_atr: float = 0.0
    be_at_r: float = 0.0
    sides: str = "both"  # both | long | short
    session: str = "all"  # all | london_ny


def in_sess(ts, session: str) -> bool:
    if session == "all":
        return True
    h = int(ts.hour)
    return (8 <= h < 17) or (13 <= h < 22)


def run_cfg(
    df: pd.DataFrame,
    cfg: Cfg,
    *,
    init: float = 100_000.0,
    risk_pct: float = 0.01,
    spread_pips: float = 0.8,
    commission_pips: float = 0.4,
    pip: float = 0.0001,
) -> dict:
    M = df["M"].to_numpy(float)
    Mp = np.roll(M, 1)
    Mp[0] = np.nan
    hM = df["htf_M"].to_numpy(float)
    hD = df["htf_direction"].to_numpy(float)
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    a = atr_ind(df, 14).to_numpy(float)
    idx = df.index

    htf_ok_long = (hD > 0) & (hM > cfg.htf_m_min)
    htf_ok_short = (hD < 0) & (hM < -cfg.htf_m_min)
    long_lv = (M > cfg.entry_thr) & htf_ok_long
    short_lv = (M < -cfg.entry_thr) & htf_ok_short
    long_e = long_lv & (Mp <= cfg.entry_thr)
    short_e = short_lv & (Mp >= -cfg.entry_thr)

    if cfg.session != "all":
        sess = np.array([in_sess(t, cfg.session) for t in idx])
        long_e = long_e & sess
        short_e = short_e & sess

    if cfg.sides == "long":
        short_e = np.zeros_like(short_e)
    elif cfg.sides == "short":
        long_e = np.zeros_like(long_e)

    long_e = np.nan_to_num(long_e.astype(float), nan=0.0).astype(bool)
    short_e = np.nan_to_num(short_e.astype(float), nan=0.0).astype(bool)
    long_lv = np.nan_to_num(long_lv.astype(float), nan=0.0).astype(bool)
    short_lv = np.nan_to_num(short_lv.astype(float), nan=0.0).astype(bool)

    cost = (spread_pips + commission_pips) * pip
    equity = float(init)
    floor = init * 0.2
    eq = np.full(len(df), np.nan)
    pnls = []
    holds = []
    wins = losses = 0

    side = 0
    entry = stop = target = risk_dist = qty = 0.0
    entry_i = -1
    pending = 0
    pending_atr = 0.0
    be_armed = False

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
                    stop = fill - cfg.sl_atr * aa
                    target = fill + cfg.tp_atr * aa if cfg.tp_atr > 0 else np.nan
                else:
                    fill = fill - 0.5 * cost
                    stop = fill + cfg.sl_atr * aa
                    target = fill - cfg.tp_atr * aa if cfg.tp_atr > 0 else np.nan
                risk_dist = abs(fill - stop)
                if risk_dist > 0:
                    qty = (equity * risk_pct) / risk_dist
                    side = pending
                    entry = fill
                    entry_i = i
                    be_armed = False
            pending = 0

        if side != 0:
            aa = a[i] if np.isfinite(a[i]) and a[i] > 0 else risk_dist / max(cfg.sl_atr, 1e-9)
            # BE / trail
            if cfg.be_at_r > 0 and risk_dist > 0:
                fav = (c[i] - entry) * side
                if fav >= cfg.be_at_r * risk_dist:
                    be_armed = True
                    stop = max(stop, entry) if side > 0 else min(stop, entry)
            if cfg.trail_atr > 0 and (be_armed or cfg.be_at_r <= 0):
                if side > 0:
                    stop = max(stop, c[i] - cfg.trail_atr * aa)
                else:
                    stop = min(stop, c[i] + cfg.trail_atr * aa)

            exit_px = None
            held = i - entry_i
            if side > 0:
                if l[i] <= stop:
                    exit_px = stop
                elif cfg.tp_atr > 0 and np.isfinite(target) and h[i] >= target:
                    exit_px = target
                else:
                    if cfg.exit_mode == "four":
                        if np.isfinite(M[i]) and M[i] < cfg.exit_thr:
                            exit_px = c[i] - 0.5 * cost
                    elif cfg.exit_mode == "zero":
                        if np.isfinite(M[i]) and M[i] < 0:
                            exit_px = c[i] - 0.5 * cost
                    elif cfg.exit_mode == "neg":
                        if np.isfinite(M[i]) and M[i] < -abs(cfg.exit_thr):
                            exit_px = c[i] - 0.5 * cost
                    elif cfg.exit_mode == "hard":
                        pass  # SL/TP only
                    elif cfg.exit_mode == "trail":
                        pass  # trail handles
                    if short_lv[i] and exit_px is None:
                        exit_px = c[i] - 0.5 * cost
            else:
                if h[i] >= stop:
                    exit_px = stop
                elif cfg.tp_atr > 0 and np.isfinite(target) and l[i] <= target:
                    exit_px = target
                else:
                    if cfg.exit_mode == "four":
                        if np.isfinite(M[i]) and M[i] > -cfg.exit_thr:
                            exit_px = c[i] + 0.5 * cost
                    elif cfg.exit_mode == "zero":
                        if np.isfinite(M[i]) and M[i] > 0:
                            exit_px = c[i] + 0.5 * cost
                    elif cfg.exit_mode == "neg":
                        if np.isfinite(M[i]) and M[i] > abs(cfg.exit_thr):
                            exit_px = c[i] + 0.5 * cost
                    if long_lv[i] and exit_px is None:
                        exit_px = c[i] + 0.5 * cost

            if exit_px is not None:
                pnl = qty * (exit_px - entry) * side
                equity += pnl
                pnls.append(pnl)
                holds.append(held)
                if pnl > 0:
                    wins += 1
                else:
                    losses += 1
                side = 0
                qty = 0.0

        if side == 0 and pending == 0:
            if long_e[i] and np.isfinite(a[i]) and a[i] > 0:
                pending, pending_atr = 1, a[i]
            elif short_e[i] and np.isfinite(a[i]) and a[i] > 0:
                pending, pending_atr = -1, a[i]
        eq[i] = equity

    if side != 0:
        pnl = qty * (c[-1] - entry) * side
        equity += pnl
        pnls.append(pnl)
        holds.append(len(df) - 1 - entry_i)
        if pnl > 0:
            wins += 1
        else:
            losses += 1
        eq[-1] = equity

    eq_s = pd.Series(eq, index=idx).ffill().fillna(init)
    final = float(eq_s.iloc[-1])
    ret = (final / init - 1.0) * 100.0
    dd = float((eq_s / eq_s.cummax() - 1.0).min() * 100.0)
    n = len(pnls)
    if n == 0:
        return dict(n=0, ret=ret, pf=0.0, wr=0.0, dd=dd, sharpe=np.nan, avg_bars=0.0, score=-999)

    arr = np.array(pnls)
    gw, gl = arr[arr > 0].sum(), -arr[arr <= 0].sum()
    pf = (gw / gl) if gl > 0 else (10.0 if gw > 0 else 0.0)
    wr = 100.0 * wins / n
    try:
        d = eq_s.resample("1D").last().dropna().pct_change().dropna()
        sh = float(d.mean() / d.std() * np.sqrt(252)) if len(d) > 5 and d.std() > 0 else 0.0
    except Exception:
        sh = 0.0
    avg_bars = float(np.mean(holds)) if holds else 0.0

    # profit-first score
    score = (
        0.04 * min(ret, 150)
        + 2.5 * min(pf, 3.0)
        + 1.5 * min(max(sh, -1), 3)
        - 0.04 * abs(dd)
        + 0.002 * min(n, 400)
    )
    if n < 30:
        score -= 15
    if ret <= 0 or pf < 1.05:
        score -= 20

    return dict(
        n=n, ret=ret, pf=pf, wr=wr, dd=dd, sharpe=sh, avg_bars=avg_bars,
        final=final, score=score,
    )


def catalog() -> list[Cfg]:
    """Compact profit-oriented grid (higher TFs first; 5m only as control)."""
    cfgs: list[Cfg] = []
    pairs = [
        ("15min", "1h"),
        ("15min", "4h"),
        ("30min", "4h"),
        ("1h", "1d"),
        ("5min", "1h"),  # original — likely weak, kept for comparison
    ]
    for base, htf in pairs:
        for entry in (15, 20, 25, 35):
            for exit_mode, exit_thr in (("zero", 0.0), ("hard", 0.0), ("neg", 10.0)):
                for htf_min in (0.0, 12.0, 25.0):
                    for sl, tp in ((1.5, 2.5), (1.5, 3.5), (2.0, 4.0)):
                        for sides in ("both", "long"):
                            name = (
                                f"{base}+{htf}_e{entry}_{exit_mode}_hm{int(htf_min)}"
                                f"_s{sl}_t{tp}_{sides}"
                            )
                            cfgs.append(Cfg(
                                name=name, base=base, htf=htf, entry_thr=float(entry),
                                exit_mode=exit_mode, exit_thr=exit_thr,
                                htf_m_min=float(htf_min), sl_atr=sl, tp_atr=tp,
                                sides=sides,
                            ))
    # trail / session extras on best TF region
    for base, htf in (("15min", "4h"), ("30min", "4h"), ("1h", "1d")):
        for entry in (20, 30):
            cfgs.append(Cfg(
                name=f"{base}+{htf}_e{entry}_trail_be",
                base=base, htf=htf, entry_thr=float(entry), exit_mode="trail",
                sl_atr=1.5, tp_atr=0.0, trail_atr=1.2, be_at_r=1.0, htf_m_min=12.0,
            ))
            cfgs.append(Cfg(
                name=f"{base}+{htf}_e{entry}_hard_sess",
                base=base, htf=htf, entry_thr=float(entry), exit_mode="hard",
                sl_atr=1.5, tp_atr=3.0, htf_m_min=15.0, session="london_ny",
            ))
    return cfgs


def main():
    os.makedirs(OUT, exist_ok=True)
    symbols = ["EURUSD", "US30"]
    print("=" * 72)
    print("PROFIT-FIRST MOMENTUM VECTOR SEARCH")
    print("=" * 72)

    all_rows = []
    frame_cache: dict = {}

    for sym in symbols:
        path = resolve_csv(os.path.join(_ROOT, "data"), sym)
        print(f"\nLoad {sym} ...", flush=True)
        m1 = load_m1_fast(path)
        pip = pip_size(sym)
        spread = 2.0 if sym == "US30" else 0.8
        print(f"  M1={len(m1):,}", flush=True)

        # prebuild unique TF frames
        needed = {(c.base, c.htf) for c in catalog()}
        built = {}
        for base, htf in sorted(needed):
            key = (sym, base, htf)
            print(f"  features {base}+{htf} ...", flush=True)
            b = resample(m1, base)
            h = resample(m1, htf)
            fr = build_mtf_momentum(b, h, htf).dropna(subset=["M", "htf_M", "htf_direction"])
            built[key] = fr
            print(f"    bars={len(fr):,}", flush=True)

        # dedupe catalog by name
        seen = set()
        cfgs = []
        for c in catalog():
            if c.name in seen:
                continue
            seen.add(c.name)
            cfgs.append(c)

        print(f"  testing {len(cfgs)} configs ...", flush=True)
        best_local = None
        for i, cfg in enumerate(cfgs):
            fr = built[(sym, cfg.base, cfg.htf)]
            met = run_cfg(fr, cfg, pip=pip, spread_pips=spread)
            row = dict(symbol=sym, **asdict(cfg), **met)
            all_rows.append(row)
            if best_local is None or met["score"] > best_local["score"]:
                best_local = {**row}
            if (i + 1) % 200 == 0:
                print(f"    ... {i+1}/{len(cfgs)} best_ret={best_local['ret']:+.1f}% "
                      f"PF={best_local['pf']:.2f} {best_local['name'][:50]}", flush=True)

        print(f"  BEST {sym}: ret={best_local['ret']:+.1f}% PF={best_local['pf']:.2f} "
              f"n={best_local['n']} Sh={best_local['sharpe']:.2f} | {best_local['name']}", flush=True)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "mv_profit_search.csv"), index=False)

    # gates: profitable
    good = df[(df.ret > 5) & (df.pf >= 1.15) & (df.n >= 40) & (df.sharpe > 0.2)].copy()
    good = good.sort_values("score", ascending=False)

    print("\n" + "=" * 72)
    print(f"PROFITABLE CONFIGS: {len(good)} / {len(df)}")
    print("=" * 72)
    if len(good):
        cols = ["symbol", "base", "htf", "entry_thr", "exit_mode", "htf_m_min", "sl_atr", "tp_atr",
                "sides", "session", "n", "ret", "pf", "wr", "dd", "sharpe", "avg_bars", "score", "name"]
        print(good[cols].head(25).to_string(index=False))
        good.head(50).to_csv(os.path.join(OUT, "mv_profit_winners.csv"), index=False)
    else:
        # relax
        soft = df[(df.ret > 0) & (df.pf >= 1.05) & (df.n >= 25)].sort_values("score", ascending=False)
        print("No strict winners. Soft positive:")
        print(soft.head(20)[["symbol", "name", "n", "ret", "pf", "wr", "dd", "sharpe", "score"]].to_string(index=False))
        soft.head(50).to_csv(os.path.join(OUT, "mv_profit_winners.csv"), index=False)

    # cross-symbol: same structural config profitable on both?
    print("\nCROSS-SYMBOL structural matches (base,htf,entry,exit,htf_min,sl,tp,sides):")
    df["struct"] = (
        df.base + "|" + df.htf + "|" + df.entry_thr.astype(str) + "|" + df.exit_mode + "|"
        + df.htf_m_min.astype(str) + "|" + df.sl_atr.astype(str) + "|" + df.tp_atr.astype(str)
        + "|" + df.sides
    )
    # average score where both symbols exist
    piv = df.pivot_table(index="struct", columns="symbol", values=["ret", "pf", "n", "score", "sharpe"], aggfunc="first")
    if ("ret", "EURUSD") in piv.columns and ("ret", "US30") in piv.columns:
        piv["both_ok"] = (
            (piv[("ret", "EURUSD")] > 0) & (piv[("pf", "EURUSD")] >= 1.1)
            & (piv[("ret", "US30")] > 0) & (piv[("pf", "US30")] >= 1.05)
            & (piv[("n", "EURUSD")] >= 30) & (piv[("n", "US30")] >= 30)
        )
        both = piv[piv["both_ok"]].copy()
        if len(both):
            both["avg_score"] = (both[("score", "EURUSD")] + both[("score", "US30")]) / 2
            both = both.sort_values("avg_score", ascending=False)
            print(both.head(15).to_string())
            both.head(30).to_csv(os.path.join(OUT, "mv_profit_both_symbols.csv"))
        else:
            print("  None profitable on BOTH EURUSD and US30 under gates.")
            # show best average
            piv["avg_ret"] = (piv[("ret", "EURUSD")] + piv[("ret", "US30")]) / 2
            piv["avg_pf"] = (piv[("pf", "EURUSD")] + piv[("pf", "US30")]) / 2
            top = piv.sort_values("avg_ret", ascending=False).head(10)
            print("  Top by avg ret:")
            print(top[[("ret", "EURUSD"), ("ret", "US30"), ("pf", "EURUSD"), ("pf", "US30"), "avg_ret", "avg_pf"]].to_string())

    print(f"\nFull grid -> {os.path.join(OUT, 'mv_profit_search.csv')}")
    print("Done.")


if __name__ == "__main__":
    main()
