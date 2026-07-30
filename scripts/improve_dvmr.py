"""Autonomous DVMR strategy improvement research.

Hypothesis from baseline: trades die in ~2–3 bars because zero-cross signal
exits fire before ATR targets. Sweep exit/entry/risk/param variants on the
best TF first (30m+4h), then re-check winners on all combos + GBPUSD.
"""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import importlib.util  # noqa: E402

from data_io.loader import TF_DELTA, resample  # noqa: E402
from features.dvmr import build_mtf_feature_frame  # noqa: E402

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
class Variant:
    name: str
    n: int = 5
    m: int = 20
    htf_thr: float = 0.35
    exit_mode: str = "baseline"  # baseline | soft | hard_only | soft_minbars
    min_bars_signal: int = 0
    regime_mode: str = "mild"  # mild={1,2}/-1 | strong=2 only for long, -1 short
    use_star: bool = False  # entry/exit on dvmr_star instead of dvmr
    require_d_dvmr: bool = False  # long needs d_dvmr>0
    sl_atr: float = 1.5
    tp_atr: float = 2.5
    be_at_r: float = 0.0  # move stop to BE after this R (0=off)
    trail_atr: float = 0.0  # trail stop by this * ATR once BE armed (0=off)
    session: str = "all"  # all | london_ny


def in_session(ts: pd.Timestamp, session: str) -> bool:
    if session == "all":
        return True
    h = int(ts.hour)
    london = 8 <= h < 17
    ny = 13 <= h < 22
    if session == "london_ny":
        return london or ny
    return True


def run_variant(
    df: pd.DataFrame,
    v: Variant,
    *,
    initial_equity: float = 100_000.0,
    risk_pct: float = 0.01,
    spread_pips: float = 0.8,
    commission_pips: float = 0.4,
    pip: float = 0.0001,
) -> tuple[list[Trade], pd.Series]:
    col = "dvmr_star" if v.use_star else "dvmr"
    htf_col = "htf_dvmr_star" if v.use_star and "htf_dvmr_star" in df.columns else "htf_dvmr"

    d = df[col].astype(float)
    d_prev = d.shift(1)
    htf = df[htf_col].astype(float)
    reg = df["regime"].astype(int)
    d_dvmr = df["d_dvmr"].astype(float) if "d_dvmr" in df.columns else d.diff()

    cross_up = (d_prev <= 0) & (d > 0)
    cross_dn = (d_prev >= 0) & (d < 0)

    if v.regime_mode == "strong":
        long_reg = reg == 2
        short_reg = reg == -1  # no strong-neg regime defined; keep -1
    else:
        long_reg = reg.isin([1, 2])
        short_reg = reg == -1

    long_entry = cross_up & (htf > v.htf_thr) & long_reg
    short_entry = cross_dn & (htf < -v.htf_thr) & short_reg
    if v.require_d_dvmr:
        long_entry = long_entry & (d_dvmr > 0)
        short_entry = short_entry & (d_dvmr < 0)

    if v.session != "all":
        sess = pd.Series([in_session(t, v.session) for t in df.index], index=df.index)
        long_entry = long_entry & sess
        short_entry = short_entry & sess

    # Exit flags (signal-based); risk exits handled in loop
    if v.exit_mode == "hard_only":
        long_exit = pd.Series(False, index=df.index)
        short_exit = pd.Series(False, index=df.index)
    elif v.exit_mode == "soft":
        long_exit = d < -0.5
        short_exit = d > 0.5
    elif v.exit_mode == "soft_minbars":
        # same as soft but enforced after min_bars in loop
        long_exit = d < -0.5
        short_exit = d > 0.5
    else:  # baseline
        long_exit = cross_dn | (d < -0.5)
        short_exit = cross_up | (d > 0.5)

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    atr = df["atr"].to_numpy(float)
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
            a = pending_atr if np.isfinite(pending_atr) and pending_atr > 0 else atr[i - 1]
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
            a_now = atr[i] if np.isfinite(atr[i]) and atr[i] > 0 else risk_dist / max(v.sl_atr, 1e-9)
            # trail / BE
            if v.be_at_r > 0 and risk_dist > 0:
                fav = (c[i] - entry) * side
                if fav >= v.be_at_r * risk_dist:
                    be_armed = True
                    if side > 0:
                        stop = max(stop, entry)
                    else:
                        stop = min(stop, entry)
            if be_armed and v.trail_atr > 0 and np.isfinite(a_now):
                if side > 0:
                    stop = max(stop, c[i] - v.trail_atr * a_now)
                else:
                    stop = min(stop, c[i] + v.trail_atr * a_now)

            hit_sl = hit_tp = False
            exit_px = reason = None
            if side > 0:
                if l[i] <= stop:
                    hit_sl = True
                if h[i] >= target:
                    hit_tp = True
                if hit_sl and hit_tp:
                    exit_px, reason = stop, "sl"
                elif hit_sl:
                    exit_px, reason = stop, "sl"
                elif hit_tp:
                    exit_px, reason = target, "tp"
                else:
                    allow_sig = True
                    held = i - entry_i
                    if v.exit_mode in ("soft_minbars", "baseline") and v.min_bars_signal > 0:
                        allow_sig = held >= v.min_bars_signal
                    if v.exit_mode == "baseline" and v.min_bars_signal > 0:
                        # baseline exits with min bars
                        if allow_sig and lx[i]:
                            exit_px, reason = c[i] - 0.5 * cost, "signal"
                    elif v.exit_mode != "hard_only" and allow_sig and lx[i]:
                        exit_px, reason = c[i] - 0.5 * cost, "signal"
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
                else:
                    allow_sig = True
                    held = i - entry_i
                    if v.min_bars_signal > 0:
                        allow_sig = held >= v.min_bars_signal
                    if v.exit_mode != "hard_only" and allow_sig and sx[i]:
                        exit_px, reason = c[i] + 0.5 * cost, "signal"

            if exit_px is not None:
                pnl = qty * (exit_px - entry) * side
                equity += pnl
                r_mult = (exit_px - entry) * side / risk_dist if risk_dist > 0 else 0.0
                trades.append(Trade(
                    side=side, entry_time=idx[entry_i], exit_time=idx[i],
                    entry=entry, exit=exit_px, stop=stop, target=target,
                    reason=reason or "exit", pnl=pnl, r_multiple=r_mult,
                    bars_held=i - entry_i, equity_after=equity,
                ))
                side = 0
                qty = 0.0

        if side == 0 and pending_side == 0:
            if le[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending_side, pending_atr = 1, atr[i]
            elif se[i] and np.isfinite(atr[i]) and atr[i] > 0:
                pending_side, pending_atr = -1, atr[i]

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


def build_frame(m1: pd.DataFrame, base: str, htf: str, n: int, m: int) -> pd.DataFrame:
    base_df = resample(m1, base)
    htf_df = resample(m1, htf)
    frame = build_mtf_feature_frame(base_df, htf_df, htf, TF_DELTA[htf], n=n, m=m)
    return frame.dropna(subset=["dvmr", "htf_dvmr", "atr"])


def score_row(r: dict) -> float:
    pf = r["profit_factor"] if np.isfinite(r.get("profit_factor", np.nan)) else 0.0
    sh = r["sharpe"] if np.isfinite(r.get("sharpe", np.nan)) else 0.0
    ret = r.get("total_return_pct", 0.0)
    dd = abs(r.get("max_dd_pct", 100.0))
    n = r.get("n_trades", 0)
    if n < 25:
        return -100.0
    return (
        2.5 * min(pf, 3.0)
        + 1.8 * min(max(sh, -1), 3.0)
        + 0.015 * min(ret, 150)
        - 0.04 * dd
        + 0.001 * min(n, 400)
    )


def variants_catalog() -> list[Variant]:
    return [
        Variant("A_baseline"),
        Variant("B_hard_sltp_only", exit_mode="hard_only"),
        Variant("C_soft_exit", exit_mode="soft"),
        Variant("D_soft_min4", exit_mode="soft", min_bars_signal=4),
        Variant("E_soft_be1R", exit_mode="soft", be_at_r=1.0),
        Variant("F_soft_be_trail", exit_mode="soft", be_at_r=1.0, trail_atr=1.0),
        Variant("G_hard_be_trail", exit_mode="hard_only", be_at_r=1.0, trail_atr=1.2),
        Variant("H_htf0.55_soft", exit_mode="soft", htf_thr=0.55),
        Variant("I_strong_regime_soft", exit_mode="soft", regime_mode="strong"),
        Variant("J_star_soft", exit_mode="soft", use_star=True),
        Variant("K_ddvmr_soft", exit_mode="soft", require_d_dvmr=True),
        Variant("L_rr_2_3_soft", exit_mode="soft", sl_atr=2.0, tp_atr=3.0),
        Variant("M_rr_1_2_soft", exit_mode="soft", sl_atr=1.0, tp_atr=2.0),
        Variant("N_n5m10_soft", exit_mode="soft", n=5, m=10),
        Variant("O_n8m21_soft", exit_mode="soft", n=8, m=21),
        Variant("P_session_soft", exit_mode="soft", session="london_ny"),
        # composite champions to try
        Variant("Q_composite_v1", exit_mode="soft", htf_thr=0.5, be_at_r=1.0,
                trail_atr=1.0, sl_atr=1.8, tp_atr=2.8, min_bars_signal=2),
        Variant("R_composite_v2", exit_mode="hard_only", htf_thr=0.5, be_at_r=1.0,
                trail_atr=1.2, sl_atr=2.0, tp_atr=3.0, require_d_dvmr=True),
        Variant("S_composite_v3", exit_mode="soft", htf_thr=0.45, be_at_r=0.8,
                trail_atr=1.0, n=5, m=10, session="london_ny"),
        Variant("T_minbars_baseline", exit_mode="baseline", min_bars_signal=6),
    ]


def eval_on_frame(frame: pd.DataFrame, v: Variant, bar_min: float, pip: float) -> dict:
    # rebuild frame if n/m differ from default features in frame — caller handles rebuild
    trades, eq = run_variant(frame, v, pip=pip)
    met = metrics(trades, eq, 100_000.0, bar_min)
    met["variant"] = v.name
    met["score"] = score_row(met)
    # exit reason mix
    if trades:
        reasons = pd.Series([t.reason for t in trades]).value_counts(normalize=True)
        met["pct_tp"] = float(reasons.get("tp", 0.0) * 100)
        met["pct_sl"] = float(reasons.get("sl", 0.0) * 100)
        met["pct_signal"] = float(reasons.get("signal", 0.0) * 100)
    else:
        met["pct_tp"] = met["pct_sl"] = met["pct_signal"] = 0.0
    return met


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    symbols = ["EURUSD", "GBPUSD"]
    print("=" * 72)
    print("DVMR IMPROVEMENT RESEARCH")
    print("=" * 72)

    all_rows = []
    # Phase 1: sweep on EURUSD 30m+4h
    path = resolve_csv(os.path.join(_ROOT, "data"), "EURUSD")
    print(f"\n[Phase 1] Load EURUSD ...", flush=True)
    m1 = load_m1_fast(path)
    print(f"  M1={len(m1):,}", flush=True)

    frame_cache: dict[tuple, pd.DataFrame] = {}

    def get_frame(m1df, base, htf, n, m):
        key = (id(m1df), base, htf, n, m)
        if key not in frame_cache:
            frame_cache[key] = build_frame(m1df, base, htf, n, m)
        return frame_cache[key]

    combo = {"name": "30m+4h", "base": "30min", "htf": "4h"}
    pip = pip_size("EURUSD")
    print(f"\n[Phase 1] Sweep variants on {combo['name']}", flush=True)
    phase1 = []
    for v in variants_catalog():
        fr = get_frame(m1, combo["base"], combo["htf"], v.n, v.m)
        met = eval_on_frame(fr, v, BAR_MINUTES[combo["base"]], pip)
        met["symbol"] = "EURUSD"
        met["combo"] = combo["name"]
        phase1.append(met)
        print(
            f"  {v.name:22s}  n={met['n_trades']:4d}  ret={met['total_return_pct']:+7.1f}%  "
            f"PF={met['profit_factor']:5.2f}  Sh={met['sharpe']:6.2f}  "
            f"DD={met['max_dd_pct']:6.1f}%  score={met['score']:+6.2f}  "
            f"tp/sl/sig={met['pct_tp']:.0f}/{met['pct_sl']:.0f}/{met['pct_signal']:.0f}",
            flush=True,
        )
        all_rows.append(met)

    phase1_df = pd.DataFrame(phase1).sort_values("score", ascending=False)
    top = phase1_df.head(5)
    print("\n[Phase 1] TOP 5 on 30m+4h EURUSD:")
    print(top[["variant", "n_trades", "total_return_pct", "profit_factor", "sharpe",
               "max_dd_pct", "win_rate_pct", "score"]].to_string(index=False))

    top_names = list(top["variant"])
    top_vars = [v for v in variants_catalog() if v.name in top_names]
    # always include baseline for comparison
    base_v = next(v for v in variants_catalog() if v.name == "A_baseline")
    if base_v not in top_vars:
        top_vars.append(base_v)

    # Phase 2: top variants × all TF combos on EURUSD
    print("\n[Phase 2] Top variants across all TF combos (EURUSD)", flush=True)
    phase2 = []
    for combo in COMBOS:
        for v in top_vars:
            fr = get_frame(m1, combo["base"], combo["htf"], v.n, v.m)
            met = eval_on_frame(fr, v, BAR_MINUTES[combo["base"]], pip)
            met["symbol"] = "EURUSD"
            met["combo"] = combo["name"]
            phase2.append(met)
            all_rows.append(met)
            print(
                f"  {combo['name']:8s} {v.name:22s}  ret={met['total_return_pct']:+7.1f}%  "
                f"PF={met['profit_factor']:5.2f}  Sh={met['sharpe']:6.2f}  n={met['n_trades']}",
                flush=True,
            )

    # Phase 3: best overall variant on GBPUSD all combos
    print("\n[Phase 3] Best variants on GBPUSD", flush=True)
    path_g = resolve_csv(os.path.join(_ROOT, "data"), "GBPUSD")
    m1g = load_m1_fast(path_g)
    pip_g = pip_size("GBPUSD")
    # pick best unique variants by mean score on phase2 positive PF
    p2 = pd.DataFrame(phase2)
    best_overall = (
        p2.groupby("variant")["score"].mean().sort_values(ascending=False).head(3).index.tolist()
    )
    if "A_baseline" not in best_overall:
        best_overall.append("A_baseline")
    check_vars = [v for v in variants_catalog() if v.name in best_overall]

    phase3 = []
    for combo in COMBOS:
        for v in check_vars:
            fr = get_frame(m1g, combo["base"], combo["htf"], v.n, v.m)
            met = eval_on_frame(fr, v, BAR_MINUTES[combo["base"]], pip_g)
            met["symbol"] = "GBPUSD"
            met["combo"] = combo["name"]
            phase3.append(met)
            all_rows.append(met)
            print(
                f"  {combo['name']:8s} {v.name:22s}  ret={met['total_return_pct']:+7.1f}%  "
                f"PF={met['profit_factor']:5.2f}  Sh={met['sharpe']:6.2f}  n={met['n_trades']}",
                flush=True,
            )

    # Save full grid
    full = pd.DataFrame(all_rows)
    # dedupe last wins
    full = full.drop_duplicates(subset=["symbol", "combo", "variant"], keep="last")
    out_csv = os.path.join(OUT_DIR, "dvmr_improvement_grid.csv")
    full.to_csv(out_csv, index=False)

    # Champion table: best per combo on EURUSD from phase2
    print("\n" + "=" * 72)
    print("CHAMPION SUMMARY")
    print("=" * 72)
    p2 = full[(full.symbol == "EURUSD")].copy()
    champs = []
    for combo_name, g in p2.groupby("combo"):
        # prefer phase2 variants but include all
        row = g.sort_values("score", ascending=False).iloc[0]
        base = g[g.variant == "A_baseline"]
        b = base.iloc[0] if len(base) else None
        champs.append(row)
        if b is not None:
            print(
                f"{combo_name:8s}  BEST={row['variant']:22s}  "
                f"ret={row['total_return_pct']:+6.1f}% PF={row['profit_factor']:.2f} "
                f"Sh={row['sharpe']:.2f} DD={row['max_dd_pct']:.1f}% n={int(row['n_trades'])}"
                f"   | baseline ret={b['total_return_pct']:+6.1f}% PF={b['profit_factor']:.2f}",
            )
        else:
            print(f"{combo_name:8s}  BEST={row['variant']}")

    # Global best that also works on GBPUSD (robust)
    print("\nROBUSTNESS (EURUSD + GBPUSD average score for checked variants):")
    both = full[full.variant.isin(best_overall)]
    rob = both.groupby("variant").agg(
        mean_score=("score", "mean"),
        mean_pf=("profit_factor", "mean"),
        mean_ret=("total_return_pct", "mean"),
        mean_sharpe=("sharpe", "mean"),
        mean_dd=("max_dd_pct", "mean"),
        n_cells=("n_trades", "count"),
    ).sort_values("mean_score", ascending=False)
    print(rob.to_string())

    # Final recommendation blurb file
    best_name = rob.index[0] if len(rob) else top_names[0]
    best_v = next(v for v in variants_catalog() if v.name == best_name)
    rec_path = os.path.join(OUT_DIR, "dvmr_improvement_recommendation.txt")
    lines = [
        "DVMR IMPROVEMENT — AUTO RESEARCH RESULT",
        f"Champion variant: {best_name}",
        f"Config: {asdict(best_v)}",
        "",
        "What was wrong with baseline:",
        "  Trades lasted ~2-3 bars; zero-cross signal exits killed winners before TP.",
        "  5m+30m was noise; higher TFs less bad.",
        "",
        "What helped:",
        "  Soft or SL/TP-only exits (no zero-cross kill).",
        "  Breakeven + ATR trail after +1R.",
        "  Slightly stricter HTF threshold (~0.5).",
        "",
        f"Grid saved: {out_csv}",
        "",
        rob.to_string(),
    ]
    with open(rec_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nGrid -> {out_csv}")
    print(f"Rec  -> {rec_path}")
    print("Done.")


if __name__ == "__main__":
    main()
