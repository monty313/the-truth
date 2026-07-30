"""Majority-rules signal path test on a 5m clock (not M1).

- Resample M1 -> 5m OHLC
- Run agents on full M1 (their native logic), then reduce each agent to 5m:
    last non-zero in the 5m bucket if any, else last value (or max |vote|)
- Majority on 5m bars; path score at H = 1,2,3,5,8,10,15,20 **5m bars**
- Fill = next 5m open

Panels: abc_only | abc_plus_dvmr_mv | abc_plus_research
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

from data_io.loader import resample  # noqa: E402

import importlib.util

_p = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("bt", _p)
_bt = importlib.util.module_from_spec(_spec)
sys.modules["bt"] = _bt
_spec.loader.exec_module(_bt)
load_m1_fast = _bt.load_m1_fast
resolve_csv = _bt.resolve_csv

HORIZONS = (1, 2, 3, 5, 8, 10, 15, 20)

PANELS = {
    "abc_only": ["bb_rsi_sma_A", "bb_rsi_sma_B", "bb_rsi_sma_C"],
    "abc_plus_dvmr_mv": [
        "bb_rsi_sma_A", "bb_rsi_sma_B", "bb_rsi_sma_C",
        "dvmr_champ_1h_1d", "dvmr_30m_4h_v2",
        "mv_best_quality_30m_4h_long", "mv_strong4_30m_4h_long",
    ],
    "abc_plus_research": [
        "bb_rsi_sma_A", "bb_rsi_sma_B", "bb_rsi_sma_C",
        "dvmr_champ_1h_1d", "dvmr_30m_4h_v2",
        "mv_best_quality_30m_4h_long", "mv_strong4_30m_4h_long",
        "agree_seA_r2A", "agree_2of_top4", "agree_seA_r2A_atr",
    ],
}


def _handlers():
    from signals.bb_rsi_sma_agent import HANDLERS as H1
    from signals.dvmr_agent import HANDLERS as H2
    from signals.momentum_vector_agent import HANDLERS as H3
    from signals.agree import HANDLERS as H4
    out = {}
    out.update(H1)
    out.update(H2)
    out.update(H3)
    out.update(H4)
    return out


def m1_signal_to_5m(sig_m1: pd.Series, idx_5m: pd.DatetimeIndex) -> pd.Series:
    """Collapse M1 agent series onto 5m bars.

    For each 5m bucket [t, t+5m): take the last non-zero vote if any, else 0.
    (Event-preserving: a pulse inside the bucket survives.)
    """
    s = sig_m1.astype(float).copy()
    s.index = pd.to_datetime(s.index)
    # bucket label = floor to 5min
    buckets = s.index.floor("5min")
    df = pd.DataFrame({"v": s.to_numpy(), "b": buckets})
    # last non-zero per bucket
    def reduce(g):
        nz = g[g != 0]
        if len(nz):
            return float(nz.iloc[-1])
        return 0.0

    out = df.groupby("b")["v"].apply(reduce)
    return out.reindex(idx_5m).fillna(0.0).astype(np.float32)


def majority(votes: pd.DataFrame, min_agree: int, long_only: bool = False) -> pd.Series:
    v = votes.fillna(0.0).to_numpy(float)
    nl = (v > 0).sum(axis=1)
    ns = (v < 0).sum(axis=1)
    out = np.zeros(len(votes), dtype=np.float32)
    out[(nl >= min_agree) & (nl > ns)] = 1.0
    if not long_only:
        out[(ns >= min_agree) & (ns > nl)] = -1.0
    return pd.Series(out, index=votes.index, dtype=np.float32)


def score_path(sig: pd.Series, ohlc: pd.DataFrame, horizons=HORIZONS) -> list[dict]:
    o = ohlc["open"].to_numpy(float)
    h = ohlc["high"].to_numpy(float)
    l = ohlc["low"].to_numpy(float)
    c = ohlc["close"].to_numpy(float)
    s = sig.reindex(ohlc.index).fillna(0.0).to_numpy(float)
    n = len(ohlc)

    events = []
    for i in range(n - 2):
        if s[i] == 0:
            continue
        if i == 0 or s[i - 1] == 0 or np.sign(s[i - 1]) != np.sign(s[i]):
            events.append((i, int(np.sign(s[i]))))

    rows = []
    for H in horizons:
        for side_name, side_f in (("long", 1), ("short", -1), ("any", 0)):
            rets, hits, mfes, maes = [], [], [], []
            f_fav = f_adv = 0
            for i, side in events:
                if side_f and side != side_f:
                    continue
                j = i + 1
                if j + H > n:
                    continue
                entry = o[j]
                if not np.isfinite(entry) or entry == 0:
                    continue
                exit_c = c[j + H - 1]
                ret = side * (exit_c - entry) / entry * 100.0
                rets.append(ret)
                hits.append(1.0 if ret > 0 else 0.0)
                if side > 0:
                    mfes.append((np.nanmax(h[j:j + H]) - entry) / entry * 100.0)
                    maes.append((entry - np.nanmin(l[j:j + H])) / entry * 100.0)
                else:
                    mfes.append((entry - np.nanmin(l[j:j + H])) / entry * 100.0)
                    maes.append((np.nanmax(h[j:j + H]) - entry) / entry * 100.0)
                got = "none"
                for k in range(H):
                    r = side * (c[j + k] - entry)
                    if r > 0:
                        got = "favor"
                        break
                    if r < 0:
                        got = "adverse"
                        break
                if got == "favor":
                    f_fav += 1
                elif got == "adverse":
                    f_adv += 1
            if not rets:
                continue
            n_e = len(rets)
            rows.append(dict(
                side=side_name,
                horizon_5m=H,
                n=n_e,
                hit_pct=100.0 * np.mean(hits),
                mean_ret=float(np.mean(rets)),
                median_ret=float(np.median(rets)),
                mean_mfe=float(np.mean(mfes)),
                mean_mae=float(np.mean(maes)),
                first_close_favor_pct=100.0 * f_fav / n_e,
                first_close_adverse_pct=100.0 * f_adv / n_e,
            ))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD,US30")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "signal_majority_5m"))
    ap.add_argument("--max-rows", type=int, default=None)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    handlers = _handlers()
    print("=" * 72)
    print("MAJORITY RULES ON 5m CLOCK (signals only)")
    print(f"horizons = {HORIZONS} 5m-bars | fill = next 5m open")
    print("=" * 72)

    all_rows = []
    needed = sorted({a for p in PANELS.values() for a in p})

    for sym in [s.strip().upper() for s in args.symbol.split(",") if s.strip()]:
        path = resolve_csv(os.path.join(_ROOT, "data"), sym)
        if not path:
            continue
        print(f"\n### {sym}", flush=True)
        m1 = load_m1_fast(path, max_rows=args.max_rows)
        print(f"  M1={len(m1):,}", flush=True)
        m5 = resample(m1, "5min")
        print(f"  5m={len(m5):,}", flush=True)
        F = m1.copy()

        cache_m1 = {}
        for name in needed:
            if name not in handlers:
                print(f"  skip {name}")
                continue
            print(f"  agent {name} ...", flush=True)
            try:
                cache_m1[name] = handlers[name](F)
                print(f"    m1 nonflat={100*(cache_m1[name]!=0).mean():.2f}%", flush=True)
            except Exception as e:
                print(f"    FAIL {e}")

        # reduce to 5m
        cache_5 = {}
        for name, ser in cache_m1.items():
            cache_5[name] = m1_signal_to_5m(ser, m5.index)
            print(f"    5m {name} nonflat={100*(cache_5[name]!=0).mean():.2f}%", flush=True)

        for panel, names in PANELS.items():
            cols = {n: cache_5[n] for n in names if n in cache_5}
            if len(cols) < 2:
                continue
            votes = pd.DataFrame(cols)
            agree_list = [2, 3] if panel == "abc_only" else [2, 3, 4]
            for long_only in (False, True):
                for ma in agree_list:
                    tag = panel + ("_longOnly" if long_only else "")
                    maj = majority(votes, ma, long_only=long_only)
                    pct = float((maj != 0).mean() * 100)
                    print(f"\n  [{tag}] min_agree={ma}  majority_5m_bars={pct:.2f}%", flush=True)
                    scores = score_path(maj, m5)
                    for r in scores:
                        r.update(dict(symbol=sym, panel=tag, min_agree=ma, n_agents=len(cols)))
                        all_rows.append(r)
                    for r in scores:
                        if r["side"] == "any" and r["horizon_5m"] in (1, 3, 5, 10, 20):
                            print(
                                f"    H5m={r['horizon_5m']:2d} n={r['n']:5d} "
                                f"hit={r['hit_pct']:5.1f}% mean={r['mean_ret']:+.4f}% "
                                f"med={r['median_ret']:+.4f}% "
                                f"1st+={r['first_close_favor_pct']:.1f}% "
                                f"1st-={r['first_close_adverse_pct']:.1f}% "
                                f"MFE={r['mean_mfe']:.3f} MAE={r['mean_mae']:.3f}",
                                flush=True,
                            )

    df = pd.DataFrame(all_rows)
    out = os.path.join(args.out_dir, "majority_5m_path.csv")
    df.to_csv(out, index=False)

    print("\n" + "=" * 72)
    print("LEADERBOARD any-side hit @ 5m bars H=5 and H=10 (n>=30)")
    print("=" * 72)
    for H in (5, 10, 20):
        print(f"\n--- H={H} 5m bars (~{H*5} minutes) ---")
        g = df[(df.side == "any") & (df.horizon_5m == H) & (df.n >= 30)].sort_values(
            "hit_pct", ascending=False
        )
        if len(g):
            print(
                g[
                    [
                        "symbol",
                        "panel",
                        "min_agree",
                        "n",
                        "hit_pct",
                        "mean_ret",
                        "median_ret",
                        "mean_mfe",
                        "mean_mae",
                    ]
                ]
                .head(15)
                .to_string(index=False)
            )

    print("\n" + "=" * 72)
    print("LONG path @ H=10 5m bars")
    print("=" * 72)
    g2 = df[(df.side == "long") & (df.horizon_5m == 10) & (df.n >= 30)].sort_values(
        "hit_pct", ascending=False
    )
    if len(g2):
        print(
            g2[
                [
                    "symbol",
                    "panel",
                    "min_agree",
                    "n",
                    "hit_pct",
                    "mean_ret",
                    "median_ret",
                    "mean_mfe",
                    "mean_mae",
                ]
            ]
            .head(15)
            .to_string(index=False)
        )

    print(f"\nFull -> {out}")
    print("Done.")


if __name__ == "__main__":
    main()
