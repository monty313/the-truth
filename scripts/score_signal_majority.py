"""Signals-only majority-rules scoreboard.

For a panel of agents, each bar:
  votes = count of +1 / -1 (zeros abstain)
  majority long  if n_long  > n_short and n_long  >= min_agree
  majority short if n_short > n_long  and n_short >= min_agree

Then measure path quality like an agent signal (not full P&L):
  fill = next M1 open
  hit rate / mean ret / MFE-MAE at H = 5, 10, 20, 30, 60 M1 bars
  first close sign within H bars

Panels:
  1) bb_rsi_sma ABC only (slots 90-92)
  2) research pack: ABC + DVMR + MV profit + agree 80-83
  3) ABC long-only majority (only long votes counted)

Usage:
  python scripts/score_signal_majority.py --symbol EURUSD,US30
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Callable

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ROOT)

import importlib.util

_p = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("bt", _p)
_bt = importlib.util.module_from_spec(_spec)
sys.modules["bt"] = _bt
_spec.loader.exec_module(_bt)
load_m1_fast = _bt.load_m1_fast
resolve_csv = _bt.resolve_csv

HORIZONS = (5, 10, 20, 30, 60)


def _load_handlers() -> dict[str, Callable]:
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


PANELS = {
    "abc_only": [
        "bb_rsi_sma_A",
        "bb_rsi_sma_B",
        "bb_rsi_sma_C",
    ],
    "abc_plus_research": [
        "bb_rsi_sma_A",
        "bb_rsi_sma_B",
        "bb_rsi_sma_C",
        "dvmr_champ_1h_1d",
        "dvmr_30m_4h_v2",
        "mv_best_quality_30m_4h_long",
        "mv_strong4_30m_4h_long",
        "agree_seA_r2A",
        "agree_2of_top4",
        "agree_seA_r2A_atr",
    ],
    "abc_plus_dvmr_mv": [
        "bb_rsi_sma_A",
        "bb_rsi_sma_B",
        "bb_rsi_sma_C",
        "dvmr_champ_1h_1d",
        "dvmr_30m_4h_v2",
        "mv_best_quality_30m_4h_long",
        "mv_strong4_30m_4h_long",
    ],
}


def majority_series(
    votes: pd.DataFrame,
    *,
    min_agree: int = 2,
    long_only: bool = False,
) -> pd.Series:
    """votes: columns are agent signals in {-1,0,+1}."""
    v = votes.fillna(0.0).to_numpy(dtype=float)
    n_long = (v > 0).sum(axis=1)
    n_short = (v < 0).sum(axis=1)
    out = np.zeros(len(votes), dtype=np.float32)
    if long_only:
        out[(n_long >= min_agree) & (n_long > n_short)] = 1.0
    else:
        out[(n_long >= min_agree) & (n_long > n_short)] = 1.0
        out[(n_short >= min_agree) & (n_short > n_long)] = -1.0
    return pd.Series(out, index=votes.index, dtype=np.float32)


def score_majority_signal(
    sig: pd.Series,
    ohlc: pd.DataFrame,
    horizons=HORIZONS,
) -> list[dict]:
    """sig on M1; entry next open; path in M1 bars."""
    o = ohlc["open"].to_numpy(float)
    h = ohlc["high"].to_numpy(float)
    l = ohlc["low"].to_numpy(float)
    c = ohlc["close"].to_numpy(float)
    s = sig.reindex(ohlc.index).fillna(0.0).to_numpy(float)
    n = len(ohlc)
    rows = []

    # entry events: signal non-zero (pulse) or rising to non-zero
    # treat every bar with sig != 0 as active suggestion; for pulse agents
    # that is the fire bar. For sticky, use change into a side.
    events = []
    prev = 0.0
    for i in range(n - 2):
        cur = s[i]
        # fire on non-zero bar (pulse) OR newly entered side (sticky)
        if cur != 0 and (cur != prev or abs(cur) > 0):
            # count pulse-style: any non-zero as event only if prev was 0 or opposite
            if prev == 0 or np.sign(prev) != np.sign(cur):
                events.append((i, int(np.sign(cur))))
            elif prev == 0:
                events.append((i, int(np.sign(cur))))
        # simpler: every bar with non-zero is too dense for sticky
        prev = cur

    # Prefer: event when signal is non-zero AND (i==0 or s[i-1]==0 or sign change)
    events = []
    for i in range(n - 2):
        if s[i] == 0:
            continue
        if i == 0 or s[i - 1] == 0 or np.sign(s[i - 1]) != np.sign(s[i]):
            events.append((i, int(np.sign(s[i]))))

    for H in horizons:
        for side_name, side_f in (("long", 1), ("short", -1), ("any", 0)):
            hits = []
            rets = []
            mfes = []
            maes = []
            first_fav = 0
            first_adv = 0
            first_none = 0
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
                    mfe = (np.nanmax(h[j:j + H]) - entry) / entry * 100.0
                    mae = (entry - np.nanmin(l[j:j + H])) / entry * 100.0
                else:
                    mfe = (entry - np.nanmin(l[j:j + H])) / entry * 100.0
                    mae = (np.nanmax(h[j:j + H]) - entry) / entry * 100.0
                mfes.append(mfe)
                maes.append(mae)
                # first close sign in 1..H
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
                    first_fav += 1
                elif got == "adverse":
                    first_adv += 1
                else:
                    first_none += 1

            if not rets:
                continue
            rets = np.array(rets)
            hits = np.array(hits)
            n_e = len(rets)
            rows.append(dict(
                side=side_name,
                horizon_m1=H,
                n=n_e,
                hit_pct=100.0 * hits.mean(),
                mean_ret=float(rets.mean()),
                median_ret=float(np.median(rets)),
                mean_mfe=float(np.mean(mfes)),
                mean_mae=float(np.mean(maes)),
                first_close_favor_pct=100.0 * first_fav / n_e,
                first_close_adverse_pct=100.0 * first_adv / n_e,
            ))
    return rows


def agreement_stats(votes: pd.DataFrame, maj: pd.Series) -> dict:
    v = votes.fillna(0.0)
    n_long = (v > 0).sum(axis=1)
    n_short = (v < 0).sum(axis=1)
    active = (maj != 0).mean() * 100.0
    return dict(
        pct_bars_majority=float(active),
        mean_long_votes=float(n_long.mean()),
        mean_short_votes=float(n_short.mean()),
        max_agree_long=int(n_long.max()),
        max_agree_short=int(n_short.max()),
        n_agents=v.shape[1],
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", default="EURUSD,US30")
    ap.add_argument("--min-agree", type=int, default=2)
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "artifacts", "signal_majority"))
    ap.add_argument("--max-rows", type=int, default=None,
                    help="Optional M1 cap for speed; default full curriculum")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    handlers = _load_handlers()
    print("=" * 72)
    print("SIGNALS-ONLY MAJORITY RULES SCOREBOARD")
    print(f"min_agree={args.min_agree} | horizons M1 bars {HORIZONS}")
    print("zeros abstain | fill=next open | path score only")
    print("=" * 72)

    all_rows = []
    meta_rows = []

    for sym in [s.strip().upper() for s in args.symbol.split(",") if s.strip()]:
        path = resolve_csv(os.path.join(_ROOT, "data"), sym)
        if not path:
            print(f"[{sym}] no data")
            continue
        print(f"\n### {sym} loading M1 ...", flush=True)
        m1 = load_m1_fast(path, max_rows=args.max_rows)
        print(f"    bars={len(m1):,}", flush=True)
        F = m1.copy()

        # compute all unique agents needed
        needed = sorted({a for panel in PANELS.values() for a in panel})
        cache = {}
        for name in needed:
            if name not in handlers:
                print(f"    SKIP missing handler {name}")
                continue
            print(f"    agent {name} ...", flush=True)
            try:
                cache[name] = handlers[name](F).reindex(F.index).fillna(0.0).astype(np.float32)
                nz = float((cache[name] != 0).mean() * 100)
                print(f"      nonflat={nz:.2f}%", flush=True)
            except Exception as e:
                print(f"      FAIL {e}")

        for panel_name, agent_names in PANELS.items():
            cols = {a: cache[a] for a in agent_names if a in cache}
            if len(cols) < 2:
                print(f"  panel {panel_name}: not enough agents")
                continue
            votes = pd.DataFrame(cols)
            for long_only in (False, True):
                tag = panel_name + ("_longOnly" if long_only else "")
                # min_agree sweep for abc_only
                agrees = [args.min_agree]
                if panel_name == "abc_only":
                    agrees = [2, 3]
                if panel_name.startswith("abc_plus"):
                    agrees = [2, 3, 4]

                for ma in agrees:
                    maj = majority_series(votes, min_agree=ma, long_only=long_only)
                    meta = agreement_stats(votes, maj)
                    meta.update(dict(symbol=sym, panel=tag, min_agree=ma))
                    meta_rows.append(meta)
                    print(
                        f"\n  [{tag}] min_agree={ma}  majority_bars={meta['pct_bars_majority']:.2f}%  "
                        f"agents={meta['n_agents']}",
                        flush=True,
                    )
                    scores = score_majority_signal(maj, F)
                    for r in scores:
                        r.update(dict(symbol=sym, panel=tag, min_agree=ma, n_agents=meta["n_agents"]))
                        all_rows.append(r)
                    # print compact any-side
                    for r in scores:
                        if r["side"] == "any" and r["horizon_m1"] in (5, 10, 20, 60):
                            print(
                                f"    H={r['horizon_m1']:3d} n={r['n']:5d} "
                                f"hit={r['hit_pct']:5.1f}% mean={r['mean_ret']:+.4f}% "
                                f"med={r['median_ret']:+.4f}% "
                                f"1st+={r['first_close_favor_pct']:.1f}% "
                                f"1st-={r['first_close_adverse_pct']:.1f}% "
                                f"MFE={r['mean_mfe']:.3f} MAE={r['mean_mae']:.3f}",
                                flush=True,
                            )

    df = pd.DataFrame(all_rows)
    meta = pd.DataFrame(meta_rows)
    out1 = os.path.join(args.out_dir, "majority_path_scoreboard.csv")
    out2 = os.path.join(args.out_dir, "majority_meta.csv")
    df.to_csv(out1, index=False)
    meta.to_csv(out2, index=False)

    print("\n" + "=" * 72)
    print("LEADERBOARD — any side, hit@10 M1 bars, n>=30")
    print("=" * 72)
    g = df[(df.side == "any") & (df.horizon_m1 == 10) & (df.n >= 30)].sort_values(
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
                    "first_close_favor_pct",
                    "mean_mfe",
                    "mean_mae",
                ]
            ].head(25).to_string(index=False)
        )

    print("\n" + "=" * 72)
    print("LEADERBOARD — LONG only, hit@10")
    print("=" * 72)
    g2 = df[(df.side == "long") & (df.horizon_m1 == 10) & (df.n >= 30)].sort_values(
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
            ].head(20).to_string(index=False)
        )

    print(f"\nScores -> {out1}")
    print(f"Meta   -> {out2}")
    print("Done.")


if __name__ == "__main__":
    main()
