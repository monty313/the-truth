"""Individual agent path accuracy on 5m clock (signals only)."""
from __future__ import annotations

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

from signals.bb_rsi_sma_agent import HANDLERS as H1  # noqa: E402
from signals.dvmr_agent import HANDLERS as H2  # noqa: E402
from signals.momentum_vector_agent import HANDLERS as H3  # noqa: E402
from signals.agree import HANDLERS as H4  # noqa: E402

HANDLERS = {}
HANDLERS.update(H1)
HANDLERS.update(H2)
HANDLERS.update(H3)
HANDLERS.update(H4)

NAMES = [
    "bb_rsi_sma_A",
    "bb_rsi_sma_B",
    "bb_rsi_sma_C",
    "dvmr_champ_1h_1d",
    "dvmr_30m_4h_v2",
    "dvmr_champ_1h_1d_pulse",
    "mv_best_quality_30m_4h_long",
    "mv_strong4_30m_4h_long",
    "mv_profit_30m_4h_both",
    "agree_seA_r2A",
    "agree_seA_r2A_atr",
    "agree_2of_top4",
    "agree_seB_r2B_epB",
]
HOR = (5, 10, 20)


def to5(sig: pd.Series, idx5: pd.DatetimeIndex) -> pd.Series:
    s = sig.astype(float).copy()
    s.index = pd.to_datetime(s.index)
    b = s.index.floor("5min")
    df = pd.DataFrame({"v": s.to_numpy(), "b": b})

    def red(g):
        nz = g[g != 0]
        return float(nz.iloc[-1]) if len(nz) else 0.0

    return df.groupby("b")["v"].apply(red).reindex(idx5).fillna(0.0)


def score(sig: pd.Series, ohlc: pd.DataFrame):
    o = ohlc["open"].to_numpy(float)
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
    for H in HOR:
        for side_name, sf in (("long", 1), ("short", -1), ("any", 0)):
            rets = []
            for i, side in events:
                if sf and side != sf:
                    continue
                j = i + 1
                if j + H > n:
                    continue
                entry = o[j]
                if not np.isfinite(entry) or entry == 0:
                    continue
                rets.append(side * (c[j + H - 1] - entry) / entry * 100.0)
            if len(rets) < 20:
                continue
            rets = np.array(rets)
            rows.append(
                dict(
                    side=side_name,
                    H=H,
                    n=len(rets),
                    hit=100.0 * (rets > 0).mean(),
                    mean=float(rets.mean()),
                    med=float(np.median(rets)),
                )
            )
    return rows


def main():
    out_dir = os.path.join(_ROOT, "artifacts", "signal_majority_5m")
    os.makedirs(out_dir, exist_ok=True)
    rows = []
    for sym in ("EURUSD", "US30"):
        print(f"### {sym}", flush=True)
        m1 = _bt.load_m1_fast(_bt.resolve_csv("data", sym))
        m5 = resample(m1, "5min")
        F = m1.copy()
        for name in NAMES:
            if name not in HANDLERS:
                print(f"  skip {name}")
                continue
            print(f"  {name} ...", flush=True)
            try:
                sm1 = HANDLERS[name](F)
            except Exception as e:
                print(f"    FAIL {e}")
                continue
            s5 = to5(sm1, m5.index)
            for r in score(s5, m5):
                r["symbol"] = sym
                r["agent"] = name
                rows.append(r)
            for r in rows:
                if r["symbol"] == sym and r["agent"] == name and r["side"] == "any" and r["H"] == 10:
                    print(
                        f"    any@10 n={r['n']} hit={r['hit']:.1f}% mean={r['mean']:+.4f}%",
                        flush=True,
                    )

    df = pd.DataFrame(rows)
    path = os.path.join(out_dir, "individual_agent_5m_path.csv")
    df.to_csv(path, index=False)

    print("\n=== RANK any-side hit@10 5m bars (n>=50) ===")
    g = df[(df.side == "any") & (df.H == 10) & (df.n >= 50)].sort_values("hit", ascending=False)
    print(g[["symbol", "agent", "n", "hit", "mean", "med"]].to_string(index=False))

    print("\n=== RANK long hit@10 (n>=50) ===")
    g2 = df[(df.side == "long") & (df.H == 10) & (df.n >= 50)].sort_values("hit", ascending=False)
    print(g2[["symbol", "agent", "n", "hit", "mean", "med"]].to_string(index=False))

    print("\n=== RANK short hit@10 (n>=50) ===")
    g3 = df[(df.side == "short") & (df.H == 10) & (df.n >= 50)].sort_values("hit", ascending=False)
    print(g3[["symbol", "agent", "n", "hit", "mean", "med"]].to_string(index=False))

    print(f"\nSaved {path}")


if __name__ == "__main__":
    main()
