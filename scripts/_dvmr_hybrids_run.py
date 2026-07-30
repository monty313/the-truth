"""Quick hybrid validation for DVMR v2 winners."""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)
sys.path.insert(0, _ROOT)
os.chdir(_ROOT)

import importlib.util

import pandas as pd

_v2_path = os.path.join(_ROOT, "scripts", "improve_dvmr_v2.py")
_spec = importlib.util.spec_from_file_location("improve_dvmr_v2", _v2_path)
_v2 = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
sys.modules["improve_dvmr_v2"] = _v2
_spec.loader.exec_module(_v2)

BAR_MINUTES = _v2.BAR_MINUTES
OUT_DIR = _v2.OUT_DIR
V2 = _v2.V2
build_frame = _v2.build_frame
eval_slice = _v2.eval_slice
load_m1_fast = _v2.load_m1_fast
pip_size = _v2.pip_size
resolve_csv = _v2.resolve_csv
walk_forward = _v2.walk_forward


def main():
    hybrids = [
        V2("h_thr50_hard", htf_thr=0.5, exit_mode="hard_only"),
        V2("h_thr50_soft_min4", htf_thr=0.5, exit_mode="soft", min_bars_signal=4),
        V2("h_thr50_hard_atr", htf_thr=0.5, exit_mode="hard_only", atr_pct_lo=0.2, atr_pct_hi=0.85),
        V2("h_thr50_soft_atr", htf_thr=0.5, exit_mode="soft", atr_pct_lo=0.2, atr_pct_hi=0.85),
        V2(
            "h_thr50_soft_min4_atr",
            htf_thr=0.5,
            exit_mode="soft",
            min_bars_signal=4,
            atr_pct_lo=0.2,
            atr_pct_hi=0.85,
        ),
        V2("h_thr50_nm515_soft", htf_thr=0.5, n=5, m=15, exit_mode="soft"),
        V2("h_thr50_nm515_hard", htf_thr=0.5, n=5, m=15, exit_mode="hard_only"),
        V2(
            "h_thr50_soft_min4_tp3",
            htf_thr=0.5,
            exit_mode="soft",
            min_bars_signal=4,
            tp_atr=3.0,
        ),
        V2("h_thr45_hard", htf_thr=0.45, exit_mode="hard_only"),
        V2(
            "h_thr50_hard_rr12_25",
            htf_thr=0.5,
            exit_mode="hard_only",
            sl_atr=1.2,
            tp_atr=2.5,
        ),
        # previous anchor
        V2("anchor_soft55", htf_thr=0.55, exit_mode="soft"),
    ]

    combos = [
        ("30m+4h", "30min", "4h"),
        ("1h+1d", "1h", "1d"),
    ]
    rows = []
    for sym in ["EURUSD", "GBPUSD"]:
        m1 = load_m1_fast(resolve_csv("data", sym))
        pip = pip_size(sym)
        cache = {}
        for cname, base, htf in combos:
            for v in hybrids:
                key = (base, htf, v.n, v.m)
                if key not in cache:
                    cache[key] = build_frame(m1, base, htf, v.n, v.m)
                fr = cache[key]
                met, _, _ = eval_slice(fr, v, BAR_MINUTES[base], pip)
                wf = walk_forward(fr, v, BAR_MINUTES[base], pip)
                rows.append(
                    dict(
                        symbol=sym,
                        combo=cname,
                        variant=v.name,
                        ret=met["total_return_pct"],
                        pf=met["profit_factor"],
                        sh=met["sharpe"],
                        dd=met["max_dd_pct"],
                        n=met["n_trades"],
                        oos_pf=wf["oos_pf"],
                        oos_ret=wf["oos_ret"],
                        pos=wf["pos_folds"],
                    )
                )
                print(
                    f"{sym:7s} {cname:8s} {v.name:28s} ret={met['total_return_pct']:+6.1f}% "
                    f"PF={met['profit_factor']:5.2f} Sh={met['sharpe']:5.2f} n={met['n_trades']:3d} "
                    f"OOS_PF={wf['oos_pf']:.2f} +f={wf['pos_folds']}",
                    flush=True,
                )

    df = pd.DataFrame(rows)
    pivot = []
    for (c, v), g in df.groupby(["combo", "variant"]):
        if set(g.symbol) != {"EURUSD", "GBPUSD"}:
            continue
        eu = g[g.symbol == "EURUSD"].iloc[0]
        gb = g[g.symbol == "GBPUSD"].iloc[0]
        robust = (
            eu.ret > 0
            and eu.pf >= 1.15
            and eu.n >= 25
            and gb.pf >= 0.9
            and gb.ret > -10
            and eu.oos_pf >= 1.15
        )
        score = (
            0.5 * (eu.pf + max(gb.pf, 0))
            + 0.02 * (eu.ret + gb.ret)
            + 0.3 * (eu.sh + gb.sh)
            - 0.02 * (abs(eu.dd) + abs(gb.dd))
        )
        pivot.append(
            dict(
                combo=c,
                variant=v,
                eu_ret=eu.ret,
                eu_pf=eu.pf,
                eu_sh=eu.sh,
                eu_n=eu.n,
                eu_oos_pf=eu.oos_pf,
                gb_ret=gb.ret,
                gb_pf=gb.pf,
                gb_sh=gb.sh,
                gb_n=gb.n,
                robust=robust,
                score=score,
            )
        )
    p = pd.DataFrame(pivot).sort_values("score", ascending=False)
    print("\n=== CROSS-SYMBOL RANK ===")
    print(p.to_string(index=False))
    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(os.path.join(OUT_DIR, "dvmr_v2_hybrids.csv"), index=False)
    p.to_csv(os.path.join(OUT_DIR, "dvmr_v2_hybrids_rank.csv"), index=False)
    rob = p[p.robust]
    print("\nROBUST winners:", len(rob))
    if len(rob):
        print(rob.head(8).to_string(index=False))
    else:
        print("None fully robust; best compromise:")
        print(p.head(8).to_string(index=False))
    print("Done.")


if __name__ == "__main__":
    main()
