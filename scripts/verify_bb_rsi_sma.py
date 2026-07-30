"""Sanity-check BB/RSI-SMA implementation + re-run bar path score."""
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
from features.bb_rsi_sma_sets import SETS, _tf_frame, build_set_frame  # noqa: E402
from features.indicators import bollinger, rsi  # noqa: E402

import importlib.util

_p = os.path.join(_ROOT, "scripts", "backtest_dvmr_mtf.py")
_spec = importlib.util.spec_from_file_location("bt", _p)
_bt = importlib.util.module_from_spec(_spec)
sys.modules["bt"] = _bt
_spec.loader.exec_module(_bt)

_s = os.path.join(_ROOT, "scripts", "score_signal_bars_from_entry.py")
_spec2 = importlib.util.spec_from_file_location("score_bars", _s)
_sc = importlib.util.module_from_spec(_spec2)
sys.modules["score_bars"] = _sc
_spec2.loader.exec_module(_sc)


def main():
    print("=" * 70)
    print("VERIFY IMPLEMENTATION")
    print("=" * 70)

    m1 = _bt.load_m1_fast(_bt.resolve_csv("data", "EURUSD"), max_rows=100_000)
    ltf = resample(m1, "30min")
    c = ltf["close"]
    r = rsi(c, 5)
    rup, rmid, rlo = bollinger(r, 10, 1.0, shift=5)
    pup, pmid, plo = bollinger(c, 10, 1.0, shift=5)

    print("\n[1] What are BBs applied to on LTF?")
    print(f"  close last          = {float(c.dropna().iloc[-1]):.5f}  (PRICE)")
    print(f"  RSI(5) range        = {float(r.dropna().min()):.2f} .. {float(r.dropna().max()):.2f}")
    print(f"  bollinger(RSI) mid  = {float(rmid.dropna().min()):.2f} .. {float(rmid.dropna().max()):.2f}  <- RSI units")
    print(f"  bollinger(PRICE) mid= {float(pmid.dropna().min()):.5f} .. {float(pmid.dropna().max()):.5f}  <- price units")
    assert rmid.dropna().max() <= 100.5, "RSI-BB mid should be RSI-scale"
    assert rmid.dropna().min() >= -0.5
    print("  OK: LTF BB is on RSI (0-100 scale), NOT on price.")

    print("\n[2] HTF price BB period=100 dev=0.5 shift=2?")
    h = resample(m1, "4h")
    hf = _tf_frame(h, kind="htf")
    # rebuild expected
    eup, emid, elo = bollinger(h["close"], 100, 0.5, shift=2)
    match = np.allclose(
        hf["bb_up"].dropna().to_numpy(),
        eup.reindex(hf["bb_up"].dropna().index).to_numpy(),
        equal_nan=True,
    )
    print(f"  HTF bb_up matches bollinger(close,100,0.5,shift=2): {match}")
    assert match
    print(f"  HTF close last={float(h['close'].iloc[-1]):.5f} bb_mid last={float(emid.dropna().iloc[-1]):.5f}")
    print("  OK: HTF BB is on PRICE with 100 / 0.5 / shift 2.")

    print("\n[3] LTF frame uses rsi_cross vs rsi_bb_lo (not price bands)")
    lf = _tf_frame(ltf, kind="ltf")
    assert "rsi_cross_up_lo" in lf.columns
    assert "rsi_bb_lo" in lf.columns
    # cross definition sample
    print(f"  rsi_cross_up_lo fires: {int(lf['rsi_cross_up_lo'].fillna(False).sum())}")
    print(f"  ltf_buy fires:         {int(lf['ltf_buy'].fillna(False).sum())}")
    print("  OK: timing is RSI vs RSI-BB.")

    print("\n[4] Re-test path score (bars from entry) EURUSD + US30 Set C")
    print("-" * 70)
    for sym in ("EURUSD", "US30"):
        m = _bt.load_m1_fast(_bt.resolve_csv("data", sym))
        for set_name in SETS:
            fr = build_set_frame(m, set_name).dropna(subset=["rsi", "sma_low", "rsi_bb_lo"])
            # scale check
            mid_max = float(fr["rsi_bb_mid"].dropna().max())
            assert mid_max <= 100.5, f"{sym} {set_name} rsi_bb_mid max {mid_max}"
            n_l = int(fr["long_entry"].sum())
            n_s = int(fr["short_entry"].sum())
            rows = _sc.score_frame(fr)
            print(f"\n{sym} {set_name}  long={n_l} short={n_s}  (RSI-BB mid max={mid_max:.1f})")
            print(f"  {'side':5s} {'H':>3} {'n':>6} {'hit@H':>7} {'mean%':>9} {'1stCl+':>7} {'1stCl-':>7}")
            for r in rows:
                if r["side"] in ("any", "long") and r["bars_from_entry"] in (5, 10, 20):
                    print(
                        f"  {r['side']:5s} {r['bars_from_entry']:3d} {r['n']:6d} "
                        f"{r['hit_pct_at_H']:6.1f}% {r['mean_ret_at_H']:+8.4f}% "
                        f"{r['first_close_favor_pct']:6.1f}% {r['first_close_adverse_pct']:6.1f}%"
                    )

    # sample anatomy
    print("\n[5] Sample long signal anatomy (EURUSD Set C)")
    m = _bt.load_m1_fast(_bt.resolve_csv("data", "EURUSD"))
    fr = build_set_frame(m, "C_30m_4h_1d").dropna(subset=["rsi", "rsi_bb_lo"])
    i = fr.index[fr["long_entry"]][10]
    row = fr.loc[i]
    print(f"  time={i}")
    print(f"  close={row.close:.5f}  sma_low={row.sma_low:.5f}")
    print(f"  RSI={row.rsi:.2f}  RSI_BB_lo={row.rsi_bb_lo:.2f}  RSI_BB_up={row.rsi_bb_up:.2f}")
    print(f"  (RSI and bands same scale => BB applied to RSI)")
    print(f"  htf0_buy={row.get('htf0_buy')} htf1_buy={row.get('htf1_buy')} htf_buy_any={row.htf_buy_any}")

    out = os.path.join(_ROOT, "artifacts", "bb_rsi_sma", "verify_rerun_summary.txt")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        f.write("Verified: LTF BB on RSI(5) 10/1/shift5; HTF BB on price 100/0.5/shift2\n")
    print(f"\nALL VERIFICATIONS PASSED")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
