"""OFFLINE probe: Wave-style CCI→BB multi-TF signal vs heuristic + equity outcomes.

Does NOT modify multi-pair policy / equity_day / PROVEN.
Uses FULL continuous M1 history so CCI(100) on 15m can warm up (day-isolated
15m only has ~90 bars → CCI100 never ready — that was a false zero-signal bug).

Usage (repo root, PYTHONPATH=.;code):
  python lineages/adaptive_rl_brain_7_31_26/probe_wave_signal.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))
_CODE = os.path.join(_ROOT, "code")
for _p in (_ROOT, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from features.indicators import bollinger, cci
from lineages.adaptive_rl_brain_7_31_26.data.mtf import resample_lineage
from lineages.adaptive_rl_brain_7_31_26.equity_day import (
    GoalEquityDay,
    load_calendar_days,
    split_practice_forward,
)
from lineages.adaptive_rl_brain_7_31_26.policy_stub import (
    ACTION_BUY,
    ACTION_HOLD,
    ACTION_SELL,
)
from lineages.adaptive_rl_brain_7_31_26.price_data import load_raw_m1

OUT_PATH = os.path.join(_HERE, "checkpoints", "wave_probe_report.json")
DATA = "XAUUSD_curriculum_2026.csv"
TF1, TF2, TF3 = "1m", "5m", "15m"
HORIZON_M1 = 25
PRACTICE_N = 50


def _ensure_vol(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "vol" not in out.columns:
        out["vol"] = 100.0
    return out


def _wave_frame(ohlc: pd.DataFrame) -> pd.DataFrame:
    o = ohlc
    cci10 = cci(o, 10)
    cci100 = cci(o, 100)
    bb10_up, _, bb10_lo = bollinger(cci10, 20, 1.0, shift=0)
    bb100_up, _, bb100_lo = bollinger(cci100, 20, 0.5, shift=0)
    return pd.DataFrame(
        {
            "cci10": cci10,
            "cci100": cci100,
            "bb_fast_up": bb10_up,
            "bb_fast_lo": bb10_lo,
            "bb_slow_up": bb100_up,
            "bb_slow_lo": bb100_lo,
            "close": o["close"],
        },
        index=o.index,
    )


def build_global_pack(m1_full: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    m1_full = _ensure_vol(m1_full)
    return {
        TF1: _wave_frame(m1_full),
        TF2: _wave_frame(resample_lineage(m1_full, TF2)),
        TF3: _wave_frame(resample_lineage(m1_full, TF3)),
    }


def _asof_row(tf_df: pd.DataFrame, ts: pd.Timestamp) -> pd.Series | None:
    sub = tf_df.loc[:ts]
    if len(sub) < 2:
        return None
    # last closed-ish: previous bar (avoid incomplete)
    return sub.iloc[-2]


def wave_signal_at(pack: Dict[str, pd.DataFrame], ts: pd.Timestamp) -> Tuple[str, int]:
    r1 = _asof_row(pack[TF1], ts)
    r2 = _asof_row(pack[TF2], ts)
    r3 = _asof_row(pack[TF3], ts)
    if r1 is None or r2 is None or r3 is None:
        return "", ACTION_HOLD
    keys_slow = ("cci100", "bb_slow_up", "bb_slow_lo")
    for r in (r1, r2, r3):
        if any(not np.isfinite(float(r[k])) for k in keys_slow):
            return "", ACTION_HOLD
    if not np.isfinite(float(r1["cci10"])) or not np.isfinite(float(r1["bb_fast_up"])):
        return "", ACTION_HOLD

    bull_macro = (
        float(r1["cci100"]) > float(r1["bb_slow_up"])
        and float(r2["cci100"]) > float(r2["bb_slow_up"])
        and float(r3["cci100"]) > float(r3["bb_slow_up"])
    )
    bear_macro = (
        float(r1["cci100"]) < float(r1["bb_slow_lo"])
        and float(r2["cci100"]) < float(r2["bb_slow_lo"])
        and float(r3["cci100"]) < float(r3["bb_slow_lo"])
    )
    if not bull_macro and not bear_macro:
        return "", ACTION_HOLD

    c10 = float(r1["cci10"])
    fu = float(r1["bb_fast_up"])
    fl = float(r1["bb_fast_lo"])
    if bull_macro:
        if c10 < fl:
            return "Buy Pullback", ACTION_BUY
        if c10 < fu:
            return "Buy", ACTION_BUY
        return "", ACTION_HOLD
    if bear_macro:
        if c10 > fu:
            return "Sell Pullback", ACTION_SELL
        if c10 > fl:
            return "Sell", ACTION_SELL
    return "", ACTION_HOLD


def forward_move(close: np.ndarray, t: int, horizon: int, side: int) -> float:
    j = min(t + horizon, len(close) - 1)
    if j <= t:
        return 0.0
    raw = float(close[j] - close[t])
    if side == ACTION_SELL:
        raw = -raw
    return raw - 0.5


def scan_signals(
    days: List[Tuple[str, pd.DataFrame]],
    pack: Dict[str, pd.DataFrame],
    *,
    max_days: int,
    tag: str,
) -> dict:
    label_c: Counter = Counter()
    path_ok: Counter = Counter()
    path_n: Counter = Counter()
    n_dec = n_wave = agree_h = 0
    for date_str, m1 in days[:max_days]:
        m1 = _ensure_vol(m1)
        day = GoalEquityDay(m1, target_pct=2.0, risk_pct=3.0, date_str=str(date_str))
        close = day._close
        # map day local index to global timestamps via m1.index
        for t in day.runner.decision_indices():
            if t >= len(close) - HORIZON_M1:
                continue
            ts = m1.index[t]
            lab, wact = wave_signal_at(pack, ts)
            h = day.recommended_action(t)
            n_dec += 1
            if wact == ACTION_HOLD:
                continue
            n_wave += 1
            label_c[lab or "signal"] += 1
            if wact == h:
                agree_h += 1
            mv = forward_move(close, t, HORIZON_M1, wact)
            key = lab or "signal"
            path_n[key] += 1
            if mv > 0:
                path_ok[key] += 1
    return {
        "window": tag,
        "n_decision_bars": n_dec,
        "n_wave_signals": n_wave,
        "signal_rate": n_wave / max(n_dec, 1),
        "agree_heuristic_rate": agree_h / max(n_wave, 1),
        "labels": dict(label_c),
        "forward_win_rate_by_label": {
            k: path_ok[k] / max(path_n[k], 1) for k in path_n
        },
        "forward_n_by_label": dict(path_n),
        "horizon_m1": HORIZON_M1,
    }


def score_equity(
    days: List[Tuple[str, pd.DataFrame]],
    pack: Dict[str, pd.DataFrame],
    target: float,
    risk: float,
    mode: str,
    max_days: int,
) -> dict:
    cleared = breached = n = entries = 0
    for date_str, m1 in days[:max_days]:
        m1 = _ensure_vol(m1)
        day = GoalEquityDay(
            m1, target_pct=target, risk_pct=risk, date_str=str(date_str)
        )
        indices = day.runner.decision_indices()
        prev_t = 0
        for t in indices:
            if day.dead or day.banked:
                break
            for bt in range(prev_t, t):
                if day.dead or day.banked:
                    break
                day._mark_bar(bt)
            prev_t = t + 1
            if day.dead or day.banked:
                break
            ts = m1.index[t]
            if mode == "heuristic":
                action = day.recommended_action(t)
            else:
                _, wave_act = wave_signal_at(pack, ts)
                if mode == "wave":
                    if day.side is None:
                        action = wave_act
                    else:
                        if day.side > 0 and wave_act == ACTION_SELL:
                            action = ACTION_SELL
                        elif day.side < 0 and wave_act == ACTION_BUY:
                            action = ACTION_BUY
                        else:
                            action = ACTION_HOLD
                else:  # hybrid: flat entry only if wave agrees with heuristic side
                    h = day.recommended_action(t)
                    if day.side is None:
                        if wave_act != ACTION_HOLD and wave_act == h:
                            action = wave_act
                        else:
                            action = ACTION_HOLD  # strict: wave confirms heuristic
                    else:
                        action = h
            day.step_action(t, int(action))
        if not day.dead and not day.banked:
            for bt in range(prev_t, len(day.m1)):
                if day.dead or day.banked:
                    break
                day._mark_bar(bt)
        t_last = len(day.m1) - 1
        day._flatten(float(day._close[t_last]), float(day._spread_px[t_last]))
        pnl = 100.0 * (day.balance - day.eq0) / day.eq0
        if day.breached:
            breached += 1
        elif pnl >= target - 1e-9:
            cleared += 1
        entries += day.n_entries
        n += 1
    return {
        "n_days": n,
        "cleared": cleared,
        "breached": breached,
        "clear_pct": 100.0 * cleared / max(n, 1),
        "breach_pct": 100.0 * breached / max(n, 1),
        "mean_entries": entries / max(n, 1),
        "target": target,
        "risk": risk,
        "mode": mode,
    }


def main() -> None:
    print("=" * 64, flush=True)
    print("WAVE PROBE v2 — full-history warmup (no policy code edits)", flush=True)
    print(f"data={DATA} TFs={TF1}/{TF2}/{TF3}", flush=True)
    print("=" * 64, flush=True)

    print("Loading full M1 + building Wave packs (slow once)...", flush=True)
    m1_full = _ensure_vol(load_raw_m1(DATA))
    pack = build_global_pack(m1_full)
    print(
        f"pack sizes 1m={len(pack[TF1])} 5m={len(pack[TF2])} 15m={len(pack[TF3])} "
        f"15m cci100 finite={int(pack[TF3]['cci100'].notna().sum())}",
        flush=True,
    )

    all_days = load_calendar_days(DATA, min_bars=900)
    practice, forward = split_practice_forward(all_days, practice_n=PRACTICE_N)
    print(f"days total={len(all_days)} practice={len(practice)} forward={len(forward)}", flush=True)

    print("\n[1/3] Signals practice...", flush=True)
    prac_sig = scan_signals(practice, pack, max_days=40, tag="practice")
    print(json.dumps(prac_sig, indent=2), flush=True)

    print("\n[2/3] Signals forward (unseen calendar)...", flush=True)
    fwd_sig = scan_signals(forward, pack, max_days=40, tag="forward")
    print(json.dumps(fwd_sig, indent=2), flush=True)

    print("\n[3/3] Equity shell @ 2.0/3.0 ...", flush=True)
    results = {}
    for mode_key in ("heuristic", "wave", "hybrid"):
        print(f"  {mode_key} practice+forward...", flush=True)
        rp = score_equity(practice, pack, 2.0, 3.0, mode_key, max_days=30)
        rf = score_equity(forward, pack, 2.0, 3.0, mode_key, max_days=30)
        results[mode_key] = {"practice": rp, "forward": rf}
        print(
            f"  {mode_key}: prac clear={rp['cleared']}/{rp['n_days']} breach={rp['breached']} | "
            f"fwd clear={rf['cleared']}/{rf['n_days']} breach={rf['breached']} "
            f"entries/d={rf['mean_entries']:.2f}",
            flush=True,
        )

    print("\n[bonus] forward @ 3.0/3.5 ...", flush=True)
    for mode_key in ("heuristic", "wave", "hybrid"):
        r = score_equity(forward, pack, 3.0, 3.5, mode_key, max_days=25)
        results[f"{mode_key}_3p0_3p5_fwd"] = r
        print(
            f"  {mode_key}: clear={r['cleared']}/{r['n_days']} breach={r['breached']} "
            f"entries/d={r['mean_entries']:.2f}",
            flush=True,
        )

    h_f = results["heuristic"]["forward"]
    w_f = results["wave"]["forward"]
    hy_f = results["hybrid"]["forward"]

    if w_f["breached"] > h_f["breached"]:
        verdict = "HURTS floor vs heuristic on this sample — do not replace my eyes."
    elif w_f["cleared"] > h_f["cleared"] and w_f["breached"] <= h_f["breached"]:
        verdict = "HELPS clear on forward with no extra breach — candidate attention tag."
    elif hy_f["cleared"] > h_f["cleared"] and hy_f["breached"] <= h_f["breached"]:
        verdict = "HYBRID helps more than raw wave — confirm-only filter candidate."
    elif w_f["n_days"] and w_f["mean_entries"] < 0.5:
        verdict = "Wave still too rare / inactive under shell — weak as sole policy."
    else:
        verdict = "NO clear win vs heuristic on this sample — research tag only; shell stays king."

    report = {
        "note": "Offline Wave probe; multi_pair policy code NOT modified",
        "warmup": "full continuous M1 history (required for 15m CCI100)",
        "data": DATA,
        "tfs": [TF1, TF2, TF3],
        "practice_signals": prac_sig,
        "forward_signals": fwd_sig,
        "equity": results,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 64, flush=True)
    print("VERDICT (forward @ 2.0/3.0)", flush=True)
    print(
        f"  heuristic clear {h_f['cleared']}/{h_f['n_days']} breach {h_f['breached']} "
        f"entries/d {h_f['mean_entries']:.2f}",
        flush=True,
    )
    print(
        f"  wave-only  clear {w_f['cleared']}/{w_f['n_days']} breach {w_f['breached']} "
        f"entries/d {w_f['mean_entries']:.2f}",
        flush=True,
    )
    print(
        f"  hybrid     clear {hy_f['cleared']}/{hy_f['n_days']} breach {hy_f['breached']} "
        f"entries/d {hy_f['mean_entries']:.2f}",
        flush=True,
    )
    print(f"  signal_rate practice={prac_sig['signal_rate']:.3f} forward={fwd_sig['signal_rate']:.3f}", flush=True)
    print(f"  agree_heuristic practice={prac_sig['agree_heuristic_rate']:.3f} forward={fwd_sig['agree_heuristic_rate']:.3f}", flush=True)
    print(f"  => {verdict}", flush=True)
    print(f"report={OUT_PATH}", flush=True)
    print("=" * 64, flush=True)


if __name__ == "__main__":
    main()
