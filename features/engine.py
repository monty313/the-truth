"""The 4-Set feature engine — senses, states, strategy signals, masks.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-25  gate signal slots via features.yaml include_signal_agent_slots (default false) — WHY: PROVEN_* brains need obs_dim 1820; expanded obs requires new train.
- 2026-07-25  append 500 obs::sig_* slots — WHY: Monty signal agents in observation; empty=0.
- 2026-07-24  align set2/set3 HTFs to Monty lock (5m/1h/4h; 15m/4h/1d) — WHY: exact TF sets. SEMANTIC obs shift; re-prove frozen brains.
- 2026-07-19  masks fail-closed on warmup, event edges, live-line variants, S2 reload to spec  — WHY: audit R1/R2 fidelity + no-look-ahead fixes.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from data_io.loader import resample, align_to_m1
from features import indicators as ind
from telemetry import tracer

SETS = {  # Monty lock 2026-07-24: A=1m/15m/30m B=5m/1h/4h C=15m/4h/1d
    "set1": {"ltf": "1min", "htfs": ["15min", "30min"], "extra": "1h"},
    "set2": {"ltf": "5min", "htfs": ["1h", "4h"], "extra": "1d"},
    "set3": {"ltf": "15min", "htfs": ["4h", "1d"], "extra": "1w"},
    "set4": {"ltf": "30min", "htfs": ["4h", "1d"], "extra": "1w"},
}
ALL_TFS = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w"]
MASK_TFS = ["15min", "30min", "1h"]


def _tf_block(m1: pd.DataFrame, tf: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    o = resample(m1, tf)
    f = pd.DataFrame(index=o.index)
    c = o["close"]
    for p in (30, 100):
        cc = ind.cci(o, p)
        f[f"cci{p}"] = cc
        f[f"cci{p}_line"] = ind.sma_shifted(cc, 2, 2)
    for tag, per in (("wide", 100), ("fast", 10)):
        up, mid, lo = ind.bollinger(c, per, 0.5, 2)
        f[f"bb_{tag}_up"], f[f"bb_{tag}_lo"] = up, lo
        upl, _, lol = ind.bollinger(c, per, 0.5, 1)
        f[f"bb_{tag}_up_live"], f[f"bb_{tag}_lo_live"] = upl, lol
    f["sma50"] = ind.sma(c, 50)
    f["env_hi_s2"], f["env_lo_s2"] = ind.envelope(o, 4, 2)
    f["env_hi_s4"], f["env_lo_s4"] = ind.envelope(o, 4, 4)
    f["env_hi_s4_live"], f["env_lo_s4_live"] = ind.envelope(o, 4, 3)
    for p, tag in ((2, "fast"), (20, "slow")):
        r = ind.rsi(c, p)
        u, m, l = ind.bollinger(r, 20, 0.5, 2)
        f[f"rsi_{tag}"], f[f"rsi_{tag}_up"], f[f"rsi_{tag}_mid"], f[f"rsi_{tag}_lo"] = r, u, m, l
    f["atr14"] = ind.atr(o, 14)
    dju, djm, djl = ind.bollinger(f["cci30"], 20, 1.0, 2)
    f["dj_up"], f["dj_lo"] = dju, djl
    f["mcf_rsi13"] = ind.rsi(c, 13)
    f["close"] = c
    return align_to_m1(f, tf, idx).add_prefix(f"{tf}::")


def _strategy_conditions(F: pd.DataFrame, tf: str, side: int) -> dict[str, pd.Series]:
    g = lambda col: F[f"{tf}::{col}"]
    c = g("close")
    if side == +1:
        s1 = (g("cci30") > g("cci30_line")) & (g("cci100") > g("cci100_line"))
        s2 = (c > g("bb_wide_up")) & (c > g("sma50"))
        s3 = c > g("env_hi_s4")
        s4 = (g("rsi_fast") > g("rsi_fast_up")) | (g("rsi_slow") > g("rsi_slow_mid"))
        reload_ = (c > g("env_hi_s2")) & (g("cci30") > g("cci30_line"))
    else:
        s1 = (g("cci30") < g("cci30_line")) & (g("cci100") < g("cci100_line"))
        s2 = (c < g("bb_wide_lo")) & (c < g("sma50"))
        s3 = c < g("env_lo_s4")
        s4 = (g("rsi_fast") < g("rsi_fast_lo")) | (g("rsi_slow") < g("rsi_slow_mid"))
        reload_ = (c < g("env_lo_s2")) & (g("cci30") < g("cci30_line"))
    return {"S1": s1, "S2": s2, "S3": s3, "S4": s4, "S2_reload": reload_}


def build_features(m1: pd.DataFrame) -> pd.DataFrame:
    idx = m1.index
    blocks = [_tf_block(m1, tf, idx) for tf in ALL_TFS]
    F = pd.concat([m1] + blocks, axis=1)
    F["spread"] = m1["spread"] if "spread" in m1.columns else 0.0

    new = {}
    for sname, spec in SETS.items():
        ltf = spec["ltf"]
        for tag, side in (("buy", +1), ("sell", -1)):
            conds = _strategy_conditions(F, ltf, side)
            for st, ser in conds.items():
                new[f"{sname}::{st}_{tag}"] = ser.astype(np.float32)
            # HTF alignment flags (simple: mid and outer HTF share side pressure)
            for i, htf in enumerate(spec["htfs"]):
                hc = _strategy_conditions(F, htf, side)
                new[f"{sname}::htf{i}_{tag}"] = (hc["S1"] & hc["S2"]).astype(np.float32)

        new_bar = F.index.to_series().diff().dt.total_seconds().fillna(9999) > 30
        for tag in ("buy", "sell"):
            for st in ("S1", "S2", "S3", "S4"):
                col = f"{sname}::{st}_{tag}"
                new[f"{col}_event"] = (new[col].astype(bool) & new_bar).astype(np.float32)
            rc = f"{sname}::S2_reload_{tag}"
            new[f"{rc}_event"] = (new[rc].astype(bool) & new_bar).astype(np.float32)

    for tf in ("15min", "1h", "4h"):
        a = F[f"{tf}::atr14"]
        new[f"obs::{tf}_cci30"] = (F[f"{tf}::cci30"] / 300.0).clip(-3, 3).values
        new[f"obs::{tf}_cci100"] = (F[f"{tf}::cci100"] / 300.0).clip(-3, 3).values
        new[f"obs::{tf}_stretch"] = ((F["close"] - F[f"{tf}::bb_wide_up"]) / a).clip(-5, 5).values
        new[f"obs::{tf}_env_gap"] = ((F["close"] - F[f"{tf}::env_hi_s4"]) / a).clip(-5, 5).values
    new["obs::spread_rel"] = (F["spread"] / F["spread"].rolling(1440, min_periods=60)
                              .median()).clip(0, 6).values
    new["obs::hour_sin"] = np.sin(2 * np.pi * F.index.hour / 24.0)
    new["obs::hour_cos"] = np.cos(2 * np.pi * F.index.hour / 24.0)
    for tf in ("15min", "1h", "4h"):
        new[f"obs::{tf}_dj"] = ((F[f"{tf}::cci30"] - F[f"{tf}::dj_up"]) / 100.0).clip(-5, 5).values
        new[f"obs::{tf}_mcf"] = (F[f"{tf}::mcf_rsi13"] / 100.0).values
        new[f"obs::{tf}_s1_strength"] = (
            ((F[f"{tf}::cci30"] - F[f"{tf}::cci30_line"]) +
             (F[f"{tf}::cci100"] - F[f"{tf}::cci100_line"])) / 600.0).clip(-3, 3).values

    # ---- 500 signal-agent slots (suggestions in obs; empty = 0) ----
    # Gated: configs/features.yaml include_signal_agent_slots
    # OFF = match PROVEN_* brains (obs_dim 1820). ON = +500 cols; retrain required.
    try:
        from core.configs import load as _load_cfg
        _feat = _load_cfg("features") or {}
        _use_sig = bool(_feat.get("include_signal_agent_slots", False))
    except Exception:
        _use_sig = False
    if _use_sig:
        try:
            from signals.encode import append_signal_obs
            append_signal_obs(F, new)
        except Exception:
            pass

    return pd.concat([F, pd.DataFrame(new, index=F.index)], axis=1)


def obs_columns(F: pd.DataFrame) -> list[str]:
    """Observation columns (set signals + obs::* including sig_000..499 + masks)."""
    sig = [c for c in F.columns
           if c.startswith(("set1::", "set2::", "set3::", "set4::"))]
    obs = [c for c in F.columns if c.startswith("obs::")]
    return sig + obs + ["mask_buy_blocked", "mask_sell_blocked"]
