"""The 4-Set feature engine — senses, states, strategy signals, masks.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty (ADR-0004, codex/regimes/*).
WHAT:  Builds, on the M1 timeline (no look-ahead):
       - per-TF indicator blocks for every TF the matrix touches
       - S1..S4 buy/sell conditions per set + strengths
       - state flags+strengths (Continuation/Pullback/Reversal) per set
       - FOREVER MASK columns (sell_blocked / buy_blocked)
       - observation matrix for the brain (frame-stacked, normalized,
         goal-conditioned self-state appended by the env at step time)
WHEN:  2026-07-19 overnight build.
WHERE: consumed by backtesting/simulator.py and training/env.py.
WHY:   One engine, shared by sim and (later) live — the identical-
       physics guarantee.
INTERCONNECTED WITH: data_io/loader (align rules), features/indicators
       (MT5 math), configs/features.yaml + timeframes.yaml,
       telemetry spans (feature_generation, state_classification).
----------------------------------------------------------------------

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-07-19  masks fail-closed on warmup, event edges, live-line variants, S2 reload to spec  — WHY: audit R1/R2 fidelity + no-look-ahead fixes.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from data_io.loader import resample, align_to_m1
from features import indicators as ind
from telemetry import tracer

SETS = {  # ADR-0004 (extra-confidence TF listed last, weighted not gating)
    "set1": {"ltf": "1min", "htfs": ["15min", "30min"], "extra": "1h"},
    "set2": {"ltf": "5min", "htfs": ["30min", "1h"], "extra": "4h"},
    "set3": {"ltf": "15min", "htfs": ["1h", "4h"], "extra": "1d"},
    "set4": {"ltf": "30min", "htfs": ["4h", "1d"], "extra": "1w"},
}
ALL_TFS = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w"]
MASK_TFS = ["15min", "30min", "1h"]          # forever-mask chain (ADR-0003)


def _tf_block(m1: pd.DataFrame, tf: str, idx: pd.DatetimeIndex) -> pd.DataFrame:
    """All per-TF raw indicator lines, aligned to M1 by last CLOSED bar."""
    o = resample(m1, tf)
    f = pd.DataFrame(index=o.index)
    c = o["close"]
    # --- S1: CCIs vs their SMA(2,+2) ---
    for p in (30, 100):
        cc = ind.cci(o, p)
        f[f"cci{p}"] = cc
        f[f"cci{p}_line"] = ind.sma_shifted(cc, 2, 2)
    # --- S2: price tunnels + SMA50 ---
    for tag, per in (("wide", 100), ("fast", 10)):
        up, mid, lo = ind.bollinger(c, per, 0.5, 2)
        f[f"bb_{tag}_up"], f[f"bb_{tag}_lo"] = up, lo
        upl, _, lol = ind.bollinger(c, per, 0.5, 1)     # live variant (R1#2)
        f[f"bb_{tag}_up_live"], f[f"bb_{tag}_lo_live"] = upl, lol
    f["sma50"] = ind.sma(c, 50)
    # --- S3 + forever masks: envelopes (LTF +2 / HTF+mask +4) ---
    f["env_hi_s2"], f["env_lo_s2"] = ind.envelope(o, 4, 2)
    f["env_hi_s4"], f["env_lo_s4"] = ind.envelope(o, 4, 4)
    # "_live" = the value the MT5 chart shows AT the forming bar (shift k-1),
    # used ONLY where an M1 price is compared against an HTF line (R1#2)
    f["env_hi_s4_live"], f["env_lo_s4_live"] = ind.envelope(o, 4, 3)
    # --- S4: RSIs with BB on the RSI ---
    for p, tag in ((2, "fast"), (20, "slow")):
        r = ind.rsi(c, p)
        u, m, l = ind.bollinger(r, 20, 0.5, 2)
        f[f"rsi_{tag}"], f[f"rsi_{tag}_up"], f[f"rsi_{tag}_mid"], f[f"rsi_{tag}_lo"] = r, u, m, l
    # --- shared scale + observation-only sauces (Dimension Jump = BB on CCI) ---
    f["atr14"] = ind.atr(o, 14)
    dju, djm, djl = ind.bollinger(f["cci30"], 20, 1.0, 2)
    f["dj_up"], f["dj_lo"] = dju, djl
    f["mcf_rsi13"] = ind.rsi(c, 13)
    f["close"] = c
    return align_to_m1(f, tf, idx).add_prefix(f"{tf}::")


def _strategy_conditions(F: pd.DataFrame, tf: str, side: int) -> dict[str, pd.Series]:
    """Per-TF boolean conditions for each strategy, one side (+1 buy/-1 sell).
    HTF role = 'permission' condition; LTF role = 'trigger' condition."""
    g = lambda col: F[f"{tf}::{col}"]
    c = g("close")
    if side == +1:
        s1_perm = (g("cci30") > g("cci30_line")) & (g("cci100") > g("cci100_line"))
        s1_trig = (g("cci100") > g("cci100_line")) & (g("cci30") < g("cci30_line"))
        s2_perm = (c > g("bb_wide_up")) & (c > g("bb_fast_up"))
        s2_trig = (c > g("bb_wide_up")) & (c < g("bb_fast_up"))
        s2_reload = (g("sma50") > g("bb_wide_up"))
        s3_perm = (c > g("env_hi_s4")) & (c > g("env_lo_s4"))
        s3_trig = (c > g("env_hi_s2")) & (c > g("env_lo_s2"))
        s4_perm = (g("rsi_fast") > g("rsi_fast_up")) & (g("rsi_slow") > g("rsi_slow_up"))
        s4_trig = (g("rsi_slow") > g("rsi_slow_mid")) & (g("rsi_fast") < g("rsi_fast_lo"))
    else:  # exact mathematical inverse (codex law)
        s1_perm = (g("cci30") < g("cci30_line")) & (g("cci100") < g("cci100_line"))
        s1_trig = (g("cci100") < g("cci100_line")) & (g("cci30") > g("cci30_line"))
        s2_perm = (c < g("bb_wide_lo")) & (c < g("bb_fast_lo"))
        s2_trig = (c < g("bb_wide_lo")) & (c > g("bb_fast_lo"))
        s2_reload = (g("sma50") < g("bb_wide_lo"))
        s3_perm = (c < g("env_hi_s4")) & (c < g("env_lo_s4"))
        s3_trig = (c < g("env_hi_s2")) & (c < g("env_lo_s2"))
        s4_perm = (g("rsi_fast") < g("rsi_fast_lo")) & (g("rsi_slow") < g("rsi_slow_lo"))
        s4_trig = (g("rsi_slow") < g("rsi_slow_mid")) & (g("rsi_fast") > g("rsi_fast_up"))
    return {"S1_perm": s1_perm, "S1_trig": s1_trig,
            "S2_perm": s2_perm, "S2_trig": s2_trig, "S2_reload": s2_reload,
            "S3_perm": s3_perm, "S3_trig": s3_trig,
            "S4_perm": s4_perm, "S4_trig": s4_trig}


def build_features(m1: pd.DataFrame) -> pd.DataFrame:
    """Full feature table on the M1 timeline. Columns:
    <tf>::<indicator>, <set>::<strategy>_<buy|sell> signals, states,
    mask_buy_blocked / mask_sell_blocked, plus normalized obs columns."""
    idx = m1.index
    with tracer.span("feature_generation", rows=len(m1)):
        blocks = [_tf_block(m1, tf, idx) for tf in ALL_TFS]
        F = pd.concat([m1[["open", "high", "low", "close", "vol", "spread"]]] + blocks,
                      axis=1)

    new: dict = {}
    with tracer.span("state_classification"):
        for sname, cfg in SETS.items():
            for side, tag in ((+1, "buy"), (-1, "sell")):
                per_tf = {tf: _strategy_conditions(F, tf, side)
                          for tf in [cfg["ltf"]] + cfg["htfs"] + [cfg["extra"]]}
                htf_and = lambda key: np.logical_and.reduce(
                    [per_tf[tf][key].fillna(False).values for tf in cfg["htfs"]])
                extra_ok = lambda key: per_tf[cfg["extra"]][key].fillna(False).values
                ltf = per_tf[cfg["ltf"]]
                for st in ("S1", "S2", "S3", "S4"):
                    sig = htf_and(f"{st}_perm") & ltf[f"{st}_trig"].fillna(False).values
                    new[f"{sname}::{st}_{tag}"] = sig.astype(np.float32)
                    new[f"{sname}::{st}_{tag}_x"] = (          # extra-confidence boost
                        sig & extra_ok(f"{st}_perm")).astype(np.float32)
                # S2 reload (touch of fast band while sma50 filter holds)
                ltf_tf = cfg["ltf"]
                fast_up = F[f"{ltf_tf}::bb_fast_up_live"]
                fast_lo = F[f"{ltf_tf}::bb_fast_lo_live"]
                touch = ((F["low"] <= fast_up) & (F["high"] >= fast_up)) if side == +1 \
                    else ((F["high"] >= fast_lo) & (F["low"] <= fast_lo))
                # spec (codex S2): SMA50 filter is the ONLY gate on reloads (R1#4)
                new[f"{sname}::S2_reload_{tag}"] = (
                    ltf["S2_reload"].fillna(False).values
                    & touch.fillna(False).values).astype(np.float32)
                new[f"{sname}::S2_reload_{tag}_gated"] = (
                    htf_and("S2_perm") & ltf["S2_reload"].fillna(False).values
                    & touch.fillna(False).values).astype(np.float32)
                # states: continuation / pullback / reversal + strength
                cont = htf_and("S1_perm") & per_tf[cfg["ltf"]]["S1_perm"].fillna(False).values
                pull = htf_and("S1_perm") & ltf["S1_trig"].fillna(False).values
                new[f"{sname}::cont_{tag}"] = cont.astype(np.float32)
                new[f"{sname}::pull_{tag}"] = pull.astype(np.float32)
            hb = pd.Series(new[f"{sname}::cont_buy"], index=F.index)
            hs = pd.Series(new[f"{sname}::cont_sell"], index=F.index)
            side_now = pd.Series(np.where(hb > 0, 1.0, np.where(hs > 0, -1.0, np.nan)),
                                 index=F.index).ffill()
            new[f"{sname}::rev_buy"] = ((side_now.shift(1) < 0) & (side_now > 0)).astype(np.float32).values
            new[f"{sname}::rev_sell"] = ((side_now.shift(1) > 0) & (side_now < 0)).astype(np.float32).values

    with tracer.span("mask_check", stage="precompute"):
        # FOREVER MASKS (ADR-0003), v2 after review R1#1/#8: warmup NaN on ANY
        # of the six lines (or price) fail-closes BOTH masks; live-line variant.
        above_all, below_all, nan_any = [], [], F["close"].isna()
        for tf in MASK_TFS:
            hi, lo = F[f"{tf}::env_hi_s4_live"], F[f"{tf}::env_lo_s4_live"]
            nan_any = nan_any | hi.isna() | lo.isna()
            above_all.append(((F["close"] > hi) & (F["close"] > lo)))
            below_all.append(((F["close"] < hi) & (F["close"] < lo)))
        nan_v = nan_any.values
        new["mask_sell_blocked"] = (np.logical_and.reduce(
            [a.values for a in above_all]) | nan_v).astype(np.float32)
        new["mask_buy_blocked"] = (np.logical_and.reduce(
            [b.values for b in below_all]) | nan_v).astype(np.float32)

    # ---- event edges: a signal is FRESH only on the first M1 row where a newly
    # closed LTF bar became visible (R2#4). Consumers wanting entries use _event.
    for sname, cfg in SETS.items():
        ltf_close = F[f"{cfg['ltf']}::close"]
        new_bar = (ltf_close != ltf_close.shift(1)).fillna(False).values
        for tag in ("buy", "sell"):
            for st in ("S1", "S2", "S3", "S4"):
                col = f"{sname}::{st}_{tag}"
                new[f"{col}_event"] = (new[col].astype(bool) & new_bar).astype(np.float32)
            rc = f"{sname}::S2_reload_{tag}"
            new[f"{rc}_event"] = (new[rc].astype(bool) & new_bar).astype(np.float32)

    # ---- tension / distance features (normalized senses, no raw prices) ----
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
    for tf in ("15min", "1h", "4h"):   # sauces observation-only (R1#10) + strengths
        new[f"obs::{tf}_dj"] = ((F[f"{tf}::cci30"] - F[f"{tf}::dj_up"]) / 100.0).clip(-5, 5).values
        new[f"obs::{tf}_mcf"] = (F[f"{tf}::mcf_rsi13"] / 100.0).values
        new[f"obs::{tf}_s1_strength"] = (
            ((F[f"{tf}::cci30"] - F[f"{tf}::cci30_line"]) +
             (F[f"{tf}::cci100"] - F[f"{tf}::cci100_line"])) / 600.0).clip(-3, 3).values
    return pd.concat([F, pd.DataFrame(new, index=F.index)], axis=1)


def obs_columns(F: pd.DataFrame) -> list[str]:
    """Observation columns fed to the brain (signals + states + masks + senses)."""
    sig = [c for c in F.columns
           if c.startswith(("set1::", "set2::", "set3::", "set4::"))]
    obs = [c for c in F.columns if c.startswith("obs::")]
    return sig + obs + ["mask_buy_blocked", "mask_sell_blocked"]
