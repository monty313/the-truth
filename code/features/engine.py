"""The 4-Set feature engine — senses, states, strategy signals, masks.

CHANGE LOG (newest first — APPEND here on every edit, with date + WHY;
keep this instruction so we never lose the thread):
- 2026-08-04  resolve SETS from configs (sets_lock mark|proven_legacy) —
  WHY: policy=Mark on chart needs 5m|30m,1h and 15m|1h,4h; PROVEN stays
  on proven_legacy. See POLICY_EQUALS_MARK_ON_CHART.md.
- 2026-07-30  CCI dual masks (5m OR 30m) + dual SMA4+4 (1m AND 15m) OR envelope — WHY: Shell wrong-side open blocks.
- 2026-07-25  RESTORE full engine (masks + S1_perm/trig states) after bad rewrite; gate signal slots via features.yaml — WHY: d6313e9 gutted masks; PROVEN_* need obs_dim 1820.
- 2026-07-25  gate signal slots via features.yaml include_signal_agent_slots (default false) — WHY: PROVEN_* brains need obs_dim 1820; expanded obs requires new train.
- 2026-07-25  append 500 obs::sig_* slots — WHY: Monty signal agents in observation; empty=0.
- 2026-07-24  align set2/set3 HTFs to Monty lock (5m/1h/4h; 15m/4h/1d) — WHY: then-lock; now proven_legacy only.
- 2026-07-19  masks fail-closed on warmup, event edges, live-line variants, S2 reload to spec  — WHY: audit R1/R2 fidelity + no-look-ahead fixes.
# NEXT EDITOR: append your change at the top with date + WHY, and keep this line.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from data_io.loader import resample, align_to_m1
from features import indicators as ind
from telemetry import tracer

# Fallback if configs missing (PROVEN-safe legacy).
_SETS_PROVEN_LEGACY = {
    "set1": {"ltf": "1min", "htfs": ["15min", "30min"], "extra": "1h"},
    "set2": {"ltf": "5min", "htfs": ["1h", "4h"], "extra": "1d"},
    "set3": {"ltf": "15min", "htfs": ["4h", "1d"], "extra": "1w"},
    "set4": {"ltf": "30min", "htfs": ["4h", "1d"], "extra": "1w"},
}
_SETS_MARK = {
    "set1": {"ltf": "1min", "htfs": ["15min", "30min"], "extra": "1h"},
    "set2": {"ltf": "5min", "htfs": ["30min", "1h"], "extra": "4h"},
    "set3": {"ltf": "15min", "htfs": ["1h", "4h"], "extra": "1d"},
    "set4": {"ltf": "30min", "htfs": ["4h", "1d"], "extra": "1w"},
}


def resolve_sets(sets_lock: str | None = None) -> dict:
    """Return the active 4-set matrix.

    sets_lock:
      proven_legacy — PROVEN_* obs (default)
      mark          — Mark-on-chart (new trains only)
    """
    lock = (sets_lock or "proven_legacy").strip().lower()
    try:
        from core.configs import load
        feat = load("features") or {}
        tf = load("timeframes") or {}
        if sets_lock is None:
            lock = str(feat.get("sets_lock", tf.get("default_sets_lock", "proven_legacy"))).strip().lower()
        if lock == "mark":
            raw = tf.get("sets_mark") or _SETS_MARK
        else:
            raw = tf.get("sets_proven_legacy") or tf.get("sets") or _SETS_PROVEN_LEGACY
        # normalize list/tuple htfs
        out = {}
        for k, cfg in raw.items():
            out[k] = {
                "ltf": cfg["ltf"],
                "htfs": list(cfg["htfs"]),
                "extra": cfg.get("extra", cfg["htfs"][-1] if cfg.get("htfs") else "1h"),
            }
        return out
    except Exception:
        return dict(_SETS_MARK if lock == "mark" else _SETS_PROVEN_LEGACY)


# Module-level default = proven_legacy (safe for PROVEN warm-start / prove_it).
SETS = resolve_sets("proven_legacy")
ALL_TFS = ["1min", "5min", "15min", "30min", "1h", "4h", "1d", "1w"]
MASK_TFS = ["15min", "30min", "1h"]       # legacy envelope forever-masks
CCI_MASK_TFS = ["5min", "30min"]          # either TF: dual CCI firm → block opposite open
SMA_GATE_TFS = ["1min", "15min"]          # both TFs: price vs SMA(4)+4 close gate


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
        s1_perm = (g("cci30") > g("cci30_line")) & (g("cci100") > g("cci100_line"))
        s1_trig = (g("cci100") > g("cci100_line")) & (g("cci30") < g("cci30_line"))
        s2_perm = (c > g("bb_wide_up")) & (c > g("bb_fast_up"))
        s2_trig = (c > g("bb_wide_up")) & (c < g("bb_fast_up"))
        s2_reload = (g("sma50") > g("bb_wide_up"))
        s3_perm = (c > g("env_hi_s4")) & (c > g("env_lo_s4"))
        s3_trig = (c > g("env_hi_s2")) & (c > g("env_lo_s2"))
        s4_perm = (g("rsi_fast") > g("rsi_fast_up")) & (g("rsi_slow") > g("rsi_slow_up"))
        s4_trig = (g("rsi_slow") > g("rsi_slow_mid")) & (g("rsi_fast") < g("rsi_fast_lo"))
    else:
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


def build_features(m1: pd.DataFrame, *, sets_lock: str | None = None) -> pd.DataFrame:
    """Build feature frame. sets_lock=None → features.yaml (default proven_legacy)."""
    global SETS
    SETS = resolve_sets(sets_lock)
    idx = m1.index
    with tracer.span(
        "feature_generation",
        rows=len(m1),
        sets_lock=str(sets_lock or "from_config"),
    ):
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
                    new[f"{sname}::{st}_{tag}_x"] = (
                        sig & extra_ok(f"{st}_perm")).astype(np.float32)
                ltf_tf = cfg["ltf"]
                fast_up = F[f"{ltf_tf}::bb_fast_up_live"]
                fast_lo = F[f"{ltf_tf}::bb_fast_lo_live"]
                touch = ((F["low"] <= fast_up) & (F["high"] >= fast_up)) if side == +1 \
                    else ((F["high"] >= fast_lo) & (F["low"] <= fast_lo))
                new[f"{sname}::S2_reload_{tag}"] = (
                    ltf["S2_reload"].fillna(False).values
                    & touch.fillna(False).values).astype(np.float32)
                new[f"{sname}::S2_reload_{tag}_gated"] = (
                    htf_and("S2_perm") & ltf["S2_reload"].fillna(False).values
                    & touch.fillna(False).values).astype(np.float32)
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
        # --- (1) Legacy envelope forever-masks: 15m/30m/1h all above/below ---
        above_all, below_all, nan_env = [], [], F["close"].isna()
        for tf in MASK_TFS:
            hi, lo = F[f"{tf}::env_hi_s4_live"], F[f"{tf}::env_lo_s4_live"]
            nan_env = nan_env | hi.isna() | lo.isna()
            above_all.append(((F["close"] > hi) & (F["close"] > lo)))
            below_all.append(((F["close"] < hi) & (F["close"] < lo)))
        env_sell = np.logical_and.reduce([a.values for a in above_all]) | nan_env.values
        env_buy = np.logical_and.reduce([b.values for b in below_all]) | nan_env.values

        # --- (2) CCI dual mask: BOTH CCI30+100 >0 and each > applied SMA (5m OR 30m)
        #         sell blocked. Mirror <0 and each < SMA → buy blocked. ---
        cci_sell_parts, cci_buy_parts = [], []
        nan_cci = np.zeros(len(F), dtype=bool)
        for tf in CCI_MASK_TFS:
            c30 = F[f"{tf}::cci30"]
            c100 = F[f"{tf}::cci100"]
            l30 = F[f"{tf}::cci30_line"]
            l100 = F[f"{tf}::cci100_line"]
            nan_cci = nan_cci | c30.isna().values | c100.isna().values \
                | l30.isna().values | l100.isna().values
            # both CCIs above 0 AND each above its own SMA line
            bull = ((c30 > 0) & (c30 > l30) & (c100 > 0) & (c100 > l100)).fillna(False).values
            bear = ((c30 < 0) & (c30 < l30) & (c100 < 0) & (c100 < l100)).fillna(False).values
            cci_sell_parts.append(bull)  # no sells in firm bull CCI
            cci_buy_parts.append(bear)   # no buys in firm bear CCI
        cci_sell = np.logical_or.reduce(cci_sell_parts) | nan_cci  # either TF
        cci_buy = np.logical_or.reduce(cci_buy_parts) | nan_cci

        # --- (3) Dual SMA(4)+4 on close: 1m AND 15m both required ---
        # buy blocked if close < SMA on both; sell blocked if close > SMA on both
        below_sma, above_sma = [], []
        nan_sma = np.zeros(len(F), dtype=bool)
        for tf in SMA_GATE_TFS:
            c = F[f"{tf}::close"]
            sma4 = ind.sma_shifted(c, 4, 4)
            nan_sma = nan_sma | c.isna().values | sma4.isna().values
            below_sma.append((c < sma4).fillna(False).values)
            above_sma.append((c > sma4).fillna(False).values)
        sma_buy = np.logical_and.reduce(below_sma) | nan_sma   # both under → no buy
        sma_sell = np.logical_and.reduce(above_sma) | nan_sma  # both over → no sell

        new["mask_sell_blocked"] = (env_sell | cci_sell | sma_sell).astype(np.float32)
        new["mask_buy_blocked"] = (env_buy | cci_buy | sma_buy).astype(np.float32)
        # diagnostics (not in obs_columns — for tests / HUD later)
        new["mask_cci_sell"] = cci_sell.astype(np.float32)
        new["mask_cci_buy"] = cci_buy.astype(np.float32)
        new["mask_sma_sell"] = sma_sell.astype(np.float32)
        new["mask_sma_buy"] = sma_buy.astype(np.float32)

    for sname, cfg in SETS.items():
        ltf_close = F[f"{cfg['ltf']}::close"]
        new_bar = (ltf_close != ltf_close.shift(1)).fillna(False).values
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
