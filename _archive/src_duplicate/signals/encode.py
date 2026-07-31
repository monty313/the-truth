"""Encode 500 strategy slots: +1 buy / -1 sell / 0 empty.

CHANGE LOG:
- 2026-07-30  wire bb_rsi_sma_agent HANDLERS (slots 90-92 sets A/B/C)
- 2026-07-29  wire dvmr_agent HANDLERS (slot 84 champion 1h+1d)
- 2026-07-25  wire smma_rsi + agree HANDLERS (slots 76-83)
- 2026-07-25  wire rsi2_ema HANDLERS slots 67-75
- 2026-07-25  stoch_ema A/B/C + HTF bias
- 2026-07-25  stoch_mtf; rsi/sma/DT/Camillion/MO
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
import yaml

N_SLOTS = 500
ROOT = Path(__file__).resolve().parents[1]
SLOTS_CFG = ROOT / "configs" / "signal_slots.yaml"
Handler = Callable[[pd.DataFrame], pd.Series]


def signal_column_names() -> list[str]:
    return [f"obs::sig_{i:03d}" for i in range(N_SLOTS)]


def load_slot_config(path: Path | None = None) -> dict:
    path = path or SLOTS_CFG
    if not path.is_file():
        return {"n_slots": N_SLOTS, "filled": {}}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {"n_slots": N_SLOTS, "filled": {}}


def load_filled_slots(path: Path | None = None, *, only_enabled: bool = True) -> dict[int, dict[str, Any]]:
    raw = load_slot_config(path)
    filled = raw.get("filled") or {}
    out: dict[int, dict[str, Any]] = {}
    for k, v in filled.items():
        try:
            idx = int(k)
        except (TypeError, ValueError):
            continue
        if not (0 <= idx < N_SLOTS) or not isinstance(v, dict):
            continue
        if only_enabled and not v.get("enabled", True):
            continue
        out[idx] = v
    return out


def _zero(F: pd.DataFrame) -> pd.Series:
    return pd.Series(0.0, index=F.index, dtype=np.float32)


def _flag_dir(F: pd.DataFrame, buy_col: str, sell_col: str) -> pd.Series:
    buy, sell = F.get(buy_col), F.get(sell_col)
    if buy is None or sell is None:
        return _zero(F)
    out = np.where(buy.to_numpy() > 0, 1.0, np.where(sell.to_numpy() > 0, -1.0, 0.0))
    return pd.Series(out, index=F.index, dtype=np.float32)


def _pull(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _flag_dir(F, f"{set_name}::pull_buy", f"{set_name}::pull_sell")


def _cont(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _flag_dir(F, f"{set_name}::cont_buy", f"{set_name}::cont_sell")


def _rev(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _flag_dir(F, f"{set_name}::rev_buy", f"{set_name}::rev_sell")


def _agree(a: pd.Series, b: pd.Series) -> pd.Series:
    aa, bb = a.to_numpy(), b.to_numpy()
    out = np.where((aa == bb) & (aa != 0), aa, 0.0)
    return pd.Series(out, index=a.index, dtype=np.float32)


def _cci_strength_dir(F: pd.DataFrame, tf: str = "15min") -> pd.Series:
    col = f"obs::{tf}_s1_strength"
    if col not in F.columns:
        c30 = F.get(f"obs::{tf}_cci30")
        if c30 is None:
            return _zero(F)
        v = c30.to_numpy()
        return pd.Series(np.where(v > 0.15, 1.0, np.where(v < -0.15, -1.0, 0.0)), index=F.index, dtype=np.float32)
    v = F[col].to_numpy()
    return pd.Series(np.where(v > 0.1, 1.0, np.where(v < -0.1, -1.0, 0.0)), index=F.index, dtype=np.float32)


def _gravity_proxy(F: pd.DataFrame) -> pd.Series:
    return _agree(_cont(F, "set2"), _cont(F, "set3"))


def _vote_dirs(*series_list) -> np.ndarray:
    mats = [s.to_numpy() for s in series_list]
    n = len(mats[0])
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        ups = sum(1 for m in mats if m[i] > 0)
        dns = sum(1 for m in mats if m[i] < 0)
        if ups > dns:
            out[i] = 1.0
        elif dns > ups:
            out[i] = -1.0
    return out


def _dt_ftmo_proxy(F: pd.DataFrame) -> pd.Series:
    parts = [_cont(F, s) for s in ("set1", "set2", "set3")] + [_pull(F, s) for s in ("set1", "set2", "set3")]
    return pd.Series(_vote_dirs(*parts), index=F.index, dtype=np.float32)


def _s11_cci_proxy(F: pd.DataFrame) -> pd.Series:
    return _cci_strength_dir(F, "15min")


def _s11_pull_proxy(F: pd.DataFrame) -> pd.Series:
    return _pull(F, "set1")


def _s11_m15_proxy(F: pd.DataFrame) -> pd.Series:
    return _cont(F, "set1")


def _phase_cci_align(F: pd.DataFrame) -> pd.Series:
    return _cont(F, "set1")


def _phase_hilo_trend(F: pd.DataFrame) -> pd.Series:
    return _cont(F, "set2")


def _phase_bb_mid(F: pd.DataFrame) -> pd.Series:
    return _cont(F, "set3")


def _phase_sma_stack(F: pd.DataFrame) -> pd.Series:
    return _cont(F, "set2")


def _phase_atr_expand(F: pd.DataFrame) -> pd.Series:
    return _cci_strength_dir(F, "1h")


def _adx_proxy(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _cont(F, set_name)


def _orb_stub(F: pd.DataFrame) -> pd.Series:
    return _zero(F)


def _regime_pulse_trend(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _cont(F, set_name)


def _regime_pulse_pullback(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _pull(F, set_name)


def _cci_surge_trend(F: pd.DataFrame, set_name: str, tf: str) -> pd.Series:
    return _agree(_cont(F, set_name), _cci_strength_dir(F, tf))


def _cci_surge_pullback(F: pd.DataFrame, set_name: str, tf: str) -> pd.Series:
    return _agree(_pull(F, set_name), _cci_strength_dir(F, tf))


def _sma_stack_trend(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _cont(F, set_name)


def _sma_stack_pullback(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _pull(F, set_name)


def _sma_reversion(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _rev(F, set_name)


_SMA_SETS = {
    "A": {"ltf": "1min", "htfs": ["15min", "30min"]},
    "B": {"ltf": "5min", "htfs": ["1h", "4h"]},
    "C": {"ltf": "15min", "htfs": ["4h", "1d"]},
}


def _sma_series(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def _tf_band_from_m1(m1: pd.DataFrame, tf: str, shift: int) -> pd.DataFrame:
    from data_io.loader import resample, align_to_m1
    o = resample(m1, tf)
    hi = _sma_series(o["high"], 4).shift(shift)
    lo = _sma_series(o["low"], 4).shift(shift)
    mid = _sma_series(o["close"], 4).shift(shift)
    band = pd.DataFrame({"hi": hi, "lo": lo, "mid": mid}, index=o.index)
    return align_to_m1(band, tf, m1.index)


def _m1_ohlc(F: pd.DataFrame) -> pd.DataFrame | None:
    need = ("open", "high", "low", "close")
    if not all(c in F.columns for c in need):
        return None
    out = F[list(need)].copy()
    out["vol"] = F["vol"] if "vol" in F.columns else 1.0
    if "spread" in F.columns:
        out["spread"] = F["spread"]
    return out


def _sma_mtf_sig(F: pd.DataFrame, set_key: str, mode: str = "mid") -> pd.Series:
    m1 = _m1_ohlc(F)
    if m1 is None:
        return _zero(F)
    cfg = _SMA_SETS[set_key]
    try:
        ltf = _tf_band_from_m1(m1, cfg["ltf"], 2)
        h1 = _tf_band_from_m1(m1, cfg["htfs"][0], 4)
        h2 = _tf_band_from_m1(m1, cfg["htfs"][1], 4)
    except Exception:
        return _zero(F)
    c = m1["close"]
    htf_up = (c > h1["hi"]) & (c > h1["lo"]) & (c > h2["hi"]) & (c > h2["lo"])
    htf_dn = (c < h1["hi"]) & (c < h1["lo"]) & (c < h2["hi"]) & (c < h2["lo"])
    if mode == "outer":
        buy = (c < ltf["lo"]) & htf_up
        sell = (c > ltf["hi"]) & htf_dn
    else:
        buy = (c < ltf["mid"]) & htf_up
        sell = (c > ltf["mid"]) & htf_dn
    out = np.where(buy.fillna(False), 1.0, np.where(sell.fillna(False), -1.0, 0.0))
    return pd.Series(out, index=F.index, dtype=np.float32)


def _sma_mtf_agree(F: pd.DataFrame, mode: str = "mid") -> pd.Series:
    sigs = [_sma_mtf_sig(F, k, mode).to_numpy() for k in ("A", "B", "C")]
    n = len(F)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        vals = [s[i] for s in sigs if s[i] != 0]
        if len(vals) >= 2 and all(v == vals[0] for v in vals):
            out[i] = vals[0]
    return pd.Series(out, index=F.index, dtype=np.float32)


_RSI_SETS = {
    "A": {"ltf": "1min", "htfs": ["15min", "30min"]},
    "B": {"ltf": "5min", "htfs": ["1h", "4h"]},
    "C": {"ltf": "15min", "htfs": ["4h", "1d"]},
}


def _rsi_bb_tf(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    from data_io.loader import resample, align_to_m1
    from features import indicators as ind
    o = resample(m1, tf)
    r = ind.rsi(o["close"], 14)
    up, mid, lo = ind.bollinger(r, 10, 1.0, shift=5)
    df = pd.DataFrame({"rsi": r, "up": up, "mid": mid, "lo": lo}, index=o.index)
    return align_to_m1(df, tf, m1.index)


def _rsi_mtf_sig(F: pd.DataFrame, set_key: str, mode: str = "combined") -> pd.Series:
    m1 = _m1_ohlc(F)
    if m1 is None:
        return _zero(F)
    cfg = _RSI_SETS[set_key]
    try:
        ltf = _rsi_bb_tf(m1, cfg["ltf"])
        h1 = _rsi_bb_tf(m1, cfg["htfs"][0])
        h2 = _rsi_bb_tf(m1, cfg["htfs"][1])
    except Exception:
        return _zero(F)
    mom_buy = (ltf["rsi"] > ltf["up"]) & (h1["rsi"] > h1["up"]) & (h2["rsi"] > h2["up"])
    mom_sell = (ltf["rsi"] < ltf["lo"]) & (h1["rsi"] < h1["lo"]) & (h2["rsi"] < h2["lo"])
    ltf_x_up = (ltf["rsi"].shift(1) <= ltf["mid"].shift(1)) & (ltf["rsi"] > ltf["mid"])
    ltf_x_dn = (ltf["rsi"].shift(1) >= ltf["mid"].shift(1)) & (ltf["rsi"] < ltf["mid"])
    pb_buy = (h1["rsi"] > h1["up"]) & (h2["rsi"] > h2["up"]) & ltf_x_up
    pb_sell = (h1["rsi"] < h1["lo"]) & (h2["rsi"] < h2["lo"]) & ltf_x_dn
    mom = np.where(mom_buy.fillna(False), 1.0, np.where(mom_sell.fillna(False), -1.0, 0.0))
    pb = np.where(pb_buy.fillna(False), 1.0, np.where(pb_sell.fillna(False), -1.0, 0.0))
    if mode == "momentum":
        out = mom
    elif mode == "pullback":
        out = pb
    else:
        out = np.where(mom != 0, mom, pb)
    return pd.Series(out, index=F.index, dtype=np.float32)


def _rsi_mtf_agree(F: pd.DataFrame) -> pd.Series:
    sigs = [_rsi_mtf_sig(F, k, "combined").to_numpy() for k in ("A", "B", "C")]
    n = len(F)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        vals = [s[i] for s in sigs if s[i] != 0]
        if len(vals) >= 2 and all(v == vals[0] for v in vals):
            out[i] = vals[0]
    return pd.Series(out, index=F.index, dtype=np.float32)


def _rsi_mtf_any(F: pd.DataFrame) -> pd.Series:
    out = np.zeros(len(F), dtype=np.float32)
    for k in ("A", "B", "C"):
        s = _rsi_mtf_sig(F, k, "combined").to_numpy()
        out = np.where(out == 0, s, out)
    return pd.Series(out, index=F.index, dtype=np.float32)


_STOCH_SETS = {
    "A": {"ltf": "1min", "htfs": ["15min", "30min"]},
    "B": {"ltf": "5min", "htfs": ["1h", "4h"]},
    "C": {"ltf": "15min", "htfs": ["4h", "1d"]},
}


def _stochastic(o: pd.DataFrame, k_period=5, d_period=3, slowing=3):
    low_min = o["low"].rolling(k_period, min_periods=k_period).min()
    high_max = o["high"].rolling(k_period, min_periods=k_period).max()
    denom = (high_max - low_min).replace(0.0, np.nan)
    raw_k = 100.0 * (o["close"] - low_min) / denom
    k = raw_k.rolling(slowing, min_periods=slowing).mean()
    d = k.rolling(d_period, min_periods=d_period).mean()
    return k, d


def _stoch_bb_tf(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    from data_io.loader import resample, align_to_m1
    from features import indicators as ind
    o = resample(m1, tf)
    k, d = _stochastic(o, 5, 3, 3)
    up, mid, lo = ind.bollinger(k, 10, 0.5, shift=5)
    df = pd.DataFrame({"k": k, "d": d, "up": up, "mid": mid, "lo": lo}, index=o.index)
    return align_to_m1(df, tf, m1.index)


def _stoch_mtf_sig(F: pd.DataFrame, set_key: str, mode: str = "combined") -> pd.Series:
    if set_key == "C" and mode == "pullback":
        return _zero(F)
    m1 = _m1_ohlc(F)
    if m1 is None:
        return _zero(F)
    cfg = _STOCH_SETS[set_key]
    try:
        ltf = _stoch_bb_tf(m1, cfg["ltf"])
        h1 = _stoch_bb_tf(m1, cfg["htfs"][0])
        h2 = _stoch_bb_tf(m1, cfg["htfs"][1])
    except Exception:
        return _zero(F)

    def bull(s):
        return (s["k"] > s["d"]) & (s["k"] > s["up"])

    def bear(s):
        return (s["k"] < s["d"]) & (s["k"] < s["lo"])

    mom_buy = bull(ltf) & bull(h1) & bull(h2)
    mom_sell = bear(ltf) & bear(h1) & bear(h2)
    pb_buy = bull(h1) & bull(h2) & (ltf["k"] < ltf["lo"])
    pb_sell = bear(h1) & bear(h2) & (ltf["k"] > ltf["up"])
    mom = np.where(mom_buy.fillna(False), 1.0, np.where(mom_sell.fillna(False), -1.0, 0.0))
    pb = np.where(pb_buy.fillna(False), 1.0, np.where(pb_sell.fillna(False), -1.0, 0.0))
    if mode == "momentum":
        out = mom
    elif mode == "pullback":
        out = pb
    else:
        out = mom if set_key == "C" else np.where(mom != 0, mom, pb)
    return pd.Series(out, index=F.index, dtype=np.float32)


def _stoch_mtf_agree(F: pd.DataFrame) -> pd.Series:
    sigs = [
        _stoch_mtf_sig(F, "A", "combined").to_numpy(),
        _stoch_mtf_sig(F, "B", "combined").to_numpy(),
        _stoch_mtf_sig(F, "C", "momentum").to_numpy(),
    ]
    n = len(F)
    out = np.zeros(n, dtype=np.float32)
    for i in range(n):
        vals = [s[i] for s in sigs if s[i] != 0]
        if len(vals) >= 2 and all(v == vals[0] for v in vals):
            out[i] = vals[0]
    return pd.Series(out, index=F.index, dtype=np.float32)


_STOCH_EMA_SETS = {
    "A": {"ltf": "1min", "htfs": ["15min", "30min"]},
    "B": {"ltf": "5min", "htfs": ["1h", "4h"]},
    "C": {"ltf": "15min", "htfs": ["4h", "1d"]},
}


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _stoch_ema_raw_on(o: pd.DataFrame) -> pd.Series:
    k, d = _stochastic(o, 5, 3, 3)
    e8 = _ema(o["close"], 8)
    c = o["close"]
    cross_up = (k.shift(1) <= d.shift(1)) & (k > d)
    cross_dn = (k.shift(1) >= d.shift(1)) & (k < d)
    long_ = cross_up & (k < 40) & (c > e8)
    short_ = cross_dn & (k > 60) & (c < e8)
    out = np.where(long_.fillna(False), 1.0, np.where(short_.fillna(False), -1.0, 0.0))
    return pd.Series(out, index=o.index, dtype=np.float32)


def _stoch_ema_bias(o: pd.DataFrame):
    k, d = _stochastic(o, 5, 3, 3)
    e8 = _ema(o["close"], 8)
    bull = (k > d) & (o["close"] > e8)
    bear = (k < d) & (o["close"] < e8)
    return bull.fillna(False), bear.fillna(False)


def _stoch_ema_htf(F: pd.DataFrame, set_key: str) -> pd.Series:
    m1 = _m1_ohlc(F)
    if m1 is None:
        return _zero(F)
    cfg = _STOCH_EMA_SETS[set_key]
    try:
        from data_io.loader import resample, align_to_m1
        ltf = resample(m1, cfg["ltf"])
        h1 = resample(m1, cfg["htfs"][0])
        h2 = resample(m1, cfg["htfs"][1])
        raw = _stoch_ema_raw_on(ltf)
        b1, s1 = _stoch_ema_bias(h1)
        b2, s2 = _stoch_ema_bias(h2)
        b1a = b1.reindex(ltf.index, method="ffill").fillna(False)
        b2a = b2.reindex(ltf.index, method="ffill").fillna(False)
        s1a = s1.reindex(ltf.index, method="ffill").fillna(False)
        s2a = s2.reindex(ltf.index, method="ffill").fillna(False)
        r = raw.to_numpy()
        long_ok = (r > 0) & b1a.to_numpy() & b2a.to_numpy()
        short_ok = (r < 0) & s1a.to_numpy() & s2a.to_numpy()
        filtered = np.where(long_ok, 1.0, np.where(short_ok, -1.0, 0.0)).astype(np.float32)
        ser = pd.Series(filtered, index=ltf.index, name="s")
        aligned = align_to_m1(ser.to_frame(), cfg["ltf"], m1.index)["s"]
        return aligned.astype(np.float32).reindex(F.index).fillna(0.0)
    except Exception:
        return _zero(F)


KIND_HANDLERS: dict[str, Handler] = {
    "zero": _zero,
    "pull_set1": lambda F: _pull(F, "set1"),
    "pull_set2": lambda F: _pull(F, "set2"),
    "pull_set3": lambda F: _pull(F, "set3"),
    "cont_set1": lambda F: _cont(F, "set1"),
    "cont_set2": lambda F: _cont(F, "set2"),
    "cont_set3": lambda F: _cont(F, "set3"),
    "rev_set1": lambda F: _rev(F, "set1"),
    "rev_set2": lambda F: _rev(F, "set2"),
    "rev_set3": lambda F: _rev(F, "set3"),
    "cam_gravity_30m_4h": _gravity_proxy,
    "cam_regime_pulse_trend_5m_30m": lambda F: _regime_pulse_trend(F, "set2"),
    "cam_regime_pulse_pullback_5m_30m": lambda F: _regime_pulse_pullback(F, "set2"),
    "cam_regime_pulse_trend_30m_4h": lambda F: _regime_pulse_trend(F, "set3"),
    "cam_regime_pulse_pullback_30m_4h": lambda F: _regime_pulse_pullback(F, "set3"),
    "cam_cci_surge_trend_5m_30m": lambda F: _cci_surge_trend(F, "set2", "1h"),
    "cam_cci_surge_pullback_5m_30m": lambda F: _cci_surge_pullback(F, "set2", "1h"),
    "cam_cci_surge_trend_30m_4h": lambda F: _cci_surge_trend(F, "set3", "4h"),
    "cam_cci_surge_pullback_30m_4h": lambda F: _cci_surge_pullback(F, "set3", "4h"),
    "cam_sma_stack_trend_5m_30m": lambda F: _sma_stack_trend(F, "set2"),
    "cam_sma_stack_pullback_5m_30m": lambda F: _sma_stack_pullback(F, "set2"),
    "cam_sma_stack_trend_30m_4h": lambda F: _sma_stack_trend(F, "set3"),
    "cam_sma_stack_pullback_30m_4h": lambda F: _sma_stack_pullback(F, "set3"),
    "cam_sma_reversion_rally_5m_30m": lambda F: _sma_reversion(F, "set2"),
    "cam_sma_reversion_rally_30m_4h": lambda F: _sma_reversion(F, "set3"),
    "cam_orb_ny_breakout": _orb_stub,
    "cam_adx_di_align_5m_30m": lambda F: _adx_proxy(F, "set2"),
    "cam_adx_di_align_30m_4h": lambda F: _adx_proxy(F, "set3"),
    "dt_ftmo_alpha": _dt_ftmo_proxy,
    "s11_cci": _s11_cci_proxy,
    "s11_pull": _s11_pull_proxy,
    "s11_m15": _s11_m15_proxy,
    "phase_cci_align": _phase_cci_align,
    "phase_hilo_trend": _phase_hilo_trend,
    "phase_bb_mid": _phase_bb_mid,
    "phase_sma_stack": _phase_sma_stack,
    "phase_atr_expand": _phase_atr_expand,
    "sma_mtf_A_mid": lambda F: _sma_mtf_sig(F, "A", "mid"),
    "sma_mtf_B_mid": lambda F: _sma_mtf_sig(F, "B", "mid"),
    "sma_mtf_C_mid": lambda F: _sma_mtf_sig(F, "C", "mid"),
    "sma_mtf_A_outer": lambda F: _sma_mtf_sig(F, "A", "outer"),
    "sma_mtf_B_outer": lambda F: _sma_mtf_sig(F, "B", "outer"),
    "sma_mtf_C_outer": lambda F: _sma_mtf_sig(F, "C", "outer"),
    "sma_mtf_agree_mid": lambda F: _sma_mtf_agree(F, "mid"),
    "sma_mtf_agree_outer": lambda F: _sma_mtf_agree(F, "outer"),
    "rsi_mtf_A_momentum": lambda F: _rsi_mtf_sig(F, "A", "momentum"),
    "rsi_mtf_A_pullback": lambda F: _rsi_mtf_sig(F, "A", "pullback"),
    "rsi_mtf_A_combined": lambda F: _rsi_mtf_sig(F, "A", "combined"),
    "rsi_mtf_B_momentum": lambda F: _rsi_mtf_sig(F, "B", "momentum"),
    "rsi_mtf_B_pullback": lambda F: _rsi_mtf_sig(F, "B", "pullback"),
    "rsi_mtf_B_combined": lambda F: _rsi_mtf_sig(F, "B", "combined"),
    "rsi_mtf_C_momentum": lambda F: _rsi_mtf_sig(F, "C", "momentum"),
    "rsi_mtf_C_pullback": lambda F: _rsi_mtf_sig(F, "C", "pullback"),
    "rsi_mtf_C_combined": lambda F: _rsi_mtf_sig(F, "C", "combined"),
    "rsi_mtf_agree": lambda F: _rsi_mtf_agree(F),
    "rsi_mtf_any": lambda F: _rsi_mtf_any(F),
    "stoch_mtf_A_momentum": lambda F: _stoch_mtf_sig(F, "A", "momentum"),
    "stoch_mtf_A_pullback": lambda F: _stoch_mtf_sig(F, "A", "pullback"),
    "stoch_mtf_A_combined": lambda F: _stoch_mtf_sig(F, "A", "combined"),
    "stoch_mtf_B_momentum": lambda F: _stoch_mtf_sig(F, "B", "momentum"),
    "stoch_mtf_B_pullback": lambda F: _stoch_mtf_sig(F, "B", "pullback"),
    "stoch_mtf_B_combined": lambda F: _stoch_mtf_sig(F, "B", "combined"),
    "stoch_mtf_C_momentum": lambda F: _stoch_mtf_sig(F, "C", "momentum"),
    "stoch_mtf_agree": lambda F: _stoch_mtf_agree(F),
    "stoch_ema_A": lambda F: _stoch_ema_htf(F, "A"),
    "stoch_ema_B": lambda F: _stoch_ema_htf(F, "B"),
    "stoch_ema_C": lambda F: _stoch_ema_htf(F, "C"),
}
from signals.rsi2_ema import HANDLERS as _RSI2_HANDLERS
KIND_HANDLERS.update(_RSI2_HANDLERS)
try:
    from signals.smma_rsi import HANDLERS as _SMMA_RSI_HANDLERS
    KIND_HANDLERS.update(_SMMA_RSI_HANDLERS)
except Exception:
    pass
try:
    from signals.agree import HANDLERS as _AGREE_HANDLERS
    KIND_HANDLERS.update(_AGREE_HANDLERS)
except Exception:
    pass
try:
    from signals.dvmr_agent import HANDLERS as _DVMR_HANDLERS
    KIND_HANDLERS.update(_DVMR_HANDLERS)
except Exception:
    pass
try:
    from signals.momentum_vector_agent import HANDLERS as _MV_HANDLERS
    KIND_HANDLERS.update(_MV_HANDLERS)
except Exception:
    pass
try:
    from signals.bb_rsi_sma_agent import HANDLERS as _BB_RSI_SMA_HANDLERS
    KIND_HANDLERS.update(_BB_RSI_SMA_HANDLERS)
except Exception:
    pass


def compute_slot(F: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    kind = (spec.get("kind") or "zero").strip()
    return KIND_HANDLERS.get(kind, _zero)(F).astype(np.float32)


def append_signal_obs(F: pd.DataFrame, new: dict | None = None) -> dict:
    if new is None:
        new = {}
    filled = load_filled_slots(only_enabled=True)
    zeros = np.zeros(len(F), dtype=np.float32)
    for i in range(N_SLOTS):
        col = f"obs::sig_{i:03d}"
        new[col] = compute_slot(F, filled[i]).to_numpy(dtype=np.float32) if i in filled else zeros
    return new


def list_kinds() -> list[str]:
    return sorted(KIND_HANDLERS.keys())
