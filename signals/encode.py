"""Encode 500 strategy slots: +1 buy / -1 sell / 0 empty.

CHANGE LOG:
- 2026-07-25  sma_mtf A/B/C mid+outer+agree — WHY: Monty tested SMA multi-TF strategy.
- 2026-07-25  slots 28+ DT/S11/live phases.
- 2026-07-25  Camillion kinds + MO natives.
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


# ---- Monty multi-TF SMA strategy (native) ----
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
}


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
