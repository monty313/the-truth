"""Encode up to 500 strategy slots into observation columns obs::sig_000..sig_499.

Values: +1 buy, -1 sell, 0 empty/flat.
Camillion alphas mapped as kinds (proxied onto Momentum One Gravity flags).

CHANGE LOG:
- 2026-07-25  Camillion alpha pack kinds + MO natives — WHY: fill slots from Camillion_Claude_RL_model.
- 2026-07-25  created — WHY: Monty 500-slot advisor obs; unfilled slots stay 0.
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
    """Camillion gravity_30m_4h: set2 cont agree set3 cont."""
    return _agree(_cont(F, "set2"), _cont(F, "set3"))


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


def _orb_stub(F: pd.DataFrame) -> pd.Series:
    return _zero(F)


def _adx_proxy(F: pd.DataFrame, set_name: str) -> pd.Series:
    return _cont(F, set_name)


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
}


def compute_slot(F: pd.DataFrame, spec: dict[str, Any]) -> pd.Series:
    kind = (spec.get("kind") or "zero").strip()
    handler = KIND_HANDLERS.get(kind, _zero)
    return handler(F).astype(np.float32)


def append_signal_obs(F: pd.DataFrame, new: dict | None = None) -> dict:
    if new is None:
        new = {}
    filled = load_filled_slots(only_enabled=True)
    n = len(F)
    zeros = np.zeros(n, dtype=np.float32)
    for i in range(N_SLOTS):
        col = f"obs::sig_{i:03d}"
        if i in filled:
            new[col] = compute_slot(F, filled[i]).to_numpy(dtype=np.float32)
        else:
            new[col] = zeros
    return new


def list_kinds() -> list[str]:
    return sorted(KIND_HANDLERS.keys())
