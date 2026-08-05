#!/usr/bin/env python3
"""Pulse-only (entry bar) direction hits for student-strategy proxies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))

from data_io.loader import align_to_m1, resample as mt5_resample  # noqa: E402


def load_tail(path: Path, max_rows: int = 120_000) -> pd.DataFrame:
    with path.open("rb") as f:
        nlines = sum(1 for _ in f)
    data_rows = nlines - 1
    skip = max(0, data_rows - max_rows)
    raw = pd.read_csv(
        path,
        sep=None,
        engine="python",
        skiprows=range(1, skip + 1) if skip else None,
    )
    raw.columns = [c.strip().strip("<>").upper() for c in raw.columns]
    ts = pd.to_datetime(
        raw["DATE"].astype(str) + " " + raw["TIME"].astype(str),
        format="mixed",
        errors="coerce",
    )
    df = pd.DataFrame(
        {
            "open": pd.to_numeric(raw["OPEN"], errors="coerce").to_numpy(),
            "high": pd.to_numeric(raw["HIGH"], errors="coerce").to_numpy(),
            "low": pd.to_numeric(raw["LOW"], errors="coerce").to_numpy(),
            "close": pd.to_numeric(raw["CLOSE"], errors="coerce").to_numpy(),
            "vol": pd.to_numeric(raw.get("TICKVOL", 1.0), errors="coerce").to_numpy(),
        },
        index=ts,
    ).dropna(subset=["open", "high", "low", "close"])
    return df[~df.index.isna()].sort_index()


def hit_pulse(side_m1: np.ndarray, close: np.ndarray, h: int):
    s = side_m1[:-h]
    ret = close[h:] - close[:-h]
    mask = s != 0
    nf = int(mask.sum())
    if nf < 20:
        return nf, None
    correct = ((s > 0) & (ret > 0)) | ((s < 0) & (ret < 0))
    return nf, float(correct[mask].mean())


def to_pulse(sig_ltf: pd.Series, tf: str, m1_index: pd.DatetimeIndex) -> np.ndarray:
    a = align_to_m1(sig_ltf, tf, m1_index).fillna(0.0)
    prev = a.shift(1).fillna(0.0)
    return a.where((a != 0) & (a != prev), 0.0).to_numpy(dtype=float)


def bb_pulse(m1: pd.DataFrame, tf: str = "5min", n: int = 20, k: float = 2.0) -> np.ndarray:
    ltf = mt5_resample(m1, tf)
    mid = ltf["close"].rolling(n).mean()
    std = ltf["close"].rolling(n).std()
    up, lo = mid + k * std, mid - k * std
    long_e = (ltf["low"] <= lo) & (ltf["close"] > lo)
    short_e = (ltf["high"] >= up) & (ltf["close"] < up)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e] = 1.0
    sig[short_e] = -1.0
    return to_pulse(sig, tf, m1.index)


def sd_pulse(m1: pd.DataFrame, tf: str = "15min") -> np.ndarray:
    ltf = mt5_resample(m1, tf)
    rng = (ltf["high"] - ltf["low"]).replace(0, np.nan)
    atr = rng.rolling(14).mean()
    is_base = rng < (0.6 * atr)
    bull_imp = (ltf["close"] - ltf["open"]) > (1.5 * atr)
    bear_imp = (ltf["open"] - ltf["close"]) > (1.5 * atr)
    base_run = is_base.rolling(3).sum() >= 3
    demand = base_run.shift(1).fillna(False) & bull_imp
    supply = base_run.shift(1).fillna(False) & bear_imp
    d_lo = ltf["low"].shift(1).where(demand).ffill()
    d_hi = ltf["high"].shift(1).where(demand).ffill()
    s_lo = ltf["low"].shift(1).where(supply).ffill()
    s_hi = ltf["high"].shift(1).where(supply).ffill()
    in_d = (ltf["low"] <= d_hi) & (ltf["high"] >= d_lo)
    in_s = (ltf["high"] >= s_lo) & (ltf["low"] <= s_hi)
    long_e = in_d & (ltf["close"] > ltf["open"]) & (ltf["close"] > d_hi)
    short_e = in_s & (ltf["close"] < ltf["open"]) & (ltf["close"] < s_lo)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    return to_pulse(sig, tf, m1.index)


def ob_pulse(m1: pd.DataFrame, tf: str = "15min") -> np.ndarray:
    ltf = mt5_resample(m1, tf)
    atr = (ltf["high"] - ltf["low"]).rolling(14).mean()
    bull = (ltf["close"] - ltf["open"]) > (1.8 * atr)
    bear = (ltf["open"] - ltf["close"]) > (1.8 * atr)
    prev_bear = ltf["close"].shift(1) < ltf["open"].shift(1)
    prev_bull = ltf["close"].shift(1) > ltf["open"].shift(1)
    bull_ob = bull & prev_bear
    bear_ob = bear & prev_bull
    ob_hi = ltf["high"].shift(1).where(bull_ob).ffill()
    sob_lo = ltf["low"].shift(1).where(bear_ob).ffill()
    long_e = (ltf["low"] <= ob_hi) & (ltf["close"] > ob_hi) & (ltf["close"] > ltf["open"])
    short_e = (ltf["high"] >= sob_lo) & (ltf["close"] < sob_lo) & (ltf["close"] < ltf["open"])
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    return to_pulse(sig, tf, m1.index)


def rsi_pulse(m1: pd.DataFrame, tf: str = "5min", period: int = 14) -> np.ndarray:
    ltf = mt5_resample(m1, tf)
    delta = ltf["close"].diff()
    up = delta.clip(lower=0).rolling(period).mean()
    dn = (-delta.clip(upper=0)).rolling(period).mean()
    rsi = 100 - (100 / (1 + up / dn.replace(0, np.nan)))
    long_e = (rsi.shift(1) < 30) & (rsi >= 30)
    short_e = (rsi.shift(1) > 70) & (rsi <= 70)
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    return to_pulse(sig, tf, m1.index)


def fib_pulse(m1: pd.DataFrame, tf: str = "15min", look: int = 30) -> np.ndarray:
    ltf = mt5_resample(m1, tf)
    hh = ltf["high"].rolling(look).max()
    ll = ltf["low"].rolling(look).min()
    rng = (hh - ll).replace(0, np.nan)
    pos = (ltf["close"] - ll) / rng
    mid = (hh + ll) / 2
    uptrend = ltf["close"] > mid
    long_e = uptrend & (pos >= 0.25) & (pos <= 0.50) & (ltf["close"] > ltf["open"])
    short_e = (~uptrend) & (pos >= 0.50) & (pos <= 0.75) & (ltf["close"] < ltf["open"])
    sig = pd.Series(0.0, index=ltf.index)
    sig[long_e.fillna(False)] = 1.0
    sig[short_e.fillna(False)] = -1.0
    return to_pulse(sig, tf, m1.index)


def main() -> int:
    builders = {
        "BB_pulse": bb_pulse,
        "SD_pulse": sd_pulse,
        "OB_pulse": ob_pulse,
        "RSI_rev_pulse": rsi_pulse,
        "Fib_pull_pulse": fib_pulse,
    }
    paths = {
        "EURUSD": ROOT / "data/raw/EURUSD_M1_curriculum.csv",
        "US30": ROOT / "data/raw/US30_M1_curriculum.csv",
        "XAUUSD": ROOT / "data/raw/XAUUSD_curriculum_2026.csv",
        "GBPUSD": ROOT / "data/raw/GBPUSD_M1_curriculum.csv",
    }
    print("PULSE (entry bar only) direction hits @ h=5/10/20 M1 bars")
    print(
        f"{'sym':7} {'proxy':14} {'n5':>6} {'h5':>7} {'n10':>6} {'h10':>7} {'n20':>6} {'h20':>7}"
    )
    for sym, p in paths.items():
        m1 = load_tail(p, 120_000)
        close = m1["close"].to_numpy(float)
        for name, fn in builders.items():
            side = fn(m1)
            row = [sym, name]
            for h in (5, 10, 20):
                nf, hit = hit_pulse(side, close, h)
                row.append(str(nf))
                row.append(f"{hit:.4f}" if hit is not None else "n/a")
            print(
                f"{row[0]:7} {row[1]:14} {row[2]:>6} {row[3]:>7} {row[4]:>6} {row[5]:>7} {row[6]:>6} {row[7]:>7}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
