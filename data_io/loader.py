"""M1 data loading, synthetic fallback, resampling, CEST trading calendar.

5W+I -----------------------------------------------------------------
WHO:   Claude for Monty.
WHAT:  (1) read_mt5_m1: parses MT5 M1 CSV exports (DATE TIME OHLC TICKVOL
       VOL SPREAD, tab/comma tolerant). (2) synthetic_m1: gold-like fake
       data for pipeline proof while the real zip is pending (ADR-0010).
       (3) resample: M1 -> any TF, left-labeled, no look-ahead.
       (4) align_to_m1: HTF frame -> M1 timeline via last CLOSED bar.
       (5) trading_days: split an M1 frame into 00:00-CEST day episodes.
WHEN:  2026-07-19 overnight build.
WHERE: consumed by features/engine.py, backtesting/simulator.py.
WHY:   Clean, honest data with exact day boundaries is the foundation;
       look-ahead here would poison everything above it.
INTERCONNECTED WITH: configs/data.yaml, configs/goals.yaml (day tz),
       tests/test_data.py (shuffle/no-look-ahead proofs).
----------------------------------------------------------------------
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from core.configs import load as _cfg

TF_RULE = {"1min": "1min", "5min": "5min", "15min": "15min", "30min": "30min",
           "1h": "1h", "4h": "4h", "1d": "1D", "1w": "W-MON"}
TF_DELTA = {"1min": pd.Timedelta("1min"), "5min": pd.Timedelta("5min"),
            "15min": pd.Timedelta("15min"), "30min": pd.Timedelta("30min"),
            "1h": pd.Timedelta("1h"), "4h": pd.Timedelta("4h"),
            "1d": pd.Timedelta("1D"), "1w": pd.Timedelta("7D")}


def read_mt5_m1(path: str, max_rows: int | None = None) -> pd.DataFrame:
    """MT5 export -> DataFrame[open, high, low, close, vol, spread] @ M1 index.
    Assumes timestamps are broker time ~ CEST family (Monty's day clock)."""
    df = pd.read_csv(path, sep=None, engine="python", nrows=max_rows)
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    ts = pd.to_datetime(df["DATE"].astype(str) + " " + df["TIME"].astype(str),
                        format="mixed", errors="coerce")
    out = pd.DataFrame({
        "open": pd.to_numeric(df["OPEN"], errors="coerce"),
        "high": pd.to_numeric(df["HIGH"], errors="coerce"),
        "low": pd.to_numeric(df["LOW"], errors="coerce"),
        "close": pd.to_numeric(df["CLOSE"], errors="coerce"),
        "vol": pd.to_numeric(df.get("TICKVOL", 1.0), errors="coerce"),
        "spread": pd.to_numeric(df.get("SPREAD", 0), errors="coerce"),
    }, index=ts).dropna(subset=["open", "high", "low", "close"])
    out = out[~out.index.isna()].sort_index()
    # Day-boundary law (ADR-0001: 00:00 CEST; audit S2): broker stamps are
    # EET-family on FTMO-style servers -> convert to the day clock so the
    # episode split, midnight flat and goal window land on Monty's clock.
    # broker_tz is an ASSUMPTION until verified on the real export (flagged).
    d = _cfg("data")
    if d.get("convert_to_day_tz", False):
        try:
            tz_b = d.get("broker_tz", "Europe/Athens")
            tz_d = _cfg("goals").get("day_boundary_tz", "Europe/Berlin")
            out.index = (out.index.tz_localize(tz_b, nonexistent="shift_forward",
                                               ambiguous="NaT")
                         .tz_convert(tz_d).tz_localize(None))
            out = out[~out.index.isna()].sort_index()
        except Exception:
            pass  # conversion is best-effort; audit report flags verification
    return out


def synthetic_m1(days: int = 10, seed: int = 7, start="2026-06-01") -> pd.DataFrame:
    """Gold-like synthetic M1 (trend/chop/shock mix). CLEARLY NOT REAL DATA —
    exists so every pipeline stage is provable before the real zip lands."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 1440, freq="1min")
    n = len(idx)
    regime = np.repeat(rng.choice([0.0, 0.06, -0.05, 0.12], size=max(1, n // 360)),
                       360)[:n]
    ret = regime * 0.22 + rng.normal(0, 0.62, n)   # gold-realistic: ~1-1.5% daily range
    shock = rng.random(n) < 0.0004                       # occasional news shock
    ret[shock] += rng.normal(0, 9.0, shock.sum())
    close = 2400 + np.cumsum(ret)
    spread_pts = np.clip(rng.normal(13, 3, n), 8, 60)     # points (0.01 units)
    high = close + np.abs(rng.normal(0.9, 0.6, n))
    low = close - np.abs(rng.normal(0.9, 0.6, n))
    openp = np.concatenate([[close[0]], close[:-1]])
    high = np.maximum.reduce([high, close, openp])
    low = np.minimum.reduce([low, close, openp])
    return pd.DataFrame({"open": openp, "high": high, "low": low, "close": close,
                         "vol": np.abs(rng.normal(80, 30, n)),
                         "spread": spread_pts}, index=idx)


def resample(m1: pd.DataFrame, tf: str) -> pd.DataFrame:
    """M1 -> native TF bars (left label/closed: bar timestamp = bar OPEN time)."""
    r = TF_RULE[tf]
    o = m1["open"].resample(r, label="left", closed="left").first()
    h = m1["high"].resample(r, label="left", closed="left").max()
    l = m1["low"].resample(r, label="left", closed="left").min()
    c = m1["close"].resample(r, label="left", closed="left").last()
    v = m1["vol"].resample(r, label="left", closed="left").sum()
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "vol": v}).dropna()


def align_to_m1(frame: pd.DataFrame | pd.Series, tf: str,
                m1_index: pd.DatetimeIndex):
    """Map native-TF values onto the M1 timeline using the last CLOSED bar.
    A bar stamped 14:00 (4h) closes at 18:00 — its values become visible at
    18:00, never before. This is the no-look-ahead guarantee."""
    f = frame.copy()
    # Bar labeled at OPEN closes at open+delta; it becomes USABLE on the M1 row
    # whose own close coincides with that close (decision happens at row close),
    # hence -1min (review R1#3/R2#7). 1min TF becomes the identity mapping.
    f.index = f.index + TF_DELTA[tf] - pd.Timedelta("1min")
    return f.reindex(m1_index, method="ffill")


def trading_days(m1: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    """Split into 00:00-broker-clock day episodes (ADR-0001: day = CEST family;
    MT5 exports are already in broker time, so calendar-day split applies)."""
    return [(str(day), g) for day, g in m1.groupby(m1.index.date)
            if len(g) >= 300]   # kills Sunday-stub phantom days (R2#3)
