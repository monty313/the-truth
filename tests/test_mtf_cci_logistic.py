"""Unit tests for as-of join + CCI velocity + logistic pipeline (no MT5)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mtf_cci_logistic_validate import (  # noqa: E402
    asof_join_htf,
    cci,
    features_on_tf,
    filter_to_sessions,
    make_pipeline,
    session_mask,
    velocity_flag,
)


def test_cci_known_values():
    # Flat typical price → MAD=0 → CCI undefined (NaN), not a crash
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="1min")
    close = pd.Series(np.full(n, 100.0), index=idx)
    high = close.copy()
    low = close.copy()
    out = cci(high, low, close, 30)
    assert out.iloc[:29].isna().all()
    assert out.iloc[29:].isna().all()

    # Mildly varying series → finite CCI after warmup
    close2 = pd.Series(100.0 + np.sin(np.linspace(0, 8, n)), index=idx)
    high2 = close2 + 0.2
    low2 = close2 - 0.2
    out2 = cci(high2, low2, close2, 30)
    assert out2.iloc[:29].isna().all()
    assert np.isfinite(out2.iloc[-1])


def test_velocity_uses_shifted_sma():
    idx = pd.date_range("2024-01-01", periods=20, freq="1min")
    # Rising then flat CCI-like series
    s = pd.Series(np.linspace(-10, 10, 20), index=idx)
    v = velocity_flag(s, 4)
    assert v.iloc[:4].isna().all()  # need SMA(4).shift(1)
    # After warmup, flag is ±1
    assert set(v.dropna().unique()).issubset({-1.0, 1.0})


def test_asof_no_lookahead():
    """HTF feature must not appear before HTF bar has closed."""
    # HTF 15m bars at 10:00 and 10:15 with distinct feature values
    htf_idx = pd.to_datetime(["2024-01-01 10:00", "2024-01-01 10:15"])
    htf = pd.DataFrame({"cci30": [111.0, 222.0]}, index=htf_idx)

    # LTF 1m bars from 10:00 through 10:30
    low_idx = pd.date_range("2024-01-01 10:00", periods=31, freq="1min")
    joined = asof_join_htf(low_idx, htf, htf_minutes=15, low_minutes=1, prefix="15m")

    # Bar 10:00 closes at 10:01 → no completed 15m bar yet (10:00 closes 10:15)
    # At low open 10:13: decision_time=10:14 → still no completed HTF (needs 10:15)
    # At low open 10:14: decision_time=10:15 → HTF 10:00 completes → value 111
    # At low open 10:29: decision_time=10:30 → HTF 10:15 completes → value 222
    assert pd.isna(joined.loc[pd.Timestamp("2024-01-01 10:00"), "cci30_15m"])
    assert pd.isna(joined.loc[pd.Timestamp("2024-01-01 10:13"), "cci30_15m"])
    assert joined.loc[pd.Timestamp("2024-01-01 10:14"), "cci30_15m"] == 111.0
    assert joined.loc[pd.Timestamp("2024-01-01 10:15"), "cci30_15m"] == 111.0
    assert joined.loc[pd.Timestamp("2024-01-01 10:28"), "cci30_15m"] == 111.0
    assert joined.loc[pd.Timestamp("2024-01-01 10:29"), "cci30_15m"] == 222.0


def test_logistic_pipeline_scaler_and_coef_shape():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 8))
    # Make feature 0 weakly predictive
    logits = 0.8 * X[:, 0]
    y = (logits + rng.normal(scale=0.5, size=500) > 0).astype(int)
    pipe = make_pipeline(seed=0)
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)[:, 1]
    assert proba.shape == (500,)
    assert pipe.named_steps["clf"].coef_.shape == (1, 8)
    # Feature 0 should have largest |coef| on standardized inputs
    coef = pipe.named_steps["clf"].coef_.ravel()
    assert abs(coef[0]) == max(abs(c) for c in coef)


def test_session_mask_london_ny():
    # One full UTC day of hourly stamps
    idx = pd.date_range("2024-06-03 00:00", periods=24, freq="1h")
    m = session_mask(idx, "london_ny")
    # 07–20 inclusive hours open (07..15 london, 12..20 ny) → hours 7..20 = 14 hours
    assert int(m.sum()) == 14
    assert not m.loc[pd.Timestamp("2024-06-03 06:00")]
    assert m.loc[pd.Timestamp("2024-06-03 07:00")]
    assert m.loc[pd.Timestamp("2024-06-03 15:00")]
    assert m.loc[pd.Timestamp("2024-06-03 20:00")]
    assert not m.loc[pd.Timestamp("2024-06-03 21:00")]

    df = pd.DataFrame({"x": range(24)}, index=idx)
    filtered = filter_to_sessions(df, "london_ny")
    assert len(filtered) == 14


def test_features_on_tf_columns():
    idx = pd.date_range("2024-01-01", periods=200, freq="5min")
    rng = np.random.default_rng(1)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, size=200))
    ohlc = pd.DataFrame(
        {
            "open": close,
            "high": close + 0.3,
            "low": close - 0.3,
            "close": close,
        },
        index=idx,
    )
    f = features_on_tf(ohlc)
    assert list(f.columns) == ["cci30", "cci100", "v30", "v100"]
    # After 100 bars warmup + velocity shift, should have finite rows
    assert f.dropna().shape[0] > 50
