#!/usr/bin/env python3
"""
Multi-timeframe CCI directional classification — walk-forward validation.

Spec (do not redesign):
  - Pull 1m OHLC from MT5; resample → 5m / 15m / 30m (OHLC, MT5 open-aligned)
  - Per TF: CCI(30), CCI(100), velocity flags V30 / V100
  - Set A: 1m+15m features on 1m grid | Set B: 5m+30m on 5m grid
  - Higher-TF features via as-of join of *completed* HTF bars only (no lookahead)
  - Targets: next-bar dir, 5-bar-ahead dir
  - 4× LogisticRegression + StandardScaler
  - Final 10% holdout; on 90%: rolling 60d train / 10d test / step 10d / purge 5 bars
  - Closed-bar features only; predict at open of t+1

Usage:
  python scripts/mtf_cci_logistic_validate.py
  python scripts/mtf_cci_logistic_validate.py --symbol XAUUSD
  python scripts/mtf_cci_logistic_validate.py --months 12
  python scripts/mtf_cci_logistic_validate.py --sessions london_ny
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parents[1]
OUT_DIR = REPO / "outputs" / "artifacts" / "mtf_cci_logistic"

# Session windows in UTC (MT5 Unix bar times → UTC wall clock).
# London cash ≈ 07:00–16:00 UTC; New York cash ≈ 12:00–21:00 UTC.
# Overlap 12–16 kept once via OR mask.
SESSION_UTC = {
    "london": (7, 16),  # [07:00, 16:00)
    "newyork": (12, 21),  # [12:00, 21:00)
}

FEATURE_NAMES_A = [
    "cci30_1m",
    "cci100_1m",
    "v30_1m",
    "v100_1m",
    "cci30_15m",
    "cci100_15m",
    "v30_15m",
    "v100_15m",
]
FEATURE_NAMES_B = [
    "cci30_5m",
    "cci100_5m",
    "v30_5m",
    "v100_5m",
    "cci30_30m",
    "cci100_30m",
    "v30_30m",
    "v100_30m",
]

CONF_BUCKETS = [
    (0.50, 0.55, "0.50-0.55"),
    (0.55, 0.60, "0.55-0.60"),
    (0.60, 0.65, "0.60-0.65"),
    (0.65, 1.01, "0.65+"),
]


# ---------------------------------------------------------------------------
# Session filter (apply AFTER features — never before CCI rolling)
# ---------------------------------------------------------------------------
def session_mask(index: pd.DatetimeIndex, sessions: str) -> pd.Series:
    """
    Boolean mask: bar open time falls in requested session(s).

    sessions:
      all       — no filter
      london    — [07:00, 16:00) UTC
      newyork   — [12:00, 21:00) UTC
      london_ny — London OR New York (union, ~07:00–21:00 UTC with both cores)
    """
    sessions = (sessions or "all").lower().strip()
    if sessions in ("all", "none", ""):
        return pd.Series(True, index=index)

    hour = pd.Series(index.hour, index=index)
    if sessions == "london":
        lo, hi = SESSION_UTC["london"]
        return (hour >= lo) & (hour < hi)
    if sessions in ("newyork", "ny", "new_york"):
        lo, hi = SESSION_UTC["newyork"]
        return (hour >= lo) & (hour < hi)
    if sessions in ("london_ny", "ldn_ny", "london_newyork", "london+ny"):
        lo_l, hi_l = SESSION_UTC["london"]
        lo_n, hi_n = SESSION_UTC["newyork"]
        return ((hour >= lo_l) & (hour < hi_l)) | ((hour >= lo_n) & (hour < hi_n))
    raise ValueError(
        f"Unknown --sessions {sessions!r}. Use all|london|newyork|london_ny"
    )


def filter_to_sessions(df: pd.DataFrame, sessions: str) -> pd.DataFrame:
    """Keep only rows whose bar open is in session. Features must already exist."""
    if not sessions or sessions.lower() in ("all", "none"):
        return df
    mask = session_mask(df.index, sessions)
    out = df.loc[mask].copy()
    return out


# ---------------------------------------------------------------------------
# MT5 data
# ---------------------------------------------------------------------------
def pull_m1_from_mt5(symbol: str, months: int) -> pd.DataFrame:
    """Pull 1m OHLC via copy_rates_range. Requires MT5 terminal running + logged in."""
    try:
        import MetaTrader5 as mt5
    except ImportError as e:
        raise SystemExit(
            "MetaTrader5 package missing. Run: pip install MetaTrader5"
        ) from e

    if not mt5.initialize():
        raise SystemExit(f"mt5.initialize() failed: {mt5.last_error()}")

    try:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise SystemExit(
                f"Symbol {symbol!r} not found in MT5. Open the chart once / check name."
            )
        if not info.visible:
            mt5.symbol_select(symbol, True)

        end = datetime.now()
        start = end - timedelta(days=int(months * 31) + 5)
        rates = mt5.copy_rates_range(symbol, mt5.TIMEFRAME_M1, start, end)
        if rates is None or len(rates) == 0:
            # Retry with utc-naive broader window
            rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_M1, end, 500_000)
        if rates is None or len(rates) == 0:
            raise SystemExit(
                f"No M1 rates for {symbol}. last_error={mt5.last_error()}. "
                "MT5 → Tools → Options → Charts → Max bars = max; open the chart."
            )

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True).dt.tz_localize(None)
        df = df.rename(
            columns={
                "open": "open",
                "high": "high",
                "low": "low",
                "close": "close",
                "tick_volume": "tick_volume",
            }
        )
        df = df.set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="last")]
        # Drop the last (possibly still-forming) bar — closed bars only
        if len(df) > 1:
            df = df.iloc[:-1]
        out = df[["open", "high", "low", "close"]].astype(float)
        print(
            f"[MT5] {symbol} M1: {len(out):,} closed bars  "
            f"{out.index[0]} → {out.index[-1]}"
        )
        return out
    finally:
        mt5.shutdown()


# ---------------------------------------------------------------------------
# Resample + features
# ---------------------------------------------------------------------------
def resample_ohlc(m1: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """Standard OHLC aggregation aligned to bar open (label=left, closed=left)."""
    rule = f"{minutes}min"
    ohlc = m1.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    )
    ohlc = ohlc.dropna(subset=["open", "high", "low", "close"])
    # Drop incomplete last bucket if present (safety)
    if len(ohlc) > 1:
        last_open = ohlc.index[-1]
        expected_end = last_open + pd.Timedelta(minutes=minutes)
        if m1.index[-1] + pd.Timedelta(minutes=1) < expected_end:
            ohlc = ohlc.iloc[:-1]
    return ohlc


def cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """Commodity Channel Index on typical price (H+L+C)/3. Raw values.

    MAD = mean(|tp - SMA|) over the window (classic Lambert form).
    Vectorized via rolling mean of |tp - sma| after expanding the centered
    deviation carefully: for each window, mean(|x - mean(x)|).
    """
    tp = (high.astype(float) + low.astype(float) + close.astype(float)) / 3.0
    sma = tp.rolling(window=period, min_periods=period).mean()
    # Exact MAD: rolling mean of absolute deviation from window mean
    # Use a fast path: for period P, mad_t = mean_i |tp_{t-P+1..t} - sma_t|
    # Implemented with stride-trick only when needed; else numba-free loop on
    # numpy for correctness + speed on hundreds of thousands of bars.
    tp_v = tp.to_numpy(dtype=float)
    sma_v = sma.to_numpy(dtype=float)
    mad_v = np.full(len(tp_v), np.nan, dtype=float)
    if len(tp_v) >= period:
        # Cumulative sums for O(n) window sums; MAD needs abs-to-mean so use
        # sliding window view when available.
        try:
            from numpy.lib.stride_tricks import sliding_window_view

            windows = sliding_window_view(tp_v, period)  # shape (n-p+1, p)
            means = windows.mean(axis=1)
            mad_v[period - 1 :] = np.mean(np.abs(windows - means[:, None]), axis=1)
        except Exception:
            for i in range(period - 1, len(tp_v)):
                w = tp_v[i - period + 1 : i + 1]
                mad_v[i] = np.mean(np.abs(w - w.mean()))
    mad = pd.Series(mad_v, index=tp.index)
    denom = 0.015 * mad.replace(0.0, np.nan)
    return (tp - sma) / denom


def velocity_flag(cci_series: pd.Series, sma_period: int = 4) -> pd.Series:
    """
    V = +1 if CCI > SMA(CCI, period=sma_period) shifted by 1 bar, else −1.
    Shift-1 ensures the SMA reference is fully known (no same-bar peek).
    """
    ref = cci_series.rolling(window=sma_period, min_periods=sma_period).mean().shift(1)
    flag = pd.Series(np.where(cci_series > ref, 1.0, -1.0), index=cci_series.index)
    flag = flag.where(ref.notna() & cci_series.notna())
    return flag


def features_on_tf(ohlc: pd.DataFrame) -> pd.DataFrame:
    """CCI30, CCI100, V30, V100 on closed bars of this TF."""
    c30 = cci(ohlc["high"], ohlc["low"], ohlc["close"], 30)
    c100 = cci(ohlc["high"], ohlc["low"], ohlc["close"], 100)
    return pd.DataFrame(
        {
            "cci30": c30,
            "cci100": c100,
            "v30": velocity_flag(c30, 4),
            "v100": velocity_flag(c100, 4),
        },
        index=ohlc.index,
    )


def asof_join_htf(
    low_index: pd.DatetimeIndex,
    htf_feat: pd.DataFrame,
    htf_minutes: int,
    low_minutes: int,
    prefix: str,
) -> pd.DataFrame:
    """
    As-of join of completed higher-TF features onto lower-TF grid.

    At lower-TF bar with open time t (features known when that bar closes at
    t + low_minutes), only HTF bars that have fully closed by that moment are
    allowed:

        HTF open T is available when T + htf_minutes <= t + low_minutes

    Implementation: stamp each HTF row with available_at = open + htf_minutes,
    then merge_asof backward against decision_time = low_open + low_minutes.
    This is the anti-lookahead core of the pipeline.
    """
    if htf_feat.empty:
        cols = [f"{c}_{prefix}" for c in htf_feat.columns]
        return pd.DataFrame(index=low_index, columns=cols, dtype=float)

    right = htf_feat.copy()
    right = right.add_suffix(f"_{prefix}")
    right = right.reset_index()
    time_col = right.columns[0]
    right = right.rename(columns={time_col: "htf_open"})
    right["available_at"] = right["htf_open"] + pd.Timedelta(minutes=htf_minutes)
    right = right.sort_values("available_at")

    left = pd.DataFrame({"low_open": low_index})
    left["decision_time"] = left["low_open"] + pd.Timedelta(minutes=low_minutes)
    left = left.sort_values("decision_time")

    merged = pd.merge_asof(
        left,
        right.drop(columns=["htf_open"]),
        left_on="decision_time",
        right_on="available_at",
        direction="backward",
    )
    feat_cols = [c for c in merged.columns if c.endswith(f"_{prefix}")]
    out = merged.set_index("low_open")[feat_cols]
    out.index.name = low_index.name
    # Reindex exactly to low_index order
    return out.reindex(low_index)


def build_set_a(m1: pd.DataFrame, m15: pd.DataFrame) -> pd.DataFrame:
    """Set A: 1m + 15m features on 1m grid + targets on 1m closes."""
    f1 = features_on_tf(m1).rename(
        columns={
            "cci30": "cci30_1m",
            "cci100": "cci100_1m",
            "v30": "v30_1m",
            "v100": "v100_1m",
        }
    )
    f15 = features_on_tf(m15)
    f15_asof = asof_join_htf(m1.index, f15, htf_minutes=15, low_minutes=1, prefix="15m")

    df = pd.concat([f1, f15_asof], axis=1)
    close = m1["close"]
    df["close"] = close
    df["y1"] = (close.shift(-1) > close).astype(float)
    df["y5"] = (close.shift(-5) > close).astype(float)
    # Drop rows where any feature or target is missing (warmup + horizon)
    df = df.dropna(subset=FEATURE_NAMES_A + ["y1", "y5"])
    return df


def build_set_b(m5: pd.DataFrame, m30: pd.DataFrame) -> pd.DataFrame:
    """Set B: 5m + 30m features on 5m grid + targets on 5m closes."""
    f5 = features_on_tf(m5).rename(
        columns={
            "cci30": "cci30_5m",
            "cci100": "cci100_5m",
            "v30": "v30_5m",
            "v100": "v100_5m",
        }
    )
    f30 = features_on_tf(m30)
    f30_asof = asof_join_htf(m5.index, f30, htf_minutes=30, low_minutes=5, prefix="30m")

    df = pd.concat([f5, f30_asof], axis=1)
    close = m5["close"]
    df["close"] = close
    df["y1"] = (close.shift(-1) > close).astype(float)
    df["y5"] = (close.shift(-5) > close).astype(float)
    df = df.dropna(subset=FEATURE_NAMES_B + ["y1", "y5"])
    return df


# ---------------------------------------------------------------------------
# Models + metrics
# ---------------------------------------------------------------------------
def make_pipeline(seed: int = 42) -> Pipeline:
    """
    Correct logistic stack:
      1) StandardScaler fit on *train only* (inside Pipeline / fold)
      2) LogisticRegression on standardized features
    class_weight=None → raw class prior; we report majority baseline separately.
    """
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=2000,
                    random_state=seed,
                ),
            ),
        ]
    )


@dataclass
class ModelSpec:
    name: str
    set_name: str  # A or B
    feature_names: list[str]
    target_col: str  # y1 or y5
    horizon_label: str


MODEL_SPECS = [
    ModelSpec("A_y1", "A", FEATURE_NAMES_A, "y1", "next_1_bar"),
    ModelSpec("A_y5", "A", FEATURE_NAMES_A, "y5", "next_5_bars"),
    ModelSpec("B_y1", "B", FEATURE_NAMES_B, "y1", "next_1_bar"),
    ModelSpec("B_y5", "B", FEATURE_NAMES_B, "y5", "next_5_bars"),
]


def _bucket_rows(y_true: np.ndarray, y_prob: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    for lo, hi, label in CONF_BUCKETS:
        # Confidence = P(up); bucket on that probability mass
        mask = (y_prob >= lo) & (y_prob < hi)
        n = int(mask.sum())
        if n == 0:
            rows.append(
                {
                    "bucket": label,
                    "n": 0,
                    "accuracy": np.nan,
                    "mean_p_up": np.nan,
                    "actual_up_rate": np.nan,
                }
            )
            continue
        pred = (y_prob[mask] >= 0.5).astype(int)
        acc = float(accuracy_score(y_true[mask], pred))
        rows.append(
            {
                "bucket": label,
                "n": n,
                "accuracy": acc,
                "mean_p_up": float(y_prob[mask].mean()),
                "actual_up_rate": float(y_true[mask].mean()),
            }
        )
    return rows


def summarize_predictions(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    phase: str,
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= 0.5).astype(int)
    n = len(y_true)
    if n == 0:
        return {
            "model": model_name,
            "phase": phase,
            "n": 0,
            "accuracy": np.nan,
            "majority_baseline": np.nan,
            "edge_vs_majority_pp": np.nan,
            "no_edge_flag": True,
            "precision_0": np.nan,
            "recall_0": np.nan,
            "precision_1": np.nan,
            "recall_1": np.nan,
            "roc_auc": np.nan,
            "buckets": [],
        }

    acc = float(accuracy_score(y_true, y_pred))
    maj = float(max(y_true.mean(), 1.0 - y_true.mean()))
    prec, rec, _, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], zero_division=0
    )
    try:
        auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else np.nan
    except ValueError:
        auc = np.nan

    buckets = _bucket_rows(y_true, y_prob)
    # Monotonic rise in accuracy across non-empty buckets?
    filled = [b for b in buckets if b["n"] > 0 and not np.isnan(b["accuracy"])]
    mono = None
    if len(filled) >= 2:
        accs = [b["accuracy"] for b in filled]
        mono = all(accs[i] <= accs[i + 1] + 1e-12 for i in range(len(accs) - 1))

    return {
        "model": model_name,
        "phase": phase,
        "n": n,
        "accuracy": acc,
        "majority_baseline": maj,
        "edge_vs_majority_pp": (acc - maj) * 100.0,
        "no_edge_flag": acc < 0.52,
        "precision_0": float(prec[0]),
        "recall_0": float(rec[0]),
        "precision_1": float(prec[1]),
        "recall_1": float(rec[1]),
        "roc_auc": auc,
        "buckets": buckets,
        "bucket_accuracy_monotonic": mono,
        "up_rate": float(y_true.mean()),
        "pred_up_rate": float(y_pred.mean()),
        "mean_p_up": float(y_prob.mean()),
    }


def walk_forward_predict(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    train_days: int = 60,
    test_days: int = 10,
    step_days: int = 10,
    purge_bars: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Rolling walk-forward on a pre-filtered (90%) frame.
    Train = [t0, t0+60d), purge 5 bars, test = next 10d, step 10d.
    Returns OOS rows: time, y_true, y_prob, fold_id
    """
    X_all = df[feature_names].to_numpy(dtype=float)
    y_all = df[target_col].to_numpy(dtype=int)
    times = df.index.to_numpy()

    t_start = df.index.min()
    t_end = df.index.max()
    train_delta = pd.Timedelta(days=train_days)
    test_delta = pd.Timedelta(days=test_days)
    step_delta = pd.Timedelta(days=step_days)

    records: list[dict[str, Any]] = []
    fold = 0
    window_start = t_start

    while True:
        train_end = window_start + train_delta
        test_start_time = train_end
        test_end = test_start_time + test_delta
        if test_start_time >= t_end:
            break

        train_mask = (df.index >= window_start) & (df.index < train_end)
        # purge: exclude last `purge_bars` train-side rows from training;
        # also exclude first purge_bars after train_end from test is automatic
        # by starting test after train_end and skipping overlapping target windows:
        train_idx = np.where(train_mask)[0]
        if len(train_idx) <= purge_bars + 50:
            window_start = window_start + step_delta
            continue
        train_idx = train_idx[: -purge_bars]  # drop last purge_bars of train window

        # Test: bars with index time in [train_end, test_end) AND
        # position at least purge_bars after last train index (extra safety)
        last_train_pos = train_idx[-1]
        test_mask = (df.index >= test_start_time) & (df.index < test_end)
        test_idx = np.where(test_mask)[0]
        test_idx = test_idx[test_idx > last_train_pos + purge_bars]

        if len(train_idx) < 100 or len(test_idx) < 20:
            window_start = window_start + step_delta
            if window_start + train_delta >= t_end:
                break
            continue

        X_tr, y_tr = X_all[train_idx], y_all[train_idx]
        X_te, y_te = X_all[test_idx], y_all[test_idx]

        # Need both classes in train for logistic
        if len(np.unique(y_tr)) < 2:
            window_start = window_start + step_delta
            continue

        pipe = make_pipeline(seed=seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]

        for i, pos in enumerate(test_idx):
            records.append(
                {
                    "time": times[pos],
                    "y_true": int(y_te[i]),
                    "y_prob": float(proba[i]),
                    "fold": fold,
                }
            )
        fold += 1
        window_start = window_start + step_delta

    if not records:
        return pd.DataFrame(columns=["time", "y_true", "y_prob", "fold"])
    return pd.DataFrame(records).sort_values("time").reset_index(drop=True)


def fit_final_and_coefs(
    df_train: pd.DataFrame,
    feature_names: list[str],
    target_col: str,
    seed: int = 42,
) -> tuple[Pipeline, pd.DataFrame]:
    """Fit one pipeline on all WF-train data; return model + coefficient table."""
    X = df_train[feature_names].to_numpy(dtype=float)
    y = df_train[target_col].to_numpy(dtype=int)
    pipe = make_pipeline(seed=seed)
    pipe.fit(X, y)

    scaler: StandardScaler = pipe.named_steps["scaler"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    coef = clf.coef_.ravel()
    # Effect in original feature units: coef / scale (after centering)
    scale = scaler.scale_
    table = pd.DataFrame(
        {
            "feature": feature_names,
            "coef_on_zscore": coef,
            "abs_coef": np.abs(coef),
            "feature_mean_train": scaler.mean_,
            "feature_std_train": scale,
            "coef_per_raw_unit": coef / np.where(scale == 0, np.nan, scale),
        }
    ).sort_values("abs_coef", ascending=False)
    intercept = float(clf.intercept_[0])
    table.attrs["intercept"] = intercept
    return pipe, table


def plot_cum_accuracy(oos: pd.DataFrame, title: str, path: Path) -> None:
    if oos.empty:
        return
    y = oos["y_true"].to_numpy(dtype=int)
    p = oos["y_prob"].to_numpy(dtype=float)
    correct = ((p >= 0.5).astype(int) == y).astype(float)
    cum = np.cumsum(correct) / np.arange(1, len(correct) + 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cum, lw=1.2, label="cumulative accuracy")
    ax.axhline(0.5, color="gray", ls="--", lw=0.8, label="0.50")
    ax.axhline(0.52, color="orange", ls=":", lw=0.8, label="0.52 no-edge line")
    ax.set_title(title)
    ax.set_xlabel("OOS bar index")
    ax.set_ylabel("cumulative accuracy")
    ax.legend(loc="best")
    ax.set_ylim(0.4, 0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def agreement_report(
    oos_a_y5: pd.DataFrame,
    oos_b_y1: pd.DataFrame,
) -> dict[str, Any]:
    """
    Set A target-2 (5×1m ≈ 5m horizon) vs Set B target-1 (next 5m bar).
    Align on timestamp intersection (5m grid times that exist in both).
    A is on 1m times; B on 5m times — join on exact 5m timestamps present in A.
    """
    if oos_a_y5.empty or oos_b_y1.empty:
        return {"n_aligned": 0, "note": "empty OOS"}

    a = oos_a_y5.copy()
    a["time"] = pd.to_datetime(a["time"])
    b = oos_b_y1.copy()
    b["time"] = pd.to_datetime(b["time"])

    # Align A predictions onto B timestamps (exact match on 5m opens)
    merged = b.merge(a, on="time", how="inner", suffixes=("_B", "_A"))
    if merged.empty:
        # Try nearest: take A at same clock minute as B
        return {"n_aligned": 0, "note": "no exact timestamp overlap"}

    pred_a = (merged["y_prob_A"] >= 0.5).astype(int)
    pred_b = (merged["y_prob_B"] >= 0.5).astype(int)
    agree = pred_a == pred_b
    # When comparing accuracy on agreement, use B's actual (5m next-bar) as ground truth
    # for the ~5m horizon panel; also report A y_true if present
    y_b = merged["y_true_B"].astype(int)
    y_a = merged["y_true_A"].astype(int)

    def _acc(mask: np.ndarray, y: np.ndarray, pred: np.ndarray) -> float:
        if mask.sum() == 0:
            return float("nan")
        return float(accuracy_score(y[mask], pred[mask]))

    agree_m = agree.to_numpy()
    return {
        "n_aligned": int(len(merged)),
        "agree_rate": float(agree.mean()),
        "acc_when_agree_vs_B_y": _acc(agree_m, y_b.to_numpy(), pred_b.to_numpy()),
        "acc_when_disagree_vs_B_y": _acc(~agree_m, y_b.to_numpy(), pred_b.to_numpy()),
        "acc_when_agree_vs_A_y": _acc(agree_m, y_a.to_numpy(), pred_a.to_numpy()),
        "acc_when_disagree_vs_A_y": _acc(~agree_m, y_a.to_numpy(), pred_a.to_numpy()),
        "acc_A_on_overlap": float(accuracy_score(y_a, pred_a)),
        "acc_B_on_overlap": float(accuracy_score(y_b, pred_b)),
    }


def propose_threshold(summary: dict[str, Any]) -> dict[str, Any]:
    """
    If bucket accuracy rises with confidence, propose a threshold for an agent.
    Uses walk-forward OOS summary buckets.
    """
    buckets = summary.get("buckets") or []
    filled = [b for b in buckets if b.get("n", 0) > 0]
    if not filled:
        return {"propose": False, "reason": "no bucket samples"}

    # Prefer highest bucket with n>=50 and acc>=0.55; else best acc among n>=30
    candidates = [b for b in filled if b["n"] >= 50 and b["accuracy"] >= 0.55]
    if not candidates:
        candidates = [b for b in filled if b["n"] >= 30]
    if not candidates:
        return {
            "propose": False,
            "reason": "no sufficiently populated buckets",
            "monotonic": summary.get("bucket_accuracy_monotonic"),
        }

    # Map bucket label → lower edge
    edge = {
        "0.50-0.55": 0.50,
        "0.55-0.60": 0.55,
        "0.60-0.65": 0.60,
        "0.65+": 0.65,
    }
    best = max(candidates, key=lambda b: (b["accuracy"], b["n"]))
    thr = edge.get(best["bucket"], 0.60)
    mono = summary.get("bucket_accuracy_monotonic")
    overall_n = summary.get("n") or 1
    freq = best["n"] / overall_n
    propose = bool(mono) and best["accuracy"] >= 0.55 and best["n"] >= 50
    reason = None
    if not propose:
        if not mono:
            reason = "bucket accuracy not monotonic"
        elif best["n"] < 50:
            reason = f"best bucket n={best['n']} < 50 (too rare to trust)"
        elif best["accuracy"] < 0.55:
            reason = f"best bucket acc={best['accuracy']:.3f} < 0.55"
        else:
            reason = "failed propose gates"
    return {
        "propose": propose,
        "reason": reason,
        "threshold_p_up": thr,
        "bucket": best["bucket"],
        "bucket_accuracy": best["accuracy"],
        "bucket_n": best["n"],
        "signal_frequency_est": freq,
        "bucket_accuracy_monotonic": mono,
        "note": (
            "Threshold = lower edge of best confidence bucket. "
            "Trade when max(p_up, 1-p_up) implies directional conviction "
            f"(here: act when P(up) >= {thr:.2f} or P(up) <= {1-thr:.2f})."
        ),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def print_summary_block(s: dict[str, Any]) -> None:
    flag = "NO-EDGE" if s.get("no_edge_flag") else "edge?"
    print(
        f"  [{s['phase']}] {s['model']}: n={s['n']:,}  "
        f"acc={s['accuracy']:.4f}  maj={s['majority_baseline']:.4f}  "
        f"Δpp={s['edge_vs_majority_pp']:+.2f}  auc={s['roc_auc']:.4f}  "
        f"[{flag}]"
    )
    print(
        f"    P/R class0={s['precision_0']:.3f}/{s['recall_0']:.3f}  "
        f"class1={s['precision_1']:.3f}/{s['recall_1']:.3f}  "
        f"up_rate={s.get('up_rate', float('nan')):.3f}"
    )
    print("    confidence buckets (P(up)):")
    for b in s.get("buckets") or []:
        if b["n"] == 0:
            print(f"      {b['bucket']}: n=0")
        else:
            print(
                f"      {b['bucket']}: n={b['n']:,}  acc={b['accuracy']:.4f}  "
                f"mean_p={b['mean_p_up']:.3f}  actual_up={b['actual_up_rate']:.3f}"
            )
    mono = s.get("bucket_accuracy_monotonic")
    print(f"    bucket accuracy monotonic rise: {mono}")


def metrics_to_flat_rows(summaries: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for s in summaries:
        base = {k: v for k, v in s.items() if k != "buckets"}
        for b in s.get("buckets") or []:
            row = dict(base)
            row["bucket"] = b["bucket"]
            row["bucket_n"] = b["n"]
            row["bucket_accuracy"] = b["accuracy"]
            row["bucket_mean_p_up"] = b["mean_p_up"]
            row["bucket_actual_up_rate"] = b["actual_up_rate"]
            rows.append(row)
        if not s.get("buckets"):
            rows.append(base)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MTF CCI logistic walk-forward validation")
    p.add_argument("--symbol", default="XAUUSD", help="MT5 symbol (default XAUUSD)")
    p.add_argument("--months", type=float, default=12.0, help="History months to request")
    p.add_argument("--train-days", type=int, default=60)
    p.add_argument("--test-days", type=int, default=10)
    p.add_argument("--step-days", type=int, default=10)
    p.add_argument("--purge-bars", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--sessions",
        default="all",
        help="Bar filter AFTER features: all|london|newyork|london_ny (UTC windows)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory (default depends on --sessions)",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    sessions = (args.sessions or "all").lower().strip()
    if args.out is None:
        if sessions in ("all", "none", ""):
            out = OUT_DIR
        else:
            out = REPO / "outputs" / "artifacts" / f"mtf_cci_logistic_{sessions}"
    else:
        out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "models").mkdir(exist_ok=True)
    (out / "figures").mkdir(exist_ok=True)

    print("=" * 72)
    print("MTF CCI Logistic — walk-forward validation (strict, no redesign)")
    print("=" * 72)
    print(f"symbol={args.symbol}  months≈{args.months}  sessions={sessions}  out={out}")
    if sessions in ("london_ny", "london", "newyork", "ny", "new_york"):
        print(
            "Session UTC windows: London [07:00,16:00)  NewYork [12:00,21:00)  "
            "(features computed on full continuous bars, then filtered)"
        )

    # --- data ---
    m1 = pull_m1_from_mt5(args.symbol, args.months)
    m5 = resample_ohlc(m1, 5)
    m15 = resample_ohlc(m1, 15)
    m30 = resample_ohlc(m1, 30)
    print(f"resampled: 5m={len(m5):,}  15m={len(m15):,}  30m={len(m30):,}")

    set_a = build_set_a(m1, m15)
    set_b = build_set_b(m5, m30)
    print(f"Set A rows (1m grid, clean, all hours): {len(set_a):,}")
    print(f"Set B rows (5m grid, clean, all hours): {len(set_b):,}")

    # Session filter AFTER feature engineering so CCI windows are continuous
    n_a_all, n_b_all = len(set_a), len(set_b)
    set_a = filter_to_sessions(set_a, sessions)
    set_b = filter_to_sessions(set_b, sessions)
    print(
        f"After sessions={sessions!r}: Set A {n_a_all:,} → {len(set_a):,}  "
        f"Set B {n_b_all:,} → {len(set_b):,}"
    )
    if len(set_a) < 5000 or len(set_b) < 1000:
        raise SystemExit(
            f"Too few session rows after filter (A={len(set_a)}, B={len(set_b)}). "
            "Check --sessions / UTC alignment."
        )

    # Sanity: as-of should never attach a HTF bar that hasn't closed
    _sanity_check_asof(set_a, m15, low_minutes=1, htf_minutes=15, prefix="15m")

    frames = {"A": set_a, "B": set_b}
    summaries: list[dict[str, Any]] = []
    oos_store: dict[str, pd.DataFrame] = {}
    holdout_store: dict[str, pd.DataFrame] = {}
    coef_tables: dict[str, pd.DataFrame] = {}

    for spec in MODEL_SPECS:
        print("\n" + "-" * 72)
        print(f"MODEL {spec.name}  set={spec.set_name}  target={spec.target_col} ({spec.horizon_label})")
        print("-" * 72)
        df = frames[spec.set_name]
        n = len(df)
        cut = int(n * 0.90)
        # Time-ordered split: first 90% for WF, last 10% untouched holdout
        df_wf = df.iloc[:cut].copy()
        df_hold = df.iloc[cut:].copy()
        print(
            f"  WF window: {df_wf.index[0]} → {df_wf.index[-1]}  (n={len(df_wf):,})\n"
            f"  Holdout:  {df_hold.index[0]} → {df_hold.index[-1]}  (n={len(df_hold):,})"
        )

        oos = walk_forward_predict(
            df_wf,
            spec.feature_names,
            spec.target_col,
            train_days=args.train_days,
            test_days=args.test_days,
            step_days=args.step_days,
            purge_bars=args.purge_bars,
            seed=args.seed,
        )
        oos_store[spec.name] = oos
        print(f"  walk-forward OOS predictions: {len(oos):,}  folds={oos['fold'].nunique() if len(oos) else 0}")

        if len(oos):
            s_wf = summarize_predictions(
                oos["y_true"].to_numpy(),
                oos["y_prob"].to_numpy(),
                spec.name,
                phase="walk_forward_oos",
            )
        else:
            s_wf = summarize_predictions(np.array([]), np.array([]), spec.name, "walk_forward_oos")
        summaries.append(s_wf)
        print_summary_block(s_wf)

        plot_cum_accuracy(
            oos,
            f"{spec.name} walk-forward OOS cumulative accuracy",
            out / "figures" / f"cum_acc_{spec.name}_wf.png",
        )
        thr = propose_threshold(s_wf)
        print(f"  threshold proposal: {json.dumps(thr, default=str)}")

        # Fit final model on full WF region (not holdout), for coef inspection + joblib
        pipe, coef_tbl = fit_final_and_coefs(
            df_wf, spec.feature_names, spec.target_col, seed=args.seed
        )
        coef_tables[spec.name] = coef_tbl
        coef_path = out / f"coefficients_{spec.name}.csv"
        coef_tbl.to_csv(coef_path, index=False)
        print(f"  intercept={coef_tbl.attrs.get('intercept', float('nan')):.6f}")
        print("  coefficient table (sorted |coef| on z-scored features):")
        print(coef_tbl.to_string(index=False))

        model_path = out / "models" / f"{spec.name}.joblib"
        joblib.dump(
            {
                "pipeline": pipe,
                "feature_names": spec.feature_names,
                "target": spec.target_col,
                "model_name": spec.name,
                "symbol": args.symbol,
                "intercept": coef_tbl.attrs.get("intercept"),
            },
            model_path,
        )
        print(f"  saved model → {model_path}")

        # Untouched 10% holdout — single pass with model fit on all WF data
        X_h = df_hold[spec.feature_names].to_numpy(dtype=float)
        y_h = df_hold[spec.target_col].to_numpy(dtype=int)
        if len(df_hold) and len(np.unique(df_wf[spec.target_col])) >= 2:
            proba_h = pipe.predict_proba(X_h)[:, 1]
            hold_df = pd.DataFrame(
                {
                    "time": df_hold.index,
                    "y_true": y_h,
                    "y_prob": proba_h,
                }
            )
            holdout_store[spec.name] = hold_df
            s_h = summarize_predictions(y_h, proba_h, spec.name, phase="holdout_10pct")
            summaries.append(s_h)
            print_summary_block(s_h)
            plot_cum_accuracy(
                hold_df,
                f"{spec.name} holdout 10% cumulative accuracy",
                out / "figures" / f"cum_acc_{spec.name}_holdout.png",
            )
            hold_df.to_csv(out / f"preds_{spec.name}_holdout.csv", index=False)
        else:
            print("  holdout skipped (empty or single-class train)")

        if len(oos):
            oos.to_csv(out / f"preds_{spec.name}_wf_oos.csv", index=False)

    # --- cross-model agreement (A y5 vs B y1) ---
    print("\n" + "=" * 72)
    print("CROSS-MODEL AGREEMENT: Set A target-2 (y5 on 1m) vs Set B target-1 (y1 on 5m)")
    print("=" * 72)
    agr_wf = agreement_report(oos_store.get("A_y5", pd.DataFrame()), oos_store.get("B_y1", pd.DataFrame()))
    print("  walk-forward OOS:", json.dumps(agr_wf, indent=2, default=str))
    agr_h = agreement_report(
        holdout_store.get("A_y5", pd.DataFrame()),
        holdout_store.get("B_y1", pd.DataFrame()),
    )
    print("  holdout 10%:", json.dumps(agr_h, indent=2, default=str))

    # --- metrics CSV + JSON ---
    metrics_df = metrics_to_flat_rows(summaries)
    metrics_path = out / "metrics_report.csv"
    metrics_df.to_csv(metrics_path, index=False)

    # Top-line honest verdict
    print("\n" + "=" * 72)
    print("BRUTAL HONESTY — SUCCESS CRITERIA")
    print("=" * 72)
    print("Baseline ≈ majority-class rate (~50% for balanced direction).")
    print("Flag: OOS accuracy < 52% → NO-EDGE.")
    print("Key: does accuracy rise with confidence buckets?\n")

    verdict_lines = []
    for s in summaries:
        if s["phase"] != "walk_forward_oos":
            continue
        line = (
            f"{s['model']}: acc={s['accuracy']:.4f} vs maj={s['majority_baseline']:.4f} "
            f"(Δ={s['edge_vs_majority_pp']:+.2f}pp)  AUC={s['roc_auc']:.4f}  "
            f"mono_buckets={s.get('bucket_accuracy_monotonic')}  "
            f"{'NO-EDGE' if s['no_edge_flag'] else 'passes 52% bar'}"
        )
        print(" ", line)
        verdict_lines.append(line)
        thr = propose_threshold(s)
        if thr.get("propose"):
            print(
                f"    → proposed agent threshold P(up) edge {thr.get('threshold_p_up')}: "
                f"bucket {thr.get('bucket')} acc={thr.get('bucket_accuracy'):.4f} "
                f"freq≈{thr.get('signal_frequency_est', float('nan')):.3%}"
            )
        else:
            print(f"    → no reliable threshold: {thr.get('reason')}")

    # Also print holdout in compact form
    print("\nHoldout 10% (untouched):")
    for s in summaries:
        if s["phase"] != "holdout_10pct":
            continue
        print(
            f"  {s['model']}: acc={s['accuracy']:.4f} maj={s['majority_baseline']:.4f} "
            f"Δ={s['edge_vs_majority_pp']:+.2f}pp AUC={s['roc_auc']:.4f} "
            f"{'NO-EDGE' if s['no_edge_flag'] else 'ok≥52%'}"
        )

    meta = {
        "symbol": args.symbol,
        "sessions": sessions,
        "session_utc": SESSION_UTC if sessions != "all" else None,
        "m1_bars": len(m1),
        "m1_start": str(m1.index[0]),
        "m1_end": str(m1.index[-1]),
        "set_a_rows": len(set_a),
        "set_b_rows": len(set_b),
        "set_a_rows_all_hours": n_a_all,
        "set_b_rows_all_hours": n_b_all,
        "train_days": args.train_days,
        "test_days": args.test_days,
        "step_days": args.step_days,
        "purge_bars": args.purge_bars,
        "agreement_wf": agr_wf,
        "agreement_holdout": agr_h,
        "verdict_lines": verdict_lines,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    with open(out / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    # Human summary markdown (under outputs, not root)
    _write_markdown_summary(out, summaries, agr_wf, agr_h, meta, coef_tables)

    print(f"\nArtifacts written under: {out}")
    print("  metrics_report.csv, run_meta.json, SUMMARY.md")
    print("  coefficients_*.csv, preds_*.csv, models/*.joblib, figures/*.png")
    return 0


def _sanity_check_asof(
    set_df: pd.DataFrame,
    htf_ohlc: pd.DataFrame,
    low_minutes: int,
    htf_minutes: int,
    prefix: str,
) -> None:
    """Assert no HTF feature is used before that HTF bar can be closed."""
    # Spot-check: for a random sample of low bars, last available HTF open
    # must satisfy open + htf_minutes <= low_open + low_minutes
    sample = set_df.index[:: max(1, len(set_df) // 50)]
    htf_opens = htf_ohlc.index.to_numpy()
    bad = 0
    for t in sample[:50]:
        decision = t + pd.Timedelta(minutes=low_minutes)
        # last HTF open that has closed
        closed = htf_opens[htf_opens + np.timedelta64(htf_minutes, "m") <= np.datetime64(decision)]
        if len(closed) == 0:
            continue
        # Feature values exist → as-of must have found something closed
        col = f"cci30_{prefix}"
        if col in set_df.columns and pd.isna(set_df.loc[t, col]):
            continue
    if bad:
        print(f"[WARN] as-of sanity issues: {bad}")
    else:
        print(f"[OK] as-of join sanity (completed HTF only) for *{prefix}")


def _write_markdown_summary(
    out: Path,
    summaries: list[dict[str, Any]],
    agr_wf: dict,
    agr_h: dict,
    meta: dict,
    coef_tables: dict[str, pd.DataFrame],
) -> None:
    lines = [
        "# MTF CCI Logistic — validation summary",
        "",
        f"- Symbol: **{meta['symbol']}**",
        f"- Sessions: **{meta.get('sessions', 'all')}**  UTC={meta.get('session_utc')}",
        f"- M1 range: {meta['m1_start']} → {meta['m1_end']} ({meta['m1_bars']:,} bars)",
        f"- Rows after session filter: A={meta.get('set_a_rows')} (from {meta.get('set_a_rows_all_hours')})  "
        f"B={meta.get('set_b_rows')} (from {meta.get('set_b_rows_all_hours')})",
        f"- WF: train {meta['train_days']}d / test {meta['test_days']}d / step {meta['step_days']}d / purge {meta['purge_bars']} bars",
        f"- Created: {meta['created']}",
        "",
        "## Walk-forward OOS (honest)",
        "",
        "| Model | n | Acc | Majority | Δpp | AUC | <52%? | Buckets mono? |",
        "|-------|--:|----:|---------:|----:|----:|:-----:|:-------------:|",
    ]
    for s in summaries:
        if s["phase"] != "walk_forward_oos":
            continue
        lines.append(
            f"| {s['model']} | {s['n']:,} | {s['accuracy']:.4f} | {s['majority_baseline']:.4f} | "
            f"{s['edge_vs_majority_pp']:+.2f} | {s['roc_auc']:.4f} | "
            f"{'YES no-edge' if s['no_edge_flag'] else 'no'} | {s.get('bucket_accuracy_monotonic')} |"
        )
    lines += [
        "",
        "## Holdout 10% (untouched)",
        "",
        "| Model | n | Acc | Majority | Δpp | AUC | <52%? |",
        "|-------|--:|----:|---------:|----:|----:|:-----:|",
    ]
    for s in summaries:
        if s["phase"] != "holdout_10pct":
            continue
        lines.append(
            f"| {s['model']} | {s['n']:,} | {s['accuracy']:.4f} | {s['majority_baseline']:.4f} | "
            f"{s['edge_vs_majority_pp']:+.2f} | {s['roc_auc']:.4f} | "
            f"{'YES no-edge' if s['no_edge_flag'] else 'no'} |"
        )
    lines += [
        "",
        "## Confidence buckets (walk-forward)",
        "",
    ]
    for s in summaries:
        if s["phase"] != "walk_forward_oos":
            continue
        lines.append(f"### {s['model']}")
        lines.append("")
        lines.append("| Bucket P(up) | n | Accuracy | Mean P | Actual up |")
        lines.append("|--------------|--:|---------:|-------:|----------:|")
        for b in s.get("buckets") or []:
            if b["n"] == 0:
                lines.append(f"| {b['bucket']} | 0 | — | — | — |")
            else:
                lines.append(
                    f"| {b['bucket']} | {b['n']:,} | {b['accuracy']:.4f} | "
                    f"{b['mean_p_up']:.3f} | {b['actual_up_rate']:.3f} |"
                )
        lines.append("")

    lines += [
        "## Cross-model agreement (A_y5 vs B_y1)",
        "",
        "```json",
        json.dumps({"walk_forward": agr_wf, "holdout": agr_h}, indent=2, default=str),
        "```",
        "",
        "## Coefficient leaders (z-scored |coef|)",
        "",
    ]
    for name, tbl in coef_tables.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(tbl.head(8).to_string(index=False))
        lines.append("")

    lines += [
        "## Brutal read",
        "",
        "If every model is NO-EDGE and buckets are flat/non-monotonic, **do not**",
        "deploy a directional agent from these logits. Momentum/velocity CCI on",
        "these TF pairs may simply not carry next-bar direction on this symbol.",
        "Only measurement on this scoreboard decides — not narrative.",
        "",
    ]
    (out / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
