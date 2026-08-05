"""Curriculum / thrust synthetic days for adaptive_rl_brain_7_31_26 sandbox.

CHANGE LOG:
- 2026-07-31  anti-hold practice data — WHY: plain synthetic_m1 is near-random
  and keeps multi-TF confluence NEUTRAL, so the policy only learns HOLD.
  Thrust legs produce directional CCI/RSI/channel votes (same idea as
  test_live_indicators). Lineage only; never touches PROVEN.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import List, Optional

import numpy as np
import pandas as pd


def thrust_m1_day(
    *,
    n_bars: int = 3600,
    direction: int = 1,
    seed: int = 0,
    start: Optional[datetime] = None,
    session_date: Optional[datetime] = None,
) -> pd.DataFrame:
    """One multi-hour M1 thrust day (default 3600 bars ≈ 60h continuous).

    Long enough that 30m/1h confirmation TFs have >=12 warm bars so Official
    Set 2 can leave NEUTRAL. direction=+1 bull, -1 bear.
    """
    rng = np.random.default_rng(int(seed))
    d = 1 if int(direction) >= 0 else -1
    close = 100.0 if d > 0 else 500.0
    closes = np.empty(n_bars, dtype=float)
    for i in range(n_bars):
        thrust = (i % 10) < 7
        step = d * (1.2 + 0.15 * float(rng.random()))
        if thrust:
            close += step
        else:
            close -= d * (0.45 + 0.1 * float(rng.random()))
        # tiny microstructure noise (does not erase trend)
        close += float(rng.normal(0.0, 0.05))
        closes[i] = close

    if session_date is None:
        session_date = datetime(2026, 6, 1) + timedelta(days=int(seed) % 40)
    if start is None:
        start = session_date.replace(hour=8, minute=0, second=0, microsecond=0)
    idx = pd.date_range(start=start, periods=n_bars, freq="1min")
    c = closes
    high = c + 0.8 + rng.random(n_bars) * 0.4
    low = c - 0.8 - rng.random(n_bars) * 0.4
    open_ = c - d * 0.15
    # ensure OHLC consistency
    high = np.maximum(high, np.maximum(open_, c))
    low = np.minimum(low, np.minimum(open_, c))
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": c,
            "vol": rng.integers(50, 200, size=n_bars),
        },
        index=idx,
    )


def curriculum_days(
    n_days: int,
    *,
    seed: int = 7,
    n_bars: int = 3600,
    pattern: str = "alternate",
) -> List[pd.DataFrame]:
    """Build n directional curriculum days (bull/bear mix).

    pattern:
      alternate — bull, bear, bull, bear, ...
      bull      — all bull
      bear      — all bear
      random    — rng choice per day
    """
    out: List[pd.DataFrame] = []
    rng = np.random.default_rng(int(seed))
    for i in range(int(n_days)):
        if pattern == "bull":
            d = 1
        elif pattern == "bear":
            d = -1
        elif pattern == "random":
            d = 1 if rng.random() < 0.5 else -1
        else:  # alternate
            d = 1 if (i % 2 == 0) else -1
        day = thrust_m1_day(
            n_bars=n_bars,
            direction=d,
            seed=int(seed) + 17 * i + (0 if d > 0 else 3),
            session_date=datetime(2026, 6, 1) + timedelta(days=i),
        )
        out.append(day)
    return out
